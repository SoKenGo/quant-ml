# flows/prefect_nightly.py
import os, subprocess, datetime as dt, json
import pandas as pd
from prefect import flow, task
from dotenv import load_dotenv
from src.utils.r2_client import _fs, s3url, list_keys

load_dotenv(override=True)

# ----------------------------
# Config (env-overridable)
# ----------------------------
def _env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]

SYMS      = _env_list("SYMBOLS", "TSLA,RKLB,NVDA")
YEARS     = _env_list("YEARS", "2018,2019,2020,2021,2022,2023,2024,2025")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")            # <- key change
RTH_ONLY  = os.getenv("RTH_ONLY", "1") == "1"       # enforce US/Eastern RTH in ingest/BT
DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")    # iex|sip (if your plan allows)
PARAM_CFG = os.getenv("PARAM_CFG", "src/config/params.yaml")
SYMS_CFG  = os.getenv("SYMS_CFG", "src/config/symbols.yaml")

AUTO_TUNE = os.getenv("AUTO_TUNE", "0") == "1"
N_TRIALS  = int(os.getenv("TUNE_TRIALS", "40"))

def sh(cmd: str):
    print("→", cmd)
    subprocess.run(cmd, shell=True, check=True)

def _read_tuned_params_if_any(symbol: str):
    # try R2 first
    try:
        keys = list_keys(f"models/xgb_params/symbol={symbol}")
        if any(k.endswith("params.json") for k in keys):
            fs = _fs()
            with fs.open(s3url(f"models/xgb_params/symbol={symbol}/params.json"), "rb") as f:
                return json.load(f)
    except Exception:
        pass
    # fallback: local params
    lp = f"models/xgb_params/symbol={symbol}/params.json"
    if os.path.exists(lp):
        return json.load(open(lp))
    return None

# ----------------------------
# PIPELINE STEPS
# ----------------------------

@task
def ingest():
    # Replace Tiingo daily with Alpaca hourly (partitioned Parquet to R2)
    yrs = " ".join(YEARS)
    for s in SYMS:
        flags = f"--symbol {s} --years {yrs} --timeframe {TIMEFRAME} --data-feed {DATA_FEED}"
        if RTH_ONLY:
            flags += " --rth-only"
        # expects: src/ingest/alpaca_h1_to_r2.py (CLI-compatible)
        sh(f"python -m src.ingest.alpaca_h1_to_r2 {flags} --push-to-r2")

@task
def make_features():
    # Hourly feature set (current year only to keep nightly fast)
    yrs = " ".join(YEARS[-1:])
    # expects: features.make_features_h1_to_cloud (to be added/updated)
    sh(
        f"python -m features.make_features_h1_to_cloud "
        f"--symbol ALL --symbols-cfg {SYMS_CFG} --years {yrs} --timeframe {TIMEFRAME} --push-to-r2"
    )

@task
def make_labels():
    # Hourly triple-barrier labels; ATR scaling for 1h handled in the module
    yrs = " ".join(YEARS[-1:])
    # expects: features.make_labels_h1_to_cloud (to be added/updated)
    sh(
        f"python -m features.make_labels_h1_to_cloud "
        f"--symbol ALL --symbols-cfg {SYMS_CFG} --years {yrs} "
        f"--timeframe {TIMEFRAME} --pt-mult 2.0 --sl-mult 1.0 --max-holding 10 --push-to-r2"
    )

@task
def regime_and_aux():
    yrs = " ".join(YEARS)
    # HMM regime on QQQ (timeframe-aware)
    # expects: src.regime.hmm to accept --timeframe (backward-compatible if ignored)
    flags = f"--years {yrs} --timeframe {TIMEFRAME} --push-to-r2"
    sh(f"python -m src.regime.hmm {flags}")

    # Kronos + mlforecast per symbol (guarded: keep existing behavior, don't fail the flow)
    for s in SYMS:
        try:
            sh(f"python -m src.alpha.kronos_signal --symbol {s} --years {yrs} --timeframe {TIMEFRAME} --push-to-r2")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] kronos_signal failed for {s}: {e}")
        try:
            sh(f"python -m src.models.mlf_train --symbol {s} --years {yrs} --timeframe {TIMEFRAME} --push-to-r2")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] mlf_train missing or failed for {s}: {e}")

@task
def train_all():
    yrs = " ".join(YEARS)
    for s in SYMS:
        print(f"== Train {s} ==")
        tuned = {}
        if AUTO_TUNE:
            # expects: src.models.tune_xgb to accept timeframe & intraday folds
            sh(
                f"python -m src.models.tune_xgb --symbol {s} --years {yrs} --timeframe {TIMEFRAME} "
                f"--n-trials {N_TRIALS} --min-train-days 252 --min-test-days 30 --drop-zero-labels"
            )
            tuned = _read_tuned_params_if_any(s) or {}
        else:
            tuned = _read_tuned_params_if_any(s) or {}

        flags = ""
        for k, cli in dict(
            n_estimators="--n-estimators",
            max_depth="--max-depth",
            learning_rate="--learning-rate",
            subsample="--subsample",
            colsample="--colsample",
        ).items():
            if k in tuned:
                flags += f" {cli} {tuned[k]}"

        # Walk-forward (hourly), dual calibration + auto class-weight
        # expects: src.models.xgb_walkforward to accept --timeframe
        sh(
            f"python -m src.models.xgb_walkforward --symbol {s} --years {yrs} --timeframe {TIMEFRAME} "
            f"--min-train-days 252 --min-test-days 30 --push-to-r2 "
            f"--scale-pos-weight auto --calibration-dual{flags}"
        )

@task
def backtest_all(run_id: str):
    yrs = " ".join(YEARS)
    for s in SYMS:
        print(f"== Backtest {s} ==")
        try:
            # Event-driven 1h, bar-close → next bar open; costs & slippage; RTH and flatten EOD
            sh(
              f"python -m src.backtest.engine --symbol {s} --years {yrs} "
              f"--timeframe {TIMEFRAME} --rth-only --flatten-eod "
              f"--run-id {run_id} --commission-per-share 0.0038 --slippage-bps 5 "
              f"--use-param-cfg {PARAM_CFG} --use-next-open "
              f"--qqq-regime --hmm-regime --push-to-r2"
            )
        except subprocess.CalledProcessError as e:
            print(f"[WARN] backtest failed for {s}: {e}")

@task
def run_portfolio_report(run_id: str):
    # Generates portfolio_<RUN_ID>.{csv,json} and summary_<RUN_ID>.csv under backtests/analysis/
    # HRP/Target-Vol can be evaluated daily even when signals are 1h (report handles resample)
    sh(f"python -m src.reporting.eval_suite --run-id {run_id} --enable-hrp --enable-target-vol --target-vol 0.10")

@task
def push_reports_to_r2(run_id: str):
    fs = _fs()
    # push portfolio csv/json
    for ext in ("csv","json"):
        lp = f"backtests/analysis/portfolio_{run_id}.{ext}"
        if os.path.exists(lp):
            with open(lp, "rb") as fin, fs.open(s3url(f"backtests/analysis/portfolio_{run_id}.{ext}"), "wb") as fout:
                fout.write(fin.read())
            print(f"Pushed {ext.upper()} to R2: {s3url(f'backtests/analysis/portfolio_{run_id}.{ext}')}")
    # push summary csv (produced by eval_suite)
    summ_lp = f"backtests/analysis/summary_{run_id}.csv"
    if os.path.exists(summ_lp):
        with open(summ_lp, "rb") as fin, fs.open(s3url(f"backtests/analysis/summary_{run_id}.csv"), "wb") as fout:
            fout.write(fin.read())
        print(f"Pushed summary to R2: {s3url(f'backtests/analysis/summary_{run_id}.csv')}")

@task
def print_top5_from_summary(run_id: str):
    path = f"backtests/analysis/summary_{run_id}.csv"
    if not os.path.exists(path):
        print(f"[WARN] summary not found: {path}")
        return
    df = pd.read_csv(path)
    if "sharpe" in df.columns:
        top5 = df.sort_values("sharpe", ascending=False).head(5)[["symbol","sharpe","cagr","maxdd"]]
        print("\n== Sharpe Top-5 ==")
        print(top5.to_string(index=False))

# ----------------------------
# FLOW
# ----------------------------
@flow(name="quant-ml-nightly")
def nightly():
    # keep UTC-based run_id; downstream code stores artifacts under this tag
    run_id = dt.datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    ingest()
    make_features()
    make_labels()
    regime_and_aux()
    train_all()
    backtest_all(run_id)
    run_portfolio_report(run_id)
    push_reports_to_r2(run_id)
    print_top5_from_summary(run_id)
    print(f"[DONE] RUN_ID={run_id}; TIMEFRAME={TIMEFRAME}; RTH_ONLY={RTH_ONLY}")

if __name__ == "__main__":
    nightly()
