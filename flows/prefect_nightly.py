# flows/prefect_nightly.py
import os, subprocess, datetime as dt, json
import pandas as pd
from prefect import flow, task
from dotenv import load_dotenv
from src.utils.r2_client import _fs, s3url, list_keys

load_dotenv(override=True)

SYMS  = ["TSLA", "RKLB", "NVDA"]
YEARS = ["2018","2019","2020","2021","2022","2023","2024","2025"]

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

@task
def ingest():
    sh("python -m src.data.ingest_tiingo_to_cloud")

@task
def make_features():
    yrs = " ".join(YEARS[-1:])  # current year only for daily run
    sh(f"python -m features.make_features_to_cloud --symbol ALL --symbols-cfg src/config/symbols.yaml --years {yrs}")

@task
def make_labels():
    yrs = " ".join(YEARS[-1:])
    sh(f"python -m features.make_labels_to_cloud --symbol ALL --symbols-cfg src/config/symbols.yaml --years {yrs} --pt-mult 2.0 --sl-mult 1.0 --max-holding 10")

@task
def regime_and_aux():
    yrs = " ".join(YEARS)
    # HMM (QQQ)
    sh(f"python -m src.regime.hmm --years {yrs} --push-to-r2")
    # Kronos + mlforecast per symbol
    for s in SYMS:
        sh(f"python -m src.alpha.kronos_signal --symbol {s} --years {yrs} --push-to-r2")
        sh(f"python -m src.models.mlf_train    --symbol {s} --years {yrs} --push-to-r2")

@task
def train_all():
    yrs = " ".join(YEARS)
    for s in SYMS:
        print(f"== Train {s} ==")
        tuned = {}
        if AUTO_TUNE:
            sh(f"python -m src.models.tune_xgb --symbol {s} --years {yrs} --n-trials {N_TRIALS} --min-train-days 252 --min-test-days 30 --drop-zero-labels")
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
        # dual calibration + per-fold scale_pos_weight
        sh(
            f"python -m src.models.xgb_walkforward --symbol {s} --years {yrs} "
            f"--min-train-days 252 --min-test-days 30 --push-to-r2 "
            f"--scale-pos-weight auto --calibration-dual{flags}"
        )

@task
def backtest_all(run_id: str):
    yrs = " ".join(YEARS)
    for s in SYMS:
        print(f"== Backtest {s} ==")
        try:
            sh(
              f"python -m src.backtest.engine --symbol {s} --years {yrs} "
              f"--run-id {run_id} --commission-per-share 0.0038 --slippage-bps 5 "
              f"--use-param-cfg src/config/params.yaml --use-next-open "
              f"--qqq-regime --hmm-regime --push-to-r2"
            )
        except subprocess.CalledProcessError as e:
            print(f"[WARN] backtest failed for {s}: {e}")

@task
def run_portfolio_report(run_id: str):
    # Generates portfolio_<RUN_ID>.{csv,json} and summary_<RUN_ID>.csv under backtests/analysis/
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

@flow(name="quant-ml-nightly")
def nightly():
    run_id = dt.datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    ingest()
    make_features()
    make_labels()
    # Regime (HMM) + Aux predictions
    regime_and_aux()
    # XGB training (dual calibration + auto spw)
    train_all()
    # Backtests with param overrides + gates
    backtest_all(run_id)
    # Portfolio + Summary + Push + Console Top-5
    run_portfolio_report(run_id)
    push_reports_to_r2(run_id)
    print_top5_from_summary(run_id)
    print(f"[DONE] RUN_ID={run_id}")

if __name__ == "__main__":
    nightly()
