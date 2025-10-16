import os, subprocess, datetime as dt, json
from prefect import flow, task
from dotenv import load_dotenv
from src.utils.r2_client import _fs, s3url, list_keys

load_dotenv(override=True)

SYMS = ["AVGO","NVDA","TSLA","IBM","DELL","LLY","AMD","META","AAPL","MSFT","GOOGL","AMZN"]
YEARS = ["2018","2019","2020","2021","2022","2023","2024","2025"]

AUTO_TUNE = os.getenv("AUTO_TUNE", "0") == "1"          # set AUTO_TUNE=1 to enable
N_TRIALS  = int(os.getenv("TUNE_TRIALS", "40"))         # tuning trials if enabled

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
def train_all():
    yrs = " ".join(YEARS)
    for s in SYMS:
        print(f"== Train {s} ==")
        if AUTO_TUNE:
            sh(f"python -m src.models.tune_xgb --symbol {s} --years {yrs} --n-trials {N_TRIALS} --min-train-days 252 --min-test-days 30 --drop-zero-labels")
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
        sh(f"python -m src.models.xgb_walkforward --symbol {s} --years {yrs} --min-train-days 252 --min-test-days 30 --push-to-r2{flags}")

@task
def backtest_all(run_id: str):
    yrs = " ".join(YEARS)
    for s in SYMS:
        print(f"== Backtest {s} ==")
        try:
            sh(
              f"python -m src.backtest.engine --symbol {s} --years {yrs} "
              f"--run-id {run_id} --commission-per-share 0.0038 --slippage-bps 5 "
              f"--entry-thr 0.55 --exit-thr 0.50 --risk-pct 0.01 --atr-mult 2.0 --use-next-open --push-to-r2"
            )
        except subprocess.CalledProcessError as e:
            print(f"[WARN] backtest failed for {s}: {e}")

@task
def report_and_push(run_id: str):
    out_csv = f"backtests/summary_{run_id}.csv"
    sh(f"python -m src.reporting.make_report --run-id {run_id} --out-csv {out_csv}")
    # push summary CSV to R2
    fs = _fs()
    with open(out_csv, "rb") as fin, fs.open(s3url(f"backtests/summary/{run_id}.csv"), "wb") as fout:
        fout.write(fin.read())
    print(f"Pushed summary to s3://{os.getenv('R2_BUCKET','quant-ml')}/backtests/summary/{run_id}.csv")

@flow(name="quant-ml-nightly")
def nightly():
    run_id = dt.datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    ingest()
    make_features()
    make_labels()
    train_all()
    backtest_all(run_id)
    report_and_push(run_id)

if __name__ == "__main__":
    nightly()
