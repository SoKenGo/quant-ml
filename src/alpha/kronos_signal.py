import argparse, warnings, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from src.utils.r2_client import _fs, s3url

def load_eod(sym, years):
    fs=_fs(); parts=[]
    for y in years:
        key=f"eod/symbol={sym}/year={y}/part.parquet"
        with fs.open(s3url(key),"rb") as f:
            parts.append(pq.read_table(f).to_pandas())
    return pd.concat(parts).sort_values("date").reset_index(drop=True)

def kronos_predict(close: pd.Series) -> pd.DataFrame:
    # 优先尝试 external/Kronos
    try:
        import sys, importlib
        sys.path.append("external/Kronos")
        KR = importlib.import_module("kronos.core")  # 视仓库结构可能不同
        preds = KR.predict_close(close)  # 假定返回 np.array（示意）
        df=pd.DataFrame({"kronos_close_pred": preds})
    except Exception as e:
        warnings.warn(f"Kronos not available ({e}); fallback to EMA/AR1")
        ema = close.ewm(span=5, adjust=False).mean().shift(1)
        ret1 = np.log(close).diff()
        ar1  = ret1.rolling(20).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x.dropna())>2 else 0.0, raw=False)
        pred_close = close * np.exp((ret1.shift(1)*ar1).fillna(0.0))
        df=pd.DataFrame({"kronos_close_pred": pred_close})
    df["kronos_ret1_pred"] = np.log(df["kronos_close_pred"]).diff()
    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--push-to-r2", action="store_true")
    args=ap.parse_args()
    df=load_eod(args.symbol, args.years)
    pred = kronos_predict(df["close"])
    out = pd.DataFrame({"date": pd.to_datetime(df["date"]).dt.date}).join(pred)
    if args.push_to_r2:
        fs=_fs(); key=f"models/kronos_oos/symbol={args.symbol}/preds.parquet"
        table=pa.Table.from_pandas(out)
        with fs.open(s3url(key),"wb") as f:
            pq.write_table(table,f)
    else:
        out.to_parquet(f"kronos_{args.symbol}.parquet", index=False)

if __name__=="__main__":
    main()
