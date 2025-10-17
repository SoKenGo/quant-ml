# -*- coding: utf-8 -*-
"""
Kronos-style predictive signal on 1H bars:
- Loads 1H from R2: h1/symbol=<SYM>/year=<YYYY>/month=<MM>/part.parquet
- If external Kronos available, calls it; otherwise falls back to EMA/AR1 hybrid.
- Writes monthly partitions to:
    models/kronos_oos_h1/symbol=<SYM>/year=<YYYY>/month=<MM>/preds.parquet
Columns:
    date, kronos_close_pred, kronos_ret1_pred
"""
from __future__ import annotations
import os, argparse, warnings
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from dotenv import load_dotenv

from src.utils.r2_client import _fs, s3url
from src.features.make_features_h1_to_cloud import read_hourly_from_r2

warnings.filterwarnings("ignore")
load_dotenv(override=True)

def fs(): return _fs()

def _kronos_predict_close(price_series: pd.Series) -> pd.Series:
    # Try external Kronos first
    try:
        import sys, importlib
        sys.path.append("external/Kronos")
        KR = importlib.import_module("kronos.core")  # adjust if different
        pred = KR.predict_close(price_series)  # expected to return np.ndarray-like
        s = pd.Series(pred, index=price_series.index)
        return s
    except Exception as e:
        warnings.warn(f"[Kronos] external not available ({e}); using EMA/AR1 fallback.")
        # Fallback: EMA + AR(1) on log-returns
        ema = price_series.ewm(span=5, adjust=False).mean()
        r = np.log(price_series).diff()
        phi = r.rolling(100).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if np.isfinite(x).sum() >= 3 else np.nan,
            raw=False
        ).fillna(0.0)
        ret_pred = r.shift(1) * phi
        close_pred = price_series * np.exp(ret_pred.fillna(0.0))
        return close_pred

def run(symbol: str, years: List[int], push_to_r2: bool):
    bucket = os.getenv("R2_BUCKET", "quant-ml")
    for y in years:
        # read 1H
        raw = read_hourly_from_r2(bucket, symbol.upper(), y)[["date","close","year","month"]].copy()
        raw = raw.sort_values("date").reset_index(drop=True)
        close_pred = _kronos_predict_close(raw["close"])
        out = pd.DataFrame({
            "date": raw["date"],
            "kronos_close_pred": close_pred,
        })
        out["kronos_ret1_pred"] = np.log(out["kronos_close_pred"]).diff()

        for (yy, mm), dfm in out.groupby([raw["year"], raw["month"]], sort=True):
            key = f"models/kronos_oos_h1/symbol={symbol.upper()}/year={yy:04d}/month={mm:02d}/preds.parquet"
            table = pa.Table.from_pandas(dfm.reset_index(drop=True))
            if push_to_r2:
                with fs().open(s3url(key), "wb") as f:
                    pq.write_table(table, f)
                print(f"[Kronos-1H] wrote R2: s3://{bucket}/{key} rows={len(dfm)}")
            else:
                os.makedirs(os.path.dirname(key), exist_ok=True)
                pq.write_table(table, key)
                print(f"[Kronos-1H] wrote local: {key} rows={len(dfm)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--push-to-r2", action="store_true")
    args = ap.parse_args()
    run(symbol=args.symbol, years=args.years, push_to_r2=args.push_to_r2)
