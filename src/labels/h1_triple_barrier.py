# -*- coding: utf-8 -*-
# Hourly triple-barrier labels → R2:
#   s3://{R2_BUCKET}/train_h1/symbol=<SYM>/year=<YYYY>/month=<MM>/labels.parquet
#
# Stores: date, y (0/1), t1 (first hit time or horizon end), weight (class balancing).
# ATR is Wilder-smoothed on 1H bars (n=14 by default).

from __future__ import annotations
import os, argparse, warnings
from typing import Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.utils.r2_client import _fs, s3url
from src.features.make_features_h1_to_cloud import read_hourly_from_r2  # robust hourly loader

warnings.filterwarnings("ignore")
load_dotenv(override=True)

def fs(): return _fs()

# -------- ATR(14h) Wilder ----------
def _atr_wilder_14(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]).abs(),
         (df["high"] - pc).abs(),
         (df["low"]  - pc).abs()],
        axis=1
    ).max(axis=1)
    # Wilder smoothing keeps intraday scale stable
    return tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

# -------- core triple barrier ----------
def _triple_barrier(
    price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    pt_mult: float,
    sl_mult: float,
    max_holding_bars: int
) -> Tuple[pd.Series, pd.Series]:
    """
    Long-only TB:
      y=1 if upper barrier hit first within horizon, else 0 if stop or horizon end.
      t1 = first hit time (upper/stop) or horizon end time.
    """
    n = len(price)
    y  = np.full(n, np.nan)
    # Use np.datetime64("NaT") for safe datetime array initialization
    t1 = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

    for i in range(n - 1):
        if not (np.isfinite(price.iloc[i]) and np.isfinite(atr.iloc[i])):
            continue
        up = price.iloc[i] + pt_mult * atr.iloc[i]
        dn = price.iloc[i] - sl_mult * atr.iloc[i]
        end = min(i + max_holding_bars, n - 1)
        label = 0
        t_hit = price.index[end]  # default: horizon end
        for j in range(i + 1, end + 1):
            if high.iloc[j] >= up:
                label = 1
                t_hit = price.index[j]
                break
            if low.iloc[j] <= dn:
                label = 0
                t_hit = price.index[j]
                break
        y[i]  = label
        t1[i] = t_hit
    return pd.Series(y, index=price.index), pd.Series(t1, index=price.index)

def _label_month(dfm: pd.DataFrame, pt: float, sl: float, horizon: int) -> pd.DataFrame:
    # dfm needs: date, high, low, close
    if dfm.empty:
        return dfm
    dfm = dfm.sort_values("date").reset_index(drop=True)
    dfm["atr14h"] = _atr_wilder_14(dfm)
    y, t1 = _triple_barrier(dfm["close"], dfm["high"], dfm["low"], dfm["atr14h"], pt, sl, horizon)

    out = dfm[["date"]].copy()
    out["y"]  = y
    out["t1"] = t1

    # Avoid label peeking across month: drop the last 'horizon' bars
    if len(out) > horizon:
        out = out.iloc[:-horizon].reset_index(drop=True)
    else:
        out = out.iloc[0:0]

    # Simple class-balancing weights (inverse frequency)
    if not out.empty and out["y"].notna().any():
        p_pos = np.nanmean(out["y"])
        # guard for degenerate months
        w_pos = 0.5 / max(p_pos, 1e-6)
        w_neg = 0.5 / max(1 - p_pos, 1e-6)
        out["weight"] = np.where(out["y"] == 1, w_pos, w_neg)
    else:
        out["weight"] = np.nan

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--bucket", default=os.getenv("R2_BUCKET","quant-ml"))
    ap.add_argument("--pt-mult", type=float, default=2.0)
    ap.add_argument("--sl-mult", type=float, default=1.0)
    ap.add_argument("--max-holding-bars", type=int, default=24)  # ~1 day
    args = ap.parse_args()

    sym = args.symbol.upper()
    for yr in args.years:
        raw = read_hourly_from_r2(args.bucket, sym, yr)  # date, high, low, close, price, volume, year, month
        for (y, m), dfm in raw.groupby(["year","month"], sort=True):
            lab = _label_month(dfm[["date","high","low","close"]].copy(),
                               args.pt_mult, args.sl_mult, args.max_holding_bars)
            key = f"train_h1/symbol={sym}/year={y:04d}/month={m:02d}/labels.parquet"
            with fs().open(s3url(key), "wb") as f:
                lab.to_parquet(f, index=False)
            print(f"[labels_h1] wrote s3://{args.bucket}/{key} rows={len(lab)}")

    print("[labels_h1] Done.")

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
