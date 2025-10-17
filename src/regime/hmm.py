# -*- coding: utf-8 -*-
# HMM regime detection (timeframe-aware: intraday 1h by default).
# Outputs monthly partitions suitable for feature merge:
#   s3://{R2_BUCKET}/regime_hmm/symbol=<SYM>/year=<YYYY>/month=<MM>/regime.parquet
#
# Columns: date, regime_state (0..K-1), regime_bull, regime_bear, regime_neutral

from __future__ import annotations
import os, argparse, warnings
import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from dotenv import load_dotenv
from typing import List, Optional

from src.utils.r2_client import _fs, s3url
from src.features.make_features_h1_to_cloud import read_hourly_from_r2

load_dotenv(override=True)
warnings.filterwarnings("ignore")

# ---------- daily (EOD) loader, used only when timeframe is daily ----------
def _pick_close(df: pd.DataFrame) -> pd.Series:
    for c in ["adjClose","adjclose","adj_close","close","Close","c"]:
        if c in df.columns:
            return pd.Series(df[c], name="close").astype(float)
    raise RuntimeError("[HMM] missing close/adjClose")

def _load_eod(symbol: str, years: List[int]) -> pd.DataFrame:
    fs = _fs(); parts=[]
    for y in years:
        key = f"eod/symbol={symbol}/year={y}/part.parquet"
        try:
            with fs.open(s3url(key), "rb") as f:
                parts.append(pd.read_parquet(f))
        except Exception as e:
            warnings.warn(f"[HMM] missing/failed: {key} ({e})")
    if not parts:
        raise RuntimeError(f"[HMM] no EOD for {symbol} {years}")
    raw = pd.concat(parts, ignore_index=True)
    px = pd.DataFrame({
        "date": pd.to_datetime(raw["date"], utc=False).dt.tz_localize(None),
        "close": _pick_close(raw)
    }).sort_values("date").dropna()
    return px

# ---------- observation builders ----------
def _build_obs_daily(px: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    ret = np.log(px["close"]).diff()
    vol = ret.rolling(vol_window, min_periods=vol_window).std()
    return pd.DataFrame({"ret": ret, "vol": vol}, index=px["date"]).dropna()

def _build_obs_intraday(px: pd.DataFrame, vol_window_bars: int = 24) -> pd.DataFrame:
    # px must have columns: date, close
    ret = np.log(px["close"]).diff()
    vol = ret.rolling(vol_window_bars, min_periods=vol_window_bars).std()
    return pd.DataFrame({"ret": ret, "vol": vol}, index=px["date"]).dropna()

# ---------- HMM backends ----------
def _fit_predict_pomegranate(X: np.ndarray, n_states=3, n_init=5, max_iter=300):
    try:
        from pomegranate.hmm import HiddenMarkovModel as PM_HMM   # type: ignore
        from pomegranate.distributions import Normal as PM_Normal # type: ignore
    except Exception:
        PM_HMM = PM_Normal = None
    if PM_HMM is None:
        try:
            from pomegranate import HiddenMarkovModel as PM_HMM    # type: ignore
            from pomegranate import NormalDistribution as PM_Normal # type: ignore
        except Exception:
            PM_HMM = None
    if PM_HMM is None:
        raise ImportError("pomegranate not available")
    mdl = PM_HMM.from_samples(PM_Normal, n_components=n_states, X=[X], n_init=n_init, max_iterations=max_iter)
    return np.asarray(mdl.predict(X), dtype=int), "pomegranate"

def _fit_predict_hmmlearn(X: np.ndarray, n_states=3):
    from hmmlearn.hmm import GaussianHMM
    mdl = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=42)
    mdl.fit(X)
    return mdl.predict(X).astype(int), "hmmlearn"

def _fit_predict_kmeans(X: np.ndarray, n_states=3):
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=n_states, n_init=10, random_state=42).fit_predict(X).astype(int)
    return lab, "kmeans"

def _rank_states_by_mean_ret(states: np.ndarray, X: np.ndarray):
    # returns indices for bear (min), neutral (mid), bull (max)
    uniq = sorted(set(states))
    means = [np.nanmean(X[states==s, 0]) if (states==s).any() else -np.inf for s in uniq]
    order = np.argsort(means)
    bear, neutral, bull = uniq[order[0]], uniq[order[len(order)//2]], uniq[order[-1]]
    return bear, neutral, bull

# ---------- write helpers ----------
def _write_partitioned(out: pd.DataFrame, symbol: str, push_to_r2: bool):
    # write monthly partitions under regime_hmm/...
    out = out.copy()
    out["year"]  = out["date"].dt.year
    out["month"] = out["date"].dt.month
    fs = _fs()

    for (y, m), dsub in out.groupby(["year","month"], sort=True):
        key = f"regime_hmm/symbol={symbol}/year={y:04d}/month={m:02d}/regime.parquet"
        if push_to_r2:
            table = pa.Table.from_pandas(dsub.drop(columns=["year","month"]))
            with fs.open(s3url(key), "wb") as f:
                pq.write_table(table, f)
            print(f"[HMM] wrote R2: {s3url(key)} rows={len(dsub)}")
        else:
            os.makedirs(os.path.dirname(key), exist_ok=True)
            dsub.drop(columns=["year","month"]).to_parquet(key, index=False)
            print(f"[HMM] wrote local: {key} rows={len(dsub)}")

# ---------- main runner ----------
def run(
    years: List[int],
    symbol: str = "QQQ",
    timeframe: str = "1h",
    n_states: int = 3,
    vol_window_bars: int = 24,
    push_to_r2: bool = False
):
    symbol = symbol.upper()
    timeframe_l = timeframe.lower()

    # 1) Load price series per timeframe
    if timeframe_l in {"1h","60m","30m","15m","5m","1m","1hour","30min","15min","5min","1min"}:
        # intraday from R2 hourly path
        try:
            dfs = []
            for y in years:
                dfy = read_hourly_from_r2(os.getenv("R2_BUCKET","quant-ml"), symbol, y)[["date","close"]]
                dfs.append(dfy)
            px = pd.concat(dfs, ignore_index=True).sort_values("date").dropna()
            obs = _build_obs_intraday(px, vol_window_bars=vol_window_bars)
        except Exception as e:
            warnings.warn(f"[HMM] intraday data unavailable ({e}); skipping write.")
            return
    else:
        # daily EOD fallback
        try:
            px = _load_eod(symbol, years)
            obs = _build_obs_daily(px, vol_window=20)
        except Exception as e:
            warnings.warn(f"[HMM] daily data unavailable ({e}); skipping write.")
            return

    if len(obs) < max(40, n_states * 8):
        warnings.warn("[HMM] not enough observations to fit; skipping write.")
        return

    X = obs[["ret","vol"]].to_numpy()

    # 2) Fit with backends
    backend_used = "pomegranate"
    try:
        states, backend_used = _fit_predict_pomegranate(X, n_states=n_states)
    except Exception as e1:
        warnings.warn(f"[HMM] pomegranate failed ({e1}); trying hmmlearn…")
        try:
            states, backend_used = _fit_predict_hmmlearn(X, n_states=n_states)
        except Exception as e2:
            warnings.warn(f"[HMM] hmmlearn failed ({e2}); using KMeans fallback.")
            states, backend_used = _fit_predict_kmeans(X, n_states=n_states)

    bear, neutral, bull = _rank_states_by_mean_ret(states, X)
    out = pd.DataFrame({
        "date": obs.index.tz_localize(None),   # naive ts for joins
        "regime_state": states.astype(int),
    })
    out["regime_bull"]    = (out["regime_state"] == bull).astype(int)
    out["regime_bear"]    = (out["regime_state"] == bear).astype(int)
    out["regime_neutral"] = (out["regime_state"] == neutral).astype(int)

    # 3) Write (monthly partitions for easy merging)
    _write_partitioned(out, symbol, push_to_r2)
    print(f"[HMM] done ({backend_used}); bull={bull}, bear={bear}, neutral={neutral}, rows={len(out)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--symbol", type=str, default="QQQ")
    ap.add_argument("--timeframe", type=str, default="1h")
    ap.add_argument("--n-states", type=int, default=3)
    ap.add_argument("--vol-window-bars", type=int, default=24, help="Rolling std window for intraday")
    ap.add_argument("--push-to-r2", action="store_true")
    args = ap.parse_args()
    run(
        years=args.years,
        symbol=args.symbol,
        timeframe=args.timeframe,
        n_states=args.n_states,
        vol_window_bars=args.vol_window_bars,
        push_to_r2=args.push_to_r2
    )
