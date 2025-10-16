# src/regime/hmm.py
import os, argparse, warnings
import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from dotenv import load_dotenv
from src.utils.r2_client import _fs, s3url

load_dotenv(override=True)
warnings.filterwarnings("ignore")

# ---------- close column picking ----------
def _pick_close(df: pd.DataFrame) -> pd.Series:
    if "adjClose" in df.columns: return pd.Series(df["adjClose"], name="close").astype(float)
    if "close" in df.columns:    return pd.Series(df["close"], name="close").astype(float)
    for c in df.columns:
        if c.lower() in ("adjclose","close","adj_close"):  # 宽松匹配
            return pd.Series(df[c], name="close").astype(float)
    raise RuntimeError("[HMM] missing close/adjClose")

# ---------- R2 EOD loader ----------
def _load_eod(symbol: str, years: list[int]) -> pd.DataFrame:
    fs = _fs(); parts=[]
    for y in years:
        key = f"eod/symbol={symbol}/year={y}/part.parquet"
        try:
            with fs.open(s3url(key), "rb") as f:
                parts.append(pq.read_table(f).to_pandas())
        except Exception as e:
            warnings.warn(f"[HMM] missing/failed: {key} ({e})")
    if not parts:
        raise RuntimeError(f"[HMM] no EOD for {symbol} {years}")
    raw = pd.concat(parts, ignore_index=True)
    px = pd.DataFrame({
        "date": pd.to_datetime(raw["date"], utc=False),
        "close": _pick_close(raw)
    }).sort_values("date").dropna()
    return px

def _build_obs(px: pd.DataFrame) -> pd.DataFrame:
    ret = np.log(px["close"]).diff()
    vol20 = ret.rolling(20, min_periods=20).std()
    obs = pd.DataFrame({"ret": ret, "vol20": vol20}, index=px["date"]).dropna()
    return obs

# ---------- backends ----------
def _fit_predict_pomegranate(X: np.ndarray, n_states=3, n_init=5, max_iter=300):
    # v2+
    try:
        from pomegranate.hmm import HiddenMarkovModel as PM_HMM   # type: ignore
        from pomegranate.distributions import Normal as PM_Normal # type: ignore
    except Exception:
        PM_HMM = PM_Normal = None
    # v1.x
    if PM_HMM is None:
        try:
            from pomegranate import HiddenMarkovModel as PM_HMM    # type: ignore
            from pomegranate import NormalDistribution as PM_Normal # type: ignore
        except Exception:
            PM_HMM = None
    if PM_HMM is None:
        raise ImportError("pomegranate not available")
    mdl = PM_HMM.from_samples(PM_Normal, n_components=n_states, X=[X], n_init=n_init, max_iterations=max_iter)
    return np.asarray(mdl.predict(X), dtype=int)

def _fit_predict_hmmlearn(X: np.ndarray, n_states=3):
    from hmmlearn.hmm import GaussianHMM
    mdl = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=42)
    mdl.fit(X)
    return mdl.predict(X).astype(int)

def _fit_predict_kmeans(X: np.ndarray, n_states=3):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=n_states, n_init=10, random_state=42).fit_predict(X).astype(int)

def _bull_state_by_mean_ret(states: np.ndarray, X: np.ndarray) -> int:
    n_states = int(states.max()) + 1
    means = [np.nanmean(X[states==s, 0]) if (states==s).any() else -np.inf for s in range(n_states)]
    return int(np.nanargmax(means))

# ---------- passthrough (allow-all) ----------
def _write_allow_all(years: list[int], push_to_r2: bool, out_key: str):
    start, end = f"{min(years)}-01-01", f"{max(years)}-12-31"
    dates = pd.bdate_range(start, end)
    out = pd.DataFrame({"date": dates.date, "hmm_state": 0, "regime_bull": 1})
    fs = _fs(); table = pa.Table.from_pandas(out)
    with fs.open(s3url(out_key), "wb") as f:
        pq.write_table(table, f)
    print(f"[HMM] wrote ALLOW-ALL fallback to R2: {s3url(out_key)} rows={len(out)}")
    return out

def run(years, symbol="QQQ", push_to_r2=False, out_key="models/regime/qqq_hmm.parquet"):
    # 1) Load QQQ
    try:
        px = _load_eod(symbol, years)
        obs = _build_obs(px)
    except Exception as e:
        warnings.warn(f"[HMM] data unavailable ({e}); writing allow-all fallback.")
        _write_allow_all(years, True if push_to_r2 else False, out_key)
        return

    if len(obs) < 40:  # 不足以做 20D 波动+HMM
        warnings.warn("[HMM] not enough observations; writing allow-all fallback.")
        _write_allow_all(years, True if push_to_r2 else False, out_key)
        return

    X = obs[["ret","vol20"]].to_numpy()

    # 2) Fit with backends
    backend = "pomegranate"
    try:
        states = _fit_predict_pomegranate(X)
    except Exception as e1:
        warnings.warn(f"[HMM] pomegranate failed ({e1}); trying hmmlearn…")
        try:
            states = _fit_predict_hmmlearn(X); backend = "hmmlearn"
        except Exception as e2:
            warnings.warn(f"[HMM] hmmlearn failed ({e2}); using KMeans fallback.")
            states = _fit_predict_kmeans(X); backend = "kmeans"

    bull = _bull_state_by_mean_ret(states, X)
    out = pd.DataFrame({
        "date": obs.index.date,
        "hmm_state": states,
        "regime_bull": (states == bull).astype(int),
    })

    # 3) Write
    if push_to_r2:
        fs = _fs(); table = pa.Table.from_pandas(out)
        with fs.open(s3url(out_key), "wb") as f:
            pq.write_table(table, f)
        print(f"[HMM] wrote to R2 ({backend}): {s3url(out_key)} rows={len(out)} bull_state={bull}")
    else:
        os.makedirs(os.path.dirname(out_key), exist_ok=True)
        out.to_parquet(out_key, index=False)
        print(f"[HMM] wrote local ({backend}): {out_key} rows={len(out)} bull_state={bull}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--symbol", type=str, default="QQQ")
    ap.add_argument("--push-to-r2", action="store_true")
    args = ap.parse_args()
    run(args.years, symbol=args.symbol, push_to_r2=args.push_to_r2)
