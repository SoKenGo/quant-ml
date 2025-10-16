import pandas as pd
from typing import Optional
from src.utils.r2_client import _fs, s3url, list_keys

def _read_parquet_from_r2(key: str) -> pd.DataFrame:
    fs = _fs()
    path = key if str(key).startswith("s3://") else s3url(key)
    with fs.open(path, "rb") as f:
        return pd.read_parquet(f)

def _normalize_dates_index(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    """
    Ensure a tz-naive, midnight-normalized DatetimeIndex.
    Prefer an explicit date column when present; otherwise assume index is datetime-like.
    """
    if date_col and date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], utc=False, errors="coerce")
        df[date_col] = df[date_col].dt.tz_localize(None).dt.normalize()
        df = df.sort_values(date_col).drop_duplicates(date_col, keep="last").set_index(date_col)
    else:
        df = df.copy()
        idx = pd.to_datetime(df.index, utc=False, errors="coerce")
        idx = pd.DatetimeIndex(idx).tz_localize(None).normalize()
        df.index = idx
        df = df.sort_index()
    return df

def _pick_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("date", "Date", "timestamp", "time", "datetime"):
        if c in df.columns:
            return c
    return None

def load_eod_from_r2(symbol: str, years: list[int]) -> pd.DataFrame:
    # Hive partitions: eod/symbol=<SYM>/year=<YYYY>/part.parquet
    frames=[]
    for y in years:
        pref = f"eod/symbol={symbol}/year={y}"
        try:
            keys = [k for k in list_keys(pref) if k.endswith(".parquet")]
        except Exception:
            keys = []
        for k in keys:
            df = _read_parquet_from_r2(k)
            dcol = _pick_date_col(df)
            df = _normalize_dates_index(df, dcol)
            cols = df.columns
            close_col = "close" if "close" in cols else ("adj_close" if "adj_close" in cols else None)
            need = {
                "open": "open",
                "high": "high",
                "low":  "low",
                "close": close_col or "close",
                "volume": "volume" if "volume" in cols else ("vol" if "vol" in cols else None),
            }
            view = {k: v for k, v in need.items() if v in cols}
            frames.append(df[list(view.values())].rename(columns={v:k for k,v in view.items()}))
    if not frames:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out

def load_probs_from_r2(symbol: str) -> pd.DataFrame:
    # models/xgb_oos/symbol=<SYM>/probs.parquet
    pref = f"models/xgb_oos/symbol={symbol}"
    try:
        keys = [k for k in list_keys(pref) if k.endswith(".parquet")]
    except Exception:
        keys = []
    if not keys:
        return pd.DataFrame(columns=["prob_up"])
    df = _read_parquet_from_r2(keys[0])
    dcol = _pick_date_col(df)
    df = _normalize_dates_index(df, dcol)
    if "prob_up" not in df.columns:
        if "prob" in df.columns:
            df = df.rename(columns={"prob": "prob_up"})
        else:
            df["prob_up"] = pd.NA
    return df[["prob_up"]].astype(float)
