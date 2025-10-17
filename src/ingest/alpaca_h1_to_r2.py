# src/ingest/alpaca_h1_to_r2.py
# Ingest Alpaca Market Data v2 bars to Cloudflare R2 as partitioned Parquet.
# Default: 1-hour RTH bars, monthly partitions:
#   s3://{R2_BUCKET}/h1/symbol=<SYM>/year=<YYYY>/month=<MM>/part.parquet
import os, argparse, time, math, random, warnings
from typing import Optional, Dict, Any, List

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from src.utils.r2_client import _fs
except Exception:
    _fs = None

load_dotenv(override=True)

# --------------------------
# Utilities
# --------------------------
def fs():
    assert _fs is not None, "src.utils.r2_client._fs not available"
    return _fs()

def s3key(prefix: str, sym: str, year: int, month: int) -> str:
    return f"{prefix.rstrip('/')}/symbol={sym.upper()}/year={year:04d}/month={month:02d}/part.parquet"

def _auth_headers() -> Dict[str, str]:
    k = os.getenv("APCA_API_KEY_ID") or os.getenv("APCA-API-KEY-ID") or ""
    s = os.getenv("APCA_API_SECRET_KEY") or os.getenv("APCA-API-SECRET-KEY") or ""
    if not k or not s:
        raise SystemExit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in env or .env")
    return {
        "APCA-API-KEY-ID": k,
        "APCA-API-SECRET-KEY": s,
    }

def _base_url() -> str:
    # Market Data v2
    return os.getenv("APCA_API_BASE", "https://data.alpaca.markets").rstrip("/")

def _normalize_timeframe(tf: str) -> str:
    tf = (tf or "1h").lower()
    mapping = {
        "1h": "1Hour", "1hour": "1Hour", "60m": "1Hour",
        "30m": "30Min", "15m": "15Min", "5m": "5Min", "1m": "1Min"
    }
    return mapping.get(tf, tf if tf in {"1Hour","30Min","15Min","5Min","1Min"} else "1Hour")

def _backoff_sleep(attempt: int, base: float = 0.5, cap: float = 8.0):
    # Exponential backoff with jitter
    t = min(cap, base * (2 ** attempt))
    time.sleep(t * (0.5 + random.random()))

def _http_get_with_retry(url: str, params: dict, headers: dict, max_tries: int = 6, timeout: int = 30):
    attempt = 0
    while True:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                if attempt >= max_tries:
                    r.raise_for_status()
                _backoff_sleep(attempt)
                attempt += 1
                continue
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt >= max_tries:
                raise
            _backoff_sleep(attempt)
            attempt += 1

# --------------------------
# Core fetch
# --------------------------
def fetch_month(
    sym: str,
    year: int,
    month: int,
    timeframe_api: str = "1Hour",
    feed: str = "iex",
    adjustment: str = "split",
    rth_only: bool = True,
    pause: float = 0.1,
) -> pd.DataFrame:
    """
    Pull a single month of bars for symbol using Alpaca v2 per-symbol endpoint:
      GET /v2/stocks/{symbol}/bars?timeframe=<tf>&start=...&end=...
    Handles pagination via next_page_token and robust retries.
    """
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end = (start + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    url = f"{_base_url()}/v2/stocks/{sym.upper()}/bars"
    params = {
        "timeframe": timeframe_api,       # e.g., "1Hour"
        "start": start.isoformat(),
        "end": end.isoformat(),
        "feed": feed,                     # "iex" (free) or "sip" (paid)
        "adjustment": adjustment,         # "raw" | "split" | "all"
        "limit": 10000,                   # Alpaca v2 maximum per page
    }

    headers = _auth_headers()
    all_rows: List[Dict[str, Any]] = []
    page_token: Optional[str] = None

    while True:
        p = dict(params)
        if page_token:
            p["page_token"] = page_token
        r = _http_get_with_retry(url, p, headers)
        js = r.json() or {}
        bars = js.get("bars") or []
        all_rows.extend(bars)
        page_token = js.get("next_page_token")
        if not page_token:
            break
        time.sleep(pause)

    if not all_rows:
        return pd.DataFrame(columns=["ts","o","h","l","c","v","wap","symbol","year","month"])

    df = pd.DataFrame(all_rows)
    # API returns: t (ISO8601), o,h,l,c,v, vw (optional), n (trades)
    df["ts"] = pd.to_datetime(df["t"], utc=True)  # keep UTC tz aware initially
    for col in ("o","h","l","c","v"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # WAP from 'vw' if present; fallback to close
    df["wap"] = pd.to_numeric(df.get("vw"), errors="coerce")
    df["wap"] = df["wap"].fillna(df["c"])

    # ---- RTH filter (US/Eastern 09:30–16:00, Mon–Fri), handles DST via timezone ----
    if rth_only:
        ny = df["ts"].dt.tz_convert("America/New_York")
        rth_mask = (
            (ny.dt.weekday < 5) &
            (
                ((ny.dt.hour > 9) & (ny.dt.hour < 16)) |
                ((ny.dt.hour == 9) & (ny.dt.minute >= 30))
            )
        )
        df = df.loc[rth_mask]

    df = df.dropna(subset=["o","h","l","c"]).drop_duplicates(subset=["t"]).sort_values("t")

    # normalize to tz-naive UTC for Parquet friendliness
    df["ts"] = df["ts"].dt.tz_convert(None)

    df["symbol"] = sym.upper()
    df["year"] = year
    df["month"] = month

    # Select/Order columns
    df = df[["ts","o","h","l","c","v","wap","symbol","year","month"]].reset_index(drop=True)
    return df

# --------------------------
# Write & resume
# --------------------------
def _exists(s3_path: str) -> bool:
    try:
        return fs().exists(s3_path)
    except Exception:
        return False

def _read_parquet(s3_path: str) -> Optional[pd.DataFrame]:
    try:
        with fs().open(s3_path, "rb") as f:
            return pd.read_parquet(f)
    except Exception:
        return None

def _write_parquet(s3_path: str, df: pd.DataFrame):
    if df.empty:
        warnings.warn(f"[alpaca] empty dataframe for {s3_path}; skip write.")
        return
    with fs().open(s3_path, "wb") as f:
        df.to_parquet(f, index=False, engine="pyarrow")
    print(f"[alpaca] wrote {s3_path} rows={len(df)}")

def write_month(bucket: str, key_prefix: str, sym: str, year: int, month: int,
                df: pd.DataFrame, behavior: str = "skip"):
    """
    behavior:
      - 'skip'  : skip write if partition exists
      - 'merge' : read existing, union by 'ts', sort, rewrite
      - 'force' : overwrite blindly
    """
    key = s3key(key_prefix, sym, year, month)
    s3_path = f"s3://{bucket}/{key}"
    exists = _exists(s3_path)

    if behavior == "skip" and exists:
        print(f"[alpaca] exists, skip: {s3_path}")
        return

    if behavior == "merge" and exists:
        old = _read_parquet(s3_path)
        if old is not None and not old.empty:
            all_df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=["ts"]).sort_values("ts")
            _write_parquet(s3_path, all_df)
            return

    # force or no existing
    _write_parquet(s3_path, df)

# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Fetch Alpaca Market Data v2 bars and write monthly Parquet to R2.")
    ap.add_argument("--symbol", required=True, help="Ticker symbol, e.g., NVDA")
    ap.add_argument("--years", nargs="+", type=int, required=True, help="Years to backfill, e.g., 2019 2020 2021 ...")
    ap.add_argument("--timeframe", default=os.getenv("TIMEFRAME","1h"), help="1h|30m|15m|5m|1m (or 1Hour/30Min/...)")
    ap.add_argument("--data-feed", default=os.getenv("ALPACA_DATA_FEED","iex"), choices=["iex","sip"])
    ap.add_argument("--adjustment", default="split", choices=["raw","split","all"])
    ap.add_argument("--rth-only", action="store_true", default=(os.getenv("RTH_ONLY","1")=="1"))
    ap.add_argument("--bucket", default=os.getenv("R2_BUCKET","quant-ml"))
    ap.add_argument("--prefix", default="h1", help="Top-level key prefix (default: h1)")
    ap.add_argument("--sleep", type=float, default=0.25, help="Pause between pages/months")
    ap.add_argument("--on-exist", default=os.getenv("ON_EXIST","skip"), choices=["skip","merge","force"],
                    help="Behavior when monthly partition already exists")
    # ap.add_argument("--months", nargs="+", type=int, help="Optional months like 9 10 11")##
    args = ap.parse_args()

    tf_api = _normalize_timeframe(args.timeframe)
    sym = args.symbol.upper()

    for y in args.years:
        # months = args.months if args.months else list(range(1, 13))
        for m in range(1, 13):
        # for m in months:
            try:
                df = fetch_month(
                    sym=sym,
                    year=y,
                    month=m,
                    timeframe_api=tf_api,
                    feed=args.data_feed,
                    adjustment=args.adjustment,
                    rth_only=args.rth_only,
                    pause=args.sleep,
                )
                key = s3key(args.prefix, sym, y, m)
                s3_path = f"s3://{args.bucket}/{key}"
                write_month(args.bucket, args.prefix, sym, y, m, df, behavior=args.on_exist)
                time.sleep(args.sleep)
            except requests.HTTPError as e:
                warnings.warn(f"[alpaca] {sym} {y}-{m:02d} HTTP {e}")
            except Exception as e:
                warnings.warn(f"[alpaca] {sym} {y}-{m:02d} failed: {e}")

    print("[alpaca] Done.")

if __name__ == "__main__":
    main()
