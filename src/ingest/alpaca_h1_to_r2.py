# src/ingest/alpaca_h1_to_r2.py
# Fetch 1H bars from Alpaca Data API v2 and write to R2:
#   h1/symbol=<SYM>/year=<YYYY>/month=<MM>/part.parquet
import os, argparse, time, warnings, requests
import pandas as pd
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

try:
    from src.utils.r2_client import _fs
except Exception:
    _fs = None

load_dotenv(override=True)

def fs():
    assert _fs is not None, "src.utils.r2_client._fs not available"
    return _fs()

def _auth_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID", ""),
        "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY", ""),
    }

def _base_url() -> str:
    return os.getenv("APCA_API_BASE", "https://data.alpaca.markets").rstrip("/")

def fetch_month(sym: str, year: int, month: int, feed: str = "iex", adjustment: str = "split",
                pause: float = 0.1) -> pd.DataFrame:
    """
    Pull a single Month of 1H bars for symbol using Alpaca per-symbol endpoint:
      GET /v2/stocks/{symbol}/bars?timeframe=1Hour&start=...&end=...
    Handles pagination via next_page_token.
    """
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end   = (start + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    url = f"{_base_url()}/v2/stocks/{sym.upper()}/bars"
    params = {
        "timeframe": "1Hour",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "feed": feed,              # "iex" on free plans; use "sip" if you have it
        "adjustment": adjustment,  # "raw" | "split" | "all"; "split" works well for adjusted prices
        "limit": 10000,
        # page_token will be added dynamically
    }

    headers = _auth_headers()
    if not headers["APCA-API-KEY-ID"] or not headers["APCA-API-SECRET-KEY"]:
        raise SystemExit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in env or .env")

    all_rows: List[Dict[str, Any]] = []
    page_token: Optional[str] = None

    while True:
        p = dict(params)
        if page_token:
            p["page_token"] = page_token
        r = requests.get(url, params=p, headers=headers, timeout=60)
        if r.status_code == 429:
            time.sleep(1.0)
            r = requests.get(url, params=p, headers=headers, timeout=60)
        r.raise_for_status()
        js = r.json() or {}
        bars = js.get("bars") or []
        all_rows.extend(bars)
        page_token = js.get("next_page_token")
        if not page_token:
            break
        time.sleep(pause)

    if not all_rows:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","year","month","symbol","price"])

    df = pd.DataFrame(all_rows)
    # Fields: t (ISO8601), o,h,l,c,v (and possibly vw, n)
    df["date"]   = pd.to_datetime(df["t"], utc=True)           # tz-aware UTC
    df["open"]   = pd.to_numeric(df["o"], errors="coerce")
    df["high"]   = pd.to_numeric(df["h"], errors="coerce")
    df["low"]    = pd.to_numeric(df["l"], errors="coerce")
    df["close"]  = pd.to_numeric(df["c"], errors="coerce")
    df["volume"] = pd.to_numeric(df["v"], errors="coerce")

    # Filter to Regular Trading Hours (America/New_York 09:30–16:00)
    ny = df["date"].dt.tz_convert("America/New_York")
    rth_mask = (
        (ny.dt.weekday < 5) &
        (
            ((ny.dt.hour > 9) & (ny.dt.hour < 16)) |
            ((ny.dt.hour == 9) & (ny.dt.minute >= 30))
        )
    )
    df = df.loc[rth_mask, ["date","open","high","low","close","volume"]]
    if df.empty:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","year","month","symbol","price"])

    # normalize to tz-naive UTC and schema
    df["date"] = df["date"].dt.tz_convert(None)
    df = df.dropna(subset=["open","high","low","close"]).sort_values("date").drop_duplicates("date")
    df["year"] = year
    df["month"] = month
    df["symbol"] = sym.upper()
    df["price"] = df["close"]
    return df.reset_index(drop=True)

def write_month(bucket: str, sym: str, year: int, month: int, df: pd.DataFrame):
    if df.empty:
        warnings.warn(f"[alpaca_h1] {sym} {year}-{month:02d} empty after RTH filter; skip write.")
        return
    key = f"h1/symbol={sym.upper()}/year={year:04d}/month={month:02d}/part.parquet"
    with fs().open(f"s3://{bucket}/{key}", "wb") as f:
        df.to_parquet(f, index=False, engine="pyarrow")
    print(f"[alpaca_h1] wrote s3://{bucket}/{key} rows={len(df)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--bucket", default=os.getenv("R2_BUCKET","quant-ml"))
    ap.add_argument("--feed", default=os.getenv("ALPACA_FEED","iex"), help="iex or sip")
    ap.add_argument("--adjustment", default="split", choices=["raw","split","all"])
    ap.add_argument("--sleep", type=float, default=0.25, help="pause between months")
    args = ap.parse_args()

    for y in args.years:
        for m in range(1, 13):
            try:
                df = fetch_month(args.symbol, y, m, feed=args.feed, adjustment=args.adjustment, pause=args.sleep)
                write_month(args.bucket, args.symbol, y, m, df)
                time.sleep(args.sleep)
            except requests.HTTPError as e:
                warnings.warn(f"[alpaca_h1] {args.symbol} {y}-{m:02d} HTTP {e}")
            except Exception as e:
                warnings.warn(f"[alpaca_h1] {args.symbol} {y}-{m:02d} failed: {e}")
    print("[alpaca_h1] Done.")

if __name__ == "__main__":
    main()
