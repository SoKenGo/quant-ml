# -*- coding: utf-8 -*-
# src/utils/r2_client.py
import os, io, re, pathlib
from typing import Iterable, Optional, Tuple, List, Dict, Any
import pandas as pd

# Try loading .env automatically
try:
    from dotenv import load_dotenv
    load_dotenv(".env", override=True)
except Exception:
    pass

# ---------------------------
# Core env helpers
# ---------------------------
def _bucket() -> str:
    """Current bucket name from env (default 'quant-ml')."""
    try:
        from dotenv import load_dotenv
        load_dotenv(".env", override=True)
    except Exception:
        pass
    b = os.getenv("R2_BUCKET", "quant-ml").strip()
    return b

def _split_url(url_or_key: str) -> Tuple[str, str]:
    """
    Accepts:
      - 's3://bucket/key'        -> (bucket, key)
      - 'bucket/key' (must start with the *env bucket* name to be treated as explicit bucket)
      - 'key'                    -> (ENV_BUCKET, key)
    """
    u = (url_or_key or "").strip().lstrip("/")

    # explicit s3:// form
    if u.startswith("s3://"):
        rest = u[5:]
        bkt, _, key = rest.partition("/")
        return bkt, key.lstrip("/")

    bkt_env = _bucket()

    # only treat as explicit bucket if it matches the current ENV bucket
    if u.startswith(bkt_env + "/"):
        return bkt_env, u[len(bkt_env) + 1 :]

    # otherwise ALWAYS default to the env bucket
    return bkt_env, u

def s3url(url_or_key: str) -> str:
    """Canonical 's3://bucket/key' string (for logs)."""
    bkt, key = _split_url(url_or_key)
    return f"s3://{bkt}/{key}"

def _fs_path(url_or_key: str) -> str:
    """'bucket/key' path-style (what s3fs prefers for R2)."""
    bkt, key = _split_url(url_or_key)
    return f"{bkt}/{key}".rstrip("/")

# ---------------------------
# S3FS client (Cloudflare R2)
# ---------------------------
def _fs():
    # reload .env each time
    try:
        from dotenv import load_dotenv
        load_dotenv(".env", override=True)
    except Exception:
        pass

    import s3fs, os
    endpoint = os.getenv("R2_ENDPOINT")
    ak       = os.getenv("R2_ACCESS_KEY_ID")
    sk       = os.getenv("R2_SECRET_ACCESS_KEY")
    if not all([endpoint, ak, sk]):
        raise RuntimeError("Missing R2 env vars (R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY)")

    cfg_kwargs = {
        "config_kwargs": {
            "signature_version": "s3v4",
            "region_name": "auto",
            "s3": {"addressing_style": "path"},
        }
    }
    return s3fs.S3FileSystem(
        key=ak,
        secret=sk,
        client_kwargs={"endpoint_url": endpoint, "region_name": "auto"},
        **cfg_kwargs,
    )

# ---------------------------
# Basic ops (path-style I/O)
# ---------------------------
def list_keys(prefix: str) -> List[str]:
    fs = _fs()
    try:
        return [f"s3://{p}" for p in fs.ls(_fs_path(prefix))]
    except FileNotFoundError:
        return []

def ls(prefix: str) -> List[str]:
    return list_keys(prefix)

def exists(url_or_key: str) -> bool:
    fs = _fs()
    try:
        return fs.exists(_fs_path(url_or_key))
    except Exception:
        return False

def upload_file(local: str, key: str):
    fs = _fs()
    fs.put(local, _fs_path(key))

def download_file(key: str, local: str):
    fs = _fs()
    local = pathlib.Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    fs.get(_fs_path(key), local.as_posix())

def put_parquet(df: pd.DataFrame, key: str):
    fs = _fs()
    try:
        with fs.open(_fs_path(key), "wb") as f:
            df.to_parquet(f, index=False)
    except Exception as e:
        print(f"[WARN] s3fs put_parquet failed ({e}); falling back to boto3.")
        put_parquet_boto(df, key)

def read_parquet(key: str) -> pd.DataFrame:
    fs = _fs()
    with fs.open(_fs_path(key), "rb") as f:
        return pd.read_parquet(f)

def put_csv(df: pd.DataFrame, key: str):
    fs = _fs()
    try:
        with fs.open(_fs_path(key), "wb") as f:
            f.write(df.to_csv(index=False).encode())
    except Exception as e:
        print(f"[WARN] s3fs put_csv failed ({e}); falling back to boto3.")
        put_csv_boto(df, key)

def read_csv(key: str) -> pd.DataFrame:
    fs = _fs()
    with fs.open(_fs_path(key), "rb") as f:
        return pd.read_csv(io.BytesIO(f.read()))

def sync_dir_to_r2(local_dir: str, prefix: str, exts: Optional[Iterable[str]]=("parquet","csv")):
    fs = _fs()
    base = pathlib.Path(local_dir)
    for p in base.rglob("*"):
        if p.is_file() and (not exts or p.suffix.lstrip(".").lower() in {e.lower() for e in exts}):
            rel = p.relative_to(base).as_posix()
            key = f"{prefix.rstrip('/')}/{rel}"
            fs.put(p.as_posix(), _fs_path(key))

# ---------------------------
# Partition helpers (optional)
# ---------------------------
_PART_RE_YEAR  = re.compile(r"(?:year|YYYY)=(\d{4})/?$")
_PART_RE_MONTH = re.compile(r"(?:month|MM)=(\d{2})/?$")
_PART_RE_DAY   = re.compile(r"(?:day|DD)=(\d{2})/?$")

def _extract_part_num(name: str, level: str) -> Optional[int]:
    name = name.rstrip("/")
    if level == "year":
        m = _PART_RE_YEAR.search(name)
    elif level == "month":
        m = _PART_RE_MONTH.search(name)
    elif level == "day":
        m = _PART_RE_DAY.search(name)
    else:
        return None
    return int(m.group(1)) if m else None

def glob(prefix: str, pattern_suffix: str = "*.parquet") -> List[str]:
    fs = _fs()
    base = _fs_path(prefix)
    keys = []
    try:
        for p in fs.ls(base):
            if p.endswith(pattern_suffix) or pattern_suffix == "*":
                keys.append(f"s3://{p}")
    except FileNotFoundError:
        pass
    return sorted(keys)

def latest_partition(base_prefix: str) -> Optional[Dict[str, Any]]:
    fs = _fs()
    bkt, base_key = _split_url(base_prefix)
    year_dirs = [d for d in ls(f"s3://{bkt}/{base_key}") if ("/YYYY=" in d or "/year=" in d)]
    if not year_dirs:
        return None
    years = []
    for d in year_dirs:
        part = _extract_part_num(d.split("/")[-1], "year")
        if part is not None: years.append((part, d))
    if not years:
        return None
    y, ydir = max(years, key=lambda x: x[0])

    month_dirs = [d for d in ls(ydir) if ("/MM=" in d or "/month=" in d)]
    if not month_dirs:
        files = glob(ydir, "*.parquet")
        return {"year": y, "month": None, "day": None, "prefix": ydir, "files": files}
    months=[]
    for d in month_dirs:
        part = _extract_part_num(d.split("/")[-1], "month")
        if part is not None: months.append((part, d))
    m, mdir = max(months, key=lambda x: x[0])

    day_dirs = [d for d in ls(mdir) if ("/DD=" in d or "/day=" in d)]
    if not day_dirs:
        files = glob(mdir, "*.parquet")
        return {"year": y, "month": m, "day": None, "prefix": mdir, "files": files}
    days=[]
    for d in day_dirs:
        part = _extract_part_num(d.split("/")[-1], "day")
        if part is not None: days.append((part, d))
    d, ddir = max(days, key=lambda x: x[0])

    files = glob(ddir, "part-*.parquet") or glob(ddir, "*.parquet")
    return {"year": y, "month": m, "day": d, "prefix": ddir, "files": files}

# ---------------------------
# Convenience for bars paths
# ---------------------------
def bars_prefix(timeframe: str, symbol: str) -> str:
    return f"bars/{timeframe.strip().lower()}/{symbol.strip().upper()}"

def bars_part_prefix(timeframe: str, symbol: str, year: int, month: int, day: Optional[int] = None) -> str:
    base = bars_prefix(timeframe, symbol)
    p = f"{base}/YYYY={year:04d}/MM={month:02d}"
    if day is not None:
        p += f"/DD={day:02d}"
    return p

# ---------------------------
# Diagnostics
# ---------------------------
def exists_bucket() -> bool:
    fs = _fs()
    try:
        fs.ls(_bucket())
        return True
    except Exception:
        return False

def debug_print():
    try:
        from dotenv import load_dotenv
        load_dotenv(".env", override=True)
    except Exception:
        pass
    print("R2_ENDPOINT =", os.getenv("R2_ENDPOINT"))
    print("R2_BUCKET   =", _bucket())
    print("R2_ACCESS_KEY_ID set:", bool(os.getenv("R2_ACCESS_KEY_ID")))
    print("R2_SECRET_ACCESS_KEY set:", bool(os.getenv("R2_SECRET_ACCESS_KEY")))

def ping_write():
    df = pd.DataFrame({"ok":[1]})
    put_parquet(df, "tmp/_ping.parquet")
    return exists("tmp/_ping.parquet")

def list_root(n=5):
    fs = _fs()
    try:
        return fs.ls(_bucket())[:n]
    except Exception as e:
        return [f"ERR: {e}"]
    
# --- boto3 fallback for writes (R2 is very happy with this) ---
def _boto3_client():
    import boto3, os
    from botocore.config import Config
    endpoint = os.getenv("R2_ENDPOINT")
    ak       = os.getenv("R2_ACCESS_KEY_ID")
    sk       = os.getenv("R2_SECRET_ACCESS_KEY")
    if not all([endpoint, ak, sk]):
        raise RuntimeError("Missing R2 env vars (R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY)")
    cfg = Config(signature_version="s3v4", region_name="auto", s3={"addressing_style": "path"})
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=ak, aws_secret_access_key=sk, config=cfg)

def _split_bucket_key(url_or_key: str):
    bkt, key = _split_url(url_or_key)
    return bkt, key

def put_bytes_boto(key: str, data: bytes, content_type: str = "application/octet-stream"):
    s3 = _boto3_client()
    bkt, k = _split_bucket_key(key)
    s3.put_object(Bucket=bkt, Key=k, Body=data, ContentType=content_type)

def put_parquet_boto(df: pd.DataFrame, key: str):
    import io as _io
    bio = _io.BytesIO()
    df.to_parquet(bio, index=False)
    put_bytes_boto(key, bio.getvalue(), content_type="application/octet-stream")

def put_csv_boto(df: pd.DataFrame, key: str):
    data = df.to_csv(index=False).encode()
    put_bytes_boto(key, data, content_type="text/csv")

