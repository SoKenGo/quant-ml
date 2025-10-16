# src/utils/r2_client.py
import os, io, pandas as pd
from typing import Iterable, Optional
from dotenv import load_dotenv

def _fs():
    # ensure .env is loaded each time we build a client
    load_dotenv(".env")
    import s3fs
    endpoint = os.getenv("R2_ENDPOINT")
    bucket   = os.getenv("R2_BUCKET", "quant-ml")
    ak       = os.getenv("R2_ACCESS_KEY_ID")
    sk       = os.getenv("R2_SECRET_ACCESS_KEY")
    assert all([endpoint, ak, sk]), "Missing R2 env vars"
    fs = s3fs.S3FileSystem(key=ak, secret=sk, client_kwargs={"endpoint_url": endpoint})
    return fs

def _bucket():
    load_dotenv(".env")
    return os.getenv("R2_BUCKET", "quant-ml")

def s3url(key: str) -> str:
    return f"s3://{_bucket()}/{key.lstrip('/')}"

def list_keys(prefix: str) -> list[str]:
    fs = _fs()
    return [f"s3://{p}" for p in fs.ls(f"{_bucket()}/{prefix}".rstrip("/"))]

def upload_file(local: str, key: str):
    fs = _fs()
    fs.put(local, s3url(key))

def download_file(key: str, local: str):
    fs = _fs()
    os.makedirs(os.path.dirname(local), exist_ok=True)
    fs.get(s3url(key), local)

def put_parquet(df: pd.DataFrame, key: str):
    fs = _fs()
    with fs.open(s3url(key), "wb") as f:
        df.to_parquet(f, index=False)

def read_parquet(key: str) -> pd.DataFrame:
    fs = _fs()
    with fs.open(s3url(key), "rb") as f:
        return pd.read_parquet(f)

def put_csv(df: pd.DataFrame, key: str):
    fs = _fs()
    with fs.open(s3url(key), "wb") as f:
        f.write(df.to_csv(index=False).encode())

def read_csv(key: str) -> pd.DataFrame:
    fs = _fs()
    with fs.open(s3url(key), "rb") as f:
        return pd.read_csv(io.BytesIO(f.read()))

def sync_dir_to_r2(local_dir: str, prefix: str, exts: Optional[Iterable[str]]=("parquet","csv")):
    import pathlib
    fs = _fs()
    base = pathlib.Path(local_dir)
    for p in base.rglob("*"):
        if p.is_file() and (not exts or p.suffix.lstrip(".") in exts):
            rel = p.relative_to(base).as_posix()
            key = f"{prefix.rstrip('/')}/{rel}"
            fs.put(p.as_posix(), s3url(key))
