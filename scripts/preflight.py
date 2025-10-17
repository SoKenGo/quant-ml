# -*- coding: utf-8 -*-
# scripts/preflight.py
import os, sys, time, json, warnings, requests
import pandas as pd
from dotenv import load_dotenv, find_dotenv

warnings.filterwarnings("ignore")

def _mask(v: str, keep: int = 3) -> str:
    if not v: return "MISSING"
    if len(v) <= keep * 2: return "*" * len(v)
    return v[:keep] + "*" * (len(v) - keep * 2) + v[-keep:]

def _load_env():
    path = find_dotenv(".env", raise_error_if_not_found=False)
    load_dotenv(path or ".env", override=True)
    print(f"[ENV] loaded: {path or '<not found, using process env>'}")

def _pick_alpaca_creds():
    # Support both APCA_* and ALPACA_* names
    kid = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    sec = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    base = os.getenv("APCA_API_BASE") or os.getenv("ALPACA_API_BASE") or "https://data.alpaca.markets"
    return kid, sec, base

def check_env():
    # R2
    r2 = {
        "R2_ENDPOINT": os.getenv("R2_ENDPOINT"),
        "R2_ACCESS_KEY_ID": os.getenv("R2_ACCESS_KEY_ID"),
        "R2_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY"),
        "R2_BUCKET": os.getenv("R2_BUCKET"),
    }
    # Alpaca
    apca_key, apca_sec, apca_base = _pick_alpaca_creds()
    missing = [k for k,v in r2.items() if not v] + (["ALPACA/APCA KEYS"] if not (apca_key and apca_sec) else [])

    print("[ENV] R2:",
          {k: _mask(v) for k,v in r2.items()})
    print("[ENV] Alpaca:",
          {"API_KEY_ID": _mask(apca_key or ""),
           "SECRET_KEY": _mask(apca_sec or ""),
           "BASE": apca_base})

    if missing:
        raise SystemExit(f"[ENV] 缺少变量: {missing}. 请检查 .env 是否在项目根目录，或已 export 到当前 shell。")

    print("[ENV] OK")

def check_r2_rw():
    from src.utils.r2_client import _fs, s3url, ls
    fs = _fs()
    b = os.getenv("R2_BUCKET","quant-ml")
    key = f"tmp/healthcheck_{int(time.time())}.txt"
    url = s3url(key)

    # 1) 写入
    with fs.open(url, "wb") as f:
        f.write(b"ok")
    # 某些 s3fs/R2 对 list 有缓存，先清一下
    try:
        fs.invalidate_cache()
    except Exception:
        pass

    # 2) 直接存在性与内容校验（比 list 更稳）
    assert fs.exists(url), f"[R2] exists() 看不到刚写入的对象: {url}"
    with fs.open(url, "rb") as f:
        data = f.read()
    assert data == b"ok", "[R2] 读回内容不一致"

    # 3) 可选：列父目录（不同实现可能要求以斜杠结尾）
    parent1 = f"{b}/tmp"
    parent2 = f"{b}/tmp/"
    keys1 = ls(parent1)
    keys2 = ls(parent2) if parent1 != parent2 else []
    keys = (keys1 or []) + (keys2 or [])
    if not any(p.endswith(key) for p in keys):
        print(f"[R2] WARN: ls('{parent1}')/ls('{parent2}') 未包含刚写入的对象（可能是 R2 权限或缓存差异），但 exists/读回已通过。")

    # 清理
    fs.rm(url)
    print("[R2] RW OK (exists + read-back)")

def check_alpaca():
    apca_key, apca_sec, apca_base = _pick_alpaca_creds()
    headers = {
        "APCA-API-KEY-ID": apca_key,
        "APCA-API-SECRET-KEY": apca_sec,
    }
    url = f"{apca_base.rstrip('/')}/v2/stocks/AAPL/bars"
    params = {"timeframe":"1Hour","limit":1,"feed":os.getenv("ALPACA_DATA_FEED","iex"),"adjustment":"split"}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    if r.status_code != 200:
        raise SystemExit(f"[Alpaca] HTTP {r.status_code} / {r.text[:200]}")
    js = r.json() or {}
    assert "bars" in js, "[Alpaca] response missing 'bars'"
    print("[Alpaca] OK")

def check_python_pkgs():
    pkgs = {}
    for name in ["pandas","numpy","pyarrow","backtrader","pypfopt","xgboost","lightgbm"]:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "N/A")
            pkgs[name] = ver
        except Exception as e:
            pkgs[name] = f"NOT FOUND ({e})"
    print("[PKG]", json.dumps(pkgs, ensure_ascii=False))

if __name__ == "__main__":
    _load_env()
    check_env()
    check_r2_rw()
    check_alpaca()
    check_python_pkgs()
    print("[PREFLIGHT] ALL OK")
