# -*- coding: utf-8 -*-
import os, pytest, pandas as pd, numpy as np

SYMS = [s.strip().upper() for s in os.getenv("SMOKE_SYMS","NVDA,TSLA,RKLB").split(",")]
YEAR = int(os.getenv("SMOKE_YEAR", "2025"))
BUCKET = os.getenv("R2_BUCKET","quant-ml")

def test_r2_hourly_schema():
    from src.features.make_features_h1_to_cloud import read_hourly_from_r2
    df = read_hourly_from_r2(BUCKET, SYMS[0], YEAR)
    assert not df.empty, "hourly parquet empty"
    need = {"date","open","high","low","close","volume","price","year","month","symbol"}
    have = set(map(str.lower, df.columns))
    assert need.issubset(have), f"missing cols: {need - have}"
    # 时间单调
    assert df["date"].is_monotonic_increasing
    # RTH 粗检（NY 9:30–16:00 小时条大多落在 9:30,10:30,...,15:30）
    ny = pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York")
    assert (ny.dt.weekday < 5).mean() > 0.6  # 较宽松

def test_features_exist():
    from src.utils.r2_client import ls
    sym = SYMS[0]
    parts = ls(f"{BUCKET}/features_h1/symbol={sym}/year={YEAR:04d}")
    assert any("/month=" in p for p in parts), "features_h1 missing monthly partitions"

def test_labels_exist_and_binary():
    from src.utils.r2_client import ls, _fs, s3url
    import pyarrow.parquet as pq
    fs = _fs()
    sym = SYMS[0]
    months = [p for p in ls(f"{BUCKET}/train_h1/symbol={sym}/year={YEAR:04d}") if "/month=" in p]
    assert months, "labels monthly partitions missing"
    objs = [p for p in ls(months[0]) if p.lower().endswith(".parquet")]
    assert objs, "labels parquet missing"
    with fs.open(s3url(objs[0]), "rb") as f:
        df = pq.read_table(f).to_pandas()
    assert "y" in df.columns
    if not df.empty and df["y"].notna().any():
        vals = set(np.unique(df["y"].dropna().astype(int)))
        assert vals.issubset({0,1}), f"labels y not binary: {vals}"

def test_hmm_regime_available():
    from src.utils.r2_client import ls
    # 支持两种形态：新（月分区）或旧（单文件）
    monthly = []
    for base in [f"{BUCKET}/regime_hmm/symbol=QQQ", f"{BUCKET}/regime/symbol=QQQ"]:
        monthly = [p for p in ls(f"{base}/year={YEAR:04d}") if "/month=" in p] or monthly
    single = [p for p in ls(f"{BUCKET}/models/regime") if p.endswith("qqq_hmm.parquet")]
    assert monthly or single, "HMM regime not found (expect monthly under regime_hmm/... or models/regime/qqq_hmm.parquet)"
