# -*- coding: utf-8 -*-
"""
Reporting suite (hourly-aware):
- Reads per-symbol equity/trades from backtests/out/<SYM>/<RUN_ID>/
- Aggregates to daily for portfolio combos (EW, Filtered-EW, HRP, Target-Vol)
- Outputs:
    backtests/analysis/portfolio_<RUN_ID>.csv
    backtests/analysis/portfolio_<RUN_ID>.json
    backtests/analysis/summary_<RUN_ID>.csv  # used by nightly top-5
    backtests/analysis/calibration_<SYM>_<RUN_ID>.csv
    backtests/analysis/drawdowns_<SYM>_<RUN_ID>.csv
Includes: WF yearly metrics, turnover, regime hit rate (HMM), avg slippage vs assumption.
"""
from __future__ import annotations

import os, json, argparse, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.portfolio.optimize import (
    hrp_weights, target_vol_weights, apply_static_weights,
    intraday_to_daily,
)

# ---------- robust R2 regime/prob loaders (reuse engine conventions) ----------
from src.utils.r2_client import _fs, s3url
from pyarrow import parquet as pq

def _fs_client(): return _fs()

def _read_parquet_from_r2(key: str) -> pd.DataFrame:
    fs = _fs_client()
    with fs.open(s3url(key), "rb") as f:
        return pq.read_table(f).to_pandas()

def _ls(prefix: str) -> List[str]:
    fs = _fs_client()
    try:
        return fs.ls(prefix if prefix.startswith("s3://") else f"s3://{prefix}")
    except FileNotFoundError:
        return []

def _load_probs_from_r2(symbol: str, years: List[int]) -> Optional[pd.Series]:
    bucket = os.getenv("R2_BUCKET", "quant-ml")
    candidates = [
        f"{bucket}/models/xgb_oos_h1/symbol={symbol.upper()}",
        f"{bucket}/models/xgb_oos/symbol={symbol.upper()}/timeframe=1h",
    ]
    parts: List[pd.DataFrame] = []
    for base in candidates:
        months=[]
        for y in years: months += [p for p in _ls(f"{base}/year={y:04d}") if "/month=" in p]
        for mdir in sorted(months):
            for obj in [p for p in _ls(mdir) if p.lower().endswith(".parquet")]:
                try: parts.append(_read_parquet_from_r2(obj))
                except Exception: pass
        if parts: break
    if not parts:
        # legacy single-file fallback
        base = f"{bucket}/models/xgb_oos/symbol={symbol.upper()}"
        for fname in ["probs_sigmoid.parquet", "probs.parquet", "probs_none.parquet"]:
            try:
                parts.append(_read_parquet_from_r2(f"{base}/{fname}")); break
            except Exception: pass
    if not parts: return None
    df = pd.concat(parts, ignore_index=True)
    col = "prob_up" if "prob_up" in df.columns else ("prob" if "prob" in df.columns else None)
    tcol = "date" if "date" in df.columns else ("ts" if "ts" in df.columns else None)
    if col is None or tcol is None: return None
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").values,
                  index=pd.to_datetime(df[tcol], errors="coerce").tz_localize(None),
                  name="prob_up").dropna().sort_index()
    return s

def _load_hmm_regime_from_r2(symbol: str = "QQQ", years: Optional[List[int]] = None) -> Optional[pd.Series]:
    bucket = os.getenv("R2_BUCKET", "quant-ml")
    parts: List[pd.DataFrame] = []
    bases = [f"{bucket}/regime_hmm/symbol={symbol.upper()}",
             f"{bucket}/regime/symbol={symbol.upper()}"]
    years = years or []
    for base in bases:
        months=[]
        if years:
            for y in years: months += [p for p in _ls(f"{base}/year={y:04d}") if "/month=" in p]
        else:
            # scan all
            for ydir in _ls(base):
                if "/year=" in ydir:
                    months += [p for p in _ls(ydir) if "/month=" in p]
        for mdir in sorted(months):
            for obj in [p for p in _ls(mdir) if p.lower().endswith(".parquet")]:
                try: parts.append(_read_parquet_from_r2(obj))
                except Exception: pass
        if parts: break
    if not parts: return None
    df = pd.concat(parts, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else ("ts" if "ts" in df.columns else None)
    st_col   = "regime_bull" if "regime_bull" in df.columns else None
    if date_col is None or st_col is None: return None
    return pd.Series(pd.to_numeric(df[st_col], errors="coerce").fillna(0).astype(int).values,
                     index=pd.to_datetime(df[date_col]).tz_localize(None),
                     name="regime_bull").sort_index()

# ---------- local I/O from backtests/out ----------
def _read_equity_series(sym: str, run_id: str, base_dir: Path) -> pd.Series:
    p = base_dir / "backtests" / "out" / sym / run_id / "equity.csv"
    df = pd.read_csv(p)
    # accept ts|date
    tcol = "ts" if "ts" in df.columns else ("date" if "date" in df.columns else None)
    ecol = "equity" if "equity" in df.columns else ("Equity" if "Equity" in df.columns else None)
    if tcol is None or ecol is None:
        raise RuntimeError(f"equity.csv malformed for {sym}")
    s = pd.Series(df[ecol].values, index=pd.to_datetime(df[tcol])).sort_index()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s

def _read_trades(sym: str, run_id: str, base_dir: Path) -> pd.DataFrame:
    p = base_dir / "backtests" / "out" / sym / run_id / "trades.csv"
    if not p.exists(): return pd.DataFrame(columns=["ts","action","size","price","value","commission","pnl","pnlcomm"])
    df = pd.read_csv(p)
    tcol = "ts" if "ts" in df.columns else ("date" if "date" in df.columns else None)
    if tcol is None: df["ts"] = pd.NaT
    else: df["ts"] = pd.to_datetime(df[tcol])
    return df

def _equity_to_returns(equity: pd.Series) -> pd.Series:
    r = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r

def _align_returns(symbols, run_id: str, base_dir: Path) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    rets = []
    cols = []
    by_year: Dict[str, List[int]] = {}
    for s in symbols:
        try:
            eq = _read_equity_series(s, run_id, base_dir)
            r  = _equity_to_returns(eq)
            rets.append(r); cols.append(s)
            yrs = sorted(set(pd.to_datetime(eq.index).year))
            by_year[s] = yrs
        except Exception:
            pass
    if not rets:
        raise RuntimeError("未找到任何标的的 equity.csv")
    wide = pd.concat(rets, axis=1)
    wide.columns = cols
    wide = wide.sort_index().fillna(0.0)  # intraday aligned returns
    return wide, by_year

def _metrics_from_returns(ret: pd.Series, rf: float = 0.0) -> dict:
    if ret.empty:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0}
    nav = (1.0 + ret).cumprod()
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    cagr  = nav.iloc[-1] ** (1.0 / years) - 1.0
    ann_mu = ret.mean() * 252 * 6.5  # hour bars -> ~6.5 bars per day
    ann_sd = ret.std(ddof=0) * np.sqrt(252 * 6.5)
    sharpe = (ann_mu - rf) / (ann_sd + 1e-12)
    roll_max = nav.cummax()
    dd = (nav / roll_max - 1.0)
    maxdd = dd.min()
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(maxdd)}

def _wf_by_year(ret: pd.Series) -> pd.DataFrame:
    if ret.empty: return pd.DataFrame(columns=["year","cagr","sharpe","maxdd"])
    df = ret.to_frame("r").copy()
    df["year"] = df.index.year
    rows=[]
    for y, g in df.groupby("year"):
        m=_metrics_from_returns(g["r"])
        rows.append(dict(year=int(y), cagr=m["CAGR"], sharpe=m["Sharpe"], maxdd=m["MaxDD"]))
    return pd.DataFrame(rows).sort_values("year")

def _turnover_and_hitrate(sym: str, trades: pd.DataFrame, eq: pd.Series,
                          hmm_gate: Optional[pd.Series]) -> Tuple[float,float,Optional[dict]]:
    """Return (avg daily notional turnover ratio, win_rate, hit_by_regime)"""
    if trades.empty: return 0.0, 0.0, None
    # Notional turnover divided by last equity (approx)
    notional = trades.get("value", pd.Series(0)).abs().sum()
    eq_last  = float(eq.iloc[-1]) if len(eq) else 1.0
    days = max(1, len(pd.Index(eq.index.normalize()).unique()))
    t_over = float(notional / max(1e-9, eq_last)) / days

    # win rate by trade-close pnl
    closes = trades[trades["action"]=="CLOSE"]
    win_rate = float((closes.get("pnlcomm", closes.get("pnl", 0.0)) > 0).mean()) if len(closes) else 0.0

    # hit rate by HMM regime at entry time
    reg = None
    if hmm_gate is not None and "BUY" in trades["action"].values:
        ent = trades[trades["action"]=="BUY"].copy()
        ent["ts"] = pd.to_datetime(ent["ts"])
        # entry outcome -> match following CLOSE pnl for same trade size sign
        # Simplify: treat "CLOSE" after this BUY as the paired exit in sequence
        exits = trades[trades["action"]=="CLOSE"].copy()
        exits["ts"] = pd.to_datetime(exits["ts"])
        exits = exits.sort_values("ts").reset_index(drop=True)
        ent = ent.sort_values("ts").reset_index(drop=True)
        n = min(len(ent), len(exits))
        ent = ent.iloc[:n]; exits = exits.iloc[:n]
        outcome = (exits.get("pnlcomm", exits.get("pnl", 0.0)).values > 0).astype(int)
        # regime at entry
        rg = hmm_gate.reindex(ent["ts"], method="ffill").fillna(0).astype(int).values
        reg = {
            "bull_hit_rate": float(outcome[rg==1].mean()) if (rg==1).any() else np.nan,
            "nonbull_hit_rate": float(outcome[rg==0].mean()) if (rg==0).any() else np.nan,
            "n_pairs": int(n),
        }
    return t_over, win_rate, reg

def _slippage_check(sym: str, trades: pd.DataFrame, assumed_bps: float,
                    open_lookup: Optional[pd.Series]) -> Optional[float]:
    """
    If open_lookup is provided (Series of 1H OPEN indexed by ts),
    compute mean absolute slippage in bps on entries: |fill-open|/open*1e4.
    Otherwise return assumed_bps.
    """
    if open_lookup is None or trades.empty: return float(assumed_bps)
    ent = trades[trades["action"].isin(["BUY","SELL"])].copy()
    if ent.empty: return float(assumed_bps)
    ent["ts"] = pd.to_datetime(ent["ts"])
    ref = open_lookup.reindex(ent["ts"])
    ok = ref.notna() & pd.to_numeric(ent["price"], errors="coerce").notna()
    if not ok.any(): return float(assumed_bps)
    slip = (ent.loc[ok, "price"].astype(float) - ref.loc[ok].astype(float)).abs() / ref.loc[ok].astype(float) * 1e4
    return float(slip.mean())

# Optional: 1H OPEN fetch for slippage check
def _load_h1_open_from_r2(symbol: str, years: List[int]) -> Optional[pd.Series]:
    bucket = os.getenv("R2_BUCKET", "quant-ml")
    parts=[]
    for y in years:
        year_prefix = f"{bucket}/h1/symbol={symbol.upper()}/year={y:04d}"
        for mdir in [p for p in _ls(year_prefix) if "/month=" in p]:
            for obj in [p for p in _ls(mdir) if p.lower().endswith(".parquet")]:
                try: parts.append(_read_parquet_from_r2(obj))
                except Exception: pass
    if not parts: return None
    df = pd.concat(parts, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    tcol = "ts" if "ts" in df.columns else ("date" if "date" in df.columns else None)
    ocol = "o" if "o" in df.columns else ("open" if "open" in df.columns else None)
    if tcol is None or ocol is None: return None
    s = pd.Series(pd.to_numeric(df[ocol], errors="coerce").values,
                  index=pd.to_datetime(df[tcol], errors="coerce").tz_localize(None)).dropna().sort_index()
    return s

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--universe", nargs="+",
                    default=["NVDA","TSLA","RKLB"])
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--enable-hrp", action="store_true")
    ap.add_argument("--enable-target-vol", action="store_true")
    ap.add_argument("--target-vol", type=float, default=0.10)
    ap.add_argument("--assumed-slippage-bps", type=float, default=5.0)
    ap.add_argument("--hmm-symbol", default="QQQ")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    run_id = args.run_id
    symbols = [s.upper() for s in args.universe]

    # 1) read intraday returns wide + years per symbol
    wide_intraday, years_by_sym = _align_returns(symbols, run_id, root)
    # also make daily-aggregated for portfolio construction
    wide_daily = intraday_to_daily(wide_intraday)

    # 2) Filter symbols by Sharpe >= 0 (daily proxy)
    shp = {s: _metrics_from_returns(intraday_to_daily(wide_intraday[[s]])[s])["Sharpe"] for s in wide_intraday.columns}
    keep_syms = [s for s in wide_intraday.columns if shp.get(s, -9) >= 0]
    if not keep_syms: keep_syms = list(wide_intraday.columns)

    # 3) Portfolio combinations (static daily weights)
    ew_w = pd.Series(1.0 / len(wide_daily.columns), index=wide_daily.columns)
    ret_ew = (wide_daily * ew_w).sum(axis=1)

    few_w = pd.Series(0.0, index=wide_daily.columns)
    few_w.loc[keep_syms] = 1.0 / len(keep_syms)
    ret_few = (wide_daily * few_w).sum(axis=1)

    portfolios = {"EW": ret_ew, "Filtered_EW": ret_few}
    weights_dump = {"EW": ew_w.to_dict(), "Filtered_EW": few_w[few_w > 0].to_dict()}

    if args.enable_hrp:
        try:
            w_hrp = hrp_weights(wide_daily[keep_syms])
            ret_hrp = apply_static_weights(wide_daily, w_hrp)
            portfolios["HRP"] = ret_hrp
            weights_dump["HRP"] = w_hrp
        except Exception as e:
            print(f"[WARN] HRP 失败: {e}")

    if args.enable_target_vol:
        try:
            w_tv = target_vol_weights(wide_daily[keep_syms], vol_target=args.target_vol)
            ret_tv = apply_static_weights(wide_daily, w_tv)
            portfolios["TargetVol"] = ret_tv
            weights_dump["TargetVol"] = w_tv
        except Exception as e:
            print(f"[WARN] TargetVol 失败: {e}")

    # 4) Per-symbol metrics, WF, turnover, regime hit-rate, slippage
    rows=[]
    bucket_slip_cache: Dict[str, Optional[pd.Series]] = {}
    hmm = _load_hmm_regime_from_r2(args.hmm_symbol, sorted(set(wide_intraday.index.year))) or None

    for s in wide_intraday.columns:
        eq = _read_equity_series(s, run_id, root)
        m  = _metrics_from_returns(_equity_to_returns(eq))
        tr = _read_trades(s, run_id, root)
        # slippage check (load 1H open lazily per symbol)
        if s not in bucket_slip_cache:
            bucket_slip_cache[s] = _load_h1_open_from_r2(s, years_by_sym.get(s, []))
        slip = _slippage_check(s, tr, args.assumed_slippage_bps, bucket_slip_cache[s])
        # turnover & hit-rate
        tovr, winr, reg = _turnover_and_hitrate(s, tr, eq, hmm)
        rows.append(dict(
            symbol=s, cagr=m["CAGR"], sharpe=m["Sharpe"], maxdd=m["MaxDD"],
            win_rate=winr, turnover=tovr, avg_slippage_bps=slip,
            bull_hit_rate=(reg or {}).get("bull_hit_rate", np.nan),
            nonbull_hit_rate=(reg or {}).get("nonbull_hit_rate", np.nan),
        ))
        # calibration table per symbol
        try:
            probs = _load_probs_from_r2(s, years_by_sym.get(s, []))
            if probs is not None and not tr.empty:
                ent = tr[tr["action"]=="BUY"].copy()
                ent["ts"] = pd.to_datetime(ent["ts"])
                exits = tr[tr["action"]=="CLOSE"].copy().sort_values("ts").reset_index(drop=True)
                ent = ent.sort_values("ts").reset_index(drop=True)
                n = min(len(ent), len(exits))
                ent = ent.iloc[:n]; exits = exits.iloc[:n]
                y = (exits.get("pnlcomm", exits.get("pnl", 0.0)) > 0).astype(int).values
                p = probs.reindex(ent["ts"], method="ffill").clip(0,1).values
                bins = np.clip((p * 10).astype(int), 0, 9)  # deciles 0..9
                dfc = pd.DataFrame({"bin": bins, "p": p, "y": y})
                cal = dfc.groupby("bin").agg(p_hat=("p","mean"), y_rate=("y","mean"), n=("y","size")).reset_index()
                out_dir = root / "backtests" / "analysis"
                out_dir.mkdir(parents=True, exist_ok=True)
                cal.to_csv(out_dir / f"calibration_{s}_{run_id}.csv", index=False)
        except Exception as e:
            print(f"[WARN] calibration for {s} failed: {e}")

        # drawdown distribution snapshot
        try:
            nav = (1.0 + _equity_to_returns(eq)).cumprod()
            dd = nav / nav.cummax() - 1.0
            q = dd.quantile([0.1,0.25,0.5,0.75,0.9])
            dd_df = pd.DataFrame({"quantile": q.index, "drawdown": q.values})
            out_dir = root / "backtests" / "analysis"
            dd_df.to_csv(out_dir / f"drawdowns_{s}_{run_id}.csv", index=False)
        except Exception:
            pass

    # 5) Portfolio outputs
    port_nav = {}
    metrics = {}
    for k, r in portfolios.items():
        nav = (1.0 + r.fillna(0.0)).cumprod()
        port_nav[k] = nav
        metrics[k] = {
            "CAGR": float((nav.iloc[-1] ** (252/len(nav)) - 1.0) if len(nav) else 0.0),
            "Sharpe": float((r.mean()*np.sqrt(252)) / (r.std(ddof=0)+1e-12)) if len(r) else 0.0,
            "MaxDD": float((nav/nav.cummax()-1.0).min()) if len(nav) else 0.0,
        }
    nav_df = pd.DataFrame(port_nav).sort_index(); nav_df.index.name="date"

    analysis_dir = root / "backtests" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    nav_df.to_csv(analysis_dir / f"portfolio_{run_id}.csv", index=True)
    with open(analysis_dir / f"portfolio_{run_id}.json", "w") as f:
        json.dump({
            "run_id": run_id,
            "included_symbols": symbols,
            "filtered_kept": keep_syms,
            "weights": weights_dump,
            "metrics": metrics,
        }, f, indent=2)

    # 6) Top-5 summary CSV (for nightly console)
    summ = pd.DataFrame(rows)
    summ = summ[["symbol","sharpe","cagr","maxdd","win_rate","turnover","avg_slippage_bps",
                 "bull_hit_rate","nonbull_hit_rate"]].sort_values("sharpe", ascending=False)
    summ.to_csv(analysis_dir / f"summary_{run_id}.csv", index=False)

    # 7) Console summary
    def _fmt_row(r): return f"CAGR {r['cagr']:.2%} | Sharpe {r['sharpe']:.2f} | MaxDD {r['maxdd']:.2%}"
    print("\n=== Portfolio Summary ===")
    for k in portfolios.keys():
        m=metrics[k]; print(f"{k:<12}: CAGR {m['CAGR']:.2%} | Sharpe {m['Sharpe']:.2f} | MaxDD {m['MaxDD']:.2%}")
    print("\n=== Per-Symbol (sorted by Sharpe) ===")
    for _, r in summ.iterrows():
        print(f"{r['symbol']:<6} : {_fmt_row(r)} | Win {r['win_rate']:.1%} | TO {r['turnover']:.3f}/day | Slip {r['avg_slippage_bps']:.1f}bps")

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
