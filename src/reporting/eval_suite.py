# -*- coding: utf-8 -*-
import os, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path

# 新增导入
from src.portfolio.optimize import hrp_weights, target_vol_weights, apply_static_weights

def _read_equity_series(sym: str, run_id: str, base_dir: Path) -> pd.Series:
    # 读取每个标的的回测权益曲线 CSV：backtests/out/<SYM>/<RUN_ID>/equity.csv
    p = base_dir / "backtests" / "out" / sym / run_id / "equity.csv"
    df = pd.read_csv(p)
    # 兼容列名：date / equity
    date_col = "date" if "date" in df.columns else "Date"
    eq_col   = "equity" if "equity" in df.columns else "Equity"
    s = pd.Series(df[eq_col].values, index=pd.to_datetime(df[date_col])).sort_index()
    # 过滤 NaN/0（避免 log 计算问题）
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s

def _equity_to_returns(equity: pd.Series) -> pd.Series:
    # 日收益：简单收益（等频bar-close，已含成本）
    r = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r

def _align_returns(symbols, run_id: str, base_dir: Path) -> pd.DataFrame:
    # 宽表：行=date, 列=symbol，值=日收益
    rets = []
    cols = []
    for s in symbols:
        try:
            eq = _read_equity_series(s, run_id, base_dir)
            r  = _equity_to_returns(eq)
            rets.append(r)
            cols.append(s)
        except Exception:
            # 某标的缺文件/空，跳过
            pass
    if not rets:
        raise RuntimeError("未找到任何标的的 equity.csv")
    wide = pd.concat(rets, axis=1)
    wide.columns = cols
    # 对齐日期并前向填充缺口（同一交易日的组合对齐）
    wide = wide.sort_index().fillna(0.0)
    return wide

def _metrics_from_returns(ret: pd.Series, rf: float = 0.0) -> dict:
    # 年化用 252
    if ret.empty:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0}
    nav = (1.0 + ret).cumprod()
    # CAGR
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    cagr  = nav.iloc[-1] ** (1.0 / years) - 1.0
    # Sharpe（日频 → 年化）
    ann_mu = ret.mean() * 252
    ann_sd = ret.std(ddof=0) * np.sqrt(252)
    sharpe = (ann_mu - rf) / (ann_sd + 1e-12)
    # MaxDD
    roll_max = nav.cummax()
    dd = (nav / roll_max - 1.0)
    maxdd = dd.min()
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(maxdd)}

def _load_symbol_sharpes(symbols, run_id: str, base_dir: Path) -> pd.Series:
    # 从 metrics.json 读取每个标的 Sharpe（若缺失，回退为基于 equity 计算）
    vals = {}
    for s in symbols:
        mj = base_dir / "backtests" / "out" / s / run_id / "metrics.json"
        try:
            with open(mj, "r") as f:
                m = json.load(f)
            sp = m.get("sharpe", None)
            if sp is None:
                # 兜底：从 equity 计算
                eq = _read_equity_series(s, run_id, base_dir)
                sp = _metrics_from_returns(_equity_to_returns(eq))["Sharpe"]
            vals[s] = float(sp)
        except Exception:
            # 兜底为 NaN
            vals[s] = np.nan
    return pd.Series(vals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--universe", nargs="+",
                    default=["AVGO","NVDA","TSLA","IBM","DELL","LLY","AMD","META","AAPL","MSFT","GOOGL","AMZN"])
    ap.add_argument("--project-root", default=".")
    # 组合扩展开关与参数
    ap.add_argument("--enable-hrp", action="store_true")
    ap.add_argument("--enable-target-vol", action="store_true")
    ap.add_argument("--target-vol", type=float, default=0.10)  # 10% 年化波动目标
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    run_id = args.run_id
    symbols = args.universe

    # 1) 对齐所有标的的日收益宽表
    wide = _align_returns(symbols, run_id, root)

    # 2) 计算 per-symbol Sharpe 用于“Filtered-EW”
    shp = _load_symbol_sharpes(symbols, run_id, root)
    keep_syms = [s for s in symbols if (s in wide.columns) and (shp.get(s, -9) >= 0)]
    if not keep_syms:
        keep_syms = [c for c in wide.columns]  # 全负时，回退为全量

    # 3) 组合日收益（静态日度再平衡）
    # 3.1 EW（全体）
    ew_w = pd.Series(1.0 / len(wide.columns), index=wide.columns)
    ret_ew = (wide * ew_w).sum(axis=1)

    # 3.2 Filtered-EW（剔除负Sharpe）
    few_w = pd.Series(0.0, index=wide.columns)
    few_w.loc[keep_syms] = 1.0 / len(keep_syms)
    ret_few = (wide * few_w).sum(axis=1)

    # 3.3 HRP（可选）
    portfolios = {
        "EW": ret_ew,
        "Filtered_EW": ret_few,
    }
    weights_dump = {
        "EW": ew_w.to_dict(),
        "Filtered_EW": few_w[few_w > 0].to_dict(),
    }

    if args.enable_hrp:
        try:
            w_hrp = hrp_weights(wide[keep_syms])  # 用筛后的标的更稳健
            ret_hrp = apply_static_weights(wide, w_hrp)
            portfolios["HRP"] = ret_hrp
            weights_dump["HRP"] = w_hrp
        except Exception as e:
            print(f"[WARN] HRP 失败: {e}")

    # 3.4 目标波动率（可选）
    if args.enable_target_vol:
        try:
            w_tv = target_vol_weights(wide[keep_syms], vol_target=args.target_vol)
            ret_tv = apply_static_weights(wide, w_tv)
            portfolios["TargetVol"] = ret_tv
            weights_dump["TargetVol"] = w_tv
        except Exception as e:
            print(f"[WARN] TargetVol 失败: {e}")

    # 4) 汇总指标 & 组合净值 DataFrame
    port_nav = {}
    metrics = {}
    for k, r in portfolios.items():
        nav = (1.0 + r.fillna(0.0)).cumprod()
        port_nav[k] = nav
        metrics[k] = _metrics_from_returns(r)

    nav_df = pd.DataFrame(port_nav).sort_index()
    nav_df.index.name = "date"

    # 5) 落盘
    analysis_dir = root / "backtests" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = analysis_dir / f"portfolio_{run_id}.csv"
    json_path = analysis_dir / f"portfolio_{run_id}.json"
    nav_df.to_csv(csv_path, index=True)

    summary = {
        "run_id": run_id,
        "included_symbols": symbols,
        "filtered_kept": keep_syms,
        "weights": weights_dump,
        "metrics": metrics,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Portfolio Summary ===")
    # 控制台表（简洁）
    def _fmt(m): return f"CAGR {m['CAGR']:.2%} | Sharpe {m['Sharpe']:.2f} | MaxDD {m['MaxDD']:.2%}"
    print(f"EW            : {_fmt(metrics['EW'])}")
    print(f"Filtered_EW   : {_fmt(metrics['Filtered_EW'])}")
    if "HRP" in metrics:
        print(f"HRP           : {_fmt(metrics['HRP'])}")
    if "TargetVol" in metrics:
        tv = metrics['TargetVol']
        print(f"TargetVol({args.target_vol:.0%}) : {_fmt(tv)}")

    print(f"\n[Saved] {csv_path}")
    print(f"[Saved] {json_path}")
