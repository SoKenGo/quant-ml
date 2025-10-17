# -*- coding: utf-8 -*-
# src/portfolio/optimize.py
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- HRP import (robust across versions) ---
try:
    from pypfopt.hierarchical_portfolio import HRPOpt
except Exception:
    try:
        from pypfopt import HRPOpt  # type: ignore
    except Exception as e:
        raise ImportError("无法导入 HRPOpt，请安装/升级 PyPortfolioOpt: pip install -U PyPortfolioOpt") from e

# Optional: mean-variance target-risk
_HAS_EF = True
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
except Exception:
    _HAS_EF = False


# -------------------------
# Helpers: aggregation
# -------------------------
def _compound_to_period(ret: pd.Series, freq: str) -> pd.Series:
    """Aggregate simple returns to period using compounding: (1+r).prod()-1"""
    return (1.0 + ret).groupby(pd.Grouper(freq=freq)).prod() - 1.0

def intraday_to_daily(returns_wide: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Aggregate intraday (e.g., 1H) simple returns to daily (or W/M) per column via compounding."""
    out = pd.DataFrame(
        {c: _compound_to_period(returns_wide[c].dropna(), freq) for c in returns_wide.columns}
    )
    return out.dropna(how="all")


# -------------------------
# Optimizers
# -------------------------
def _ivp_weights(returns_wide_d: pd.DataFrame) -> Dict[str, float]:
    """Inverse-variance portfolio (fallback when HRP/EF not desired)."""
    vol = returns_wide_d.std().replace(0.0, np.nan).dropna()
    w = (1.0 / (vol ** 2))  # inverse variance
    w = w / w.sum()
    return w.to_dict()

def hrp_weights(returns_wide_d: pd.DataFrame) -> Dict[str, float]:
    """HRP on DAILY returns (rows=date, cols=symbol)."""
    hrp = HRPOpt(returns_wide_d.dropna(how="all"))
    return hrp.optimize()

def target_vol_weights(returns_wide_d: pd.DataFrame, vol_target: float = 0.10) -> Dict[str, float]:
    """
    Target portfolio volatility (annualized). Uses PyPortfolioOpt EF if available,
    otherwise falls back to inverse-variance normalized (no leverage).
    """
    if _HAS_EF:
        mu = expected_returns.mean_historical_return(returns_wide_d, frequency=252)
        S = risk_models.sample_cov(returns_wide_d, frequency=252)
        ef = EfficientFrontier(mu, S)
        w = ef.efficient_risk(vol_target)
        return w
    warnings.warn("[opt] EfficientFrontier unavailable; falling back to IVP normalized.")
    return _ivp_weights(returns_wide_d)


# -------------------------
# Turnover controls
# -------------------------
def _band_and_step(prev: pd.Series, new: pd.Series, band: float = 0.02, max_step: float = 0.10) -> pd.Series:
    """
    No-trade band & max-step throttling on weights.
    - band: ignore changes within ±band (absolute)
    - max_step: limit per-rebalance absolute change per asset
    Returns adjusted weights (renormalized to sum<=1, then clipped to [0,1]).
    """
    prev = prev.fillna(0.0)
    new = new.fillna(0.0).reindex(prev.index).fillna(0.0)

    delta = new - prev
    # no-trade band
    delta = delta.where(delta.abs() >= band, 0.0)
    # max step
    delta = delta.clip(lower=-max_step, upper=max_step)

    w = prev + delta
    w = w.clip(lower=0.0)
    s = w.sum()
    if s > 1.0:
        w = w / s  # keep unlevered
    return w


# -------------------------
# Scheduling / application
# -------------------------
def compute_weight_schedule(
    returns_intraday: pd.DataFrame,
    method: str = "hrp",               # "hrp" | "target_vol" | "ivp"
    rebalance: str = "D",              # "D", "W", "M" (never faster than daily)
    lookback_days: int = 252,          # rolling window on DAILY returns
    band: float = 0.02,
    max_step: float = 0.10,
    vol_target: Optional[float] = None # only for method="target_vol"
) -> pd.DataFrame:
    """
    Build a schedule of weights at the chosen (>= daily) frequency.
    - Intraday inputs are first aggregated to DAILY for optimization.
    - Returns a DataFrame 'W' indexed by rebalance timestamp (period end), columns=symbol weights.
    """
    # 1) aggregate to daily for optimizer
    daily = intraday_to_daily(returns_intraday, freq="D").dropna(how="all")
    # 2) rebalance dates at desired (>= daily) frequency
    #    Use period end timestamps based on 'rebalance'
    period_end = daily.index.to_series().groupby(pd.Grouper(freq=rebalance)).last().dropna()

    W = []
    cols = returns_intraday.columns
    prev_w = pd.Series(0.0, index=cols)

    for dt in period_end:
        hist = daily.loc[:dt].tail(lookback_days)
        if hist.dropna(how="all").shape[0] < max(60, int(lookback_days * 0.25)):
            # warmup not enough daily obs; keep previous
            W.append(pd.Series(prev_w, name=dt))
            continue

        if method == "hrp":
            raw_w = hrp_weights(hist)
        elif method == "target_vol":
            raw_w = target_vol_weights(hist, vol_target=vol_target or 0.10)
        elif method == "ivp":
            raw_w = _ivp_weights(hist)
        else:
            raise ValueError(f"Unknown method: {method}")

        new_w = pd.Series(raw_w, index=cols).fillna(0.0)
        adj_w = _band_and_step(prev_w, new_w, band=band, max_step=max_step)
        W.append(pd.Series(adj_w, name=dt))
        prev_w = adj_w

    W = pd.DataFrame(W)
    return W


def apply_weight_schedule(
    returns_intraday: pd.DataFrame,
    weight_schedule: pd.DataFrame,
    costs_bps: float = 0.0,
    lag_bars: int = 1
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Broadcast daily/weekly/monthly weights to intraday bars (ffill) with an execution lag (default 1 bar).
    Transaction costs are charged on rebalance bars: cost = costs_bps/1e4 * sum(|Δw|).
    Returns:
      port_ret (Series), weights_ts (DataFrame aligned to intraday index)
    """
    # Align to intraday timeline, forward-fill, then lag by 1 bar to avoid look-ahead
    weights_ts = weight_schedule.reindex(returns_intraday.index, method="ffill").shift(lag_bars).fillna(0.0)

    # Portfolio returns
    port_ret = (returns_intraday * weights_ts).sum(axis=1)

    # Costs on rebalances (rows where weights changed)
    if costs_bps > 0.0:
        changed = weights_ts.ne(weights_ts.shift()).any(axis=1)
        turn = (weights_ts.loc[changed] - weights_ts.shift().loc[changed]).abs().sum(axis=1)
        tc = (costs_bps / 1e4) * turn  # proportional to notional turnover
        port_ret.loc[changed] = port_ret.loc[changed] - tc

    return port_ret, weights_ts


# -------------------------
# Convenience wrapper
# -------------------------
def build_portfolio_intraday(
    returns_intraday: pd.DataFrame,
    method: str = "hrp",
    rebalance: str = "D",
    lookback_days: int = 252,
    band: float = 0.02,
    max_step: float = 0.10,
    vol_target: Optional[float] = None,
    costs_bps: float = 0.0,
    lag_bars: int = 1
) -> Dict[str, object]:
    """
    End-to-end:
      1) Compute daily-or-slower weight schedule from intraday returns
      2) Broadcast to intraday with 1-bar lag
      3) Apply costs at rebalance times
    Returns dict with 'weights_schedule', 'weights_ts', 'portfolio_returns'
    """
    W = compute_weight_schedule(
        returns_intraday=returns_intraday,
        method=method,
        rebalance=rebalance,
        lookback_days=lookback_days,
        band=band,
        max_step=max_step,
        vol_target=vol_target,
    )
    port_ret, weights_ts = apply_weight_schedule(
        returns_intraday=returns_intraday,
        weight_schedule=W,
        costs_bps=costs_bps,
        lag_bars=lag_bars,
    )
    return {"weights_schedule": W, "weights_ts": weights_ts, "portfolio_returns": port_ret}
