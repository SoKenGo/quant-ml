# -*- coding: utf-8 -*-
# src/portfolio/optimize.py
# -*- coding: utf-8 -*-
import pandas as pd
from typing import Dict

# --- 兼容不同版本的 HRP 导入路径 ---
try:
    # 新/常见路径
    from pypfopt.hierarchical_portfolio import HRPOpt
except Exception:
    try:
        # 有些版本支持顶层导入
        from pypfopt import HRPOpt  # type: ignore
    except Exception as e:
        raise ImportError(
            "无法导入 HRPOpt，请检查 PyPortfolioOpt 版本。"
            " 尝试: pip install -U PyPortfolioOpt"
        ) from e

from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier  # 依赖 cvxpy

def _ivp_weights(returns_wide: pd.DataFrame) -> Dict[str, float]:
    vol = returns_wide.std().replace(0.0, pd.NA).dropna()
    w = (1.0 / vol)
    w = w / w.sum()
    return w.to_dict()

def hrp_weights(returns_wide: pd.DataFrame) -> Dict[str, float]:
    # 行=date, 列=symbol，元素为日收益
    hrp = HRPOpt(returns_wide)
    return hrp.optimize()

def target_vol_weights(returns_wide: pd.DataFrame, vol_target: float = 0.10) -> Dict[str, float]:
    # 均值-方差前沿上的目标风险组合（需要 cvxpy）
    mu = expected_returns.mean_historical_return(returns_wide)
    S  = risk_models.sample_cov(returns_wide)
    ef = EfficientFrontier(mu, S)
    w  = ef.efficient_risk(vol_target)
    return w

def apply_static_weights(returns_wide: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """给定静态权重，生成组合日收益序列（按日等频再平衡）。"""
    w = pd.Series(weights).reindex(returns_wide.columns).fillna(0.0)
    port_ret = (returns_wide * w).sum(axis=1)
    return port_ret
