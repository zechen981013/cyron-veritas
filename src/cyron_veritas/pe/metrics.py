"""基金与项目业绩指标：XIRR、TVPI / DPI / RVPI、MOIC、Kaplan-Schoar PME；限售股流动性折扣（AAP 模型）。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


def xirr(flows: list[tuple[pd.Timestamp, float]]) -> float:
    """不规则现金流 IRR（流出为负、流入为正）。无解时返回 nan。"""
    flows = [(pd.Timestamp(d), float(v)) for d, v in flows if abs(v) > 1e-9]
    if not flows or all(v >= 0 for _, v in flows) or all(v <= 0 for _, v in flows):
        return float("nan")
    t0 = min(d for d, _ in flows)
    yrs = np.array([(d - t0).days / 365 for d, _ in flows])
    vals = np.array([v for _, v in flows])
    f = lambda r: float(np.sum(vals / (1 + r) ** yrs))
    try:
        return brentq(f, -0.99, 10.0)
    except ValueError:
        return float("nan")


def multiples(paid_in: float, distributed: float, nav: float) -> dict:
    if paid_in <= 0:
        return {"TVPI": float("nan"), "DPI": float("nan"), "RVPI": float("nan")}
    return {"TVPI": (distributed + nav) / paid_in, "DPI": distributed / paid_in, "RVPI": nav / paid_in}


def ks_pme(contribs: list[tuple], dists: list[tuple], nav: float, index: pd.Series, asof) -> float:
    """Kaplan-Schoar PME：把每笔现金流按指数涨跌换算到估值日再比。> 1 表示跑赢指数。"""
    idx = index.sort_index().ffill()
    at = lambda d: float(idx.loc[:pd.Timestamp(d)].iloc[-1])
    iT = at(asof)
    fv_c = sum(v * iT / at(d) for d, v in contribs)
    fv_d = sum(v * iT / at(d) for d, v in dists)
    return (fv_d + nav) / fv_c if fv_c else float("nan")


def aap_discount(sigma: float, years: float) -> float:
    """限售股流动性折扣：亚式看跌期权（AAP）模型，LoMD = N(σ√T / 2) − N(−σ√T / 2)。
    中国证券投资基金业协会《证券投资基金投资流通受限股票估值指引（试行）》推荐的方法；σ 用股票自身年化波动率。"""
    if years <= 0 or sigma <= 0:
        return 0.0
    x = sigma * math.sqrt(years) / 2
    return float(norm.cdf(x) - norm.cdf(-x))
