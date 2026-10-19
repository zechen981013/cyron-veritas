"""风险指标：全部是纯函数，输入收益率序列或权重，输出数字。不调用大模型，可单测。

约定：收益率是日简单收益率（小数）；年化按 252 个交易日。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    """历史模拟法 VaR：返回正数，表示 1 日损失比例。"""
    return float(-np.quantile(returns.dropna(), 1 - level))


def historical_cvar(returns: pd.Series, level: float = 0.95) -> float:
    """预期损失（ES / CVaR）：超过 VaR 的那些日子的平均损失。"""
    r = returns.dropna()
    cutoff = np.quantile(r, 1 - level)
    return float(-r[r <= cutoff].mean())


def parametric_var(returns: pd.Series, level: float = 0.95) -> float:
    """正态分布假设下的 VaR（方差-协方差法）。"""
    r = returns.dropna()
    return float(-(r.mean() + stats.norm.ppf(1 - level) * r.std(ddof=1)))


def max_drawdown(returns: pd.Series) -> dict:
    """最大回撤与当前回撤（基于净值曲线）。"""
    nav = (1 + returns.fillna(0)).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1
    trough = dd.idxmin()
    start = nav.loc[:trough].idxmax()
    return {"max_drawdown": float(-dd.min()), "peak_date": start, "trough_date": trough,
            "current_drawdown": float(-dd.iloc[-1])}


def beta(asset: pd.Series, bench: pd.Series) -> float:
    df = pd.concat([asset, bench], axis=1).dropna()
    if len(df) < 30:
        return float("nan")
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=1)
    return float(cov[0, 1] / cov[1, 1])


def concentration(weights: pd.Series) -> dict:
    w = weights.sort_values(ascending=False)
    hhi = float((w ** 2).sum())
    return {"top1": float(w.iloc[0]), "top1_name": str(w.index[0]), "top5": float(w.iloc[:5].sum()),
            "hhi": hhi, "effective_n": float(1 / hhi) if hhi else float("nan")}


def risk_contribution(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """各持仓对组合波动率的贡献占比（合计 = 1）：w_i * (Σw)_i / (w'Σw)。"""
    cols = [c for c in weights.index if c in returns.columns]
    w = weights[cols].values
    cov = returns[cols].cov().values
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return pd.Series(0.0, index=cols)
    return pd.Series(w * (cov @ w) / port_var, index=cols)


def kupiec_pof(exceptions: int, n: int, level: float = 0.95) -> dict:
    """Kupiec 失败比例检验：VaR 被突破的次数是否和置信水平相符。p 值 < 0.05 说明模型可能低估或高估风险。"""
    p = 1 - level
    x = exceptions
    if n == 0:
        return {"lr": float("nan"), "p_value": float("nan")}
    phat = x / n
    def ll(prob):  # 对数似然，处理 0 次或全部突破的边界
        a = (n - x) * np.log(1 - prob) if prob < 1 else (0 if x == n else -np.inf)
        b = x * np.log(prob) if prob > 0 else (0 if x == 0 else -np.inf)
        return a + b
    lr = -2 * (ll(p) - ll(phat))
    return {"lr": float(lr), "p_value": float(1 - stats.chi2.cdf(lr, df=1))}


def var_backtest(returns: pd.Series, window: int = 250, level: float = 0.95, test_days: int = 250) -> dict:
    """滚动回测：每天用前 window 天的历史 VaR 预测当天，统计实际亏损超过 VaR 的天数。"""
    r = returns.dropna()
    test_days = min(test_days, len(r) - window)
    exceptions, rows = 0, []
    for i in range(len(r) - test_days, len(r)):
        v = historical_var(r.iloc[i - window:i], level)
        hit = bool(r.iloc[i] < -v)
        exceptions += hit
        rows.append((r.index[i], -v, r.iloc[i], hit))
    test = kupiec_pof(exceptions, test_days, level)
    return {"level": level, "window": window, "days": test_days, "exceptions": exceptions,
            "expected": (1 - level) * test_days, **test,
            "series": pd.DataFrame(rows, columns=["date", "var", "ret", "hit"]).set_index("date")}
