"""DCF（FCFF 法）与 WACC。纯函数：同样的假设永远得到同样的结果，Excel 模型用同一套公式，可以互相核对。

  FCFF_t = EBIT_t × (1 − 税率) + D&A_t − 资本开支_t − ΔNWC_t
  终值    = FCFF_N × (1 + g) / (WACC − g)                     （戈登增长模型，年末折现）
  企业价值 = Σ FCFF_t / (1+WACC)^t + 终值 / (1+WACC)^N
  股权价值 = 企业价值 − 有息负债 + 现金 + 其他类现金资产（需年报佐证）− 少数股东权益
  每股价值 = 股权价值 / 总股本 × 汇率（报表币种 → 股价币种）
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DCFInputs:
    base_year: int
    base_revenue: float            # 亿元
    base_nwc: float                # 亿元（基期经营性营运资本）
    revenue_growth: list[float]    # 每年增速，长度 = 预测年数
    ebit_margin: list[float]
    tax_rate: float
    da_pct: float                  # 占收入
    capex_pct: float
    nwc_pct: float                 # 经营性营运资本 / 收入
    wacc: float
    terminal_growth: float
    debt: float
    cash: float
    minority: float
    shares: float                  # 亿股
    notes: dict = field(default_factory=dict)  # 每个假设的依据（给报告和 Excel 用）
    extra_cash: float = 0.0        # 亿元：计入其他科目的定期存款 / 理财等，必须有年报出处
    fx_to_price: float = 1.0       # 报表币种 → 股价币种（例如腾讯报表人民币、股价港元）

    def validate(self) -> list[str]:
        errs = []
        n = len(self.revenue_growth)
        if n < 3 or n > 10:
            errs.append("预测期应为 3-10 年")
        if len(self.ebit_margin) != n:
            errs.append("ebit_margin 长度要和 revenue_growth 一致")
        if not (0.03 <= self.wacc <= 0.20):
            errs.append("WACC 应在 3%-20% 之间")
        if self.terminal_growth >= self.wacc - 0.01:
            errs.append("永续增长率必须明显低于 WACC（至少低 1 个百分点）")
        if not (-0.02 <= self.terminal_growth <= 0.04):
            errs.append("永续增长率应在 -2% 到 4% 之间（不应超过长期名义 GDP 增速）")
        if self.shares <= 0:
            errs.append("股本必须为正")
        if self.extra_cash < 0:
            errs.append("其他类现金资产不能为负")
        if self.extra_cash > 0 and not self.notes.get("extra_cash"):
            errs.append("填了其他类现金资产（extra_cash）就必须在 notes 里写明年报出处（科目、金额、页码）")
        return errs


def project(x: DCFInputs) -> pd.DataFrame:
    n = len(x.revenue_growth)
    years = [x.base_year + i for i in range(1, n + 1)]
    rev, nwc_prev, rows = x.base_revenue, x.base_nwc, []
    for i in range(n):
        rev = rev * (1 + x.revenue_growth[i])
        ebit = rev * x.ebit_margin[i]
        nopat = ebit * (1 - x.tax_rate)
        da, capex = rev * x.da_pct, rev * x.capex_pct
        nwc = rev * x.nwc_pct
        d_nwc = nwc - nwc_prev
        fcff = nopat + da - capex - d_nwc
        df = 1 / (1 + x.wacc) ** (i + 1)
        rows.append(dict(year=years[i], t=i + 1, revenue=rev, growth=x.revenue_growth[i], ebit=ebit, ebit_margin=x.ebit_margin[i],
                         nopat=nopat, da=da, capex=capex, nwc=nwc, d_nwc=d_nwc, fcff=fcff, discount=df, pv_fcff=fcff * df))
        nwc_prev = nwc
    return pd.DataFrame(rows).set_index("year")


def value(x: DCFInputs, wacc: float | None = None, g: float | None = None) -> dict:
    if wacc is not None or g is not None:
        x = DCFInputs(**{**asdict(x), "wacc": wacc if wacc is not None else x.wacc,
                         "terminal_growth": g if g is not None else x.terminal_growth})
    p = project(x)
    n = len(p)
    tv = p["fcff"].iloc[-1] * (1 + x.terminal_growth) / (x.wacc - x.terminal_growth)
    pv_tv = tv / (1 + x.wacc) ** n
    ev = p["pv_fcff"].sum() + pv_tv
    equity = ev - x.debt + x.cash + x.extra_cash - x.minority
    return {"projection": p, "pv_fcff_sum": float(p["pv_fcff"].sum()), "terminal_value": float(tv), "pv_terminal": float(pv_tv),
            "enterprise_value": float(ev), "equity_value": float(equity), "per_share": float(equity / x.shares * x.fx_to_price),
            "tv_share_of_ev": float(pv_tv / ev)}


def sensitivity(x: DCFInputs, wacc_step: float = 0.005, g_step: float = 0.005, n: int = 2) -> pd.DataFrame:
    """每股价值对 WACC（行）和永续增长率（列）的敏感性表。"""
    waccs = [round(x.wacc + i * wacc_step, 4) for i in range(-n, n + 1)]
    gs = [round(x.terminal_growth + j * g_step, 4) for j in range(-n, n + 1)]
    data = [[value(x, w, g)["per_share"] if g < w - 0.005 else np.nan for g in gs] for w in waccs]
    return pd.DataFrame(data, index=waccs, columns=gs)


def wacc(rf: float, beta: float, erp: float, cost_of_debt: float, tax_rate: float, equity_value: float, debt: float) -> dict:
    """CAPM 股权成本 + 税后债务成本，按市值权重加权。"""
    ke = rf + beta * erp
    kd_after = cost_of_debt * (1 - tax_rate)
    total = equity_value + debt
    we, wd = equity_value / total, debt / total
    return {"rf": rf, "beta": beta, "erp": erp, "cost_of_equity": ke, "cost_of_debt": cost_of_debt,
            "after_tax_cost_of_debt": kd_after, "weight_equity": we, "weight_debt": wd, "wacc": we * ke + wd * kd_after}


def blume_beta(raw: float) -> float:
    """Blume 调整：beta 向 1 回归（彭博默认做法）。"""
    return 0.67 * raw + 0.33


def implied_growth(x: DCFInputs, price: float, lo: float = -0.30, hi: float = 0.60, tol: float = 1e-6) -> float | None:
    """反向 DCF：其他假设不变，预测期每年收入增速取同一个值 g，求让每股价值 = 现价的 g（市场隐含增速）。
    用来做「预期差」：市场价格隐含的增长 vs 我们的假设 vs 历史。区间内找不到解（如利润为负）返回 None。"""
    n = len(x.revenue_growth)

    def f(g: float) -> float:
        y = DCFInputs(**{**asdict(x), "revenue_growth": [g] * n})
        if y.terminal_growth >= y.wacc:
            return float("nan")
        return value(y)["per_share"] - price

    a, b = f(lo), f(hi)
    if np.isnan(a) or np.isnan(b) or a * b > 0:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        m = f(mid)
        if abs(m) < tol * max(1.0, price):
            return mid
        if (m > 0) == (b > 0):
            hi, b = mid, m
        else:
            lo, a = mid, m
    return (lo + hi) / 2
