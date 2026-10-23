"""因子库：月末截面因子（只用当时可得的数据）+ 预处理。

时点规则：
  因子在月末收盘后计算，次一交易日（下月第一个交易日）收盘价成交——研究和回测都按这个时点算远期收益，避免前视。
  基本面只用已过披露截止日的报告期（marketdata.available_period）；业绩报表是年初至今累计值，按报告期年化（不是 TTM）。

Alpha 因子（预期方向为正，已按方向取号）：
  EP     每股收益（年化）/ 股价                       价值
  BP     每股净资产 / 股价                            价值
  ROE    年化 ROE                                     质量
  GM     销售毛利率                                   质量
  CFQ    每股经营现金流 / 每股收益（截断到 [-3, 3]）  质量（利润含金量）
  NPG    净利润同比                                   成长
  REVG   营收同比                                     成长
  MOM    过去 12 个月剔除最近 1 个月的收益            动量
  REV1M  −最近 1 个月收益                             短期反转（A 股显著）
  LOWVOL −60 日收益波动                               低波
  LOWTO  −20 日平均换手率                             低换手（A 股流动性溢价）

风险模型的风格因子另见 riskmodel.py（规模、Beta、动量、波动、价值、流动性）。
预处理：MAD 去极值（中位数 ± 5 × MAD）→ 标准化 → 对行业哑变量和规模做回归取残差（行业、市值中性）→ 再标准化。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from cyron_veritas.fund import marketdata as M
from cyron_veritas.fund import risk as R

ALPHA = ["EP", "BP", "ROE", "GM", "CFQ", "NPG", "REVG", "MOM", "REV1M", "LOWVOL", "LOWTO"]
GROUP = {"EP": "价值", "BP": "价值", "ROE": "质量", "GM": "质量", "CFQ": "质量", "NPG": "成长", "REVG": "成长",
         "MOM": "动量", "REV1M": "反转", "LOWVOL": "低波", "LOWTO": "流动性"}
NAME = {"EP": "盈利收益率", "BP": "账面市值比", "ROE": "ROE", "GM": "毛利率", "CFQ": "现金流 / 利润", "NPG": "净利增速",
        "REVG": "营收增速", "MOM": "12-1 月动量", "REV1M": "1 月反转", "LOWVOL": "低波动", "LOWTO": "低换手"}
ANNUALIZE = {"0331": 4.0, "0630": 2.0, "0930": 4 / 3, "1231": 1.0}


@lru_cache(maxsize=1)
def panel() -> dict:
    """全样本宽表：收盘、总收益、换手率、流通市值、成交额。"""
    codes = sorted(M.universe()["code"])
    b = M.bars(codes)
    mv = R.market_view()
    close = M.wide(b, "close")
    vol = M.wide(b, "volume")
    to = M.wide(b, "turnover")
    float_shares = (vol / to.replace(0, np.nan)).rolling(20, min_periods=5).median()     # 流通股本 = 成交股数 / 换手率
    ret = mv["ret"].reindex(close.index)
    return {"close": close, "ret": ret, "turnover": to, "fmv": close * float_shares, "amount": M.wide(b, "amount"),
            "tri": (1 + ret.fillna(0)).cumprod(), "listed": close.notna().cumsum()}


def month_ends(start="2023-01-01", end=None) -> list[pd.Timestamp]:
    idx = panel()["close"].loc[start:end].index
    s = idx.to_series()
    return list(s.groupby(idx.to_period("M")).last())


def next_trading_day(d: pd.Timestamp) -> pd.Timestamp | None:
    idx = panel()["close"].index
    after = idx[idx > d]
    return after[0] if len(after) else None


def raw(d: pd.Timestamp) -> pd.DataFrame:
    """d 日收盘后的原始因子（行 = 股票）。只纳入上市 ≥ 120 个交易日、当日有成交的股票。"""
    p = panel()
    close, ret, to = p["close"].loc[:d], p["ret"].loc[:d], p["turnover"].loc[:d]
    px = close.iloc[-1]
    ok = px.notna() & (p["listed"].loc[:d].iloc[-1] >= 120)
    period = M.available_period(d)
    f = M.fundamentals(period).reindex(px.index)
    q = ANNUALIZE[period[4:]]
    tri = p["tri"].loc[:d]
    listed = p["listed"].loc[:d].iloc[-1]
    mom = (tri.iloc[-22] / tri.iloc[-253] - 1).where(listed >= 253) if len(tri) >= 253 else pd.Series(np.nan, index=px.index)
    out = pd.DataFrame({
        "EP": f["eps"] * q / px,
        "BP": f["bvps"] / px,
        "ROE": f["roe"] * q,
        "GM": f["gross_margin"],
        "CFQ": (f["ocfps"] / f["eps"].where(f["eps"].abs() > 1e-6)).clip(-3, 3),
        "NPG": f["np_yoy"].clip(-200, 500),
        "REVG": f["rev_yoy"].clip(-100, 300),
        "MOM": mom,
        "REV1M": -(tri.iloc[-1] / tri.iloc[-22] - 1),
        "LOWVOL": -ret.iloc[-60:].std(),
        "LOWTO": -to.iloc[-20:].mean(),
        "SIZE": np.log(p["fmv"].loc[:d].iloc[-1]),
    })
    return out[ok]


def mad_z(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if not mad or np.isnan(mad):
        return (s - s.mean()) / s.std()
    s = s.clip(med - 5 * mad, med + 5 * mad)
    return (s - s.mean()) / s.std()


def neutralize(z: pd.Series, industry: pd.Series, size: pd.Series) -> pd.Series:
    """对行业哑变量 + 规模做最小二乘，取残差（缺失值保持缺失）。"""
    ok = z.notna() & size.notna()
    if ok.sum() < 50:
        return z
    X = pd.get_dummies(industry[ok]).astype(float)
    X["SIZE"] = size[ok]
    beta, *_ = np.linalg.lstsq(X.values, z[ok].values, rcond=None)
    res = pd.Series(np.nan, index=z.index)
    res[ok] = z[ok].values - X.values @ beta
    return (res - res.mean()) / res.std()


def processed(d: pd.Timestamp) -> pd.DataFrame:
    """去极值、标准化、行业市值中性化后的 alpha 因子 z 分数，外加原始 SIZE 和 industry 列。"""
    r = raw(d)
    ind = pd.Series(M.sw_industry()).reindex(r.index).fillna("未分类")
    size = mad_z(r["SIZE"])
    out = pd.DataFrame({k: neutralize(mad_z(r[k]), ind, size) for k in ALPHA}, index=r.index)
    out["SIZE"] = size
    out["industry"] = ind
    return out
