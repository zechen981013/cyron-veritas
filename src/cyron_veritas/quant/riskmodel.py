"""多因子风险模型（Barra 思路的简化版）。

暴露：申万一级行业哑变量（没有截距，行业因子合起来就是市场）+ 6 个风格因子 z 分数（月末更新）：
  规模 ln(流通市值)、Beta（250 日对股票池流通市值加权收益，至少 120 日）、动量（12-1 月）、波动（60 日）、价值（BP）、流动性（20 日换手）。
因子收益：每个交易日对全部股票做截面加权最小二乘（权重 √流通市值），暴露用上一个月末的值（不用当天信息）。
协方差：因子收益 EWMA（半衰期 90 个交易日）× 252；特异方差：残差平方 EWMA（半衰期 60）× 252，
  观测不足 40 天的股票用截面中位数，再向截面中位数收缩 30%（减少估计噪声）。
组合主动风险 = (w − w_b)ᵀ (X F Xᵀ + D) (w − w_b)。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from cyron_veritas.config import ROOT
from cyron_veritas.fund import marketdata as M
from cyron_veritas.quant import factors as F

OUT = ROOT / "data" / "quant"
STYLES = ["规模", "Beta", "动量", "波动", "价值", "流动性"]
HL_COV, HL_SPEC, SHRINK = 90, 60, 0.3


def _ewma_weights(n: int, hl: int) -> np.ndarray:
    w = 0.5 ** (np.arange(n)[::-1] / hl)
    return w / w.sum()


def exposures(d: pd.Timestamp) -> pd.DataFrame:
    """d 日收盘后的暴露矩阵（行 = 股票，列 = 行业 + 风格）。"""
    p = F.panel()
    r = F.raw(d)
    ret = p["ret"].loc[:d]
    fmv = p["fmv"].loc[:d].iloc[-1]
    mkt = (ret.fillna(0) * fmv.reindex(ret.columns).fillna(0)).sum(axis=1) / fmv.reindex(ret.columns).fillna(0).sum()
    rr, mm = ret.iloc[-250:], mkt.iloc[-250:]
    beta = rr.apply(lambda s: s.cov(mm) / mm.var() if s.notna().sum() >= 120 else np.nan)
    st = pd.DataFrame({"规模": F.mad_z(r["SIZE"]), "Beta": F.mad_z(beta.reindex(r.index)), "动量": F.mad_z(r["MOM"]),
                       "波动": F.mad_z(-r["LOWVOL"]), "价值": F.mad_z(r["BP"]), "流动性": F.mad_z(-r["LOWTO"])}).fillna(0)
    ind = pd.get_dummies(pd.Series(M.sw_industry()).reindex(r.index).fillna("未分类")).astype(float)
    return pd.concat([ind, st], axis=1)


@lru_cache(maxsize=1)
def factor_returns(start="2023-07-01", refresh=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐日因子收益（行 = 交易日）和残差（行 = 交易日，列 = 股票）。缓存到 data/quant/。"""
    OUT.mkdir(parents=True, exist_ok=True)
    pf, pr = OUT / "factor_returns.parquet", OUT / "residuals.parquet"
    if pf.exists() and pr.exists() and not refresh:
        return pd.read_parquet(pf), pd.read_parquet(pr)
    p = F.panel()
    ends = F.month_ends(start)
    fr, res = [], []
    for i, e in enumerate(ends):
        X = exposures(e)
        nxt = ends[i + 1] if i + 1 < len(ends) else p["ret"].index[-1]
        days = p["ret"].loc[(p["ret"].index > e) & (p["ret"].index <= nxt)]
        w = np.sqrt(p["fmv"].loc[e].reindex(X.index))
        for d, y in days.iterrows():
            ok = y.reindex(X.index).notna() & w.notna()
            if ok.sum() < 100:
                continue
            Xo, yo, so = X[ok].values, y.reindex(X.index)[ok].values, np.sqrt(w[ok].values)
            keep = Xo.any(axis=0)                                   # 当天没有成分的行业列去掉，避免奇异
            b, *_ = np.linalg.lstsq(Xo[:, keep] * so[:, None], yo * so, rcond=None)
            beta = pd.Series(0.0, index=X.columns)
            beta[X.columns[keep]] = b
            fr.append(beta.rename(d))
            res.append(pd.Series(yo - Xo[:, keep] @ b, index=X.index[ok], name=d))
    fr, res = pd.DataFrame(fr), pd.DataFrame(res)
    fr.to_parquet(pf)
    res.to_parquet(pr)
    return fr, res


def model_at(d: pd.Timestamp, codes: list[str] | None = None) -> dict:
    """d 日的风险模型：X（暴露）、F（年化因子协方差）、D（年化特异方差）。只用 d 及以前的数据。"""
    fr, res = factor_returns()
    fr, res = fr.loc[:d], res.loc[:d]
    X = exposures(d)
    if codes is not None:
        X = X.reindex(codes).fillna(0)
    cols = X.columns
    f = fr.reindex(columns=cols).fillna(0).iloc[-500:]
    w = _ewma_weights(len(f), HL_COV)
    fc = f - (w[:, None] * f.values).sum(axis=0)
    Fm = pd.DataFrame((fc.values * w[:, None]).T @ fc.values * 252, index=cols, columns=cols)
    r = res.iloc[-250:].reindex(columns=X.index)
    ws = _ewma_weights(len(r), HL_SPEC)
    spec = pd.Series(np.nansum(r.values ** 2 * ws[:, None], axis=0) / np.maximum(np.nansum(np.where(r.notna(), ws[:, None], 0), axis=0), 1e-12),
                     index=X.index) * 252
    n_obs = r.notna().sum()
    med = spec[n_obs >= 40].median()
    spec = spec.where(n_obs >= 40, med)
    spec = (1 - SHRINK) * spec + SHRINK * med
    return {"X": X, "F": Fm, "D": spec, "date": d}


def active_risk(model: dict, active: pd.Series) -> dict:
    """主动权重的风险分解：总跟踪误差、因子部分（行业 / 风格）、特异部分（年化）。"""
    X, Fm, D = model["X"], model["F"], model["D"]
    a = active.reindex(X.index).fillna(0)
    y = X.T @ a                                                      # 主动暴露
    fac = float(y @ Fm @ y)
    spec = float((a ** 2 * D).sum())
    ind_cols = [c for c in X.columns if c not in STYLES]
    yi, ys = y[ind_cols], y[STYLES]
    ind_var = float(yi @ Fm.loc[ind_cols, ind_cols] @ yi)
    sty_var = float(ys @ Fm.loc[STYLES, STYLES] @ ys)
    return {"te": float(np.sqrt(max(fac + spec, 0))), "factor_var": fac, "specific_var": spec, "industry_var": ind_var, "style_var": sty_var,
            "exposure": y}
