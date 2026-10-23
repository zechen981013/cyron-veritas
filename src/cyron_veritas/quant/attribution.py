"""指数增强产品的超额收益分解（逐日精确相加，按月汇总）与风险模型检验。

记号（d 日，权重都用 d−1 日收盘后的值）：
  r_p 产品单位净值收益；r_h = Σ w_i r_i 持仓按昨收权重的收益（w 占净资产）；s = Σ w_i 股票仓位；
  r_b = Σ wb_i r_i 基准成分股组合收益（wb 为 d−1 时点成分权重）；r_idx 基准指数涨跌；a = w − s·wb 主动权重。
  r_p − r_idx = (r_p − r_h)                 费用、交易成本与成交价差
              + Σ a_i X_i f                  行业、风格因子（X 为上一个月末暴露，f 为当日因子收益）
              + Σ a_i (r_i − X_i f)          特异收益（选股）
              − (1 − s) r_b                  现金拖累
              + (r_b^价格 − r_idx)            幸存者偏差（成分股不含股息的收益 − 指数）
              + (r_b − r_b^价格)              成分股股息（指数是价格指数，不含股息）
  每一项逐日相加严格等于 r_p − r_idx；按月加总是算术和，与几何超额有很小差别（单列「复利差」）。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from cyron_veritas.fund import db
from cyron_veritas.fund import marketdata as M
from cyron_veritas.fund import product as P
from cyron_veritas.fund import risk as R
from cyron_veritas.quant import factors as F
from cyron_veritas.quant import riskmodel as RMOD

COLS = ["费用与交易成本", "行业", "风格", "选股（特异）", "现金拖累", "幸存者偏差", "成分股股息"]


@lru_cache(maxsize=1)
def _price_returns() -> pd.DataFrame:
    """不含现金分红的收益（送转按除权调整）。"""
    codes = sorted(M.universe()["code"])
    close = R.market_view()["close"]
    ca = M.corporate_actions(codes)
    bonus = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for r in ca.itertuples():
        if r.code in close.columns and r.ex_date in close.index:
            bonus.at[r.ex_date, r.code] += r.bonus_per_share
    pr = close * (1 + bonus) / close.shift() - 1
    return pr.where(pr.abs() < 0.25)


def decompose(code: str) -> pd.DataFrame:
    """逐日分解（行 = 交易日）。"""
    cfg = P.load(code)
    nav = R.nav_series(code)
    hold = db.df("select date, code, mv from holdings where product = :p", p=code)
    hold["date"] = pd.to_datetime(hold["date"])
    W = hold.pivot(index="date", columns="code", values="mv").div(nav["net_assets"], axis=0).fillna(0)
    ret = R.market_view()["ret"]
    pret = _price_returns()
    idx = M.index_close(P.bench_symbol(cfg)).pct_change()
    fr, res = RMOD.factor_returns()
    styles = RMOD.STYLES
    ends = F.month_ends("2023-07-01")
    X_cache: dict = {}
    rows = []
    dates = nav.index
    for i in range(1, len(dates)):
        d0, d = dates[i - 1], dates[i]
        w = W.loc[d0]
        w = w[w > 0]
        s = float(w.sum())
        wb, _ = R.bench_weights(P.bench_index(cfg), d0)
        r = ret.loc[d]
        rh = float((w * r.reindex(w.index).fillna(0)).sum())
        rb = float((wb * r.reindex(wb.index).fillna(0)).sum())
        rbp = float((wb * pret.loc[d].reindex(wb.index).fillna(0)).sum())
        rp = float(nav.loc[d, "unit_nav"] / nav.loc[d0, "unit_nav"] - 1)
        ri = float(idx.get(d, np.nan))
        a = w.reindex(w.index.union(wb.index)).fillna(0) - s * wb.reindex(w.index.union(wb.index)).fillna(0)
        e = max(x for x in ends if x < d)
        if e not in X_cache:
            X_cache[e] = RMOD.exposures(e)
        X = X_cache[e].reindex(a.index).fillna(0)
        f = fr.loc[d].reindex(X.columns).fillna(0) if d in fr.index else pd.Series(0.0, index=X.columns)
        contrib = (X.mul(a, axis=0)).sum() * f                               # 每个因子的贡献
        ind_c = float(contrib.drop(styles).sum())
        sty = contrib[styles]
        spec = float((a * (r.reindex(a.index).fillna(0) - X @ f)).sum())
        rows.append({"date": d, "产品": rp, "指数": ri, "超额": rp - ri, "费用与交易成本": rp - rh, "行业": ind_c, "风格": float(sty.sum()),
                     "选股（特异）": spec, "现金拖累": -(1 - s) * rb, "幸存者偏差": rbp - ri, "成分股股息": rb - rbp,
                     **{f"风格·{k}": float(v) for k, v in sty.items()}})
    return pd.DataFrame(rows).set_index("date")


def monthly(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby(daily.index.to_period("M"))
    m = g.sum()
    m["产品"] = g["产品"].apply(lambda s: (1 + s).prod() - 1)
    m["指数"] = g["指数"].apply(lambda s: (1 + s).prod() - 1)
    m["超额"] = m["产品"] - m["指数"]
    m["复利差"] = m["超额"] - m[COLS].sum(axis=1)
    m.index = m.index.astype(str)
    return m


def summary(daily: pd.DataFrame) -> dict:
    """全期年化（各项日度算术和 × 252 / 天数）与「扣除幸存者偏差后的超额」。"""
    n = len(daily)
    ann = {k: float(daily[k].sum() * 252 / n) for k in COLS}
    total_ex = float(((1 + daily["产品"]).prod()) ** (252 / n) - ((1 + daily["指数"]).prod()) ** (252 / n))
    return {"年化超额": total_ex, "分项年化": ann, "扣除幸存者偏差后年化超额": total_ex - ann["幸存者偏差"]}


def bias_test(code: str) -> pd.DataFrame:
    """风险模型检验：每次调仓的事前跟踪误差 vs 下一个调仓周期内实际主动收益的波动（年化）。
    偏差统计量 = 实际 / 预测，理想值 1；持续 > 1 说明模型低估风险。"""
    reb = db.df("select date, ex_ante_te from quant_rebalances where product = :p order by date", p=code)
    reb["date"] = pd.to_datetime(reb["date"])
    d = decompose_cached(code)
    active = d["超额"] - d["幸存者偏差"] - d["成分股股息"]                     # 相对成分股组合的主动收益（风险模型预测的对象）
    rows = []
    for i, r in reb.iterrows():
        end = reb["date"].iloc[i + 1] if i + 1 < len(reb) else active.index[-1]
        x = active.loc[(active.index > r["date"]) & (active.index <= end)]
        if len(x) >= 10 and r["ex_ante_te"] and r["ex_ante_te"] >= 0.005:      # 建仓首月模型未预热、几乎贴着基准，不纳入
            real = float(x.std() * np.sqrt(252))
            rows.append({"date": r["date"].date(), "事前跟踪误差": float(r["ex_ante_te"]), "事后跟踪误差": real, "偏差统计量": real / float(r["ex_ante_te"])})
    return pd.DataFrame(rows)


_DEC: dict = {}


def decompose_cached(code: str) -> pd.DataFrame:
    """同一进程内按产品缓存（逐日分解要几十秒）；产品重跑后重启服务即可。"""
    if code not in _DEC:
        _DEC[code] = decompose(code)
    return _DEC[code]
