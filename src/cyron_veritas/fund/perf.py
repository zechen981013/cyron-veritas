"""业绩与归因：净值指标、月度收益表、Brinson 行业归因、风格暴露。

Brinson（BHB，单期算术，按月）：
  配置效应 = Σ (组合行业权重 − 基准行业权重) × (基准行业收益 − 基准总收益)
  选股效应 = Σ 基准行业权重 × (组合行业收益 − 基准行业收益)
  交互效应 = Σ (组合行业权重 − 基准行业权重) × (组合行业收益 − 基准行业收益)
  基准总收益 = 产品基准指数（沪深300 / 中证500）实际涨跌；基准行业权重 / 行业收益 = 月初基准成分股按申万一级归类、成分股当月总收益按权重平均。
  月初成分权重优先用 Tushare 历史权重（精确）；没有 token 时用最新权重按个股总收益倒推（漂移调整，仍有幸存者偏差）。
  成分股拼出的模型基准与真实指数之差（分红再投资 + 幸存者偏差）单列「基准近似误差」。
  （试过用申万一级行业指数做基准行业收益：申万指数覆盖全 A 股、含大量小盘股，和沪深300 成分股表现差得远，
   33 个月近似误差累计 25%，弃用。）
  组合行业收益用月初持仓在当月的总收益（含分红送转）计算——即买入持有假设，月内调仓造成的差异计入「其他（交易与现金）」。

风格暴露（月末，截面 z 分数，股票池 = 中证 800，1%/99% 缩尾）：
  规模 = ln(流通市值)；价值 = 每股净资产 / 股价；动量 = 过去 12 个月剔除最近 1 个月收益；波动 = 60 日收益标准差；
  流动性 = 20 日平均换手率；Beta = 250 日对沪深300 回归。组合暴露 = 持仓权重加权，基准暴露 = 产品基准成分权重加权。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cyron_veritas.fund import marketdata as M
from cyron_veritas.fund import product as P
from cyron_veritas.fund import risk as R

RF = 0.015  # 无风险利率（夏普比率用）


def nav_metrics(code: str, benchmark: str | None = None) -> dict:
    benchmark = benchmark or P.bench_symbol(P.load(code))
    nav = R.nav_series(code)["unit_nav"]
    b = M.index_close(benchmark).reindex(nav.index).ffill()
    r, rb = nav.pct_change().dropna(), b.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    ann_b = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    dd = 1 - (nav / nav.cummax()).min()
    monthly = nav.resample("ME").last().pct_change().dropna()
    monthly_b = b.resample("ME").last().pct_change().dropna()
    excess = (r - rb.reindex(r.index)).dropna()
    te = excess.std() * np.sqrt(252)
    table = pd.DataFrame({"组合": monthly, "基准": monthly_b.reindex(monthly.index)})
    table["超额"] = table["组合"] - table["基准"]
    return {"start": str(nav.index[0].date()), "end": str(nav.index[-1].date()), "unit_nav": float(nav.iloc[-1]),
            "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1), "bench_total_return": float(b.iloc[-1] / b.iloc[0] - 1),
            "ann_return": float(ann), "bench_ann_return": float(ann_b), "ann_excess": float(ann - ann_b), "vol": float(vol),
            "max_drawdown": float(dd), "sharpe": float((ann - RF) / vol) if vol else np.nan, "calmar": float(ann / dd) if dd else np.nan,
            "tracking_error": float(te), "info_ratio": float((ann - ann_b) / te) if te else np.nan,
            "monthly_win_rate": float((monthly > 0).mean()), "monthly_excess_win_rate": float((table["超额"] > 0).mean()),
            "monthly": table}


def brinson(code: str, hist_weights: pd.DataFrame | None = "auto") -> pd.DataFrame:
    """月度 Brinson（BHB）。基准总收益 = 产品基准指数实际涨跌。
    基准行业权重与行业收益来自基准成分股：月初成分权重按申万一级归类，行业收益 = 成分股当月总收益按权重平均。
    月初成分权重：有 Tushare 历史权重时精确取当期；没有时用最新权重倒推（见 risk.bench_weights）。
    由成分股拼出的「模型基准」与真实指数之差（分红再投资 + 幸存者偏差）单列「基准近似误差」，不混进配置 / 选股效应。"""
    cfg = P.load(code)
    mv = R.market_view()
    ret = mv["ret"]
    ind = M.sw_industry()
    bidx = P.bench_index(cfg)
    hist = M.index_weights_hist(bidx) if isinstance(hist_weights, str) else hist_weights
    idx = M.index_close(P.bench_symbol(cfg))
    nav = R.nav_series(code)
    month_ends = nav.index.to_series().groupby(nav.index.to_period("M")).last()
    rows = []
    for i in range(len(month_ends) - 1):
        d0, d1 = month_ends.iloc[i], month_ends.iloc[i + 1]
        h = R.holdings_on(code, d0)
        net = float(nav.loc[d0, "net_assets"])
        if h.empty:
            continue
        period = ret.loc[(ret.index > d0) & (ret.index <= d1)]
        cum = (1 + period.fillna(0)).prod() - 1                          # 当月总收益（含分红送转）
        pw = h.set_index("code")["mv"] / net
        g = pw.groupby(pw.index.map(lambda c: ind.get(c, "未分类")))
        wp = g.sum()
        rp = g.apply(lambda s: float((s * cum.reindex(s.index).fillna(0)).sum() / s.sum()))
        bw, method = R.bench_weights(bidx, d0, hist)
        bw = bw[bw.index.isin(cum.index)]
        bw = bw / bw.sum()
        bg = bw.groupby(bw.index.map(lambda c: ind.get(c, "未分类")))
        wb = bg.sum()
        rb = bg.apply(lambda s: float((s * cum.reindex(s.index).fillna(0)).sum() / s.sum()))
        allidx = wp.index.union(wb.index)
        wp_, wb_, rp_, rb_ = (x.reindex(allidx).fillna(0) for x in (wp, wb, rp, rb))
        rp_ = rp_.where(wp_ > 0, rb_)
        rb_model = float((wb_ * rb_).sum())
        rb_actual = float(idx.loc[:d1].iloc[-1] / idx.loc[:d0].iloc[-1] - 1)
        alloc = float(((wp_ - wb_) * (rb_ - rb_model)).sum())
        select = float((wb_ * (rp_ - rb_)).sum())
        inter = float(((wp_ - wb_) * (rp_ - rb_)).sum())
        port_ret = float(nav.loc[d1, "unit_nav"] / nav.loc[d0, "unit_nav"] - 1)
        rows.append({"month": str(d1.to_period("M")), "组合收益": port_ret, "基准收益": rb_actual, "超额": port_ret - rb_actual,
                     "配置效应": alloc, "选股效应": select, "交互效应": inter,
                     "其他（交易与现金）": port_ret - rb_model - alloc - select - inter, "基准近似误差": rb_model - rb_actual,
                     "超配最多": (wp_ - wb_).idxmax(), "低配最多": (wp_ - wb_).idxmin(), "基准权重来源": method})
    return pd.DataFrame(rows)


def _z(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s = s.clip(lo, hi)
    return (s - s.mean()) / s.std()


def style_factors(d: pd.Timestamp) -> pd.DataFrame:
    """中证 800 截面风格因子 z 分数（行 = 股票）。"""
    codes = sorted(M.universe()["code"])
    b = M.bars(codes)
    b = b[b["date"] <= d]
    close, vol, to = M.wide(b, "close"), M.wide(b, "volume"), M.wide(b, "turnover")
    mv = R.market_view()
    ret = mv["ret"].loc[:d]
    px = close.iloc[-1]
    float_shares = (vol / to).iloc[-20:].median()                       # 流通股本 = 成交量 / 换手率
    f = M.fundamentals(M.available_period(d))
    bench = mv["bench"].loc[:d].iloc[-250:]
    r250 = ret.iloc[-250:]
    beta = r250.apply(lambda s: s.cov(bench) / bench.var() if s.notna().sum() > 120 else np.nan)
    raw = pd.DataFrame({
        "规模": np.log(px * float_shares),
        "价值": f["bvps"].reindex(px.index) / px,
        "动量": (1 + ret.iloc[-252:-21]).prod() - 1,
        "波动": ret.iloc[-60:].std(),
        "流动性": to.iloc[-20:].mean(),
        "Beta": beta,
    })
    return raw.apply(_z)


def style_exposure(code: str, d=None) -> pd.DataFrame:
    nav = R.nav_series(code)
    d = pd.Timestamp(d) if d else nav.index[-1]
    z = style_factors(d)
    h = R.holdings_on(code, d)
    w = h.set_index("code")["mv"]
    w = w / w.sum()
    cfg = P.load(code)
    bw, _ = R.bench_weights(P.bench_index(cfg), d)
    bw = bw[bw.index.isin(z.index)] / bw[bw.index.isin(z.index)].sum()
    port = z.reindex(w.index).mul(w, axis=0).sum()
    bench = z.reindex(bw.index).mul(bw, axis=0).sum()
    return pd.DataFrame({"组合": port, "基准": bench, "主动暴露": port - bench}).round(3)
