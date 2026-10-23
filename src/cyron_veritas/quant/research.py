"""因子检验与合成：IC / RankIC、分组收益、ICIR 加权合成（只用当时已知的 IC）。

远期收益口径：月末 e_k 收盘后算因子，次一交易日 t1_k 收盘成交，持有到下个月的 t1_{k+1}。
合成权重：在 e_m 时点，只有远期窗口已经结束的 IC（k ≤ m − 2；第 m − 1 期的窗口到 e_m 的次一交易日才结束）是已知的；取最近 12 期（至少 6 期）的 RankIC 均值 / 标准差（ICIR），
负 ICIR 的因子权重置 0（不临时翻转方向），按权重绝对值和归一。合成因子的 IC 因此是样本外的。
另算一个不估计权重的等权合成做对照；产品用哪个是事先定的（ICIR 加权），不按回测结果挑。
检验股票池 = 中证 800 当前成分（有幸存者偏差，见 README）。日线从 2023-01 开始，「上市满 120 个交易日」要到 2023-07 才有股票满足，面板从 2023-07 起。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cyron_veritas.config import ROOT
from cyron_veritas.quant import factors as F

OUT = ROOT / "data" / "quant"
LOOKBACK, MIN_OBS = 12, 6


def build_panel(start="2023-07-01", refresh=False, say=print) -> pd.DataFrame:
    """每个月末的处理后因子 + 远期收益（长表：date, code, 因子..., industry, fwd）。缓存到 data/quant/factor_panel.parquet。"""
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "factor_panel.parquet"
    if p.exists() and not refresh:
        return pd.read_parquet(p)
    tri = F.panel()["tri"]
    ends = F.month_ends(start)
    rows = []
    for i, e in enumerate(ends):
        t1 = F.next_trading_day(e)
        t2 = F.next_trading_day(ends[i + 1]) if i + 1 < len(ends) else None
        z = F.processed(e)
        z["fwd"] = (tri.loc[t2] / tri.loc[t1] - 1).reindex(z.index) if (t1 is not None and t2 is not None) else np.nan
        z["date"] = e
        rows.append(z.reset_index(names="code"))
        say(f"  因子 {e.date()}  股票 {len(z)}")
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(p)
    return df


def ic_table(panel: pd.DataFrame, cols=F.ALPHA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐月 IC（Pearson）和 RankIC（Spearman）。行 = 月末，列 = 因子。"""
    ic, ric = {}, {}
    for d, g in panel.dropna(subset=["fwd"]).groupby("date"):
        ic[d] = g[list(cols)].corrwith(g["fwd"])
        ric[d] = g[list(cols)].rank().corrwith(g["fwd"].rank())
    return pd.DataFrame(ic).T, pd.DataFrame(ric).T


def summary(ic: pd.DataFrame, ric: pd.DataFrame, quint: pd.DataFrame | None = None) -> pd.DataFrame:
    n = ric.notna().sum()
    out = pd.DataFrame({"因子": [F.NAME.get(c, c) for c in ric.columns], "类别": [F.GROUP.get(c, "合成") for c in ric.columns],
                        "IC 均值": ic.mean(), "RankIC 均值": ric.mean(), "RankIC 标准差": ric.std(),
                        "ICIR": ric.mean() / ric.std(), "t 值": ric.mean() / ric.std() * np.sqrt(n), "IC>0 占比": (ric > 0).mean(), "月数": n})
    if quint is not None:
        out["多空年化"] = quint.groupby(level=0)["多空"].mean().reindex(out.index) * 12
    return out


def quintiles(panel: pd.DataFrame, cols=F.ALPHA, q=5) -> pd.DataFrame:
    """每月按因子分 q 组等权，返回 (因子, 月末) × [Q1..Qq, 多空] 的月收益。"""
    rows = []
    for d, g in panel.dropna(subset=["fwd"]).groupby("date"):
        for c in cols:
            x = g[[c, "fwd"]].dropna()
            if len(x) < q * 20:
                continue
            grp = pd.qcut(x[c].rank(method="first"), q, labels=[f"Q{i}" for i in range(1, q + 1)])
            r = x.groupby(grp, observed=True)["fwd"].mean()
            rows.append({"factor": c, "date": d, **r.to_dict(), "多空": r[f"Q{q}"] - r["Q1"]})
    return pd.DataFrame(rows).set_index(["factor", "date"])


def weights_at(ric: pd.DataFrame, m: int) -> pd.Series:
    """第 m 个月末的合成权重：只用 ric 的第 0..m−2 行。第 m−1 行的远期窗口到 t1_m（e_m 的次一交易日）才结束，在 e_m 还不知道。"""
    hist = ric.iloc[max(0, m - 1 - LOOKBACK):max(0, m - 1)]
    ok = hist.notna().sum() >= MIN_OBS
    icir = (hist.mean() / hist.std()).where(ok, 0).fillna(0).clip(lower=0)
    return icir / icir.abs().sum() if icir.abs().sum() > 0 else icir


def composite(panel: pd.DataFrame, ric: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """合成因子 ALPHA（每月按当时权重加权、再标准化），返回 (合成因子长序列, 权重表)。"""
    dates = sorted(panel["date"].unique())
    ric = ric.reindex(dates)
    out, wt = [], {}
    for m, d in enumerate(dates):
        w = weights_at(ric, m)
        wt[d] = w
        g = panel[panel["date"] == d].set_index("code")
        if w.abs().sum() == 0:
            s = pd.Series(np.nan, index=g.index)
        else:
            s = (g[w.index].fillna(0) * w).sum(axis=1)
            s = (s - s.mean()) / s.std()
        out.append(pd.DataFrame({"date": d, "code": s.index, "ALPHA": s.values}))
    return pd.concat(out, ignore_index=True), pd.DataFrame(wt).T


def run(refresh=False, say=print) -> dict:
    """完整因子检验：面板 → 单因子 IC / 分组 → 合成 → 合成因子样本外 IC。结果写 data/quant/。"""
    panel = build_panel(refresh=refresh, say=say)
    ic, ric = ic_table(panel)
    comp, wt = composite(panel, ric)
    panel = panel.merge(comp, on=["date", "code"], how="left")
    ew = panel[F.ALPHA].mean(axis=1)                                  # 对照：等权合成（不估计权重）
    panel["ALPHA_EW"] = ew.groupby(panel["date"]).transform(lambda s: (s - s.mean()) / s.std())
    cic, cric = ic_table(panel, ["ALPHA", "ALPHA_EW"])
    ic, ric = ic.join(cic), ric.join(cric)
    qt = quintiles(panel, F.ALPHA + ["ALPHA", "ALPHA_EW"])
    summ = summary(ic, ric, qt)
    summ.loc["ALPHA", ["因子", "类别"]] = ["ICIR 加权合成（样本外，产品使用）", "合成"]
    summ.loc["ALPHA_EW", ["因子", "类别"]] = ["等权合成（对照）", "合成"]
    corr = panel.groupby("date")[F.ALPHA].corr(method="spearman").groupby(level=1).mean().loc[F.ALPHA, F.ALPHA]
    for name, obj in {"ic": ic, "rank_ic": ric, "weights": wt, "summary": summ, "corr": corr}.items():
        obj.to_parquet(OUT / f"{name}.parquet")
    qt.to_parquet(OUT / "quintiles.parquet")
    panel[["date", "code", "ALPHA"]].to_parquet(OUT / "alpha.parquet")
    return {"months": int(ric.notna().any(axis=1).sum()), "summary": summ}


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / f"{name}.parquet")
