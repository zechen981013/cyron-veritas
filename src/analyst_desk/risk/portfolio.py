"""组合估值和风险计算：读持仓 → 拉行情和汇率 → 折算人民币 → 算各项风险指标。

历史模拟用的是「当前持仓回放」：假设今天的持仓过去一直不变，用各标的过去的日收益率算组合收益率序列。
这是银行和基金做 VaR 的常见做法，衡量的是「现在这个组合」的风险，而不是过去的真实业绩。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analyst_desk.data import market
from analyst_desk.risk import metrics as M

LOOKBACK_DAYS = 500  # 风险指标用最近约两年的交易日

# 历史压力情景：用当前持仓回放这些时间段
HIST_SCENARIOS = {
    "2020 新冠暴跌": ("2020-02-19", "2020-03-23"),
    "2022 港股中概股暴跌": ("2022-03-01", "2022-03-15"),
    "2024 年初 A 股流动性危机": ("2024-01-02", "2024-02-05"),
}
# 假设情景：按市场、币种、行业给冲击（小数）
HYPO_SCENARIOS = {
    "A 股下跌 10%": {"market": {"A": -0.10}},
    "港股下跌 15%": {"market": {"HK": -0.15}},
    "美股下跌 10%": {"market": {"US": -0.10}},
    "美元、港元兑人民币贬值 5%": {"currency": {"USD": -0.05, "HKD": -0.05}},
    "科技与互联网板块下跌 20%": {"sector": {"互联网": -0.20, "半导体": -0.20, "科技硬件": -0.20}},
}


@dataclass
class PortfolioRisk:
    as_of: str
    positions: pd.DataFrame
    total_value: float
    asset_returns: pd.DataFrame
    port_returns: pd.Series
    facts: dict = field(default_factory=dict)


def load_holdings(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticker": str})
    required = {"account", "ticker", "name", "market", "currency", "sector", "quantity", "cost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"持仓文件缺少列: {missing}")
    df["id"] = np.where(df["market"] == "CASH", "CASH_" + df["currency"], df["ticker"])
    # 同一标的在多个券商账户里的持仓合并（成本按数量加权）
    df["cost_amt"] = df["quantity"] * df["cost"]
    agg = df.groupby("id", sort=False).agg(
        ticker=("ticker", "first"), name=("name", "first"), market=("market", "first"), currency=("currency", "first"),
        sector=("sector", "first"), quantity=("quantity", "sum"), cost_amt=("cost_amt", "sum"),
        accounts=("account", lambda s: "、".join(sorted(set(s)))))
    agg["cost"] = agg["cost_amt"] / agg["quantity"]
    return agg.drop(columns="cost_amt")


def build(holdings: pd.DataFrame, as_of: dt.date | None = None) -> PortfolioRisk:
    fx = market.fx_history()
    px_cny, local_last = {}, {}
    for pid, row in holdings.iterrows():
        if row["market"] == "CASH":
            continue
        local = market.price_history(row["ticker"], row["market"])
        rate = fx[row["currency"]].reindex(local.index, method="ffill")
        px_cny[pid] = (local * rate).dropna()
        local_last[pid] = (float(local.iloc[-1]), local.index[-1])

    prices = pd.DataFrame(px_cny).sort_index()
    prices = prices[prices.index.dayofweek < 5].ffill(limit=5)
    rets = prices.pct_change(fill_method=None).iloc[1:]
    rets = rets.iloc[-LOOKBACK_DAYS:]

    fx_last = fx.iloc[-1]
    pos = holdings.copy()
    pos["price_local"] = [local_last[i][0] if i in local_last else 1.0 for i in pos.index]
    pos["price_date"] = [str(local_last[i][1].date()) if i in local_last else "" for i in pos.index]
    pos["fx"] = [float(fx_last[c]) for c in pos["currency"]]
    pos["value_cny"] = pos["quantity"] * pos["price_local"] * pos["fx"]
    pos["pnl_local_pct"] = np.where(pos["market"] == "CASH", 0.0, pos["price_local"] / pos["cost"] - 1)
    total = float(pos["value_cny"].sum())
    pos["weight"] = pos["value_cny"] / total

    w = pos.loc[[c for c in rets.columns], "weight"]
    port = (rets.fillna(0) * w).sum(axis=1)  # 现金日收益按 0
    as_of_str = str((as_of or rets.index[-1].date()))
    pr = PortfolioRisk(as_of=as_of_str, positions=pos, total_value=total, asset_returns=rets, port_returns=port)
    pr.facts = compute_facts(pr, fx_last)
    return pr


def _scenario_hist(pr: PortfolioRisk, start: str, end: str) -> float | None:
    """当前持仓在历史区间的累计收益（某标的当时没有数据就按 0 处理，并记下覆盖率）。"""
    fx = market.fx_history()
    total = 0.0
    covered = 0.0
    for pid, row in pr.positions.iterrows():
        if row["market"] == "CASH":
            continue
        local = market.price_history(row["ticker"], row["market"])
        seg = local[(local.index >= start) & (local.index <= end)]
        if len(seg) < 2:
            continue
        rate = fx[row["currency"]].reindex(seg.index, method="ffill")
        r = (seg.iloc[-1] * rate.iloc[-1]) / (seg.iloc[0] * rate.iloc[0]) - 1
        total += row["weight"] * r
        covered += row["weight"]
    return total if covered > 0 else None


def _scenario_hypo(pr: PortfolioRisk, shock: dict) -> float:
    pos = pr.positions
    impact = 0.0
    for _, row in pos.iterrows():
        r = 0.0
        r += shock.get("market", {}).get(row["market"], 0.0)
        r += shock.get("sector", {}).get(row["sector"], 0.0)
        ccy = shock.get("currency", {}).get(row["currency"], 0.0)
        impact += row["weight"] * ((1 + r) * (1 + ccy) - 1)
    return impact


def compute_facts(pr: PortfolioRisk, fx_last: pd.Series) -> dict:
    """把所有风险数字算好放进一个字典。日报里的每个数字都从这里来。"""
    pos, r, rets = pr.positions, pr.port_returns, pr.asset_returns
    w_all = pos["weight"]
    w_risky = pos.loc[pos["market"] != "CASH", "weight"]

    bench = market.price_history("CSI300", "INDEX").pct_change().reindex(r.index)
    dd = M.max_drawdown(r)
    rc = M.risk_contribution(w_risky, rets)
    corr = rets.corr()
    pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]]
    pairs.sort(key=lambda x: -x[2])
    names = pos["name"].to_dict()

    facts = {
        "as_of": pr.as_of,
        "total_value": pr.total_value,
        "n_positions": int((pos["market"] != "CASH").sum()),
        "accounts": sorted({a for s in pos["accounts"] for a in s.split("、")}),
        "fx": {"USD": float(fx_last["USD"]), "HKD": float(fx_last["HKD"])},
        "weights": {names[i]: float(v) for i, v in w_all.sort_values(ascending=False).items()},
        "pnl_pct": {names[i]: float(v) for i, v in pos.loc[pos["market"] != "CASH", "pnl_local_pct"].items()},
        "by_sector": pos.groupby("sector")["weight"].sum().sort_values(ascending=False).to_dict(),
        "by_market": pos.groupby("market")["weight"].sum().sort_values(ascending=False).to_dict(),
        "by_currency": pos.groupby("currency")["weight"].sum().sort_values(ascending=False).to_dict(),
        "concentration": {**M.concentration(w_risky), "top1_name": names[M.concentration(w_risky)["top1_name"]]},
        "cash_weight": float(pos.loc[pos["market"] == "CASH", "weight"].sum()),
        "vol_ann": M.annualized_vol(r),
        "var95_hist": M.historical_var(r, 0.95),
        "var99_hist": M.historical_var(r, 0.99),
        "cvar95_hist": M.historical_cvar(r, 0.95),
        "var95_param": M.parametric_var(r, 0.95),
        "max_drawdown": dd["max_drawdown"],
        "mdd_peak": str(dd["peak_date"].date()),
        "mdd_trough": str(dd["trough_date"].date()),
        "current_drawdown": dd["current_drawdown"],
        "beta_csi300": M.beta(r, bench),
        "risk_contribution": {names[i]: float(v) for i, v in rc.sort_values(ascending=False).items()},
        "top_corr_pairs": [(names[a], names[b], c) for a, b, c in pairs[:3]],
        "lookback_days": int(len(r)),
        "lookback_start": str(r.index[0].date()),
        "position_vol": {names[c]: M.annualized_vol(rets[c]) for c in rets.columns},
    }
    facts["var95_amount"] = facts["var95_hist"] * pr.total_value
    facts["var99_amount"] = facts["var99_hist"] * pr.total_value
    facts["cvar95_amount"] = facts["cvar95_hist"] * pr.total_value

    stress = {}
    for name, (a, b) in HIST_SCENARIOS.items():
        v = _scenario_hist(pr, a, b)
        if v is not None:
            stress[name] = {"type": "历史", "period": f"{a} 至 {b}", "pnl_pct": v, "pnl_amount": v * pr.total_value}
    for name, shock in HYPO_SCENARIOS.items():
        v = _scenario_hypo(pr, shock)
        stress[name] = {"type": "假设", "period": "", "pnl_pct": v, "pnl_amount": v * pr.total_value}
    facts["stress"] = stress

    bt = M.var_backtest(r, window=250, level=0.95, test_days=min(250, len(r) - 250))
    facts["var_backtest"] = {k: v for k, v in bt.items() if k != "series"}
    pr.backtest_series = bt["series"]
    return facts
