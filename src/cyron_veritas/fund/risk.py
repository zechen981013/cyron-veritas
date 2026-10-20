"""风控与合规：每日风险快照、合同限额检查（事件留痕）、流动性、交易前检查。

数据来源只有中台数据库（估值表解析结果、股票池）和公开行情；不读模拟器状态。
收益率用「含分红送转的总收益」：除权日 r = (收盘 × (1 + 每股送转) + 每股派息) / 前收盘 − 1，避免把除息当成亏损。
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

import numpy as np
import pandas as pd

from cyron_veritas.fund import db
from cyron_veritas.fund import marketdata as M
from cyron_veritas.fund import product as P
from cyron_veritas.risk import metrics as RM

OK, WARN, BREACH = "正常", "预警", "超限"


@lru_cache(maxsize=4)
def _market(codes_key: tuple) -> dict:
    codes = list(codes_key)
    b = M.bars(codes)
    close = M.wide(b, "close")
    vol = M.wide(b, "volume") * 100
    ca = M.corporate_actions(codes)
    adj = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    bonus = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for r in ca.itertuples():
        if r.code in close.columns and r.ex_date in close.index:
            adj.at[r.ex_date, r.code] += r.cash_per_share
            bonus.at[r.ex_date, r.code] += r.bonus_per_share
    tr = (close * (1 + bonus) + adj) / close.shift() - 1
    return {"close": close, "ret": tr.where(tr.abs() < 0.25), "adv20": vol.rolling(20, min_periods=5).mean(),
            "bench": M.index_close("sh000300").pct_change()}


def market_view() -> dict:
    return _market(tuple(sorted(M.universe()["code"])))


def holdings_on(code: str, d) -> pd.DataFrame:
    return db.df("select * from holdings where product = :p and date = :d", p=code, d=pd.Timestamp(d).date())


def nav_series(code: str) -> pd.DataFrame:
    n = db.df("select * from nav where product = :p order by date", p=code)
    n["date"] = pd.to_datetime(n["date"])
    return n.set_index("date")


def snapshot(code: str, d) -> dict:
    d = pd.Timestamp(d)
    cfg = P.load(code)
    lim = cfg["contract_limits"]
    mv = market_view()
    nav = nav_series(code)
    row = nav.loc[d]
    h = holdings_on(code, d)
    net = float(row["net_assets"])
    ind = M.sw_industry()
    h["industry"] = h["code"].map(ind).fillna("未分类")
    w = h.set_index("code")["mv"] / net
    by_ind = (h.groupby("industry")["mv"].sum() / net).sort_values(ascending=False)
    # 流动性：可变现天数 = 持仓 /（20 日均量 × 参与率）
    adv = mv["adv20"].loc[:d].iloc[-1].reindex(h["code"]).values
    days = h["qty"].values / (np.maximum(adv, 1) * lim["participation_rate"])
    h["days_to_liquidate"] = days
    liq = {k: float(h.loc[h["days_to_liquidate"] <= k, "mv"].sum() / max(h["mv"].sum(), 1)) for k in (1, 5, 20)}
    # 市场风险：当前持仓回放近 250 个交易日
    r = mv["ret"].loc[:d].iloc[-250:]
    port = (r[w.index.intersection(r.columns)].fillna(0) * w.reindex(r.columns.intersection(w.index))).sum(axis=1)
    bench = mv["bench"].reindex(port.index)
    navs = nav["unit_nav"].loc[:d]
    peak = navs.cummax()
    snap = {
        "date": str(d.date()), "unit_nav": float(row["unit_nav"]), "net_assets": net, "n_holdings": int(len(h)),
        "stock_weight": float(h["mv"].sum() / net), "cash_weight": float(row["cash"] / net),
        "top1": float(w.max()) if len(w) else 0.0, "top1_name": h.loc[h["mv"].idxmax(), "name"] if len(h) else "",
        "top5": float(w.sort_values(ascending=False).iloc[:5].sum()), "hhi": float((w ** 2).sum()),
        "by_industry": by_ind.round(4).to_dict(), "liquidity": liq,
        "slowest": h.sort_values("days_to_liquidate", ascending=False).head(5)[["code", "name", "days_to_liquidate"]].round(2).to_dict("records"),
        "var95_1d": RM.historical_var(port, 0.95), "cvar95_1d": RM.historical_cvar(port, 0.95), "var99_1d": RM.historical_var(port, 0.99),
        "vol_ann": RM.annualized_vol(port), "beta": RM.beta(port, bench),
        "max_drawdown": float(1 - (navs / peak).min()), "current_drawdown": float(1 - navs.iloc[-1] / peak.iloc[-1]),
        "to_warning_line": float(row["unit_nav"] / lim["warning_line"] - 1), "to_stop_line": float(row["unit_nav"] / lim["stop_line"] - 1),
        "weights": w.round(5).to_dict(),
    }
    return snap


def check_limits(code: str, snap: dict) -> list[dict]:
    cfg = P.load(code)
    lim = cfg["contract_limits"]
    warn_ratio = 0.9
    out = []

    def add(rule, label, value, limit, direction="max", detail=""):
        if direction == "max":
            st = BREACH if value > limit else (WARN if value > limit * warn_ratio else OK)
        else:
            st = BREACH if value < limit else (WARN if value < limit / warn_ratio else OK)
        out.append({"rule": rule, "label": label, "value": value, "limit": limit, "direction": direction, "status": st, "detail": detail})

    over = [f"{k}" for k, v in snap["weights"].items() if v > lim["single_name_max"]]
    add("single_name_max", "单只股票占净值", snap["top1"], lim["single_name_max"], detail=("超限：" + "、".join(over)) if over else snap["top1_name"])
    top_ind, top_v = next(iter(snap["by_industry"].items())) if snap["by_industry"] else ("", 0.0)
    add("industry_max", "单一申万一级行业占净值", top_v, lim["industry_max"], detail=top_ind)
    add("cash_min", "现金类资产占净值", snap["cash_weight"], lim["cash_min"], "min")
    add("liquidity_5d_min", "5 日内可变现比例", snap["liquidity"][5], lim["liquidity_5d_min"], "min")
    add("warning_line", "单位净值 vs 预警线", snap["unit_nav"], lim["warning_line"], "min", "触及后不得新增仓位")
    add("stop_line", "单位净值 vs 止损线", snap["unit_nav"], lim["stop_line"], "min", "触及后须清仓")
    banned = {b["code"] for b in cfg.get("banned", [])}
    held_banned = [c for c in snap["weights"] if c in banned]
    out.append({"rule": "banned", "label": "持有禁投池证券", "value": float(len(held_banned)), "limit": 0.0, "direction": "max",
                "status": BREACH if held_banned else OK, "detail": "、".join(held_banned)})
    return out


def daily(code: str, d) -> dict:
    """盘后：风险快照 + 限额检查 → 入库；非正常状态写事件表（同一天同一规则只记一次）。"""
    snap = snapshot(code, d)
    checks = check_limits(code, snap)
    day = pd.Timestamp(d).date()
    db.replace_rows(db.risk, [{"product": code, "date": day, "metrics": {**{k: v for k, v in snap.items() if k != "weights"}, "limits": checks}}],
                    (db.risk.c.product == code) & (db.risk.c.date == day))
    events = [{"product": code, "date": day, "rule": c["rule"], "value": c["value"], "limit": c["limit"], "status": c["status"],
               "detail": f"{c['label']}：{c['detail']}", "handled_by": None, "handled_at": None, "note": None} for c in checks if c["status"] != OK]
    db.replace_rows(db.limit_events, events, (db.limit_events.c.product == code) & (db.limit_events.c.date == day) & db.limit_events.c.handled_by.is_(None))
    return {"snapshot": snap, "checks": checks}


def core_pool(code: str, d) -> set[str]:
    p = db.df("select code, pool, since from stock_pool where product = :p and since <= :d order by since, id", p=code, d=pd.Timestamp(d).date())
    last = p.groupby("code").tail(1)
    return set(last.loc[last["pool"] == "核心池", "code"])


def pretrade(code: str, d, orders: list[dict]) -> dict:
    """交易前检查。orders: [{code, side(buy/sell), qty}]，按当日收盘价估算成交后的持仓，逐条给出通过 / 拒绝及原因。"""
    d = pd.Timestamp(d)
    cfg = P.load(code)
    lim = cfg["contract_limits"]
    mv = market_view()
    px = mv["close"].loc[:d].ffill().iloc[-1]
    nav = nav_series(code).loc[d]
    h = holdings_on(code, d).set_index("code")
    qty = h["qty"].to_dict()
    cash = float(nav["cash"])
    banned = {b["code"]: b["reason"] for b in cfg.get("banned", [])}
    core = core_pool(code, d)
    ind = M.sw_industry()
    results = []
    for o in orders:
        c, side, q = str(o["code"]).zfill(6), o["side"], float(o["qty"])
        reasons = []
        price = float(px.get(c, np.nan))
        if np.isnan(price):
            reasons.append("没有行情（代码错误或未上市）")
        if side == "buy":
            if c in banned:
                reasons.append(f"禁投池：{banned[c]}")
            if c not in core:
                reasons.append("不在核心池（需先经投委会批准入池）")
            if nav["unit_nav"] < lim["warning_line"]:
                reasons.append(f"单位净值 {nav['unit_nav']:.4f} 低于预警线 {lim['warning_line']}，不得新增仓位")
            if q % 100:
                reasons.append("买入须为 100 股整数倍")
        elif q > qty.get(c, 0):
            reasons.append(f"卖出数量 {q:.0f} 超过持仓 {qty.get(c, 0):.0f}")
        ok = not reasons
        if ok and not np.isnan(price):
            amt = q * price
            qty[c] = qty.get(c, 0) + (q if side == "buy" else -q)
            cash += -amt if side == "buy" else amt
        results.append({"code": c, "side": side, "qty": q, "price": price, "pass": ok, "reasons": reasons})
    # 组合层面：模拟成交后的持仓再跑一遍限额
    net = float(nav["net_assets"])
    after = pd.Series({c: q * float(px.get(c, 0)) for c, q in qty.items() if q > 0})
    w = after / net
    ind_w = w.groupby(w.index.map(lambda x: ind.get(x, "未分类"))).sum()
    port_checks = []
    for c, v in w.items():
        if v > lim["single_name_max"]:
            port_checks.append(f"成交后 {c} 占净值 {v:.1%} > 单票上限 {lim['single_name_max']:.0%}")
    for i, v in ind_w.items():
        if v > lim["industry_max"]:
            port_checks.append(f"成交后 {i} 行业占净值 {v:.1%} > 行业上限 {lim['industry_max']:.0%}")
    if cash / net < lim["cash_min"]:
        port_checks.append(f"成交后现金占净值 {cash / net:.1%} < 下限 {lim['cash_min']:.0%}")
    res = {"date": str(d.date()), "orders": results, "portfolio_checks": port_checks,
           "pass": all(r["pass"] for r in results) and not port_checks}
    db.add(db.orders, {"product": code, "date": d.date(), "orders": orders, "result": res, "created_at": dt.datetime.now()})
    return res


def episodes(code: str, only_open: bool = True) -> list[dict]:
    """把逐日事件按规则合并成「连续触发区间」：同一规则在相邻交易日连续触发算一次事件，风控岗按区间处理、一次留痕。"""
    q = "select * from limit_events where product = :p" + (" and handled_by is null" if only_open else "") + " order by rule, date"
    ev = db.df(q, p=code)
    if ev.empty:
        return []
    days = pd.to_datetime(db.df("select date from nav where product = :p order by date", p=code)["date"])
    pos = {d.date(): i for i, d in enumerate(days)}
    ev["date"] = pd.to_datetime(ev["date"]).dt.date
    ev["i"] = ev["date"].map(pos)
    ev["ep"] = ((ev["rule"] != ev["rule"].shift()) | (ev["i"].diff() != 1)).cumsum()
    out = []
    for _, g in ev.groupby("ep"):
        worst = BREACH if (g["status"] == BREACH).any() else WARN
        peak = g["value"].min() if g["rule"].iloc[0] in ("cash_min", "liquidity_5d_min", "warning_line", "stop_line") else g["value"].max()
        out.append({"ids": g["id"].tolist(), "rule": g["rule"].iloc[0], "status": worst, "start": str(g["date"].iloc[0]), "end": str(g["date"].iloc[-1]),
                    "days": len(g), "peak": float(peak), "limit": float(g["limit"].iloc[-1]), "detail": g["detail"].iloc[-1],
                    "handled_by": g["handled_by"].iloc[-1], "note": g["note"].iloc[-1]})
    return sorted(out, key=lambda e: e["end"], reverse=True)


def handle_events(code: str, ids: list[int], handled_by: str, note: str) -> int:
    with db.engine().begin() as c:
        r = c.execute(db.limit_events.update().where(db.limit_events.c.product == code).where(db.limit_events.c.id.in_(ids))
                      .values(handled_by=handled_by, handled_at=dt.datetime.now(), note=note))
    return r.rowcount
