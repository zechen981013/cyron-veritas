"""模拟器：按真实行情和公司行为逐日运行一只股票多头私募，产出「外部世界」的数据——
托管行每日估值表（Excel）、券商成交流水、登记机构申赎记录、股票池与投委会决议。

中台的其余部分（解析、对账、风控、归因、报告）只读这些产出，不读模拟器内部状态——接真实数据时把模拟器换掉即可。

账务规则见 docs/PLATFORM_DESIGN.md 第 2 节；选股规则：
  每月第一个交易日调仓；只用已过披露截止日的报告期（marketdata.available_period）；
  条件：上市 ≥ 120 个交易日、非禁投、0 < PE < 60、年化 ROE > 8%、净利同比 > 0、每股经营现金流 > 0；
  打分：ROE、低 PE、营收增速、现金流 / 利润 四项的截面排名之和；选前 25 只等权（单一行业最多 7 只）。
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from cyron_veritas.fund import db
from cyron_veritas.fund import marketdata as M
from cyron_veritas.fund import product as P
from cyron_veritas.fund import sheet as S

ANNUALIZE = {"0331": 4.0, "0630": 2.0, "0930": 4 / 3, "1231": 1.0}


class Market:
    """模拟器用到的行情视图（宽表）。"""

    def __init__(self, codes: list[str]):
        b = M.bars(codes)
        self.close, self.high, self.low = M.wide(b, "close"), M.wide(b, "high"), M.wide(b, "low")
        self.volume = M.wide(b, "volume") * 100  # 手 → 股
        self.names = M.universe().set_index("code")["name"].to_dict()
        self.exchange = {c: "SH" if c.startswith("6") else "SZ" for c in codes}
        ca = M.corporate_actions(codes)
        self.actions = {d: g for d, g in ca.groupby(ca["ex_date"].dt.normalize())}
        self.industry = M.sw_industry()
        self.calendar = M.index_close("sh000300").index
        idx = self.close.index.union(self.calendar)
        self.close_ff = self.close.reindex(idx).ffill()       # 停牌日沿用最近收盘价（预先算好，按位置取值）
        self.has_bar = self.close.reindex(idx).notna()

    def tradable(self, code: str, d) -> bool:
        return code in self.has_bar.columns and bool(self.has_bar.at[d, code])

    def last_price(self, code: str, d) -> float:
        return float(self.close_ff.at[d, code]) if code in self.close_ff.columns else np.nan

    def limit_state(self, code: str, d, prev) -> str:
        """一字涨停 / 一字跌停：开盘到收盘同价且涨跌幅接近板（主板 10%，创业板 / 科创板 20%）。"""
        if not self.tradable(code, d) or prev is None:
            return ""
        c, h, l_ = self.close.at[d, code], self.high.at[d, code], self.low.at[d, code]
        if pd.isna(h) or pd.isna(l_):
            return ""
        p0 = self.last_price(code, prev)
        band = 0.2 if code.startswith(("300", "301", "688")) else 0.1
        if h == l_ and p0 and c >= p0 * (1 + band - 0.005):
            return "up"
        if h == l_ and p0 and c <= p0 * (1 - band + 0.005):
            return "down"
        return ""


def select(mkt: Market, d: pd.Timestamp, prev: pd.Timestamp, cfg: dict) -> list[str]:
    st = cfg["strategy"]
    period = M.available_period(d)
    f = M.fundamentals(period).copy()
    q = ANNUALIZE[period[4:]]
    px = mkt.close.loc[:prev].iloc[-1]
    listed = mkt.close.loc[:prev].notna().sum()
    banned = {b["code"] for b in cfg.get("banned", [])}
    f = f[f.index.isin(px.index)]
    f["price"] = px.reindex(f.index)
    f["pe"] = f["price"] / (f["eps"] * q)
    f["roe_ann"] = f["roe"] * q
    f["cf_quality"] = f["ocfps"] / f["eps"]
    ok = ((f["pe"] > 0) & (f["pe"] < 60) & (f["roe_ann"] > 8) & (f["np_yoy"] > 0) & (f["ocfps"] > 0)
          & (listed.reindex(f.index) >= st["min_listed_days"]) & ~f.index.isin(banned) & f["price"].notna())
    f = f[ok]
    f["score"] = (f["roe_ann"].rank(pct=True) + (-f["pe"]).rank(pct=True) + f["rev_yoy"].rank(pct=True)
                  + f["cf_quality"].clip(upper=3).rank(pct=True))
    picks, per_ind = [], {}
    for code in f.sort_values("score", ascending=False).index:
        ind = mkt.industry.get(code, "未分类")
        if per_ind.get(ind, 0) >= 7:
            continue
        picks.append(code)
        per_ind[ind] = per_ind.get(ind, 0) + 1
        if len(picks) >= st["n_holdings"]:
            break
    return picks


def _fee(cfg, side: str, amount: float) -> float:
    t = cfg["trading"]
    fee = max(amount * t["commission"], t["min_commission"]) + amount * t["transfer_fee"]
    if side == "sell":
        fee += amount * t["stamp_duty_sell"]
    return fee


def run(code: str = "CV-VG1", end: str | None = None, write_sheets: bool = True, say=print) -> dict:
    cfg = P.load(code)
    uni = M.universe()
    mkt = Market(uni["code"].tolist())
    cal = mkt.calendar[(mkt.calendar >= pd.Timestamp(cfg["inception"])) & (mkt.calendar <= pd.Timestamp(end or mkt.close.index[-1]))]
    out_dir = P.sheet_dir(code)
    for old in out_dir.glob("*.xlsx"):
        old.unlink()
    for t in (db.trades, db.flows, db.pool, db.ic):
        db.replace_rows(t, [], t.c.product == code)

    cash = float(cfg["initial_capital"])
    shares = float(cfg["initial_capital"])
    pos: dict[str, dict] = {}            # code → {qty, cost}
    recv: dict[str, dict] = {}           # code → {amount, pay_date}
    mgmt_pay = cust_pay = 0.0
    prev_d, prev_unit, prev_net = None, 1.0, cash
    flows = {pd.Timestamp(f["date"]): f for f in cfg.get("flows", [])}
    pending_flows = sorted(flows)
    core: set[str] = set()
    trade_rows, flow_rows, pool_rows, ic_rows = [], [], [], []
    st = cfg["strategy"]
    lim = cfg["contract_limits"]
    last_month = None

    for d in cal:
        # 1) 应收股利到账（除息日的次一交易日）
        for c in [c for c, r in recv.items() if r["pay_date"] <= d]:
            cash += recv.pop(c)["amount"]
        # 2) 公司行为：除权除息
        for _, a in mkt.actions.get(d, pd.DataFrame()).iterrows() if d in mkt.actions else []:
            c = a["code"]
            if c in pos:
                q0 = pos[c]["qty"]
                if a["cash_per_share"] > 0:
                    recv[c] = {"amount": recv.get(c, {}).get("amount", 0) + q0 * a["cash_per_share"], "pay_date": d + pd.Timedelta(days=1)}
                if a["bonus_per_share"] > 0:
                    pos[c]["qty"] = q0 + math.floor(q0 * a["bonus_per_share"])
        # 3) 申购赎回（按上一交易日净值确认）
        while pending_flows and pending_flows[0] <= d:
            f = flows[pending_flows.pop(0)]
            n = f["amount"] / prev_unit
            if f["type"] == "subscribe":
                cash += f["amount"]
                shares += n
            else:
                shares -= n
                cash -= f["amount"]
            flow_rows.append({"product": code, "date": d.date(), "type": f["type"], "amount": f["amount"], "shares": n, "nav": prev_unit})
        # 4) 调仓（每月第一个交易日；首日建仓）
        rebalance = prev_d is None or d.month != last_month
        last_month = d.month
        stop_hit = prev_unit < lim["stop_line"]
        warn_hit = prev_unit < lim["warning_line"]
        if rebalance or stop_hit or cash < -1:
            mv = sum(p["qty"] * mkt.last_price(c, prev_d or d) for c, p in pos.items())
            nav_now = cash + mv + sum(r["amount"] for r in recv.values()) - mgmt_pay - cust_pay
            if stop_hit:
                targets = {}
            elif rebalance:
                picks = select(mkt, d, prev_d or d - pd.Timedelta(days=1), cfg)
                new_in, out = [c for c in picks if c not in core], [c for c in core if c not in picks]
                for c in new_in:  # 入池：模拟投委会决议（研究员提交 → 投委会批准）
                    ic_rows.append({"product": code, "code": c, "name": mkt.names.get(c, c), "request": "入核心池", "requested_by": "研究部（模型初筛）",
                                    "requested_at": d.date(), "material": f"季度筛选 {M.available_period(d)}：ROE / 估值 / 增长 / 现金流综合排名前 {st['n_holdings']}",
                                    "decision": "批准", "votes": {"赞成": 3, "反对": 0, "弃权": 0}, "decided_at": d.date(), "note": "模拟决议"})
                    pool_rows.append({"product": code, "code": c, "name": mkt.names.get(c, c), "pool": "核心池", "since": d.date(), "until": None,
                                      "reason": "投委会批准入池", "decision_id": None})
                for c in out:
                    pool_rows.append({"product": code, "code": c, "name": mkt.names.get(c, c), "pool": "观察池", "since": d.date(), "until": None,
                                      "reason": "季度筛选排名下降，调出核心池", "decision_id": None})
                core = set(picks)
                w = min((1 - st["cash_target"]) / max(1, len(picks)), st["max_weight"])
                targets = {c: w for c in picks}
            else:
                targets = {c: p["qty"] * mkt.last_price(c, prev_d) / nav_now for c, p in pos.items()}
            # 先卖后买
            for c in sorted(set(pos) | set(targets), key=lambda c: targets.get(c, 0)):
                if not mkt.tradable(c, d):
                    continue
                px = float(mkt.close.at[d, c])
                tgt_qty = math.floor(targets.get(c, 0) * nav_now / px / 100) * 100
                cur = pos.get(c, {"qty": 0, "cost": 0.0})["qty"]
                delta = tgt_qty - cur
                state = mkt.limit_state(c, d, prev_d)
                if delta < 0 and state != "down":
                    q_ = -delta if tgt_qty > 0 else cur
                    price = px * (1 - cfg["trading"]["slippage"])
                    amt = q_ * price
                    fee = _fee(cfg, "sell", amt)
                    cash += amt - fee
                    p = pos[c]
                    p["cost"] -= p["cost"] * q_ / p["qty"]
                    p["qty"] -= q_
                    if p["qty"] <= 0:
                        pos.pop(c)
                    trade_rows.append({"product": code, "date": d.date(), "code": c, "side": "sell", "qty": q_, "price": price, "amount": amt,
                                       "fee": fee, "reason": "止损清仓" if stop_hit else "调仓"})
            for c, w in targets.items():
                if not mkt.tradable(c, d) or warn_hit:
                    continue
                px = float(mkt.close.at[d, c])
                tgt_qty = math.floor(w * nav_now / px / 100) * 100
                cur = pos.get(c, {"qty": 0})["qty"]
                delta = tgt_qty - cur
                if delta <= 0 or mkt.limit_state(c, d, prev_d) == "up":
                    continue
                price = px * (1 + cfg["trading"]["slippage"])
                amt = delta * price
                fee = _fee(cfg, "buy", amt)
                if amt + fee > cash - lim["cash_min"] * nav_now * 0.5:
                    delta = math.floor((cash - lim["cash_min"] * nav_now * 0.5) / price / 100) * 100
                    if delta <= 0:
                        continue
                    amt, fee = delta * price, _fee(cfg, "buy", delta * price)
                cash -= amt + fee
                p = pos.setdefault(c, {"qty": 0, "cost": 0.0})
                p["qty"] += delta
                p["cost"] += amt + fee
                trade_rows.append({"product": code, "date": d.date(), "code": c, "side": "buy", "qty": delta, "price": price, "amount": amt,
                                   "fee": fee, "reason": "调仓"})
        # 5) 费用计提（按自然日）、季末支付
        days = (d - prev_d).days if prev_d is not None else 1
        mgmt_pay += prev_net * cfg["fees"]["management"] * days / 365
        cust_pay += prev_net * cfg["fees"]["custody"] * days / 365
        nxt = cal[cal > d]
        if len(nxt) and nxt[0].quarter != d.quarter:
            cash -= mgmt_pay + cust_pay
            mgmt_pay = cust_pay = 0.0
        # 6) 估值（停牌沿用最近收盘价）与估值表
        positions = [{"code": c, "name": mkt.names.get(c, c), "exchange": mkt.exchange[c], "qty": p["qty"], "cost": p["cost"],
                      "price": mkt.last_price(c, d), "suspended": not mkt.tradable(c, d)} for c, p in pos.items()]
        settlement = min(max(cash, 0) * 0.1, 2_000_000)
        receivable = [{"code": c, "name": mkt.names.get(c, c), "amount": r["amount"]} for c, r in recv.items()]
        fname = out_dir / f"{code}_估值表_{d:%Y%m%d}.xlsx"
        if write_sheets:
            summ = S.write(fname, cfg, d, positions, cash - settlement, settlement, receivable, mgmt_pay, cust_pay, shares)
        else:
            mv = sum(p["qty"] * p["price"] for p in positions)
            net = cash + mv + sum(r["amount"] for r in receivable) - mgmt_pay - cust_pay
            summ = {"net_assets": net, "unit_nav": round(net / shares, 4)}
        prev_unit, prev_net, prev_d = summ["unit_nav"], summ["net_assets"], d
        if d.day <= 7 and d.dayofweek == 0:
            say(f"  {d:%Y-%m-%d} 单位净值 {summ['unit_nav']:.4f} 规模 {summ['net_assets'] / 1e8:.2f} 亿 持仓 {len(pos)}")

    db.add(db.trades, trade_rows)
    db.add(db.flows, flow_rows)
    db.add(db.ic, ic_rows)
    db.add(db.pool, pool_rows)
    db.replace_rows(db.products, [{"code": code, "name": cfg["name"], "type": cfg["type"], "config": _jsonable(cfg)}], db.products.c.code == code)
    return {"days": len(cal), "final_unit_nav": prev_unit, "final_net_assets": prev_net, "trades": len(trade_rows), "sheets": str(out_dir)}


def _jsonable(x):
    import json
    return json.loads(json.dumps(x, default=str))
