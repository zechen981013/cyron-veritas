"""产品账簿：二级市场产品共用的逐日记账（股票多头和量化指增都用它）。

每个交易日的顺序固定：
  start_day：应收股利到账 → 除权除息（现金分红记应收、送转加股数）→ 申购赎回（按上一日净值确认）
  策略在中间调用 buy / sell 下单（费用、成本、成交流水都在这里算）
  end_day：管理费托管费按自然日计提、季末支付 → 按收盘价估值（停牌沿用最近收盘价）→ 写托管估值表
账务规则见 docs/PLATFORM_DESIGN.md 第 2 节。策略只决定买卖什么、买卖多少，不碰现金和份额。
"""
from __future__ import annotations

import json
import math

import pandas as pd

from cyron_veritas.fund import db
from cyron_veritas.fund import product as P
from cyron_veritas.fund import sheet as S


def fee(cfg: dict, side: str, amount: float) -> float:
    t = cfg["trading"]
    f = max(amount * t["commission"], t["min_commission"]) + amount * t["transfer_fee"]
    if side == "sell":
        f += amount * t["stamp_duty_sell"]
    return f


class Ledger:
    def __init__(self, code: str, cfg: dict, mkt, cal: pd.DatetimeIndex, write_sheets: bool = True):
        self.code, self.cfg, self.mkt, self.cal, self.write_sheets = code, cfg, mkt, cal, write_sheets
        self.out_dir = P.sheet_dir(code)
        self.cash = float(cfg["initial_capital"])
        self.shares = float(cfg["initial_capital"])
        self.pos: dict[str, dict] = {}            # code → {qty, cost}
        self.recv: dict[str, dict] = {}           # code → {amount, pay_date}
        self.mgmt_pay = self.cust_pay = 0.0
        self.prev_d, self.prev_unit, self.prev_net = None, 1.0, self.cash
        self.flows = {pd.Timestamp(f["date"]): f for f in cfg.get("flows", [])}
        self.pending_flows = sorted(self.flows)
        self.trade_rows: list[dict] = []
        self.flow_rows: list[dict] = []

    def reset_outputs(self, extra_tables=()) -> None:
        """重跑前清掉这个产品上一次的估值表和流水。"""
        for old in self.out_dir.glob("*.xlsx"):
            old.unlink()
        for t in (db.trades, db.flows, *extra_tables):
            db.replace_rows(t, [], t.c.product == self.code)

    # ---- 每日开始 ----
    def start_day(self, d: pd.Timestamp) -> None:
        for c in [c for c, r in self.recv.items() if r["pay_date"] <= d]:      # 应收股利到账（除息日的次一交易日）
            self.cash += self.recv.pop(c)["amount"]
        acts = self.mkt.actions.get(d)
        for _, a in (acts.iterrows() if acts is not None else []):            # 除权除息
            c = a["code"]
            if c in self.pos:
                q0 = self.pos[c]["qty"]
                if a["cash_per_share"] > 0:
                    self.recv[c] = {"amount": self.recv.get(c, {}).get("amount", 0) + q0 * a["cash_per_share"], "pay_date": d + pd.Timedelta(days=1)}
                if a["bonus_per_share"] > 0:
                    self.pos[c]["qty"] = q0 + math.floor(q0 * a["bonus_per_share"])
        while self.pending_flows and self.pending_flows[0] <= d:                # 申购赎回（按上一交易日净值确认）
            f = self.flows[self.pending_flows.pop(0)]
            n = f["amount"] / self.prev_unit
            if f["type"] == "subscribe":
                self.cash += f["amount"]
                self.shares += n
            else:
                self.shares -= n
                self.cash -= f["amount"]
            self.flow_rows.append({"product": self.code, "date": d.date(), "type": f["type"], "amount": f["amount"], "shares": n, "nav": self.prev_unit})

    # ---- 估值 ----
    def receivable(self) -> float:
        return sum(r["amount"] for r in self.recv.values())

    def payable(self) -> float:
        return self.mgmt_pay + self.cust_pay

    def nav_at(self, d) -> float:
        """按某日收盘价（停牌沿用最近收盘价）估算的净资产。"""
        mv = sum(p["qty"] * self.mkt.last_price(c, d) for c, p in self.pos.items())
        return self.cash + mv + self.receivable() - self.payable()

    def qty(self, c: str) -> float:
        return self.pos.get(c, {"qty": 0})["qty"]

    # ---- 交易 ----
    def sell(self, d, c: str, q: float, price: float, reason: str) -> None:
        amt = q * price
        f = fee(self.cfg, "sell", amt)
        self.cash += amt - f
        p = self.pos[c]
        p["cost"] -= p["cost"] * q / p["qty"]
        p["qty"] -= q
        if p["qty"] <= 0:
            self.pos.pop(c)
        self.trade_rows.append({"product": self.code, "date": d.date(), "code": c, "side": "sell", "qty": q, "price": price, "amount": amt,
                                "fee": f, "reason": reason})

    def buy(self, d, c: str, q: float, price: float, reason: str) -> None:
        amt = q * price
        f = fee(self.cfg, "buy", amt)
        self.cash -= amt + f
        p = self.pos.setdefault(c, {"qty": 0, "cost": 0.0})
        p["qty"] += q
        p["cost"] += amt + f
        self.trade_rows.append({"product": self.code, "date": d.date(), "code": c, "side": "buy", "qty": q, "price": price, "amount": amt,
                                "fee": f, "reason": reason})

    # ---- 每日结束 ----
    def end_day(self, d: pd.Timestamp) -> dict:
        days = (d - self.prev_d).days if self.prev_d is not None else 1
        self.mgmt_pay += self.prev_net * self.cfg["fees"]["management"] * days / 365
        self.cust_pay += self.prev_net * self.cfg["fees"]["custody"] * days / 365
        nxt = self.cal[self.cal > d]
        if len(nxt) and nxt[0].quarter != d.quarter:                             # 季末支付
            self.cash -= self.payable()
            self.mgmt_pay = self.cust_pay = 0.0
        mkt = self.mkt
        positions = [{"code": c, "name": mkt.names.get(c, c), "exchange": mkt.exchange[c], "qty": p["qty"], "cost": p["cost"],
                      "price": mkt.last_price(c, d), "suspended": not mkt.tradable(c, d)} for c, p in self.pos.items()]
        settlement = min(max(self.cash, 0) * 0.1, 2_000_000)
        receivable = [{"code": c, "name": mkt.names.get(c, c), "amount": r["amount"]} for c, r in self.recv.items()]
        if self.write_sheets:
            summ = S.write(self.out_dir / f"{self.code}_估值表_{d:%Y%m%d}.xlsx", self.cfg, d, positions, self.cash - settlement, settlement,
                           receivable, self.mgmt_pay, self.cust_pay, self.shares)
        else:
            mv = sum(p["qty"] * p["price"] for p in positions)
            net = self.cash + mv + sum(r["amount"] for r in receivable) - self.payable()
            summ = {"net_assets": net, "unit_nav": round(net / self.shares, 4)}
        self.prev_unit, self.prev_net, self.prev_d = summ["unit_nav"], summ["net_assets"], d
        return summ

    def finish(self) -> None:
        db.add(db.trades, self.trade_rows)
        db.add(db.flows, self.flow_rows)
        cfg = json.loads(json.dumps(self.cfg, default=str))
        db.replace_rows(db.products, [{"code": self.code, "name": cfg["name"], "type": cfg["type"], "config": cfg}], db.products.c.code == self.code)
