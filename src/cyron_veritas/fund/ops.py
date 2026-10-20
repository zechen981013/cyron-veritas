"""运营：估值表入库 + 估值核对（托管净值 vs 自估净值）。

估值核对的做法（运营岗每天收到托管估值表后做的事）：
  自估净值 = 上一日估值表的股票持仓 × 今日收盘价 + 上一日现金 + 应收 − 上一日应付费用 − 今日新增计提
  差异 = 托管净值 − 自估净值，逐项解释：当日成交（成本与印花税佣金）、除权除息（分红到账、送转）、申购赎回、季末费用支付、停牌估值；
  解释后的残差绝对值 < 净值的 0.01% 视为一致，否则标「待核查」。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from cyron_veritas.fund import db
from cyron_veritas.fund import marketdata as M
from cyron_veritas.fund import product as P
from cyron_veritas.fund import sheet as S

TOLERANCE = 1e-4


def load_sheets(code: str, since: dt.date | None = None) -> int:
    """解析估值表目录里的全部文件（或某日之后的）→ nav / holdings 表。返回入库张数。"""
    files = sorted(P.sheet_dir(code).glob("*.xlsx"))
    n = 0
    nav_rows, hold_rows = [], []
    for f in files:
        r = S.parse(f)
        if since and r["date"] <= since:
            continue
        s = r["summary"]
        nav_rows.append({"product": code, "date": r["date"], "unit_nav": s.get("unit_nav"), "acc_nav": s.get("acc_nav"),
                         "net_assets": s.get("net_assets"), "shares": s.get("shares"), "total_assets": s.get("total_assets"),
                         "total_liab": s.get("total_liab"), "cash": (s.get("bank") or 0) + (s.get("settlement") or 0),
                         "stock_mv": s.get("stock_mv"), "receivable": s.get("receivable") or 0,
                         "fee_payable": (s.get("mgmt_payable") or 0) + (s.get("custody_payable") or 0),
                         "sheet_file": f.name, "loaded_at": dt.datetime.now()})
        hold_rows += [{"product": code, "date": r["date"], **h} for h in r["holdings"]]
        n += 1
    where_nav = (db.nav.c.product == code) & ((db.nav.c.date > since) if since else True)
    where_h = (db.holdings.c.product == code) & ((db.holdings.c.date > since) if since else True)
    db.replace_rows(db.nav, nav_rows, where_nav)
    db.replace_rows(db.holdings, hold_rows, where_h)
    return n


def reconcile(code: str) -> pd.DataFrame:
    cfg = P.load(code)
    nav = db.df("select * from nav where product = :p order by date", p=code)
    hold = db.df("select date, code, qty, price, mv from holdings where product = :p", p=code)
    trades = db.df("select date, code, side, qty, amount, fee from trades where product = :p", p=code)
    flows = db.df("select date, type, amount from flows where product = :p", p=code)
    for t in (nav, hold, trades, flows):
        if len(t):
            t["date"] = pd.to_datetime(t["date"])
    codes = sorted(set(hold["code"]))
    close = M.wide(M.bars(codes), "close")
    ca = M.corporate_actions(codes)
    rows = []
    for i in range(1, len(nav)):
        p_, c_ = nav.iloc[i - 1], nav.iloc[i]
        d0, d1 = p_["date"], c_["date"]
        h0 = hold[hold["date"] == d0]
        px = close.loc[:d1].ffill().iloc[-1] if len(close) else pd.Series(dtype=float)
        mv_est = float((h0["qty"] * h0["code"].map(px)).sum())
        days = (d1 - d0).days
        accrual = p_["net_assets"] * (cfg["fees"]["management"] + cfg["fees"]["custody"]) * days / 365
        est = p_["cash"] + p_["receivable"] + mv_est - p_["fee_payable"] - accrual
        diff = c_["net_assets"] - est
        expl = {}
        tr = trades[trades["date"] == d1]
        if len(tr):  # 成交：卖出按成交价与收盘价之差、费用；买入同理（模拟中成交价含冲击成本）
            sold = tr[tr.side == "sell"]
            bought = tr[tr.side == "buy"]
            px_s = sold["code"].map(px)
            px_b = bought["code"].map(px)
            expl["当日成交（冲击成本与税费）"] = float((sold["amount"] - sold["qty"] * px_s - sold["fee"]).sum()
                                             + (bought["qty"] * px_b - bought["amount"] - bought["fee"]).sum())
        fl = flows[flows["date"] == d1]
        if len(fl):
            expl["申购赎回"] = float(fl.apply(lambda r: r["amount"] if r["type"] == "subscribe" else -r["amount"], axis=1).sum())
        ex = ca[(ca["ex_date"] > d0) & (ca["ex_date"] <= d1) & ca["code"].isin(h0["code"])]
        if len(ex):  # 除权除息：持仓市值按除权价下降，应收股利 / 送转股补回
            q = h0.set_index("code")["qty"]
            expl["除权除息"] = float(sum(q[r.code] * (r.cash_per_share + r.bonus_per_share * px.get(r.code, 0)) for r in ex.itertuples()))
        residual = diff - sum(expl.values())
        status = "一致" if abs(residual) <= TOLERANCE * c_["net_assets"] else "待核查"
        rows.append({"product": code, "date": d1.date(), "sheet_nav": float(c_["net_assets"]), "est_nav": float(est), "diff": float(diff),
                     "explained": {**{k: round(v, 2) for k, v in expl.items()}, "残差": round(float(residual), 2)}, "status": status})
    db.replace_rows(db.reconcile, rows, db.reconcile.c.product == code)
    return pd.DataFrame(rows)
