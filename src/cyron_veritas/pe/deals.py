"""交易条款：从尽调结果里找到基金跟投的那一轮，用代码算持股比例、后续稀释、上市时持股数和每股成本。

投后估值：原文写投后 → 直接用；写投前 → 投前 + 本轮新增资金（价格 × 新增注册资本）；只有价格 → 价格 × 本轮后注册资本（股本）。
持股比例 = 投资额 / 投后估值；之后每一次增资按「增资前 / 增资后」稀释；股改、转增是同比例的，不稀释。
上市时持股数 = 持股比例 × 发行前总股本；每股成本 = 投资额 / 持股数。
锁定期：锁定承诺是逐家股东作出的，按本轮领投方（deal.lead）在招股书里的承诺：上市后 N 个月、取得后 M 个月，取两者较晚者；
找不到 → 按科创板一般规定上市日 + 12 个月，并标「锁定期待复核」。
跟投的是员工持股平台、代持还原这类轮次不合格（外部基金参与不了）。
"""
from __future__ import annotations

import re

import pandas as pd

from cyron_veritas.fund import product as P
from cyron_veritas.pe import dd

NOT_INVESTABLE = re.compile(r"员工持股|持股平台|代持|股权激励")


def _match(rounds: list[dict], sel: dict) -> tuple[int, dict]:
    for i, r in enumerate(rounds):
        if r["date"] != str(sel["date"])[:7] or r["kind"] != sel["kind"]:
            continue
        if "price" in sel and (r.get("price") is None or abs(r["price"] - sel["price"]) > 1e-6):
            continue
        if "valuation" in sel and (r.get("valuation") is None or abs(r["valuation"] - sel["valuation"]) > 1e-6):
            continue
        return i, r
    raise LookupError(f"尽调结果里没有这一轮：{sel}")


def post_money(r: dict) -> float:
    """投后估值（元）。注册资本 / 股本单位是万元 / 万股，估值单位是亿元。"""
    new_money = (r["price"] * (r["capital_after"] - r["capital_before"]) * 1e4
                 if r.get("price") and r.get("capital_after") and r.get("capital_before") else None)
    if r.get("valuation") and r.get("valuation_type") == "投后":
        return r["valuation"] * 1e8
    if r.get("valuation") and r.get("valuation_type") == "投前" and new_money:
        return r["valuation"] * 1e8 + new_money
    if r.get("price") and r.get("capital_after"):
        return r["price"] * r["capital_after"] * 1e4
    raise ValueError(f"算不出投后估值：{r}")


def terms(deal: dict) -> dict:
    d = dd.load(deal["code"])
    it = d["items"]
    rounds = it["rounds"]["rounds"]
    i, r = _match(rounds, deal["round"])
    if NOT_INVESTABLE.search(r.get("note", "") + r.get("investors", "")):
        raise ValueError(f"{d['company']} {r['date']} 这一轮是员工持股 / 代持类，外部基金不能参与")
    pm = post_money(r)
    # 交叉校验：原文写的估值 vs 价格 × 本轮后注册资本（股本），两种口径差 1% 以内视为确认（即使估值的引用页码不对）
    implied = r["price"] * r["capital_after"] * 1e4 if r.get("price") and r.get("capital_after") else None
    stated = pm if r.get("valuation") else None                    # 原文没写估值时，投后估值本身就是「价格 × 股本」，比较没有意义
    cross = {"stated": stated, "price_x_capital": implied, "diff": abs(implied / stated - 1) if implied and stated else None}
    cross["confirmed"] = cross["diff"] is not None and cross["diff"] < 0.01
    own = deal["amount"] / pm
    dilution = []
    for later in rounds[i + 1:]:
        if later["kind"] == "增资" and later.get("capital_before") and later.get("capital_after") and later["date"] >= r["date"]:
            if (later["capital_before"], later["capital_after"]) == (r.get("capital_before"), r.get("capital_after")):
                continue                                   # 同一次增资里不同价格的两档（如机构价 + 员工平台价），不是后续稀释
            f = later["capital_before"] / later["capital_after"]
            own *= f
            dilution.append({"date": later["date"], "factor": f, "page": later["page"]})
    pre = it["ipo"]["pre_ipo_shares"]["value"]
    shares = own * pre
    lk = dd.lockup_for(deal["code"], deal["lead"])
    listing = pd.Timestamp(d["listing_date"])
    close = pd.Timestamp(deal["close"])
    if lk["found"] and (lk.get("months_from_listing") or lk.get("months_from_acquisition")):
        ends = []
        if lk.get("months_from_listing"):
            ends.append(listing + pd.DateOffset(months=lk["months_from_listing"]))
        if lk.get("months_from_acquisition"):
            ends.append(close + pd.DateOffset(months=lk["months_from_acquisition"]))
        unlock = max(ends + [listing + pd.DateOffset(months=12)])     # 法定底线：首发前股份上市后 12 个月内不得转让
        parts = [f"上市后 {lk['months_from_listing']} 个月"] * bool(lk.get("months_from_listing")) + \
                [f"取得后 {lk['months_from_acquisition']} 个月"] * bool(lk.get("months_from_acquisition"))
        lk_note, lk_review = f"{deal['lead']}：{' / '.join(parts)}（第 {lk['page']} 页）", bool(lk.get("unverified"))
    else:
        unlock = listing + pd.DateOffset(months=12)
        lk_note, lk_review = f"招股书里没找到 {deal['lead']} 的锁定承诺，按科创板一般规定上市后 12 个月，待复核", True
    return {"code": deal["code"], "company": d["company"], "sector": deal["sector"], "amount": deal["amount"], "close": str(close.date()),
            "round": r, "round_index": i, "post_money": pm, "ownership_at_close": deal["amount"] / pm, "dilution": dilution,
            "ownership_pre_ipo": own, "pre_ipo_shares": pre, "shares": shares, "cost_per_share": deal["amount"] / shares,
            "listing_date": str(listing.date()), "issue_price": it["ipo"]["issue_price"]["value"],
            "unlock_date": str(unlock.date()), "lockup_note": lk_note, "lockup_review": lk_review,
            "cross_check": cross,
            "unverified": [u for u in d["unverified"] if (u["field"].startswith(f"rounds[{i}]") and not cross["confirmed"]) or u["field"].startswith("ipo.")],
            "pages": {"round": r["page"], "pre_ipo_shares": it["ipo"]["pre_ipo_shares"]["page"], "issue_price": it["ipo"]["issue_price"]["page"]}}


def all_terms(code: str = "CV-PE1") -> list[dict]:
    return [terms(x) for x in P.load(code)["deals"]]
