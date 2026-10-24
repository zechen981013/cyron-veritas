"""PE 基金逐日运行：出资（capital call）→ 交割 → 上市 → 锁定 → 减持 → 分配（瀑布）；每个季度末估值（IPEV）、算基金指标。

事件顺序（每个交易日）：虚构项目的事件 → 管理费 / 基金费用（半年预收）→ 项目交割 → 上市转换 → 分红送转 → 解禁后减持 → 季度末估值与分配。
出资：项目交割和费用按认缴比例向合伙人发缴款通知（提前 10 天）；退出期的费用优先用基金现金支付。
GP 自己 1% 的认缴不交管理费、不参与业绩报酬：投资款按认缴比例出，分配按实缴比例直接拿，其余进入 LP 瀑布。
估值（中基协《私募投资基金非上市股权投资估值指引》思路）：
  未上市、交割后 12 个月内 → 最近融资价格（成本）；超过 12 个月 → 成本 × 申万行业指数涨跌（市场法校准）；
  已上市限售 → 收盘价 ×（1 − AAP 流动性折扣），σ 取最近 120 个交易日年化波动；已解禁 → 收盘价；虚构项目 → 按事件调整。
减持：解禁后 6 个月内分批；每日 ≤ 成交量 10%，任意 90 天 ≤ 公司总股本 1%（集中竞价）；股价低于每股成本暂缓，每季度投委会复议；
  成立满 6 年后不论价格都要减持完。
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

from cyron_veritas.fund import db
from cyron_veritas.fund import marketdata as M
from cyron_veritas.fund import product as P
from cyron_veritas.pe import dd, market
from cyron_veritas.pe import deals as D
from cyron_veritas.pe import metrics as X
from cyron_veritas.pe.waterfall import Waterfall


def _months(d, n):
    return pd.Timestamp(d) + pd.DateOffset(months=n)


def run(code: str = "CV-PE1", say=print) -> dict:
    cfg = P.load(code)
    tm, ex = cfg["terms"], cfg["exit_policy"]
    inception = pd.Timestamp(cfg["inception"])
    commit = float(cfg["commitment"])
    lps = cfg["lps"]
    gp_lp = next(x for x in lps if x["type"] == "GP")
    lp_class = [x for x in lps if x["type"] != "GP"]
    lp_commit = sum(x["commitment"] for x in lp_class)
    invest_end = _months(inception, 12 * tm["investment_period_years"])
    force_exit = _months(inception, 12 * ex["must_exit_before_years"])

    # ---- 项目 ----
    holds: dict[str, dict] = {}
    for t in D.all_terms(code):
        b = market.bars(t["code"])
        holds[t["code"]] = {"id": t["code"], "name": t["company"], "sector": t["sector"], "synthetic": False, "terms": t, "status": "待交割",
                            "close": pd.Timestamp(t["close"]), "cost_total": t["amount"], "cost_left": t["amount"], "qty": 0.0,
                            "cost_ps": t["cost_per_share"], "listing": pd.Timestamp(t["listing_date"]), "unlock": pd.Timestamp(t["unlock_date"]),
                            "total_shares": t["pre_ipo_shares"] + (dd.load(t["code"])["items"]["ipo"]["issue_shares"]["value"] or 0) * 1e4,
                            "bars": b, "acts": market.actions(t["code"]), "sold90": deque(), "last_ic_q": None}
    for s in cfg.get("synthetic_deals", []):
        holds[s["id"]] = {"id": s["id"], "name": s["name"], "sector": s["sector"], "synthetic": True, "spec": s, "status": "待交割",
                          "close": pd.Timestamp(s["close"]), "cost_total": s["amount"], "cost_left": s["amount"], "mark": 1.0, "mark_note": "最近融资价格（成本）"}
    cal_idx = M.index_close("sh000300")
    last_day = min(cal_idx.index[-1], *[h["bars"].index[-1] for h in holds.values() if not h["synthetic"]])
    cal = cal_idx.index[(cal_idx.index >= inception) & (cal_idx.index <= last_day)]
    sw = M.sw_index_close()

    # ---- 账户状态 ----
    cash = 0.0
    tx, capital, marks, quarters, ic_events = [], [], [], [], []
    wf = Waterfall(tm["hurdle"], tm["carry"], tm["catch_up"])
    paid = {x["id"]: 0.0 for x in lps}
    lp_flows: list[tuple] = []                        # LP 类别现金流（负 = 出资，正 = 分配）
    deal_flows: dict[str, list] = {h: [] for h in holds}
    next_bill = inception

    def call(d, amount, purpose, partners):
        """按认缴比例向 partners 发缴款通知。"""
        nonlocal cash
        tot = sum(x["commitment"] for x in partners)
        for x in partners:
            a = amount * x["commitment"] / tot
            paid[x["id"]] += a
            capital.append({"product": code, "date": d.date(), "notice_date": (d - pd.Timedelta(days=tm["call_notice_days"])).date(), "lp": x["id"],
                            "type": "缴款", "amount": a, "detail": {"用途": purpose}})
            if x["type"] != "GP":
                wf.contribute(d, a)
                lp_flows.append((d, -a))
        cash += amount

    def fee_due(d):
        half = tm["fee_billing_months"] / 12
        exp = commit * tm["fund_expense"] * half
        if d < invest_end:
            return lp_commit * tm["mgmt_fee_invest"] * half, exp, "投资期：认缴额 × 2%"
        base = sum(h["cost_left"] for h in holds.values() if h["status"] not in ("待交割", "已退出", "已核销")) * lp_commit / commit
        return base * tm["mgmt_fee_exit"] * half, exp, "退出期：未退出项目成本 × 2%"

    def fair_value(h, d) -> tuple[float, str, float, float, str]:
        """(公允价值, 方法, 单价, 折扣, 备注)"""
        if h["status"] in ("待交割", "已退出", "已核销"):
            return 0.0, h["status"], 0.0, 0.0, ""
        if h["synthetic"]:
            return h["cost_left"] * h["mark"], h["mark_note"], 0.0, 0.0, ""
        if d < h["listing"]:
            months = (d - h["close"]).days / 30.4
            if months <= 12:
                return h["cost_left"], "最近融资价格（成本）", h["cost_ps"], 0.0, ""
            col = h["sector"]
            r = float(sw[col].loc[:d].iloc[-1] / sw[col].loc[:h["close"]].iloc[-1]) if col in sw.columns else 1.0
            return h["cost_left"] * r, f"市场法：成本 × 申万{col}指数涨跌（{r - 1:+.1%}）", h["cost_ps"] * r, 0.0, ""
        b = h["bars"]
        px = float(b["close"].loc[:d].iloc[-1])
        if d < h["unlock"]:
            r = np.log(b["close"].loc[:d]).diff().dropna().iloc[-120:]
            sigma = float(r.std() * math.sqrt(252)) if len(r) >= 20 else 0.6
            T = (h["unlock"] - d).days / 365
            disc = X.aap_discount(sigma, T)
            return h["qty"] * px * (1 - disc), "限售股：收盘价 ×（1 − AAP 流动性折扣）", px, disc, f"σ={sigma:.0%}，剩余 {T:.2f} 年"
        return h["qty"] * px, "收盘价（已解禁）", px, 0.0, ""

    quarter_ends = set(pd.Series(cal, index=cal).groupby(cal.to_period("Q")).last()) | {cal[-1]}
    all_out = lambda: all(h["status"] in ("已退出", "已核销") for h in holds.values())
    liquidated = None

    for d in cal:
        # 1) 虚构项目的事件
        for h in holds.values():
            if not h["synthetic"] or h["status"] in ("待交割", "已退出", "已核销"):
                continue
            for ev in h["spec"].get("story", []):
                if pd.Timestamp(ev["date"]) == d or (pd.Timestamp(ev["date"]) < d and not ev.get("_done") and pd.Timestamp(ev["date"]) >= cal[0]):
                    if ev.get("_done"):
                        continue
                    ev["_done"] = True
                    if "recovery" in ev:
                        bb = h["spec"]["buyback"]
                        yrs = (pd.Timestamp(bb["deadline"]) - h["close"]).days / 365
                        claim = h["cost_total"] * (1 + bb["rate"] * yrs)
                        got = claim * ev["recovery"]
                        cash += got
                        deal_flows[h["id"]].append((d, got))
                        tx.append({"product": code, "deal": h["id"], "date": d.date(), "type": "回购退出", "qty": 0, "price": 0, "amount": got,
                                   "note": f"回购主张 {claim / 1e4:,.0f} 万元，实收 {ev['recovery']:.0%}：{ev['note']}"})
                        h["status"], h["cost_left"], h["exit_date"] = "已退出", 0.0, d
                    elif ev["mark"] == 0:
                        tx.append({"product": code, "deal": h["id"], "date": d.date(), "type": "核销", "qty": 0, "price": 0, "amount": 0.0, "note": ev["note"]})
                        h["status"], h["exit_date"] = "已核销", d
                        h["written_off"] = h["cost_left"]
                        h["cost_left"] = 0.0
                    else:
                        h["mark"], h["mark_note"] = ev["mark"], f"按事件调整：{ev['note']}"
                    ic_events.append({"deal": h["id"], "date": d, "event": "投后事件", "note": ev["note"]})
        # 2) 管理费 / 基金费用（半年预收）；项目全部退出后进入清算，不再收费
        if d >= next_bill and not all_out():
            fee, exp, basis = fee_due(d)
            need = fee + exp
            if d >= invest_end and cash - commit * tm["cash_reserve"] >= need:
                cash -= need
                src = "基金现金"
            else:
                call(d, need, f"管理费 {fee / 1e4:,.0f} 万 + 基金费用 {exp / 1e4:,.0f} 万（{basis}）", lp_class)
                cash -= need
                src = "LP 缴款"
            tx.append({"product": code, "deal": "FUND", "date": d.date(), "type": "管理费", "qty": 0, "price": 0, "amount": -fee, "note": f"{basis}；来源：{src}"})
            tx.append({"product": code, "deal": "FUND", "date": d.date(), "type": "基金费用", "qty": 0, "price": 0, "amount": -exp, "note": f"来源：{src}"})
            next_bill = _months(next_bill, tm["fee_billing_months"])
        # 3) 交割
        for h in holds.values():
            if h["status"] == "待交割" and h["close"] <= d:
                call(d, h["cost_total"], f"{h['name']} 投资款", lps)
                cash -= h["cost_total"]
                deal_flows[h["id"]].append((d, -h["cost_total"]))
                h["status"] = "未上市"
                note = (f"跟投 {h['terms']['round']['date']} {h['terms']['round']['kind']}，投后估值 {h['terms']['post_money'] / 1e8:.2f} 亿，"
                        f"持股 {h['terms']['ownership_at_close']:.3%}（招股书第 {h['terms']['pages']['round']} 页）") if not h["synthetic"] else \
                    f"投后估值 {h['spec']['post_money'] / 1e8:.1f} 亿（虚构项目）"
                tx.append({"product": code, "deal": h["id"], "date": d.date(), "type": "投资", "qty": 0, "price": 0, "amount": -h["cost_total"], "note": note})
        # 4) 上市 → 5) 分红送转 → 6) 减持
        for h in holds.values():
            if h["synthetic"] or h["status"] in ("待交割", "已退出", "已核销"):
                continue
            if h["status"] == "未上市" and d >= h["listing"]:
                h["status"], h["qty"] = "限售", float(round(h["terms"]["shares"]))
                tx.append({"product": code, "deal": h["id"], "date": d.date(), "type": "上市", "qty": h["qty"], "price": h["terms"]["issue_price"], "amount": 0,
                           "note": f"发行价 {h['terms']['issue_price']} 元，持股 {h['qty'] / 1e4:,.1f} 万股，每股成本 {h['cost_ps']:.2f} 元；解禁日 {h['unlock'].date()}（{h['terms']['lockup_note']}）"})
            if h["qty"] <= 0:
                continue
            acts = h["acts"]
            for a in acts[acts["ex_date"] == d].itertuples():
                if a.cash_per_share > 0:
                    got = h["qty"] * a.cash_per_share
                    cash += got
                    deal_flows[h["id"]].append((d, got))
                    tx.append({"product": code, "deal": h["id"], "date": d.date(), "type": "分红", "qty": h["qty"], "price": a.cash_per_share, "amount": got, "note": "现金分红"})
                if a.bonus_per_share > 0:
                    add = math.floor(h["qty"] * a.bonus_per_share)
                    h["qty"] += add
                    h["cost_ps"] /= 1 + a.bonus_per_share
                    h["total_shares"] *= 1 + a.bonus_per_share
                    tx.append({"product": code, "deal": h["id"], "date": d.date(), "type": "送转", "qty": add, "price": 0, "amount": 0, "note": f"每股送转 {a.bonus_per_share:g}"})
            if h["status"] == "限售" and d >= h["unlock"]:
                h["status"] = "流通"
                ic_events.append({"deal": h["id"], "date": d, "event": "解禁", "note": f"开始按减持计划分批卖出（{ex['sell_months']} 个月）"})
            if h["status"] != "流通" or d not in h["bars"].index:
                continue
            bar = h["bars"].loc[d]
            px = float(bar["close"])
            forced = d >= force_exit
            if ex["defer_below_cost"] and px < h["cost_ps"] and not forced:
                q = pd.Timestamp(d).to_period("Q")
                if h["last_ic_q"] != q:
                    h["last_ic_q"] = q
                    ic_events.append({"deal": h["id"], "date": d, "event": "投委会复议",
                                      "note": f"股价 {px:.2f} 元低于每股成本 {h['cost_ps']:.2f} 元，暂缓减持"})
                continue
            end = max(_months(h["unlock"], ex["sell_months"]), d + pd.Timedelta(days=28))
            days_left = max(int(((cal > d) & (cal <= end)).sum()) + 1, 1)
            want = h["qty"] / days_left
            while h["sold90"] and h["sold90"][0][0] < d - pd.Timedelta(days=90):
                h["sold90"].popleft()
            room90 = h["total_shares"] * ex["max_total_share_90d"] - sum(q for _, q in h["sold90"])
            cap = min(float(bar["volume"]) * ex["max_volume_share"], max(room90, 0))
            q = min(max(want, 100.0), cap, h["qty"])                   # 至少一手，避免尾数一直卖不掉
            if h["qty"] - q < 100 and cap >= h["qty"]:
                q = h["qty"]                                           # 剩下不足一手（送转产生的零股）一起卖
            else:
                q = math.floor(q / 100) * 100
            if q <= 0:
                continue
            got = q * px * (1 - ex["cost"])
            cost_out = h["cost_left"] * q / h["qty"]
            h["qty"] -= q
            h["cost_left"] -= cost_out
            h["sold90"].append((d, q))
            cash += got
            deal_flows[h["id"]].append((d, got))
            tx.append({"product": code, "deal": h["id"], "date": d.date(), "type": "减持", "qty": q, "price": px, "amount": got,
                       "note": f"成本 {cost_out / 1e4:,.0f} 万，收益倍数 {got / cost_out:.2f}x"})
            if h["qty"] <= 0:
                h["status"], h["exit_date"], h["cost_left"] = "已退出", d, 0.0
                ic_events.append({"deal": h["id"], "date": d, "event": "退出完成", "note": "全部减持完毕"})
        # 7) 季度末：估值、分配、指标
        if d in quarter_ends:
            fvs = {}
            for h in holds.values():
                fv, method, px, disc, note = fair_value(h, d)
                fvs[h["id"]] = fv
                if h["status"] not in ("待交割",) and not (h["status"] in ("已退出", "已核销") and h.get("exit_date") and h["exit_date"] < d - pd.Timedelta(days=95)):
                    marks.append({"product": code, "date": d.date(), "deal": h["id"], "status": h["status"], "method": method,
                                  "qty": float(h.get("qty", 0)), "price": px, "discount": disc, "fv": fv, "cost": h["cost_left"], "note": note})
            # 分配：现金超过「保留现金 + 未来半年费用」就分
            fee, exp, _ = fee_due(d)
            winding_up = all_out() and d >= invest_end
            buffer = 0.0 if winding_up else commit * tm["cash_reserve"] + (fee + exp if d >= invest_end else 0)
            dist = cash - buffer
            if (dist > commit * 0.005 or (winding_up and dist > 1)) and d >= invest_end - pd.Timedelta(days=366):
                tot_paid = sum(paid.values())
                gp_part = dist * paid[gp_lp["id"]] / tot_paid
                lp_part = dist - gp_part
                r = wf.distribute(d, lp_part)
                cash -= dist
                lp_flows.append((d, r["LP 合计"]))
                for x in lp_class:
                    share = x["commitment"] / lp_commit
                    capital.append({"product": code, "date": d.date(), "notice_date": d.date(), "lp": x["id"], "type": "分配",
                                    "amount": r["LP 合计"] * share, "detail": {k: round(r[k] * share, 2) for k in ("返还实缴", "门槛收益", "超额收益（LP 80%）")}})
                capital.append({"product": code, "date": d.date(), "notice_date": d.date(), "lp": gp_lp["id"], "type": "分配", "amount": gp_part,
                                "detail": {"GP 跟投部分（按实缴比例）": round(gp_part, 2)}})
                if r["GP 合计"] > 0:
                    capital.append({"product": code, "date": d.date(), "notice_date": d.date(), "lp": gp_lp["id"], "type": "业绩报酬", "amount": r["GP 合计"],
                                    "detail": {k: round(r[k], 2) for k in ("GP 追补", "超额收益（GP 20%）")}})
            nav = cash + sum(fvs.values())
            tot_paid = sum(paid.values())
            lp_paid = sum(paid[x["id"]] for x in lp_class)
            lp_share_nav = nav * (lp_paid / tot_paid) if tot_paid else 0.0
            hyp = wf.hypothetical(d, lp_share_nav)
            lp_dist = sum(v for dd_, v in lp_flows if v > 0)
            mult = X.multiples(lp_paid, lp_dist, hyp["lp_nav"])
            g_in = sum(-v for fl in deal_flows.values() for _, v in fl if v < 0)
            g_out = sum(v for fl in deal_flows.values() for _, v in fl if v > 0)
            gross_flows = [x for fl in deal_flows.values() for x in fl] + [(d, sum(fvs.values()))]
            contribs = [(t_, -v) for t_, v in lp_flows if v < 0]
            dists = [(t_, v) for t_, v in lp_flows if v > 0]
            quarters.append({"product": code, "date": d.date(), "metrics": {
                "nav": nav, "cash": cash, "portfolio_fv": sum(fvs.values()), "lp_paid_in": lp_paid, "lp_distributed": lp_dist, "lp_nav": hyp["lp_nav"],
                "accrued_carry": hyp["accrued_carry"], "carry_paid": wf.gp_carry, "unfunded": lp_commit - lp_paid, "called_pct": lp_paid / lp_commit,
                **mult, "net_irr": X.xirr(lp_flows + [(d, hyp["lp_nav"])]), "gross_moic": (g_out + sum(fvs.values())) / g_in if g_in else float("nan"),
                "gross_irr": X.xirr(gross_flows), "cum_net_cf": sum(v for _, v in lp_flows),
                "pme": {nm: X.ks_pme(contribs, dists, hyp["lp_nav"], M.index_close(sym), d) for sym, nm in cfg["pme_indices"].items()},
                "waterfall_paid": dict(wf.paid), "liquidated": bool(winding_up)}})
            if winding_up:
                liquidated = d
                ic_events.append({"deal": "FUND", "date": d, "event": "基金清算", "note": "项目全部退出，分配剩余现金，基金进入注销程序"})
                break

    # ---- 落库 ----
    for t in (db.pe_tx, db.pe_marks, db.pe_capital, db.pe_quarters, db.pe_pipeline):
        db.replace_rows(t, [], t.c.product == code)
    db.add(db.pe_tx, tx)
    db.add(db.pe_marks, marks)
    db.add(db.pe_capital, capital)
    db.add(db.pe_quarters, [{**q, "metrics": _clean(q["metrics"])} for q in quarters])
    db.add(db.pe_pipeline, _pipeline(code, cfg, holds, ic_events))
    db.replace_rows(db.products, [{"code": code, "name": cfg["name"], "type": cfg["type"], "config": _clean(cfg)}], db.products.c.code == code)
    last = quarters[-1]["metrics"]
    say(f"  {cal[-1].date()} LP 实缴 {last['lp_paid_in'] / 1e8:.2f} 亿，已分配 {last['lp_distributed'] / 1e8:.2f} 亿，"
        f"LP 净值 {last['lp_nav'] / 1e8:.2f} 亿；TVPI {last['TVPI']:.2f}，DPI {last['DPI']:.2f}，净 IRR {last['net_irr']:.1%}")
    return {"asof": str(cal[-1].date()), "liquidated": str(liquidated.date()) if liquidated is not None else None, **{k: last[k] for k in ("TVPI", "DPI", "RVPI", "net_irr", "gross_irr", "gross_moic", "pme")},
            "tx": len(tx), "quarters": len(quarters)}


def _pipeline(code: str, cfg: dict, holds: dict, ic_events: list) -> list[dict]:
    rows = []
    for h in holds.values():
        c = h["close"]
        ev = [{"date": str((c - pd.Timedelta(days=k)).date()), "event": e} for k, e in ((150, "线索"), (110, "立项"), (75, "尽调"), (25, "投决：批准"))]
        ev.append({"date": str(c.date()), "event": "交割"})
        if not h["synthetic"]:
            ev += [{"date": str(h["listing"].date()), "event": "上市"}]
        ev += [{"date": str(pd.Timestamp(e["date"]).date()), "event": e["event"], "note": e["note"]} for e in ic_events if e["deal"] == h["id"]]
        stage = {"已退出": "退出", "已核销": "退出"}.get(h["status"], "投后")
        rows.append({"product": code, "deal": h["id"], "company": h["name"], "sector": h["sector"], "stage": stage, "status": h["status"],
                     "events": sorted(ev, key=lambda x: x["date"]), "reason": "", "synthetic": int(h["synthetic"])})
    for p in cfg.get("pipeline_only", []):
        d = dd.load(p["code"])
        listing = pd.Timestamp(d["listing_date"])
        base = listing - pd.DateOffset(months=20)
        order = ["线索", "立项", "尽调", "投决"]
        ev = [{"date": str((base + pd.Timedelta(days=30 * i)).date()), "event": s} for i, s in enumerate(order[: order.index(p["stage"]) + 1])]
        if p.get("decision"):
            ev[-1]["event"] = f"投决：{p['decision']}"
        ev[-1]["note"] = p["reason"]
        b = market.bars(p["code"])
        ipo = d["items"]["ipo"]["issue_price"]["value"]
        last = float(b["close"].iloc[-1])
        ev.append({"date": str(listing.date()), "event": "上市（事后回看）", "note": f"发行价 {ipo} 元，最新收盘 {last:.2f} 元（未复权，{last / ipo - 1:+.0%}）"})
        rows.append({"product": code, "deal": p["code"], "company": d["company"], "sector": d["items"]["business"]["industry"][:20], "stage": p["stage"],
                     "status": "未投资", "events": ev, "reason": p["reason"], "synthetic": 0})
    return rows


def _clean(x):
    import json
    return json.loads(json.dumps(x, default=lambda o: str(o) if not isinstance(o, float) else None).replace("NaN", "null"))
