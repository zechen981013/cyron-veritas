"""模拟器：按真实行情和公司行为逐日运行一只股票多头私募，产出「外部世界」的数据——
托管行每日估值表（Excel）、券商成交流水、登记机构申赎记录、股票池与投委会决议。

中台的其余部分（解析、对账、风控、归因、报告）只读这些产出，不读模拟器内部状态——接真实数据时把模拟器换掉即可。

账务规则见 docs/PLATFORM_DESIGN.md 第 2 节；选股规则：
  每月第一个交易日调仓；只用已过披露截止日的报告期（marketdata.available_period）；
  条件：上市 ≥ 120 个交易日、非禁投、0 < PE < 60、年化 ROE > 8%、净利同比 > 0、每股经营现金流 > 0；
  打分：ROE、低 PE、营收增速、现金流 / 利润 四项的截面排名之和；选前 25 只等权（单一行业最多 7 只）。
  股票池按季度复核（新报告期可用后），现有持仓排名在前 50 内保留；其余月份只调回等权。超限后次一交易日「风控整改」只卖不买。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cyron_veritas.fund import db
from cyron_veritas.fund import marketdata as M
from cyron_veritas.fund import product as P
from cyron_veritas.fund.ledger import Ledger, fee

ANNUALIZE = {"0331": 4.0, "0630": 2.0, "0930": 4 / 3, "1231": 1.0}


class Market:
    """模拟器用到的行情视图（宽表）。"""

    def __init__(self, codes: list[str]):
        b = M.bars(codes)
        self.close, self.high, self.low = M.wide(b, "close"), M.wide(b, "high"), M.wide(b, "low")
        self.volume = M.wide(b, "volume")  # 股
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


def select(mkt: Market, d: pd.Timestamp, prev: pd.Timestamp, cfg: dict, current: set[str] = frozenset()) -> tuple[list[str], list[dict]]:
    """返回（新核心池, 被投委会退回的候选）。现有持仓只要仍满足硬条件、排名在前 2N 内就保留（缓冲区，压换手）；
    其余名额按排名补，候选若「净利增速 > 200% 但营收增速 < 10%」——利润增长疑似来自非经常性损益——投委会要求补充材料，本期不入池。"""
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
    ranked = f.sort_values("score", ascending=False)
    keep = [c for c in ranked.index[: 2 * st["n_holdings"]] if c in current]
    picks, per_ind, sent_back = [], {}, []
    for code in keep + [c for c in ranked.index if c not in keep]:
        ind = mkt.industry.get(code, "未分类")
        if per_ind.get(ind, 0) >= 7:
            continue
        r = ranked.loc[code]
        if code not in current and r["np_yoy"] > 200 and r["rev_yoy"] < 10:
            sent_back.append({"code": code, "np_yoy": float(r["np_yoy"]), "rev_yoy": float(r["rev_yoy"])})
            continue
        picks.append(code)
        per_ind[ind] = per_ind.get(ind, 0) + 1
        if len(picks) >= st["n_holdings"]:
            break
    return picks, sent_back


def _breaches(mkt: Market, pos: dict, cash: float, recv: dict, payable: float, d, lim: dict) -> dict:
    """用上一交易日收盘价检查合同限额，返回超限项 {规则: 实际值}。"""
    mv = {c: p["qty"] * mkt.last_price(c, d) for c, p in pos.items()}
    net = cash + sum(mv.values()) + sum(r["amount"] for r in recv.values()) - payable
    if net <= 0 or not mv:
        return {}
    out = {}
    if cash / net < lim["cash_min"]:
        out["现金"] = cash / net
    single = {c: v / net for c, v in mv.items() if v / net > lim["single_name_max"]}
    if single:
        out["单票"] = single
    ind: dict[str, float] = {}
    for c, v in mv.items():
        k = mkt.industry.get(c, "未分类")
        ind[k] = ind.get(k, 0) + v / net
    over = {k: v for k, v in ind.items() if v > lim["industry_max"]}
    if over:
        out["行业"] = over
    return out


def _fix_targets(mkt: Market, pos: dict, d, nav_now: float, fix: dict, st: dict, lim: dict) -> dict:
    """整改目标权重：单票压到 max_weight，超限行业等比压到行业上限的 95%，现金不足时全组合等比减仓到 cash_target。"""
    w0 = {c: p["qty"] * mkt.last_price(c, d) / nav_now for c, p in pos.items()}
    w = dict(w0)
    for c in fix.get("单票", {}):
        w[c] = min(w[c], st["max_weight"])
    for k in fix.get("行业", {}):
        names = [c for c in w if mkt.industry.get(c, "未分类") == k]
        tot = sum(w[c] for c in names)
        for c in names:                      # 单票压完后行业可能已回到限额内，只缩不放
            w[c] *= min(1.0, lim["industry_max"] * 0.95 / tot)
    if "现金" in fix:
        tot = sum(w.values())
        w = {c: v * min(1.0, (1 - st["cash_target"]) / tot) for c, v in w.items()}
    return {c: v for c, v in w.items() if v < w0[c] - 1e-9}     # 只返回要减仓的股票，其余不动


def run(code: str = "CV-VG1", end: str | None = None, write_sheets: bool = True, say=print) -> dict:
    cfg = P.load(code)
    uni = M.universe()
    mkt = Market(uni["code"].tolist())
    cal = mkt.calendar[(mkt.calendar >= pd.Timestamp(cfg["inception"])) & (mkt.calendar <= pd.Timestamp(end or mkt.close.index[-1]))]
    L = Ledger(code, cfg, mkt, cal, write_sheets)
    L.reset_outputs((db.pool, db.ic))
    core: set[str] = set()
    pool_rows, ic_rows = [], []
    st = cfg["strategy"]
    lim = cfg["contract_limits"]
    last_month = None
    last_period = None

    for d in cal:
        L.start_day(d)
        prev_d = L.prev_d
        # 调仓（每月第一个交易日；首日建仓）
        rebalance = prev_d is None or d.month != last_month
        last_month = d.month
        stop_hit = L.prev_unit < lim["stop_line"]
        warn_hit = L.prev_unit < lim["warning_line"]
        fix = _breaches(mkt, L.pos, L.cash, L.recv, L.payable(), prev_d, lim) if prev_d is not None and not rebalance else {}
        if rebalance or stop_hit or fix or L.cash < -1:
            nav_now = L.nav_at(prev_d or d)
            if stop_hit:
                targets = {}
            elif rebalance:
                # 股票池：新报告期可用后（5 / 9 / 11 月初）才做季度复核；其余月份只把核心池调回等权
                period = M.available_period(d)
                review = period != last_period
                last_period = period
                picks, sent_back = select(mkt, d, prev_d or d - pd.Timedelta(days=1), cfg, core) if review else (sorted(core), [])
                for b in sent_back[:3]:
                    ic_rows.append({"product": code, "code": b["code"], "name": mkt.names.get(b["code"], b["code"]), "request": "入核心池",
                                    "requested_by": "研究部（模型初筛）", "requested_at": d.date(),
                                    "material": f"季度筛选 {period}：综合排名靠前；净利同比 {b['np_yoy']:.0f}%、营收同比 {b['rev_yoy']:.0f}%",
                                    "decision": "补充材料", "votes": {"赞成": 1, "反对": 2, "弃权": 0}, "decided_at": d.date(),
                                    "note": "利润增速远高于营收增速，疑似非经常性损益驱动，要求研究员补充扣非数据后再议"})
                new_in, out = [c for c in picks if c not in core], [c for c in core if c not in picks]
                for i, c in enumerate(new_in):  # 入池：模拟投委会决议（研究员提交 → 投委会批准）
                    ic_rows.append({"product": code, "code": c, "name": mkt.names.get(c, c), "request": "入核心池", "requested_by": "研究部（模型初筛）",
                                    "requested_at": d.date(), "material": f"季度筛选 {period}：ROE / 估值 / 增长 / 现金流综合排名前 {st['n_holdings']}",
                                    "decision": "批准", "votes": {"赞成": 3, "反对": 0, "弃权": 0} if i % 3 else {"赞成": 2, "反对": 1, "弃权": 0},
                                    "decided_at": d.date(), "note": "模拟决议"})
                    pool_rows.append({"product": code, "code": c, "name": mkt.names.get(c, c), "pool": "核心池", "since": d.date(), "until": None,
                                      "reason": "投委会批准入池", "decision_id": None})
                for c in out:
                    pool_rows.append({"product": code, "code": c, "name": mkt.names.get(c, c), "pool": "观察池", "since": d.date(), "until": None,
                                      "reason": "季度复核：跌出前 2N 或不再满足硬条件，调出核心池", "decision_id": None})
                core = set(picks)
                w = min((1 - st["cash_target"]) / max(1, len(picks)), st["max_weight"])
                targets = {c: w for c in picks}
            else:  # 风控整改：上一交易日超限，今日按合同要求调回限额内（只卖不买）
                targets = _fix_targets(mkt, L.pos, prev_d, nav_now, fix, st, lim)
            reason = "止损清仓" if stop_hit else ("调仓" if rebalance else "风控整改：" + "、".join(fix))
            # 先卖后买
            names = set(targets) if fix and not (rebalance or stop_hit) else set(L.pos) | set(targets)
            for c in sorted(names, key=lambda c: targets.get(c, 0)):
                if not mkt.tradable(c, d):
                    continue
                px = float(mkt.close.at[d, c])
                ref = px if not fix or rebalance else mkt.last_price(c, prev_d)   # 整改按昨收算目标股数，避免当日涨跌带来额外交易
                tgt_qty = math.floor(targets.get(c, 0) * nav_now / ref / 100) * 100
                cur = L.qty(c)
                delta = tgt_qty - cur
                if delta < 0 and mkt.limit_state(c, d, prev_d) != "down":
                    L.sell(d, c, -delta if tgt_qty > 0 else cur, px * (1 - cfg["trading"]["slippage"]), reason)
            for c, w in targets.items():
                if not mkt.tradable(c, d) or warn_hit or not rebalance:
                    continue
                px = float(mkt.close.at[d, c])
                delta = math.floor(w * nav_now / px / 100) * 100 - L.qty(c)
                if delta <= 0 or mkt.limit_state(c, d, prev_d) == "up":
                    continue
                price = px * (1 + cfg["trading"]["slippage"])
                floor = (lim["cash_min"] + st["cash_target"]) / 2 * nav_now   # 买入后现金不低于（下限 + 目标）/ 2，给费用和赎回留缓冲
                if delta * price + fee(cfg, "buy", delta * price) > L.cash - floor:
                    delta = math.floor((L.cash - floor) / price / 100) * 100
                    if delta <= 0:
                        continue
                L.buy(d, c, delta, price, "调仓")
        summ = L.end_day(d)
        if d.day <= 7 and d.dayofweek == 0:
            say(f"  {d:%Y-%m-%d} 单位净值 {summ['unit_nav']:.4f} 规模 {summ['net_assets'] / 1e8:.2f} 亿 持仓 {len(L.pos)}")

    L.finish()
    db.add(db.ic, ic_rows)
    db.add(db.pool, pool_rows)
    return {"days": len(cal), "final_unit_nav": L.prev_unit, "final_net_assets": L.prev_net, "trades": len(L.trade_rows), "sheets": str(L.out_dir)}
