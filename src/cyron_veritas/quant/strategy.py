"""量化指数增强产品（CV-QE1）的逐日模拟：月末优化 → 次日起分多日执行 → 同一套账簿、估值表、净值、合规。

时点：
  月末 e 收盘后：合成因子（只用 e 以前已知的 IC 定权重）、风险模型（e 及以前的数据）、基准权重（e 时点）→ 优化出目标权重；
  目标股数 = 目标权重 × e 日净值 / e 日收盘价，此后固定，直到下一次调仓覆盖。
  执行从 e 的次一交易日开始：每天先卖后买，每只股票成交不超过当日成交量 × 参与率，没做完的部分次日继续；
  一字涨停买不进、一字跌停卖不出、停牌不交易；买入后现金不低于合同下限；
  成交价 = 收盘价 ×（1 ± 冲击成本），冲击成本 = 半价差 5bp + k × 20 日波动 × √(成交量占比)。
风险模型偏差校正：事前跟踪误差系统性低估时（事后 / 事前 > 1），把风险模型整体放大，让风险厌恶系数和跟踪误差上限按真实风险起作用。
所有调仓记录（优化状态、放宽 / 收紧的约束、事前跟踪误差、预期超额、换手、执行完成率、平均冲击成本）写入 quant_rebalances 表。
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from cyron_veritas.config import ROOT
from cyron_veritas.fund import db
from cyron_veritas.fund import marketdata as M
from cyron_veritas.fund import product as P
from cyron_veritas.fund import risk as R
from cyron_veritas.fund.ledger import Ledger
from cyron_veritas.fund.simulate import Market
from cyron_veritas.quant import factors as F
from cyron_veritas.quant import optimizer as O
from cyron_veritas.quant import research as Q
from cyron_veritas.quant import riskmodel as RMOD

OUT = ROOT / "data" / "quant"


def _alpha(cfg: dict) -> pd.DataFrame:
    a = Q.load("alpha") if cfg["strategy"].get("alpha", "icir") == "icir" else None
    if a is None:
        p = Q.build_panel()
        ew = p[F.ALPHA].mean(axis=1)
        a = pd.DataFrame({"date": p["date"], "code": p["code"], "ALPHA": ew.groupby(p["date"]).transform(lambda s: (s - s.mean()) / s.std())})
    return a


def run(code: str = "CV-QE1", end: str | None = None, write_sheets: bool = True, say=print) -> dict:
    cfg = P.load(code)
    st, lim = cfg["strategy"], cfg["contract_limits"]
    mkt = Market(M.universe()["code"].tolist())
    cal = mkt.calendar[(mkt.calendar >= pd.Timestamp(cfg["inception"])) & (mkt.calendar <= pd.Timestamp(end or mkt.close.index[-1]))]
    L = Ledger(code, cfg, mkt, cal, write_sheets)
    L.reset_outputs((db.quant_rebalances,))
    alpha = _alpha(cfg)
    ret = R.market_view()["ret"]
    vol20 = ret.rolling(20, min_periods=10).std()
    bidx = P.bench_index(cfg)
    ends = set(F.month_ends(str(pd.Timestamp(cfg["inception"]) - pd.Timedelta(days=40))))
    ends = {e for e in ends if e < cal[-1]}
    targets: dict[str, int] = {}
    rebal_rows, weights_rows = [], []
    cur = None                                   # 当前调仓的执行统计
    first = True
    nav_hist: dict[pd.Timestamp, float] = {}
    bret = R.bench_returns(P.bench_symbol(cfg))

    for d in cal:
        L.start_day(d)
        prev_d = L.prev_d
        # ---- 建仓日：用成立前最后一个月末的模型（成立日之前的信息） ----
        sig_date = None
        if first:
            sig_date = max(e for e in ends if e < d)
            first = False
        elif prev_d is not None and prev_d in ends:
            sig_date = prev_d
        if sig_date is not None:
            if cur:
                rebal_rows.append(_close_exec(cur, targets, L))
            nav_e = L.nav_at(sig_date) if L.pos else L.cash
            model = RMOD.model_at(sig_date)
            z = alpha[alpha["date"] == sig_date].set_index("code")["ALPHA"]
            wb, wsrc = R.bench_weights(bidx, sig_date)
            px_e = mkt.close_ff.loc[sig_date]
            w0 = pd.Series({c: p["qty"] * px_e.get(c, np.nan) / nav_e for c, p in L.pos.items()}).dropna()
            frozen = {c for c in L.pos if not mkt.tradable(c, sig_date)}
            # 风险模型偏差校正：事后跟踪误差（60 日）/ 最近 3 次调仓的事前跟踪误差，截断到 [1, 2.5]，协方差和特异方差乘它的平方
            nv = pd.Series(nav_hist)
            ex = (nv.pct_change() - bret.reindex(nv.index)).dropna().iloc[-60:]
            te_post = float(ex.std() * np.sqrt(252)) if len(ex) >= 40 else None
            prior = [r["ex_ante_te"] for r in rebal_rows[-3:] if r.get("ex_ante_te")]
            bias = float(np.clip(te_post / np.mean(prior), 1.0, 2.5)) if te_post and prior else 1.0
            if bias > 1:
                model = {**model, "F": model["F"] * bias ** 2, "D": model["D"] * bias ** 2}
            # 事后跟踪误差超过目标的 90%：风控要求下次调仓把事前上限收紧到 70%
            te_scale = 0.7 if te_post and te_post > lim["te_max"] * 0.9 else 1.0
            res = O.optimize(z, model, wb, w0, cfg, frozen=frozen, bench_members=set(wb.index), initial=not L.pos, te_scale=te_scale)
            notes = []
            if bias > 1:
                notes.append(f"风险模型偏差校正 ×{bias:.2f}（事后 {te_post:.1%}）")
            if te_scale < 1:
                notes.append(f"事后跟踪误差 {te_post:.1%} 偏高，事前上限收紧到 {lim['te_max'] * te_scale:.1%}")
            res["relaxed"] = notes + res.get("relaxed", [])
            w = res["weights"]
            targets = {c: int(math.floor(v * nav_e / px_e[c] / 100) * 100) for c, v in w.items() if px_e.get(c, 0) > 0}
            for c in L.pos:                                                  # 优化结果里没有的持仓 → 目标 0
                targets.setdefault(c, 0)
            cur = {"product": code, "date": sig_date.date(), "exec_start": d.date(), "status": res["status"], "relaxed": res.get("relaxed", []),
                   "ex_ante_te": res.get("ex_ante_te"), "exp_alpha": res.get("alpha"), "turnover": res.get("turnover"),
                   "n_holdings": res.get("n_holdings"), "member_weight": res.get("member_weight"), "bench_source": wsrc,
                   "exposure": {k: round(float(v), 4) for k, v in res.get("exposure", pd.Series(dtype=float)).items()},
                   "target_value": float(sum(abs(q - L.qty(c)) * px_e.get(c, 0) for c, q in targets.items())),
                   "filled_value": 0.0, "impact_cost": 0.0, "nav": nav_e}
            weights_rows += [{"date": sig_date, "code": c, "weight": float(v), "bench": float(wb.get(c, 0))} for c, v in w.items()]
        # ---- 执行：先卖后买，按参与率分多日完成 ----
        if targets:
            vol_d = mkt.volume.loc[d] if d in mkt.volume.index else pd.Series(dtype=float)
            sig = vol20.loc[:d].iloc[-1] if d in vol20.index else pd.Series(dtype=float)
            floor = lim["cash_min"] * L.nav_at(prev_d or d) * 1.2
            for side in ("sell", "buy"):
                for c, tgt in list(targets.items()):
                    held = L.qty(c)
                    need = tgt - held
                    if (side == "sell" and need >= 0) or (side == "buy" and need <= 0) or not mkt.tradable(c, d):
                        continue
                    state = mkt.limit_state(c, d, prev_d)
                    if (side == "sell" and state == "down") or (side == "buy" and state == "up"):
                        continue
                    cap = math.floor(float(vol_d.get(c, 0) or 0) * st["participation"] / 100) * 100
                    q = min(abs(need), cap)
                    if side == "sell" and tgt == 0 and abs(need) <= cap:
                        q = held                                              # 清仓时零股一起卖
                    px = float(mkt.close.at[d, c])
                    s20 = float(sig.get(c, 0.02) if pd.notna(sig.get(c, np.nan)) else 0.02)
                    impact = 0.0005 + st["impact_k"] * s20 * math.sqrt(q / max(float(vol_d.get(c, 0) or 0), 1))
                    price = px * (1 - impact) if side == "sell" else px * (1 + impact)
                    if side == "buy":
                        afford = math.floor((L.cash - floor) / (price * (1 + 0.0003)) / 100) * 100
                        q = min(q, afford)
                    if q <= 0:
                        continue
                    (L.sell if side == "sell" else L.buy)(d, c, q, price, "组合优化调仓")
                    if cur:
                        cur["filled_value"] += q * px
                        cur["impact_cost"] += q * px * impact
        summ = L.end_day(d)
        nav_hist[d] = summ["unit_nav"]
        if d.day <= 7 and d.dayofweek == 0:
            say(f"  {d:%Y-%m-%d} 单位净值 {summ['unit_nav']:.4f} 规模 {summ['net_assets'] / 1e8:.2f} 亿 持仓 {len(L.pos)}")

    if cur:
        rebal_rows.append(_close_exec(cur, targets, L))
    L.finish()
    db.add(db.quant_rebalances, rebal_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(weights_rows).to_parquet(OUT / f"targets_{code}.parquet")
    return {"days": len(cal), "final_unit_nav": L.prev_unit, "final_net_assets": L.prev_net, "trades": len(L.trade_rows),
            "rebalances": len(rebal_rows), "sheets": str(L.out_dir)}


def _close_exec(cur: dict, targets: dict, L: Ledger) -> dict:
    """调仓窗口结束（下一次调仓开始）时统计执行情况：完成率、未完成金额、平均冲击成本（bp）。"""
    px = {c: L.mkt.last_price(c, L.prev_d) for c in targets}
    left = float(sum(abs(q - L.qty(c)) * (px[c] or 0) for c, q in targets.items()))
    tv = cur["target_value"]
    return {**{k: v for k, v in cur.items() if k not in ("target_value", "filled_value", "impact_cost", "nav")},
            "fill_rate": float(1 - left / tv) if tv else 1.0, "unfilled_value": left, "traded_value": cur["filled_value"],
            "impact_bp": float(cur["impact_cost"] / cur["filled_value"] * 1e4) if cur["filled_value"] else 0.0,
            "exec_end": L.prev_d.date() if L.prev_d is not None else None, "created_at": dt.datetime.now()}
