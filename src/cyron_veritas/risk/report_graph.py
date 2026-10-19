"""组合风控日报：LangGraph 固定流程。

    计算（代码）→ 限额检查（规则）→ 大模型写解读 → 数字溯源校验 ─┬─ 通过 → 生成报告
                                         ↑                 ├─ 不通过且没重写过 → 带着错误重写一次
                                         └─────────────────┘
                                                           └─ 还不通过 → 改用模板文字（保证日报一定能出）

为什么用固定流程而不是让 Agent 自由发挥：风控日报每天的步骤一样，数字必须可复核；
大模型只在「解读」这一个节点出现，并且出口有校验。
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from cyron_veritas import verify as V
from cyron_veritas.config import OUTPUT_DIR, chat_model
from cyron_veritas.risk import limits as L
from cyron_veritas.risk import portfolio as P

MAX_REWRITES = 1


class Narrative(BaseModel):
    summary: str = Field(description="今日概览，3-5 句话")
    key_risks: list[str] = Field(description="主要风险点，3-5 条，每条一句话")
    limit_comments: list[str] = Field(description="对每条超限或预警限额的说明，没有则为空列表")
    review_items: list[str] = Field(description="需要投资经理复核的事项，2-4 条，不能是买卖建议")


class State(TypedDict, total=False):
    holdings_path: str
    pr: object
    facts: dict
    limits: list
    facts_table: str
    narrative: dict
    attempts: int
    check: dict
    feedback: str
    source: str
    outputs: dict
    llm_error: str


def facts_table(f: dict, lim: list[dict]) -> str:
    """把计算结果格式化成大模型唯一可以引用的事实表。"""
    lines = [f"日期: {f['as_of']}", f"组合总市值: {V.wan(f['total_value'])}（{f['n_positions']} 个持仓，{len(f['accounts'])} 个账户）",
             f"汇率: 美元 {V.num(f['fx']['USD'], 4)}，港元 {V.num(f['fx']['HKD'], 4)}", "", "[持仓权重 / 浮动盈亏（本币）]"]
    for name, w in f["weights"].items():
        pnl = f["pnl_pct"].get(name)
        lines.append(f"- {name}: 权重 {V.pct(w)}" + (f"，浮盈亏 {V.pct(pnl)}" if pnl is not None else ""))
    lines += ["", "[敞口]",
              "行业: " + "，".join(f"{k} {V.pct(v)}" for k, v in f["by_sector"].items()),
              "市场: " + "，".join(f"{k} {V.pct(v)}" for k, v in f["by_market"].items()),
              "币种: " + "，".join(f"{k} {V.pct(v)}" for k, v in f["by_currency"].items()),
              "", "[集中度]",
              f"最大持仓 {f['concentration']['top1_name']} {V.pct(f['concentration']['top1'])}，前五大合计 {V.pct(f['concentration']['top5'])}，"
              f"有效持仓数 {V.num(f['concentration']['effective_n'], 1)}",
              "", f"[风险指标]（近 {f['lookback_days']} 个交易日，自 {f['lookback_start']} 起，按当前持仓回放）",
              f"年化波动率 {V.pct(f['vol_ann'])}；相对沪深300 的 Beta {V.num(f['beta_csi300'])}",
              f"1日 95% VaR（历史法）{V.pct(f['var95_hist'], 2)}，约 {V.wan(f['var95_amount'])}；参数法 {V.pct(f['var95_param'], 2)}",
              f"1日 99% VaR（历史法）{V.pct(f['var99_hist'], 2)}，约 {V.wan(f['var99_amount'])}",
              f"1日 95% CVaR {V.pct(f['cvar95_hist'], 2)}，约 {V.wan(f['cvar95_amount'])}",
              f"最大回撤 {V.pct(f['max_drawdown'])}（{f['mdd_peak']} 至 {f['mdd_trough']}）；当前回撤 {V.pct(f['current_drawdown'])}",
              "", "[风险贡献（占组合波动）]"]
    lines += [f"- {k}: {V.pct(v)}" for k, v in f["risk_contribution"].items()]
    lines += ["", "[相关性最高的持仓对]"] + [f"- {a} 与 {b}: {V.num(c)}" for a, b, c in f["top_corr_pairs"]]
    lines += ["", "[压力测试（当前持仓）]"]
    lines += [f"- {k}（{v['type']}{'，' + v['period'] if v['period'] else ''}）: {V.pct(v['pnl_pct'])}，约 {V.wan(v['pnl_amount'])}"
              for k, v in f["stress"].items()]
    bt = f["var_backtest"]
    lines += ["", "[VaR 回测]",
              f"最近 {bt['days']} 天，95% VaR 被突破 {bt['exceptions']} 次（理论值 {V.num(bt['expected'], 1)} 次），"
              f"Kupiec 检验 p 值 {V.num(bt['p_value'])}"]
    lines += ["", "[限额检查]"]
    for r in lim:
        op = "≥" if r["direction"] == "min" else "≤"
        lines.append(f"- {r['label']}（{op} {V.pct(r['limit'])}）: 实际 {V.pct(r['value'])}（{r['note']}）→ {r['status']}")
    return "\n".join(lines)


SYSTEM = """你是基金公司的风险分析师，负责写组合风控日报的文字解读。
规则：
1. 只能使用【事实表】里出现过的数字，原样引用（包括小数位和单位），不要自己计算、换算或四舍五入；
2. 不给任何买卖建议（不出现买入、卖出、加仓、减仓、目标价等词），只指出风险和需要复核的事项；
3. 先讲最重要的风险：超限项、集中度、风险贡献最大的持仓、压力测试里最坏的情景；
4. 语言简洁专业，每条一句话。"""


def node_compute(state: State) -> State:
    pr = P.build(P.load_holdings(state["holdings_path"]))
    lim = L.check(pr.facts)
    return {"pr": pr, "facts": pr.facts, "limits": lim, "facts_table": facts_table(pr.facts, lim), "attempts": 0}


def node_narrate(state: State) -> State:
    model = chat_model(0.2).with_structured_output(Narrative, method="function_calling")
    user = f"【事实表】\n{state['facts_table']}\n\n请写今天的风控日报解读。"
    if state.get("feedback"):
        user += f"\n\n上一版没有通过校验：{state['feedback']}。请改正后重写。"
    try:
        out: Narrative = model.invoke([("system", SYSTEM), ("user", user)])
    except Exception as e:  # 模型不可用时直接走模板，日报照样生成
        return {"llm_error": f"{type(e).__name__}: {e}"[:300], "attempts": state.get("attempts", 0) + 1}
    return {"narrative": out.model_dump(), "attempts": state.get("attempts", 0) + 1}


def _text(n: dict) -> str:
    return "\n".join([n["summary"], *n["key_risks"], *n["limit_comments"], *n["review_items"]])


def node_verify(state: State) -> State:
    if state.get("llm_error"):
        return {"check": {"ok": False}, "feedback": state["llm_error"]}
    allowed = V.allowed_set(state["facts_table"])
    res = V.check(_text(state["narrative"]), allowed)
    fb = ""
    if not res["ok"]:
        parts = []
        if res["unknown_numbers"]:
            parts.append(f"这些数字在事实表里找不到: {', '.join(res['unknown_numbers'])}")
        if res["advice_words"]:
            parts.append(f"出现了投资建议措辞: {', '.join(res['advice_words'])}")
        fb = "；".join(parts)
    return {"check": res, "feedback": fb, "source": "大模型（数字已核对）" if res["ok"] else state.get("source", "")}


def node_fallback(state: State) -> State:
    """模板文字：只用事实表里的数字，保证报告一定能生成。"""
    f, lim = state["facts"], state["limits"]
    bad = [r for r in lim if r["status"] != L.OK]
    worst = min(f["stress"].items(), key=lambda kv: kv[1]["pnl_pct"])
    top_rc = next(iter(f["risk_contribution"].items()))
    n = {"summary": f"组合总市值 {V.wan(f['total_value'])}，1日 95% VaR {V.pct(f['var95_hist'], 2)}，"
                    f"当前回撤 {V.pct(f['current_drawdown'])}；限额检查有 {len(bad)} 项需要关注。",
         "key_risks": [f"{top_rc[0]} 贡献了组合 {V.pct(top_rc[1])} 的波动。",
                       f"压力测试最坏情景为「{worst[0]}」，组合 {V.pct(worst[1]['pnl_pct'])}。"],
         "limit_comments": [f"{r['label']}: 实际 {V.pct(r['value'])}（{r['note']}），状态 {r['status']}。" for r in bad],
         "review_items": ["请复核超限项是否已获批准或有调整计划。"]}
    why = "大模型调用失败" if state.get("llm_error") else "大模型输出未通过数字校验"
    return {"narrative": n, "source": f"模板（{why}）",
            "check": V.check(_text(n), V.allowed_set(state["facts_table"]))}


def node_render(state: State) -> State:
    from cyron_veritas.report.render import render_risk

    out_dir = OUTPUT_DIR / "risk" / state["facts"]["as_of"]
    paths = render_risk(state["pr"], state["limits"], state["narrative"], state["check"], state["source"],
                        state["facts_table"], out_dir)
    return {"outputs": paths}


def route_after_verify(state: State) -> str:
    if state["check"]["ok"]:
        return "render"
    if state.get("llm_error"):
        return "fallback"
    return "narrate" if state["attempts"] <= MAX_REWRITES else "fallback"


def build_graph():
    g = StateGraph(State)
    g.add_node("compute", node_compute)
    g.add_node("narrate", node_narrate)
    g.add_node("verify", node_verify)
    g.add_node("fallback", node_fallback)
    g.add_node("render", node_render)
    g.set_entry_point("compute")
    g.add_edge("compute", "narrate")
    g.add_edge("narrate", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"render": "render", "narrate": "narrate", "fallback": "fallback"})
    g.add_edge("fallback", "render")
    g.add_edge("render", END)
    return g.compile()


def run(holdings_path: str | Path) -> State:
    return build_graph().invoke({"holdings_path": str(holdings_path)})
