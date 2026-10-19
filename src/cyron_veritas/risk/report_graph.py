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

import re

from markupsafe import Markup, escape

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
    fact_map: dict
    names_text: str
    narrative_raw: dict


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


def fact_map(f: dict, lim: list[dict]) -> dict[str, tuple[str, str]]:
    """结构化事实：键 → (含义, 格式化后的值)。大模型写解读时只能用 {{键}} 引用，代码负责填数，
    这样每个数字都绑定在它描述的对象上（不会把 A 的权重写成 B 的）。"""
    m: dict[str, tuple[str, str]] = {
        "total_value": ("组合总市值", V.wan(f["total_value"])), "n_positions": ("持仓数", str(f["n_positions"])),
        "n_accounts": ("账户数", str(len(f["accounts"]))),
        "top1": (f"最大持仓（{f['concentration']['top1_name']}）权重", V.pct(f["concentration"]["top1"])),
        "top5": ("前五大持仓合计", V.pct(f["concentration"]["top5"])),
        "eff_n": ("有效持仓数", V.num(f["concentration"]["effective_n"], 1)),
        "cash": ("现金占比", V.pct(f["cash_weight"])), "vol": ("年化波动率", V.pct(f["vol_ann"])),
        "beta": ("相对沪深300 的 Beta", V.num(f["beta_csi300"])),
        "var95": ("1日 95% VaR（历史法）", V.pct(f["var95_hist"], 2)), "var95_amt": ("1日 95% VaR 金额", V.wan(f["var95_amount"])),
        "var95_param": ("1日 95% VaR（参数法）", V.pct(f["var95_param"], 2)),
        "var99": ("1日 99% VaR（历史法）", V.pct(f["var99_hist"], 2)), "var99_amt": ("1日 99% VaR 金额", V.wan(f["var99_amount"])),
        "cvar95": ("1日 95% CVaR", V.pct(f["cvar95_hist"], 2)), "cvar95_amt": ("1日 95% CVaR 金额", V.wan(f["cvar95_amount"])),
        "mdd": ("最大回撤", V.pct(f["max_drawdown"])), "mdd_period": ("最大回撤区间", f"{f['mdd_peak']} 至 {f['mdd_trough']}"),
        "cur_dd": ("当前回撤", V.pct(f["current_drawdown"])),
        "fx_usd": ("美元兑人民币", V.num(f["fx"]["USD"], 4)), "fx_hkd": ("港元兑人民币", V.num(f["fx"]["HKD"], 4)),
        "bt_days": ("VaR 回测天数", str(f["var_backtest"]["days"])), "bt_exc": ("VaR 被突破次数", str(f["var_backtest"]["exceptions"])),
        "bt_exp": ("VaR 理论突破次数", V.num(f["var_backtest"]["expected"], 1)), "bt_p": ("Kupiec 检验 p 值", V.num(f["var_backtest"]["p_value"])),
    }
    for k, v in f["weights"].items():
        m[f"w.{k}"] = (f"{k} 权重", V.pct(v))
    for k, v in f["pnl_pct"].items():
        m[f"pnl.{k}"] = (f"{k} 浮动盈亏（本币）", V.pct(v))
    for k, v in f["risk_contribution"].items():
        m[f"rc.{k}"] = (f"{k} 风险贡献", V.pct(v))
    for grp, name in (("by_sector", "sector"), ("by_market", "market"), ("by_currency", "ccy")):
        for k, v in f[grp].items():
            m[f"{name}.{k}"] = (f"{k} 敞口", V.pct(v))
    for i, (a, b, c) in enumerate(f["top_corr_pairs"], start=1):
        m[f"corr{i}"] = (f"{a} 与 {b} 的相关系数", V.num(c))
    for k, v in f["stress"].items():
        m[f"stress.{k}"] = (f"压力情景「{k}」组合损益", V.pct(v["pnl_pct"]))
        m[f"stress_amt.{k}"] = (f"压力情景「{k}」损益金额", V.wan(v["pnl_amount"]))
    for r in lim:
        m[f"lim.{r['key']}"] = (f"{r['label']} 实际值（{r['note']}）", V.pct(r["value"]))
        m[f"limv.{r['key']}"] = (f"{r['label']} 限值", ("≥ " if r["direction"] == "min" else "≤ ") + V.pct(r["limit"]))
    return m


def keyed_table(m: dict, lim: list[dict]) -> str:
    lines = ["键 | 含义 | 值"] + [f"{{{{{k}}}}} | {lab} | {val}" for k, (lab, val) in m.items()]
    lines += ["", "[限额状态]"] + [f"- {r['label']}（{r['note']}）：{r['status']}" for r in lim]
    return "\n".join(lines)


PH_RE = re.compile(r"\{\{([^{}]+)\}\}")


def fill(text: str, m: dict, html: bool = False) -> str:
    def rep(mt):
        k = mt.group(1).strip()
        if k not in m:
            return mt.group(0)
        lab, val = m[k]
        return f'<span class="fact" title="{escape(k)}：{escape(lab)}">{escape(val)}</span>' if html else val
    if not html:
        return PH_RE.sub(rep, text)
    out, last = [], 0
    for mt in PH_RE.finditer(text):
        out.append(str(escape(text[last:mt.start()])))
        out.append(rep(mt))
        last = mt.end()
    out.append(str(escape(text[last:])))
    return Markup("".join(out))


def check_placeholders(text: str, m: dict, names_text: str) -> dict:
    keys = [k.strip() for k in PH_RE.findall(text)]
    unknown = sorted({k for k in keys if k not in m})
    raw = V.check(PH_RE.sub("", text), V.allowed_set(names_text))  # 占位符以外不许出现数字（名称里自带的数字除外，如「2020 新冠暴跌」）
    ok = not unknown and not raw["unknown_numbers"] and not raw["advice_words"] and bool(keys)
    return {"ok": ok, "placeholders": len(keys), "unknown_keys": unknown, "raw_numbers": raw["unknown_numbers"],
            "unknown_numbers": raw["unknown_numbers"] + [f"{{{{{k}}}}}" for k in unknown], "advice_words": raw["advice_words"],
            "numbers_total": len(keys) + len(raw["unknown_numbers"]), "numbers_traced": len(keys) - sum(keys.count(k) for k in unknown)}


SYSTEM = """你是基金公司的风险分析师，负责写组合风控日报的文字解读。
规则：
1. 所有数字一律用【事实表】里的占位符 {{键}} 引用，例如「英伟达权重 {{w.英伟达}}，贡献了 {{rc.英伟达}} 的组合波动」；
   不要直接写任何阿拉伯数字（代码会把占位符替换成准确数值），也不要自己计算新数字；
   情景名称、限额名称里自带的数字可以照写（如「2020 新冠暴跌」「1日 95% VaR」）；
2. 不给任何买卖建议（不出现买入、卖出、加仓、减仓、目标价等词），只指出风险和需要复核的事项；
3. 先讲最重要的风险：超限项、集中度、风险贡献最大的持仓、压力测试里最坏的情景；
4. 语言简洁专业，每条一句话。"""


def node_compute(state: State) -> State:
    pr = P.build(P.load_holdings(state["holdings_path"]))
    lim = L.check(pr.facts)
    m = fact_map(pr.facts, lim)
    return {"pr": pr, "facts": pr.facts, "limits": lim, "facts_table": keyed_table(m, lim), "fact_map": m,
            "names_text": " ".join(list(pr.facts["stress"]) + [r["label"] for r in lim] + list(pr.facts["weights"])), "attempts": 0}


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
    res = check_placeholders(_text(state["narrative"]), state["fact_map"], state["names_text"])
    fb = ""
    if not res["ok"]:
        parts = []
        if not res["placeholders"]:
            parts.append("没有使用任何占位符，数字必须写成 {{键}}")
        if res["unknown_keys"]:
            parts.append(f"这些键在事实表里不存在: {', '.join(res['unknown_keys'])}")
        if res["raw_numbers"]:
            parts.append(f"占位符以外出现了数字: {', '.join(res['raw_numbers'])}（改用占位符）")
        if res["advice_words"]:
            parts.append(f"出现了投资建议措辞: {', '.join(res['advice_words'])}")
        fb = "；".join(parts)
    return {"check": res, "feedback": fb, "source": "大模型（结构化引用，数字由代码填入）" if res["ok"] else state.get("source", "")}


def node_fallback(state: State) -> State:
    """模板文字：只用事实表里的数字，保证报告一定能生成。"""
    f, lim = state["facts"], state["limits"]
    bad = [r for r in lim if r["status"] != L.OK]
    worst = min(f["stress"].items(), key=lambda kv: kv[1]["pnl_pct"])
    top_rc = next(iter(f["risk_contribution"].items()))
    n = {"summary": "组合总市值 {{total_value}}，1日 95% VaR {{var95}}，当前回撤 {{cur_dd}}；"
                    + ("限额检查有超限或预警项需要关注。" if bad else "各项限额均正常。"),
         "key_risks": [f"{top_rc[0]} 贡献了组合 {{{{rc.{top_rc[0]}}}}} 的波动。",
                       f"压力测试最坏情景为「{worst[0]}」，组合损益 {{{{stress.{worst[0]}}}}}。"],
         "limit_comments": [f"{r['label']}: 实际 {{{{lim.{r['key']}}}}}（限值 {{{{limv.{r['key']}}}}}），状态{r['status']}。" for r in bad],
         "review_items": ["请复核超限项是否已获批准或有调整计划。"]}
    why = "大模型调用失败" if state.get("llm_error") else "大模型输出未通过数字校验"
    return {"narrative": n, "source": f"模板（{why}）",
            "check": check_placeholders(_text(n), state["fact_map"], state["names_text"])}


def node_render(state: State) -> State:
    from cyron_veritas.report.render import render_risk

    out_dir = OUTPUT_DIR / "risk" / state["facts"]["as_of"]
    m, raw = state["fact_map"], state["narrative"]
    filled = {k: (fill(v, m) if isinstance(v, str) else [fill(x, m) for x in v]) for k, v in raw.items()}
    as_html = {k: (fill(v, m, True) if isinstance(v, str) else [fill(x, m, True) for x in v]) for k, v in raw.items()}
    paths = render_risk(state["pr"], state["limits"], filled, state["check"], state["source"],
                        state["facts_table"], out_dir, narrative_html=as_html)
    return {"outputs": paths, "narrative": filled, "narrative_raw": raw}


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
