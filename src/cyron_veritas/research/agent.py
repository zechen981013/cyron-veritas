"""DCF 研究 Agent（DeepAgents）：模拟研究员写一份公司研究备忘录的工作流。

    研究主管（主 Agent）── write_todos 列计划 ──→ task 派活
        ├── 财务分析师（子 Agent）：近 6 年财报、利润质量、营运资本、异常科目
        ├── 估值分析师（子 Agent）：以历史数据为锚定假设 → WACC → DCF（基准 / 乐观 / 悲观）→ 敏感性
        └── 行业研究员（子 Agent）：近一年新闻、催化剂与风险（带链接）
    主 Agent 汇总成投资备忘录 → 代码做数字溯源 → 导出带公式的 Excel 模型

为什么这里用 DeepAgents：研究是开放式任务（查什么、查几轮、发现异常要不要追问），适合「自己列计划 + 派子 Agent」；
而风控日报步骤固定，用 LangGraph 固定流程。两种编排方式按任务性质选。
护栏：工具全部只读、数字全部由工具计算；模型调用、工具调用、派活次数都有上限；虚拟文件系统只在内存里。
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, TodoListMiddleware, ToolCallLimitMiddleware
from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from cyron_veritas import verify as V
from cyron_veritas.config import OUTPUT_DIR, chat_model
from cyron_veritas.research.tools import Ledger, financials_summary, make_tools
from cyron_veritas.valuation.excel import build_workbook

HIDDEN = {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute"}
MAIN_CALLS, SUB_CALLS, MAX_TASKS = 14, 10, 5


class HideFilesystemTools(AgentMiddleware):
    """不给模型看虚拟文件系统和 execute 工具，省掉无意义的调用。"""

    def wrap_model_call(self, request, handler):
        keep = [t for t in request.tools if (t.get("name") if isinstance(t, dict) else getattr(t, "name", None)) not in HIDDEN]
        return handler(request.override(tools=keep))


def _limits(model_calls: int, per_tool: dict[str, int]) -> list:
    return [HideFilesystemTools(), ModelCallLimitMiddleware(run_limit=model_calls, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=16, exit_behavior="end"),
            *(ToolCallLimitMiddleware(tool_name=t, run_limit=n, exit_behavior="continue") for t, n in per_tool.items())]


COMMON = """数字规则：所有数字必须直接来自工具返回结果，原样引用并注明证据编号（如 [E3]）；不要自己心算或推算新数字。
比率在工具里是小数（0.102），写报告时写成百分比（10.2%）。金额单位是亿元。"""

SUBAGENTS = {
    "financial-analyst": (["get_financials", "get_market_data", "search_annual_report"],
        {"get_financials": 2, "get_market_data": 2, "search_annual_report": 3},
        "财务分析师：分析公司近 6 年财务表现和利润质量，返回带证据编号的发现。",
        f"""你是买方研究员里的财务分析师。用工具取数后，分析：
1. 增长：收入增速的趋势和波动；
2. 盈利能力：毛利率、EBIT 利润率、ROE 的变化；
3. 利润质量：经营现金流 / 归母净利润（大于 1 说明利润含金量高）、应收和存货周转天数；
4. 营运资本、资本开支、现金与有息负债；
5. 异常：某一年突然跳变的指标，用 search_annual_report 到年报附注里查原因（例如「其他流动资产」「定期存款」「应收票据」），
   引用年报页码说明；查不到就写「年报中未找到原因」，并提醒估值时如何处理。
输出 5-8 条发现，每条一句结论 + 关键数字 + 证据编号。{COMMON}"""),
    "valuation-analyst": (["get_financials", "get_market_data", "get_wacc_inputs", "run_dcf", "search_annual_report"],
        {"get_financials": 2, "get_market_data": 2, "get_wacc_inputs": 2, "run_dcf": 4, "search_annual_report": 2},
        "估值分析师：基于历史数据定假设，计算 WACC，运行基准 / 乐观 / 悲观三个情景的 DCF，返回结果。",
        f"""你是估值分析师。步骤：
1. 取财务数据和 WACC 参考值；
2. 为基准情景定假设，每个假设都要锚定历史（例如「EBIT 利润率取近 3 年均值 10.1%」）或给出明确理由，写进 notes；
   营运资本比率波动大时用近 3 年均值；永续增长率不超过 3%；
   财务数据里 cash_like_other（其他类现金资产候选）较大时，用 search_annual_report 查「一年内到期的非流动资产」「其他非流动资产」附注，
   能确认是定期存款 / 理财的部分填进 extra_cash（亿元，年报多为千元要换算），notes['extra_cash'] 写明科目、金额和页码；不能确认就填 0；
3. 运行 run_dcf 三次：scenario=base、bull、bear（乐观 / 悲观只调整收入增速和利润率，并说明调整理由）；
   如果工具返回假设错误，修改后重试；
4. 返回：三个情景的每股价值和相对现价、终值占 EV 比重、敏感性范围、最关键的两个假设。{COMMON}"""),
    "filing-analyst": (["search_annual_report"], {"search_annual_report": 5},
        "年报研究员：从最新年报原文中提取业务结构、海外收入、战略、风险披露、股东回报等要点，每条附页码。",
        f"""你是年报研究员，只根据年报原文工作。用 search_annual_report 检索（最多 5 次），整理：
1. 业务结构：分业务板块 / 分地区收入及占比；
2. 管理层对行业和公司的判断、未来发展规划；
3. 公司披露的主要风险及应对措施；
4. 股东回报：分红、回购；
5. 其他重要事项（重大诉讼、担保、关联交易等，如有）。
每条一句话 + 关键数字 + 年报页码（如「年报 p.49」）+ 证据编号。年报金额多为千元，原样引用并注明单位。{COMMON}"""),
    "industry-researcher": (["search_news"], {"search_news": 3},
        "行业研究员：搜索近一年公司和行业新闻，整理催化剂和风险，带链接和日期。",
        """你是行业研究员。最多搜索 3 次（公司近况、行业竞争、政策或宏观风险），返回：
3-5 条近期重要事件（日期 + 一句话 + 链接）、2-3 个潜在催化剂、2-3 个主要风险。只写搜索结果里有的内容，注明证据编号。"""),
}

# 金融企业（银行、保险、券商）换一套财务分析和估值子 Agent；年报研究员、行业研究员共用
FIN_SUBAGENTS = {
    "financial-analyst": (["get_fin_metrics", "get_market_data", "search_annual_report"],
        {"get_fin_metrics": 2, "get_market_data": 2, "search_annual_report": 4},
        "财务分析师（金融企业）：分析盈利能力、收入结构、资产质量、资本与分红，返回带证据编号的发现。",
        f"""你是金融行业研究员。用 get_fin_metrics 取数后，分析：
1. 盈利能力：ROE、ROA 的趋势，净利润增速；
2. 收入结构：净利息收入占比、手续费收入的变化；
3. 资产质量与风险：信用成本（信用减值 / 贷款）、贷存比、权益乘数；
   不良贷款率、拨备覆盖率、资本充足率（核心一级）这些指标不在工具数据里，用 search_annual_report 到年报里查（注明页码）；
4. 资本与股东回报：BVPS 增长、每股股利、派息率；
5. 异常：明显跳变的指标及年报中的解释。
输出 5-8 条发现，每条一句结论 + 关键数字 + 证据编号 / 年报页码。{COMMON}"""),
    "valuation-analyst": (["get_fin_metrics", "get_market_data", "get_wacc_inputs", "run_fin_valuation"],
        {"get_fin_metrics": 2, "get_market_data": 2, "get_wacc_inputs": 2, "run_fin_valuation": 4},
        "估值分析师（金融企业）：用 PB-ROE 和两阶段 DDM 估值，运行基准 / 乐观 / 悲观三个情景，返回结果。",
        f"""你是金融企业估值分析师。金融企业不用 DCF，用 PB-ROE 和 DDM。步骤：
1. 取指标和股权成本参考（get_wacc_inputs 的 cost_of_equity；银行的 WACC 和债务权重没有意义，忽略）；
2. 基准情景假设：长期 ROE 锚定近 3 年均值并考虑趋势（ROE 持续下行时要往下调并说明）；Ke 用 CAPM 结果；
   永续增长率 g 不超过 3% 且低于 ROE；第一阶段 EPS 增速锚定近 3 年利润增速；派息率锚定近 3 年均值；依据写进 notes；
3. 运行 run_fin_valuation 三次：base、bull、bear（乐观 / 悲观主要调整 ROE 和 g1，说明理由）；假设报错就修改后重试；
   结果里的 market_implied_ke 是按现价倒推的股权成本：和模型 Ke 相差 3 个百分点以上时，必须说明分歧来源
   （市场可能在定价资产质量、地产敞口、ROE 继续下行等风险），并考虑在 bear 情景里用接近市场隐含值的 Ke；
4. 返回：三个情景的 PB-ROE 与 DDM 每股价值、平均值与相对现价、当前 P/B 与合理 P/B、市场隐含 Ke、敏感性范围、最关键的两个假设。{COMMON}"""),
}

MAIN_PROMPT = f"""你是基金公司的研究主管，要为一家上市公司写一份投资研究备忘录。
工作方式：先用 write_todos 列出计划，再用 task 把工作分派给子 Agent（financial-analyst、valuation-analyst、industry-researcher），
以及 filing-analyst（年报原文），最多派 {MAX_TASKS} 次任务；拿到结果后由你汇总成备忘录。你自己也可以调用 get_market_data 核对股价。

备忘录用中文 Markdown，结构固定：
# 公司名（代码）研究备忘录
## 一、结论摘要（3-4 句：DCF 基准情景每股价值与现价对比、估值区间、最关键的判断依据。不给买入 / 卖出建议，只陈述估值与风险）
## 二、财务分析（增长、盈利、利润质量、营运资本与资本开支、异常项及年报中的解释）
## 三、年报要点（业务结构、管理层判断、风险披露、股东回报；每条注明「年报 p.X」）
## 四、估值（假设表：假设 | 取值 | 依据；三个情景结果表；敏感性范围；终值占比的提示）
## 五、催化剂与风险
## 六、需要进一步核实的问题（2-4 条）
## 证据列表（证据编号 → 工具和用途，新闻附链接）

{COMMON}
不出现「买入」「卖出」「加仓」「减仓」「目标价」「推荐」等措辞。"""


FIN_NOTE = """
本公司是金融企业：估值一节改为「四、估值（PB-ROE 与 DDM：假设表；三个情景的 PB-ROE、DDM、平均每股价值和相对现价；当前与合理 P/B；敏感性范围）」，
财务分析一节写盈利能力、收入结构、资产质量（不良率、拨备覆盖率、资本充足率注明年报页码）、资本与分红；不要写自由现金流和营运资本。"""


NON_A_NOTE = """
本公司不是 A 股：暂无年报原文检索（没有 filing-analyst 和 search_annual_report），「三、年报要点」一节写「港股 / 美股年报检索暂不支持，以下要点来自财务数据和新闻」并据此简要概括；
金额单位以工具返回的币种为准（港股报表可能是人民币，股价是港元；每股价值已换算成股价币种）。"""


def build(ledger: Ledger, model=None, financial: bool = False, market: str = "A"):
    from deepagents import create_deep_agent

    model = model or chat_model(0.2)
    tools = make_tools(ledger)
    spec = {**SUBAGENTS, **FIN_SUBAGENTS} if financial else dict(SUBAGENTS)
    if market != "A":  # 港美股没有年报检索
        spec.pop("filing-analyst", None)
        spec = {k: ([t for t in tn if t != "search_annual_report"], {a: b for a, b in per.items() if a != "search_annual_report"}, d, pr)
                for k, (tn, per, d, pr) in spec.items()}
    subagents = [{"name": name, "description": desc, "system_prompt": prompt, "tools": [tools[t] for t in tnames],
                  "middleware": _limits(SUB_CALLS, per)} for name, (tnames, per, desc, prompt) in spec.items()]
    prompt = (MAIN_PROMPT.replace("DCF 基准情景每股价值", "基准情景每股价值") + (FIN_NOTE if financial else "")
              + (NON_A_NOTE if market != "A" else ""))
    return create_deep_agent(model=model, tools=[tools["get_market_data"]], system_prompt=prompt, subagents=subagents,
                             middleware=[TodoListMiddleware(), *_limits(MAIN_CALLS, {"task": MAX_TASKS})], name="research-lead")


class ProgressHandler(BaseCallbackHandler):
    """把工具调用实时转成事件（Web 界面显示 Agent 正在做什么）。子 Agent 的调用也会经过这里。"""

    LABEL = {"write_todos": "列计划", "task": "派活给子 Agent", "get_financials": "取财报数据", "get_market_data": "取行情与 Beta",
             "get_wacc_inputs": "计算 WACC", "run_dcf": "运行 DCF", "search_news": "搜索新闻", "search_annual_report": "检索年报原文",
             "get_fin_metrics": "取金融企业指标", "run_fin_valuation": "运行 PB-ROE / DDM"}

    def __init__(self, say):
        self.say = say

    def on_tool_start(self, serialized, input_str, **kw):
        name = (serialized or {}).get("name") or kw.get("name") or "tool"
        if name in HIDDEN:
            return
        detail = ""
        args = kw.get("inputs")  # LangChain 传的结构化参数；没有时再尝试解析字符串
        if not isinstance(args, dict):
            try:
                args = json.loads(input_str) if isinstance(input_str, str) and input_str.startswith("{") else {}
            except Exception:
                args = {}
        if isinstance(args, dict):
            if name == "task":
                detail = f"{args.get('subagent_type', '')}：{str(args.get('description', ''))[:80]}"
            elif name == "write_todos":
                detail = "；".join(t.get("content", "") for t in args.get("todos", []) if isinstance(t, dict))[:200]
            elif name == "run_dcf":
                detail = f"情景 {args.get('scenario')}，WACC {args.get('wacc')}，g {args.get('terminal_growth')}"
            elif name == "run_fin_valuation":
                detail = f"情景 {args.get('scenario')}，ROE {args.get('roe')}，Ke {args.get('ke')}，g {args.get('g')}"
            else:
                detail = str(args.get("query") or args.get("ticker") or "")[:80]
        self.say({"type": "tool", "name": name, "label": self.LABEL.get(name, name), "detail": detail})


def _allowed_from_ledger(ledger: Ledger) -> set[str]:
    """账本里每个数字的常见写法：原值、百分比（×100）、不同小数位。"""
    vals: list[float] = []

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            vals.append(float(o))
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v)
                m = re.search(r"[-\d.]+$", str(k))
                if m:
                    try:
                        vals.append(float(m.group()))
                    except ValueError:
                        pass
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            vals.extend(float(t) for t in V.numbers(o))

    for item in ledger.items:
        walk(item["args"])
        walk(item["result"])
    out = set()
    for v in vals:
        for x in (v, abs(v), abs(v) * 100):
            for nd in range(0, 5):
                out.add(repr(round(round(x, nd), 6)))
    return out | {repr(float(i)) for i in range(0, 11)}


# ======================================================================
# 外层流程（LangGraph）：研究 Agent → 估值假设人工审批（interrupt）→ 定稿
#
#     research ──→ review ──→ finalize
#                    │ interrupt：把 Agent 建议的基准假设和结果交给分析师
#                    │ 分析师：批准 / 修改后重算 / 驳回（Command(resume=...) 继续）
#
# 不在 DeepAgents 内部打断：子 Agent 是在工具调用里嵌套运行的，在那里暂停和恢复不稳定；
# 审批放在外层图的节点里，Agent 跑完后统一审批，状态清楚、可恢复。
# 账本、备忘录这类大对象放在进程内 RUNS 里，图的状态只存可序列化的小字段。
# ======================================================================
RUNS: dict[str, dict] = {}
DCF_KEYS = ["revenue_growth", "ebit_margin", "tax_rate", "da_pct", "capex_pct", "nwc_pct", "wacc", "terminal_growth", "extra_cash"]
FIN_KEYS = ["roe", "ke", "g", "g1", "payout1"]


class FlowState(TypedDict, total=False):
    run_id: str
    ticker: str
    review: bool
    decision: dict
    result: dict


def _base_run(ledger: Ledger) -> dict | None:
    return next((r for r in ledger.dcf_runs if r["scenario"] == "base"), ledger.dcf_runs[0] if ledger.dcf_runs else None)


def _proposal(ctx: dict) -> dict:
    base = _base_run(ctx["ledger"])
    if not base:
        return {"available": False}
    x = base["inputs"]
    keys = FIN_KEYS if ctx["financial"] else DCF_KEYS
    return {"available": True, "method": "PB-ROE / DDM" if ctx["financial"] else "DCF（FCFF）", "company": ctx["fin"]["company"],
            "ticker": ctx["ticker"], "assumptions": {k: getattr(x, k) for k in keys}, "notes": x.notes,
            "per_share": base["per_share"], "price": base["result"]["price"],
            "market_check": {k: base["result"][k] for k in ("current_pb", "justified_pb", "market_implied_ke") if k in base["result"]},
            "scenarios": [(r["scenario"], r["per_share"]) for r in ctx["ledger"].dcf_runs]}


def node_research(state: FlowState) -> FlowState:
    ctx = RUNS[state["run_id"]]
    say, ticker = ctx["say"], state["ticker"]
    from cyron_veritas.rag import index as I

    fin = ctx["fin"]
    year = fin["latest_year"]
    if ctx["market"] == "A" and f"{ticker}_{year}" not in I.manifest():  # A 股先把最新年报入库（第一次约 1 分钟）
        try:
            I.ingest(ticker, year, progress=lambda m: say({"type": "status", "text": f"年报入库：{m}"}))
        except Exception as e:
            say({"type": "status", "text": f"年报入库失败（{e}），本次不使用年报检索"})
    agent = build(ctx["ledger"], ctx.get("model"), financial=ctx["financial"], market=ctx["market"])
    kind = "金融企业，用 PB-ROE 和 DDM 估值" if ctx["financial"] else "非金融企业，用 DCF 估值"
    mname = {"A": "A 股", "HK": "港股", "US": "美股"}[ctx["market"]]
    task = f"请研究 {fin['company']}（{mname} {ticker}，{kind}），按要求写投资研究备忘录。今天是 {dt.date.today()}。"
    say({"type": "status", "text": "研究主管开始列计划、派子 Agent"})
    out = agent.invoke({"messages": [{"role": "user", "content": task}]},
                       {"recursion_limit": 150, "callbacks": [ctx["usage"], ProgressHandler(say)]})
    msgs = out.get("messages", [])
    memo = next((m.content for m in reversed(msgs) if isinstance(m, AIMessage) and m.content and not m.tool_calls), "")
    if isinstance(memo, list):
        memo = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in memo)
    ctx.update(memo=memo, todos=out.get("todos", []),
               calls=[c["name"] for m in msgs if isinstance(m, AIMessage) for c in (m.tool_calls or [])])
    return {}


def node_review(state: FlowState) -> FlowState:
    if not state.get("review"):
        return {"decision": {"action": "auto"}}
    ctx = RUNS[state["run_id"]]
    prop = _proposal(ctx)
    if not prop["available"]:
        return {"decision": {"action": "auto"}}
    if not ctx.get("review_sent"):  # 恢复时 LangGraph 会重新执行本节点，事件只发一次
        ctx["say"]({"type": "review", "proposal": prop})
        ctx["review_sent"] = True
    decision = interrupt(prop)  # 暂停：等分析师批准 / 修改 / 驳回
    return {"decision": decision if isinstance(decision, dict) else {"action": "approve"}}


def _approval_section(ctx: dict, decision: dict, prop: dict, new: dict | None) -> str:
    act = decision.get("action")
    who = decision.get("reviewer") or "分析师"
    if act == "auto":
        return "\n\n## 七、估值假设审批\n本次运行未开启人工审批，估值结果为 Agent 建议值，正式使用前需分析师复核。\n"
    head = {"approve": "已批准", "edit": "已修改后批准", "reject": "已驳回"}[act]
    lines = [f"\n\n## 七、估值假设审批（{head}，{who}，{dt.datetime.now():%Y-%m-%d %H:%M}）", "",
             "| 假设 | Agent 建议 | 审批后 |", "|---|---|---|"]
    final = decision.get("assumptions") or {}
    for k, v in prop["assumptions"].items():
        lines.append(f"| {k} | {v} | {final.get(k, v) if act != 'reject' else '—'} |")
    if decision.get("comment"):
        lines += ["", f"审批意见：{decision['comment']}"]
    if act == "edit" and new:
        if new.get("ok"):
            ps = new.get("per_share") or new.get("average_per_share")
            lines += ["", f"按审批后的假设重新计算：基准情景每股价值 **{ps} 元**（Agent 建议值 {prop['per_share']} 元），"
                          f"相对现价 {new['vs_price']:.1%}。[{new['evidence_id']}]"]
        else:
            lines += ["", f"审批后的假设未通过校验：{'；'.join(new.get('errors', []))}，估值沿用 Agent 建议值。"]
    if act == "reject":
        lines += ["", "**估值假设被驳回：本备忘录的估值结论不可使用，Excel 模型未生成。**"]
    return "\n".join(lines) + "\n"


def node_finalize(state: FlowState) -> FlowState:
    ctx = RUNS[state["run_id"]]
    ticker, fin, ledger, decision = state["ticker"], ctx["fin"], ctx["ledger"], state.get("decision") or {"action": "auto"}
    prop = _proposal(ctx)
    new, final_inputs = None, (_base_run(ledger) or {}).get("inputs")
    if decision.get("action") == "edit" and prop.get("available"):
        tools = make_tools(ledger)
        a = {**prop["assumptions"], **(decision.get("assumptions") or {})}
        notes = {**prop["notes"], "审批": f"{decision.get('reviewer') or '分析师'}修改：{decision.get('comment', '')}"}
        tool = tools["run_fin_valuation"] if ctx["financial"] else tools["run_dcf"]
        new = json.loads(tool.invoke({"ticker": ticker, "scenario": "approved", **a, "notes": notes}))
        if new.get("ok"):
            final_inputs = ledger.dcf_runs[-1]["inputs"]
    memo = ctx["memo"] + _approval_section(ctx, decision, prop, new) if prop.get("available") else ctx["memo"]

    # 数字溯源：备忘录里的数字能否在证据账本里找到（证据编号、年份、小整数不计入）
    body = re.sub(r"\[E\d+\]|E\d+", "", memo)
    check = V.check(body, _allowed_from_ledger(ledger))
    out_dir = (ctx.get("out_root") or OUTPUT_DIR / "research") / f"{ticker}_{dt.date.today():%Y%m%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    if final_inputs is not None and decision.get("action") != "reject":
        from cyron_veritas.data.financials import annual, financial_institution

        price = _base_run(ledger)["result"]["price"]
        if ctx["financial"]:
            from cyron_veritas.valuation.excel import build_fin_workbook
            paths["excel"] = str(build_fin_workbook(final_inputs, fin["company"], price, history=financial_institution(ticker)["table"],
                                                    path=out_dir / f"{ticker}_估值模型.xlsx"))
        else:
            w = next((i["result"] for i in ledger.items if i["tool"] == "get_wacc_inputs" and "wacc" in i["result"]), None)
            paths["excel"] = str(build_workbook(final_inputs, fin["company"], price, history=annual(ticker)["table"],
                                                wacc_detail=w, path=out_dir / f"{ticker}_DCF模型.xlsx"))
    trace = (f"\n\n---\n**数字溯源检查（代码自动生成）**：备忘录共 {check['numbers_total']} 个数字，"
             f"{check['numbers_traced']} 个能在证据账本中找到"
             + (f"；找不到出处的：{', '.join(check['unknown_numbers'])}（需人工复核）" if check["unknown_numbers"] else "，全部可追溯")
             + (f"；出现了投资建议措辞：{', '.join(check['advice_words'])}" if check["advice_words"] else "") + "。")
    (out_dir / "memo.md").write_text(memo + trace, encoding="utf-8")
    (out_dir / "evidence.json").write_text(json.dumps(ledger.items, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    stats = {"seconds": round(time.time() - ctx["t0"], 1), "financial": ctx["financial"], "main_tool_calls": ctx.get("calls", []),
             "evidence_items": len(ledger.items), "dcf_runs": [(r["scenario"], r["per_share"]) for r in ledger.dcf_runs],
             "review": decision, "tokens": {k: v for k, v in ctx["usage"].usage_metadata.items()}, "trace": check,
             "todos": ctx.get("todos", [])}
    (out_dir / "run_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths.update(memo=str(out_dir / "memo.md"), evidence=str(out_dir / "evidence.json"), stats=str(out_dir / "run_stats.json"))
    return {"result": {"memo": memo, "check": check, "paths": paths, "stats": stats}}


_FLOW = None


def flow():
    global _FLOW
    if _FLOW is None:
        from langgraph.checkpoint.memory import MemorySaver

        g = StateGraph(FlowState)
        g.add_node("research", node_research)
        g.add_node("review", node_review)
        g.add_node("finalize", node_finalize)
        g.add_edge(START, "research")
        g.add_edge("research", "review")
        g.add_edge("review", "finalize")
        g.add_edge("finalize", END)
        _FLOW = g.compile(checkpointer=MemorySaver())
    return _FLOW


def start(ticker: str, review: bool = False, model=None, out_root: Path | None = None, on_event=None, run_id: str | None = None) -> dict:
    """开始一次研究。review=True 时会在估值审批处暂停，返回 {"status": "review", "proposal": ...}；否则直接返回结果。"""
    from cyron_veritas.data.financials import market_of

    mkt, ticker = market_of(ticker)
    fin = financials_summary(ticker)
    if fin["is_financial"] and mkt != "A":
        raise SystemExit(f"{fin['company']} 是港股 / 美股金融企业：PB-ROE / DDM 目前只支持 A 股金融企业的数据口径。")
    run_id = run_id or f"{ticker}-{time.time_ns()}"
    RUNS[run_id] = {"ticker": ticker, "market": mkt, "fin": fin, "financial": bool(fin["is_financial"]), "ledger": Ledger(), "model": model,
                    "out_root": out_root, "say": on_event or (lambda e: None), "usage": UsageMetadataCallbackHandler(), "t0": time.time()}
    out = flow().invoke({"run_id": run_id, "ticker": ticker, "review": review}, {"configurable": {"thread_id": run_id}})
    return _wrap(run_id, out)


def resume(run_id: str, decision: dict) -> dict:
    """分析师的审批决定：{"action": "approve" | "edit" | "reject", "assumptions": {...}, "comment": "...", "reviewer": "..."}"""
    out = flow().invoke(Command(resume=decision), {"configurable": {"thread_id": run_id}})
    return _wrap(run_id, out)


def _wrap(run_id: str, out: dict) -> dict:
    if "__interrupt__" in out:
        return {"status": "review", "run_id": run_id, "proposal": out["__interrupt__"][0].value}
    RUNS.pop(run_id, None)
    return {"status": "done", "run_id": run_id, **out["result"]}


def run(ticker: str, model=None, out_root: Path | None = None, on_event=None) -> dict:
    """不经审批直接跑完（命令行和评测用）。"""
    return start(ticker, review=False, model=model, out_root=out_root, on_event=on_event)
