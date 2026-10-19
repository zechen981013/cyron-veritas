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
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage

from cyron_veritas import verify as V
from cyron_veritas.config import OUTPUT_DIR, chat_model
from cyron_veritas.research.tools import Ledger, financials_summary, make_tools
from cyron_veritas.valuation.excel import build_workbook

HIDDEN = {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute"}
MAIN_CALLS, SUB_CALLS, MAX_TASKS = 12, 10, 4


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
    "financial-analyst": (["get_financials", "get_market_data"], {"get_financials": 2, "get_market_data": 2},
        "财务分析师：分析公司近 6 年财务表现和利润质量，返回带证据编号的发现。",
        f"""你是买方研究员里的财务分析师。用工具取数后，分析：
1. 增长：收入增速的趋势和波动；
2. 盈利能力：毛利率、EBIT 利润率、ROE 的变化；
3. 利润质量：经营现金流 / 归母净利润（大于 1 说明利润含金量高）、应收和存货周转天数；
4. 营运资本、资本开支、现金与有息负债；
5. 异常：某一年突然跳变的指标，说明可能的原因（例如科目重分类、一次性项目），并提醒估值时如何处理。
输出 5-8 条发现，每条一句结论 + 关键数字 + 证据编号。{COMMON}"""),
    "valuation-analyst": (["get_financials", "get_market_data", "get_wacc_inputs", "run_dcf"],
        {"get_financials": 2, "get_market_data": 2, "get_wacc_inputs": 2, "run_dcf": 4},
        "估值分析师：基于历史数据定假设，计算 WACC，运行基准 / 乐观 / 悲观三个情景的 DCF，返回结果。",
        f"""你是估值分析师。步骤：
1. 取财务数据和 WACC 参考值；
2. 为基准情景定假设，每个假设都要锚定历史（例如「EBIT 利润率取近 3 年均值 10.1%」）或给出明确理由，写进 notes；
   营运资本比率波动大时用近 3 年均值；永续增长率不超过 3%；
3. 运行 run_dcf 三次：scenario=base、bull、bear（乐观 / 悲观只调整收入增速和利润率，并说明调整理由）；
   如果工具返回假设错误，修改后重试；
4. 返回：三个情景的每股价值和相对现价、终值占 EV 比重、敏感性范围、最关键的两个假设。{COMMON}"""),
    "industry-researcher": (["search_news"], {"search_news": 3},
        "行业研究员：搜索近一年公司和行业新闻，整理催化剂和风险，带链接和日期。",
        """你是行业研究员。最多搜索 3 次（公司近况、行业竞争、政策或宏观风险），返回：
3-5 条近期重要事件（日期 + 一句话 + 链接）、2-3 个潜在催化剂、2-3 个主要风险。只写搜索结果里有的内容，注明证据编号。"""),
}

MAIN_PROMPT = f"""你是基金公司的研究主管，要为一家 A 股公司写一份投资研究备忘录。
工作方式：先用 write_todos 列出计划，再用 task 把工作分派给三个子 Agent（financial-analyst、valuation-analyst、industry-researcher），
最多派 {MAX_TASKS} 次任务；拿到结果后由你汇总成备忘录。你自己也可以调用 get_market_data 核对股价。

备忘录用中文 Markdown，结构固定：
# 公司名（代码）研究备忘录
## 一、结论摘要（3-4 句：DCF 基准情景每股价值与现价对比、估值区间、最关键的判断依据。不给买入 / 卖出建议，只陈述估值与风险）
## 二、财务分析（增长、盈利、利润质量、营运资本与资本开支、异常项）
## 三、估值（假设表：假设 | 取值 | 依据；三个情景结果表；敏感性范围；终值占比的提示）
## 四、催化剂与风险
## 五、需要进一步核实的问题（2-4 条）
## 证据列表（证据编号 → 工具和用途，新闻附链接）

{COMMON}
不出现「买入」「卖出」「加仓」「减仓」「目标价」「推荐」等措辞。"""


def build(ledger: Ledger, model=None):
    from deepagents import create_deep_agent

    model = model or chat_model(0.2)
    tools = make_tools(ledger)
    subagents = [{"name": name, "description": desc, "system_prompt": prompt, "tools": [tools[t] for t in tnames],
                  "middleware": _limits(SUB_CALLS, per)} for name, (tnames, per, desc, prompt) in SUBAGENTS.items()]
    return create_deep_agent(model=model, tools=[tools["get_market_data"]], system_prompt=MAIN_PROMPT, subagents=subagents,
                             middleware=[TodoListMiddleware(), *_limits(MAIN_CALLS, {"task": MAX_TASKS})], name="research-lead")


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


def run(ticker: str, model=None, out_root: Path | None = None) -> dict:
    fin = financials_summary(ticker)
    if fin["is_financial"]:
        raise SystemExit(f"{fin['company']} 是金融企业，DCF（FCFF）不适用；银行和保险应使用 PB-ROE、DDM 或内含价值法（下一版支持）。")
    ledger = Ledger()
    usage = UsageMetadataCallbackHandler()
    t0 = time.time()
    agent = build(ledger, model)
    task = f"请研究 {fin['company']}（{ticker}），按要求写投资研究备忘录。今天是 {dt.date.today()}。"
    out = agent.invoke({"messages": [{"role": "user", "content": task}]}, {"recursion_limit": 150, "callbacks": [usage]})
    msgs = out.get("messages", [])
    memo = next((m.content for m in reversed(msgs) if isinstance(m, AIMessage) and m.content and not m.tool_calls), "")
    if isinstance(memo, list):
        memo = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in memo)

    # 数字溯源：备忘录里的数字能否在证据账本里找到（证据编号、年份、小整数不计入）
    body = re.sub(r"\[E\d+\]|E\d+", "", memo)
    allowed = _allowed_from_ledger(ledger)
    check = V.check(body, allowed)

    out_dir = (out_root or OUTPUT_DIR / "research") / f"{ticker}_{dt.date.today():%Y%m%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    base = next((r for r in ledger.dcf_runs if r["scenario"] == "base"), ledger.dcf_runs[0] if ledger.dcf_runs else None)
    if base:
        from cyron_veritas.research.tools import wacc_inputs
        from cyron_veritas.data.financials import annual
        w = next((i["result"] for i in ledger.items if i["tool"] == "get_wacc_inputs" and "wacc" in i["result"]), None)
        paths["excel"] = str(build_workbook(base["inputs"], fin["company"], base["result"]["price"], history=annual(ticker)["table"],
                                            wacc_detail=w, path=out_dir / f"{ticker}_DCF模型.xlsx"))
    trace = (f"\n\n---\n**数字溯源检查（代码自动生成）**：备忘录共 {check['numbers_total']} 个数字，"
             f"{check['numbers_traced']} 个能在证据账本中找到"
             + (f"；找不到出处的：{', '.join(check['unknown_numbers'])}（需人工复核）" if check["unknown_numbers"] else "，全部可追溯")
             + (f"；出现了投资建议措辞：{', '.join(check['advice_words'])}" if check["advice_words"] else "") + "。")
    (out_dir / "memo.md").write_text(memo + trace, encoding="utf-8")
    (out_dir / "evidence.json").write_text(json.dumps(ledger.items, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    calls = [c["name"] for m in msgs if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
    stats = {"seconds": round(time.time() - t0, 1), "main_tool_calls": calls, "evidence_items": len(ledger.items),
             "dcf_runs": [(r["scenario"], r["result"]["per_share"]) for r in ledger.dcf_runs],
             "tokens": {k: v for k, v in usage.usage_metadata.items()}, "trace": check, "todos": out.get("todos", [])}
    (out_dir / "run_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths.update(memo=str(out_dir / "memo.md"), evidence=str(out_dir / "evidence.json"), stats=str(out_dir / "run_stats.json"))
    return {"memo": memo, "check": check, "paths": paths, "stats": stats}
