"""Cyron Veritas MCP Server：把投研与风控能力暴露给 Claude Desktop / Cursor 等 MCP 客户端。

    uv run veritas mcp        # stdio 传输

原则和 Web 一样：工具全部只读，数字由代码计算；大模型客户端负责提问和组织回答。
年报问答、业绩会纪要会调用本项目配置的大模型（.env），其余工具纯计算 / 取数。
"""
from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer(
    name="cyron-veritas",
    title="Cyron Veritas · AI 投研与风控",
    version="0.4",
    instructions=("A 股 / 港股 / 美股的投研与风控工具：组合风险、财报与估值（DCF、PB-ROE、DDM）、年报原文检索与带页码问答、"
                  "业绩会 / 投资者交流纪要。所有数字由代码计算并可追溯；不提供投资建议。"),
)
RO = ToolAnnotations(read_only_hint=True, open_world_hint=True)


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


@server.tool(annotations=RO)
def portfolio_risk(portfolio: str = "sample_portfolio.csv") -> dict:
    """计算组合风险：敞口、集中度、VaR / CVaR、风险贡献、压力测试、VaR 回测、限额检查。
    portfolio 为 portfolio/ 目录下的 CSV 文件名（列：account,ticker,name,market,currency,sector,quantity,cost）。"""
    from cyron_veritas.config import PORTFOLIO_DIR
    from cyron_veritas.risk import limits as L
    from cyron_veritas.risk import portfolio as P

    path = PORTFOLIO_DIR / portfolio
    if path.parent != PORTFOLIO_DIR or not path.exists():
        return {"error": f"找不到持仓文件 {portfolio}"}
    pr = P.build(P.load_holdings(path))
    f = pr.facts
    keep = ["as_of", "total_value", "weights", "by_sector", "by_market", "by_currency", "concentration", "vol_ann", "var95_hist",
            "var99_hist", "cvar95_hist", "var95_amount", "max_drawdown", "current_drawdown", "beta_csi300", "risk_contribution",
            "top_corr_pairs", "stress", "var_backtest"]
    return _jsonable({"facts": {k: f[k] for k in keep}, "limits": L.check(f), "unit": "金额为人民币元，比率为小数"})


@server.tool(annotations=RO)
def company_financials(ticker: str) -> dict:
    """近 6 年关键财务数据与比率（收入、利润率、ROE、现金流、营运资本、资本开支、负债）。
    代码：A 股 6 位（000333）、港股 5 位（00700）、美股字母（AAPL）。"""
    from cyron_veritas.research.tools import financials_summary

    return _jsonable(financials_summary(ticker))


@server.tool(annotations=RO)
def market_snapshot(ticker: str) -> dict:
    """最新股价、52 周高低、近一年涨跌、市值、PE / PB（近似）、Beta（含 R² 和 Blume 调整），自动处理报表与股价币种。"""
    from cyron_veritas.research.tools import market_snapshot as ms

    return _jsonable(ms(ticker))


@server.tool(annotations=RO)
def dcf_valuation(ticker: str, revenue_growth: list[float], ebit_margin: list[float], tax_rate: float, da_pct: float,
                  capex_pct: float, nwc_pct: float, wacc: float, terminal_growth: float, extra_cash: float = 0.0,
                  notes: dict[str, str] | None = None) -> dict:
    """FCFF 两阶段 DCF。增速、利润率为未来 5 年的小数列表；返回每股价值（股价币种）、相对现价、终值占比、敏感性表。
    假设不合理（如永续增长率 ≥ WACC）会返回错误。extra_cash（亿，报表币种）需在 notes['extra_cash'] 写明出处。"""
    from cyron_veritas.research.tools import dcf_run

    res, _ = dcf_run(ticker, "mcp", revenue_growth, ebit_margin, tax_rate, da_pct, capex_pct, nwc_pct, wacc, terminal_growth,
                     notes=notes or {}, extra_cash=extra_cash)
    return _jsonable(res)


@server.tool(annotations=RO)
def financial_institution_valuation(ticker: str, roe: float, ke: float, g: float, g1: float, payout1: float) -> dict:
    """A 股银行 / 保险 / 券商估值：PB-ROE（合理 P/B =（ROE−g）/（Ke−g））和两阶段 DDM；返回每股价值、市场隐含股权成本、敏感性。"""
    from cyron_veritas.research.tools import fin_run

    res, _ = fin_run(ticker, "mcp", roe, ke, g, g1, payout1)
    return _jsonable(res)


@server.tool(annotations=RO)
def annual_report_search(ticker: str, query: str) -> dict:
    """在已入库的最新 A 股年报原文中检索（bge-m3 + BM25 混合检索 + 重排），返回原文片段、章节和页码。"""
    from cyron_veritas.research.tools import annual_report_search as s

    return _jsonable(s(ticker, query))


@server.tool(annotations=RO)
def annual_report_qa(ticker: str, year: int, question: str) -> dict:
    """基于年报原文回答问题，每句标注 [p.页码]；返回回答、引用页、引用与数字校验结果和证据片段。年报需先入库（ingest_annual_report）。"""
    from cyron_veritas.rag import qa

    r = qa.ask(question, ticker, year)
    return _jsonable({"answer": r["answer"], "check": r["check"],
                      "evidence": [{k: e[k] for k in ("page_start", "page_end", "section", "text")} for e in r["evidence"]]})


@server.tool(annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=True, open_world_hint=True))
def ingest_annual_report(ticker: str, year: int) -> dict:
    """从巨潮资讯网下载 A 股年报并入库（切块、向量化；首次约 1 分钟，已入库会跳过）。"""
    from cyron_veritas.rag import index as I

    m = I.ingest(ticker, year)
    return _jsonable({k: m[k] for k in ("company", "year", "title", "n_chunks", "page_url")})


@server.tool(annotations=RO)
def investor_relations_records(ticker: str) -> list[dict]:
    """列出 A 股公司在巨潮资讯网发布的投资者关系活动记录（业绩说明会、路演、调研）。"""
    from cyron_veritas.calls import run as R

    return _jsonable([{k: r[k] for k in ("record_id", "title", "date", "url")} for r in R.records(ticker)[:30]])


@server.tool(annotations=RO)
def earnings_call_summary(ticker: str, record_id: str | None = None, compare: bool = True) -> dict:
    """生成业绩会 / 投资者交流纪要：指引、经营数据、投资者关注点、回避的问题、语气；可与上一次对比。每条带原文引用并校验。"""
    from cyron_veritas.calls import run as R

    res = R.run(ticker, record_id, compare)
    return _jsonable({"markdown": res["markdown"], "check": res["result"]["check"],
                      "compare_check": (res["compare"] or {}).get("check")})


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
