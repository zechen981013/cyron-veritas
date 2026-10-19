import asyncio

from cyron_veritas.mcp_server import server


def test_mcp_tools_registered_and_read_only():
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"portfolio_risk", "company_financials", "dcf_valuation", "annual_report_qa", "earnings_call_summary"} <= names
    ro = {t.name for t in tools if t.annotations and t.annotations.read_only_hint}
    assert names - ro == {"ingest_annual_report"}                     # 只有入库会写本地数据
