"""命令行入口：

    uv run veritas risk  [--portfolio portfolio/sample_portfolio.csv]   生成组合风控日报
    uv run veritas research 000333                                    运行 DCF 研究 Agent
"""
from __future__ import annotations

import argparse
import json
import sys

from cyron_veritas.config import PORTFOLIO_DIR


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="veritas", description="AI 投研助手：组合风控日报 + DCF 研究 Agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("risk", help="生成组合风控日报")
    r.add_argument("--portfolio", default=str(PORTFOLIO_DIR / "sample_portfolio.csv"))
    s = sub.add_parser("research", help="运行 DCF 研究 Agent")
    s.add_argument("ticker", help="A 股 6 位代码，例如 000333（银行、保险、券商自动改用 PB-ROE / DDM）")
    s.add_argument("--review", action="store_true", help="估值假设需要人工审批（在终端里批准 / 修改 / 驳回）")
    sub.add_parser("mcp", help="以 MCP Server（stdio）方式运行，供 Claude Desktop / Cursor 调用")
    fd = sub.add_parser("fund", help="资管中台：模拟产品、估值表入库、每日风控、交易前检查、月报")
    fd.add_argument("action", choices=["simulate", "backfill", "daily", "pretrade", "report", "perf"])
    fd.add_argument("product", nargs="?", default="CV-VG1")
    fd.add_argument("--date")
    fd.add_argument("--month", help="月报月份，如 2026-09")
    fd.add_argument("--orders", help="交易前检查的指令 JSON 文件：[{code, side, qty}]")
    w = sub.add_parser("serve", help="启动 Web 界面")
    w.add_argument("--port", type=int, default=8100)
    w.add_argument("--host", default="127.0.0.1")
    c = sub.add_parser("calls", help="业绩会 / 投资者交流纪要（巨潮投资者关系活动记录表，默认最新一次并与上一次对比）")
    c.add_argument("ticker")
    c.add_argument("--record", help="记录编号（默认最新一次）")
    c.add_argument("--no-compare", action="store_true")
    au = sub.add_parser("transcribe", help="业绩会录音 → Whisper 转写 → 纪要")
    au.add_argument("audio")
    au.add_argument("--company", default="")
    i = sub.add_parser("ingest", help="下载并入库一份年报")
    i.add_argument("ticker")
    i.add_argument("year", type=int)
    a = ap.parse_args(argv)

    if a.cmd == "fund":
        from cyron_veritas.fund import pipeline

        if a.action == "simulate":
            from cyron_veritas.fund import simulate
            print(simulate.run(a.product))
        elif a.action == "backfill":
            print(pipeline.backfill(a.product))
        elif a.action == "daily":
            pipeline.daily(a.product, a.date)
        elif a.action == "pretrade":
            from cyron_veritas.fund import risk as R
            orders = json.loads(open(a.orders, encoding="utf-8").read())
            print(json.dumps(R.pretrade(a.product, a.date or str(R.nav_series(a.product).index[-1].date()), orders), ensure_ascii=False, indent=2))
        elif a.action == "report":
            from cyron_veritas.fund import report
            print(report.build(a.product, a.month))
        elif a.action == "perf":
            from cyron_veritas.fund import perf
            m = perf.nav_metrics(a.product)
            print(json.dumps({k: v for k, v in m.items() if k != "monthly"}, ensure_ascii=False, indent=2, default=str))
        return
    if a.cmd == "mcp":
        from cyron_veritas.mcp_server import main as mcp_main

        mcp_main()
        return
    if a.cmd == "serve":
        from cyron_veritas.web.server import serve

        serve(a.host, a.port)
        return
    if a.cmd == "calls":
        from cyron_veritas.calls import run as R

        res = R.run(a.ticker, a.record, compare=not a.no_compare, say=lambda e: print("  ·", e.get("text", "")))
        print(res["markdown"])
        print("\n已保存：", res["path"])
        return
    if a.cmd == "transcribe":
        from cyron_veritas.calls import run as R

        res = R.from_audio(a.audio, a.company, say=lambda e: print("  ·", e.get("text", "")))
        print(res["markdown"])
        return
    if a.cmd == "ingest":
        from cyron_veritas.rag import index as I

        print(I.ingest(a.ticker, a.year, progress=print))
        return

    if a.cmd == "risk":
        from cyron_veritas.risk import report_graph

        st = report_graph.run(a.portfolio)
        lim = [x for x in st["limits"] if x["status"] != "正常"]
        print(f"日期 {st['facts']['as_of']}  总市值 {st['facts']['total_value'] / 1e4:,.1f} 万元  "
              f"VaR95 {st['facts']['var95_hist']:.2%}  需关注限额 {len(lim)} 项")
        print(f"解读来源：{st['source']}  数字溯源 {st['check'].get('numbers_traced')}/{st['check'].get('numbers_total')}")
        print(json.dumps(st["outputs"], ensure_ascii=False, indent=2))
    elif a.cmd == "research":
        from cyron_veritas.research import agent

        res = agent.start(a.ticker, review=a.review, on_event=lambda e: e.get("type") in ("status", "tool") and
                          print(f"  · {e.get('label') or ''} {e.get('detail') or e.get('text') or ''}"))
        while res["status"] == "review":
            p = res["proposal"]
            print(f"\n== 估值假设待审批（{p['method']}）：Agent 建议基准情景每股 {p['per_share']} 元，现价 {p['price']} 元 ==")
            for k, v in p["assumptions"].items():
                print(f"  {k:16s} = {v}    依据：{p['notes'].get(k, '')}")
            ans = input("回车批准；输入 key=value（多个用空格分隔）修改后重算；输入 reject 驳回：").strip()
            if not ans:
                decision = {"action": "approve"}
            elif ans == "reject":
                decision = {"action": "reject", "comment": input("驳回理由：")}
            else:
                edits = {}
                for kv in ans.split():
                    k, v = kv.split("=", 1)
                    edits[k] = json.loads(v) if v.startswith("[") else float(v)
                decision = {"action": "edit", "assumptions": edits, "comment": input("修改理由：")}
            res = agent.resume(res["run_id"], decision)
        st = res["stats"]
        print(f"用时 {st['seconds']}s  证据 {st['evidence_items']} 条  DCF 情景 {st['dcf_runs']}")
        print(f"数字溯源 {res['check']['numbers_traced']}/{res['check']['numbers_total']}  未溯源 {res['check']['unknown_numbers']}")
        print(json.dumps(res["paths"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
