"""命令行入口：

    uv run desk risk  [--portfolio portfolio/sample_portfolio.csv]   生成组合风控日报
    uv run desk research 000333                                    运行 DCF 研究 Agent
"""
from __future__ import annotations

import argparse
import json
import sys

from analyst_desk.config import PORTFOLIO_DIR


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="desk", description="AI 投研助手：组合风控日报 + DCF 研究 Agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("risk", help="生成组合风控日报")
    r.add_argument("--portfolio", default=str(PORTFOLIO_DIR / "sample_portfolio.csv"))
    s = sub.add_parser("research", help="运行 DCF 研究 Agent")
    s.add_argument("ticker", help="A 股 6 位代码，例如 000333")
    a = ap.parse_args(argv)

    if a.cmd == "risk":
        from analyst_desk.risk import report_graph

        st = report_graph.run(a.portfolio)
        lim = [x for x in st["limits"] if x["status"] != "正常"]
        print(f"日期 {st['facts']['as_of']}  总市值 {st['facts']['total_value'] / 1e4:,.1f} 万元  "
              f"VaR95 {st['facts']['var95_hist']:.2%}  需关注限额 {len(lim)} 项")
        print(f"解读来源：{st['source']}  数字溯源 {st['check'].get('numbers_traced')}/{st['check'].get('numbers_total')}")
        print(json.dumps(st["outputs"], ensure_ascii=False, indent=2))
    elif a.cmd == "research":
        from analyst_desk.research import agent

        res = agent.run(a.ticker)
        st = res["stats"]
        print(f"用时 {st['seconds']}s  证据 {st['evidence_items']} 条  DCF 情景 {st['dcf_runs']}")
        print(f"数字溯源 {res['check']['numbers_traced']}/{res['check']['numbers_total']}  未溯源 {res['check']['unknown_numbers']}")
        print(json.dumps(res["paths"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
