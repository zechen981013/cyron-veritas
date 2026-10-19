"""把风控结果渲染成 HTML（带图，单文件）+ Markdown + facts.json（原始数字，便于复核）。"""
from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from cyron_veritas import verify as V
from cyron_veritas.report import charts

TEMPLATES = Path(__file__).parent / "templates"


def _env() -> Environment:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True)
    env.filters.update(pct=V.pct, wan=V.wan, num=V.num)
    return env


def render_risk(pr, limits, narrative, check, source, facts_table, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    f = pr.facts
    imgs = {
        "weights": charts.barh(f["weights"], "持仓权重", limit=0.15),
        "sector": charts.barh(f["by_sector"], "行业敞口", limit=0.35),
        "rc": charts.barh(f["risk_contribution"], "风险贡献（占组合波动）"),
        "nav": charts.nav_drawdown(pr.port_returns),
        "bt": charts.var_backtest(pr.backtest_series),
    }
    pos = pr.positions.reset_index().to_dict("records")
    ctx = dict(f=f, limits=limits, n=narrative, check=check, source=source, pos=pos, img=imgs)
    html = _env().get_template("risk.html.j2").render(**ctx)
    md = _env().get_template("risk.md.j2").render(**ctx)
    (out_dir / "risk_report.html").write_text(html, encoding="utf-8")
    (out_dir / "risk_report.md").write_text(md, encoding="utf-8")
    (out_dir / "facts_table.txt").write_text(facts_table, encoding="utf-8")
    (out_dir / "facts.json").write_text(json.dumps(f, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"html": str(out_dir / "risk_report.html"), "md": str(out_dir / "risk_report.md"),
            "facts": str(out_dir / "facts.json")}
