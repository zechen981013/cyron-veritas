"""LP 季度报告（HTML + PDF）：基金概况、该 LP 的资本账户、投资组合与估值、本季度出资分配与退出、GP 观点。

和投资者月报一样：所有数字由代码填进占位符，大模型只写「GP 观点」，校验占位符、占位符外没有数字、方向词与数字一致。
"""
from __future__ import annotations

import json
import subprocess
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel, Field

from cyron_veritas.config import OUTPUT_DIR, chat_model
from cyron_veritas.fund import db
from cyron_veritas.fund import product as P
from cyron_veritas.fund.report import CHROME, _pct, _png, check_direction
from cyron_veritas.risk.report_graph import check_placeholders, fill


class GPLetter(BaseModel):
    portfolio: str = Field(description="本季度投资组合进展，2-3 句：上市、解禁、减持、估值变化，只用占位符引用数字")
    performance: str = Field(description="基金业绩，2 句：TVPI、DPI、净 IRR 与 PME 的含义")
    outlook: str = Field(description="下季度工作重点，2 句；不承诺收益")


def _m(x):
    return x if isinstance(x, dict) else json.loads(x)


def _double_negative(text: str, facts: dict) -> list[str]:
    """「减少 {{x}}」而 x 本身是负数 → 双重否定（读者会理解成增加）。"""
    import re

    out = []
    for m in re.finditer(r"(减少|下降|下调|回落|亏损)了?\s*\{\{(\w+)\}\}", text):
        val = facts.get(m.group(2), ("", ""))[1]
        if val.startswith(("-", "−")):
            out.append(f"「{m.group(0)}」：{m.group(2)} 本身是负数 {val}，写成「变化为」或直接用数值")
    return out


def data(code: str, quarter: str, lp: str) -> dict:
    q_end = pd.Period(quarter, "Q").end_time.normalize()
    qs = db.df("select date, metrics from pe_quarters where product = :p order by date", p=code)
    qs["date"] = pd.to_datetime(qs["date"])
    qs = qs[qs["date"] <= q_end]
    if qs.empty:
        raise LookupError(f"{quarter} 没有季度数据")
    row = qs.iloc[-1]
    m = _m(row["metrics"])
    prev = _m(qs.iloc[-2]["metrics"]) if len(qs) > 1 else None
    d, d0 = row["date"], (qs.iloc[-2]["date"] if len(qs) > 1 else pd.Timestamp("1900-01-01"))
    marks = db.df("select * from pe_marks where product = :p and date = :d", p=code, d=d.date())
    names = db.df("select deal, company from pe_pipeline where product = :p", p=code).set_index("deal")["company"]
    marks["company"] = marks["deal"].map(names)
    cap = db.df("select * from pe_capital where product = :p", p=code)
    cap["date"] = pd.to_datetime(cap["date"])
    mine = cap[(cap["lp"] == lp) & (cap["date"] <= d)]
    tx = db.df("select * from pe_tx where product = :p", p=code)
    tx["date"] = pd.to_datetime(tx["date"])
    q_tx = tx[(tx["date"] > d0) & (tx["date"] <= d)]
    cfg = P.load(code)
    lpc = next(x for x in cfg["lps"] if x["id"] == lp)
    lp_commit = sum(x["commitment"] for x in cfg["lps"] if x["type"] != "GP")
    share = lpc["commitment"] / lp_commit if lpc["type"] != "GP" else None
    return {"cfg": cfg, "m": m, "prev": prev, "date": d, "marks": marks, "mine": mine, "q_tx": q_tx, "lp": lpc, "share": share}


def build(code: str, quarter: str, lp: str = "LP02", with_llm: bool = True) -> dict:
    x = data(code, quarter, lp)
    cfg, m, d, marks, mine, q_tx, lpc, share = x["cfg"], x["m"], x["date"], x["marks"], x["mine"], x["q_tx"], x["lp"], x["share"]
    called = mine[mine["type"] == "缴款"]["amount"].sum()
    dist = mine[mine["type"] == "分配"]["amount"].sum()
    my_nav = m["lp_nav"] * share if share else 0.0
    exits = q_tx[q_tx["type"].isin(["减持", "回购退出"])].groupby("deal")["amount"].sum()
    listed_q = q_tx[q_tx["type"] == "上市"]["deal"].tolist()
    held = marks[~marks["status"].isin(["已退出", "已核销"])]
    pme = m.get("pme") or {}
    tvpi_chg = m["TVPI"] - x["prev"]["TVPI"] if x["prev"] else 0.0
    # 本季度 LP 总价值变化（累计分配 + 资本账户净值）：TVPI 的变化只来自这里，分配本身不改变 TVPI
    tv = m["lp_distributed"] + m["lp_nav"]
    tv_prev = (x["prev"]["lp_distributed"] + x["prev"]["lp_nav"]) if x["prev"] else tv
    paid_chg = m["lp_paid_in"] - (x["prev"]["lp_paid_in"] if x["prev"] else m["lp_paid_in"])
    value_chg = tv - tv_prev - paid_chg
    facts = {
        "tvpi": ("TVPI（LP 净）", f"{m['TVPI']:.2f}x"), "dpi": ("DPI", f"{m['DPI']:.2f}x"), "rvpi": ("RVPI", f"{m['RVPI']:.2f}x"),
        "tvpi_chg": ("TVPI 本季度变化", f"{tvpi_chg:+.2f}x"),
        "value_chg": ("本季度 LP 总价值变化（估值变动 + 已实现收益，扣除新增实缴）", f"{value_chg / 1e4:+,.0f} 万元"),
        "irr": ("净 IRR", _pct(m["net_irr"], 1) if m.get("net_irr") is not None else "不适用"),
        "gross_moic": ("项目层面毛回报倍数", f"{m['gross_moic']:.2f}x"),
        "pme300": ("KS-PME（沪深300）", f"{pme.get('沪深300', float('nan')):.2f}"),
        "called_pct": ("实缴比例", _pct(m["called_pct"], 1)), "n_held": ("在投项目数", str(len(held))),
        "exit_amt": ("本季度退出回款", f"{exits.sum() / 1e4:,.0f} 万元"), "n_listed": ("本季度上市项目数", str(len(listed_q))),
        "my_called": ("贵方累计实缴", f"{called / 1e4:,.0f} 万元"), "my_dist": ("贵方累计分配", f"{dist / 1e4:,.0f} 万元"),
        "my_nav": ("贵方资本账户净值", f"{my_nav / 1e4:,.0f} 万元"), "carry_acc": ("应计业绩报酬", f"{m['accrued_carry'] / 1e4:,.0f} 万元"),
    }
    names_text = " ".join(v for _, v in facts.values() if not any(ch.isdigit() for ch in v)) + " 沪深300 中证500 " + " ".join(marks["company"].dropna())
    comm, check = None, None
    if with_llm:
        table = "\n".join(f"{{{{{k}}}}} | {lab} | {val}" for k, (lab, val) in facts.items())
        model = chat_model(0.3).with_structured_output(GPLetter, method="function_calling")
        msgs = [("system", "你是私募股权基金 GP，给 LP 写季度报告的「GP 观点」。数字只能用事实表里的占位符 {{键}}，不要写阿拉伯数字；"
                           "不承诺收益，不评价二级市场股价走势的未来方向。"
                           "注意：TVPI =（累计分配 + 净值）/ 实缴，分配本身不改变 TVPI；TVPI 的变化只来自估值变动和已实现收益（见 value_chg），"
                           "不要用「分配摊薄」解释 TVPI 变化。"), ("user", f"基金：{cfg['name']}，季度：{quarter}\n【事实表】\n{table}")]
        signs = {"TVPI": tvpi_chg, "估值": value_chg, "价值": value_chg}
        for _ in range(2):
            comm = model.invoke(msgs).model_dump()
            text = "\n".join(comm.values())
            check = check_placeholders(text, facts, names_text)
            check["direction_errors"] = check_direction(text, signs) + \
                [f"「{w}」：分配不改变 TVPI，不能用摊薄解释" for w in ("摊薄",) if w in text] + _double_negative(text, facts)
            check["ok"] = check["ok"] and not check["direction_errors"]
            if check["ok"]:
                break
            msgs.append(("user", f"未通过校验：未知键 {check['unknown_keys']}，占位符外的数字 {check['raw_numbers']}，方向错误 {check['direction_errors']}。请重写。"))
    if comm:                                       # 事实值已带单位，去掉模型在占位符后重复写的单位
        import re

        for k, (_, val) in facts.items():
            unit = next((u for u in ("万元", "x") if val.endswith(u)), None)
            if unit:
                comm = {kk: re.sub(r"\{\{" + k + r"\}\}\s*" + re.escape(unit), "{{" + k + "}}", vv) for kk, vv in comm.items()}
    comm_html = {k: fill(v, facts, html=True) for k, v in (comm or {}).items()}

    qs = db.df("select date, metrics from pe_quarters where product = :p order by date", p=code)
    qs["date"] = pd.to_datetime(qs["date"])
    qs = qs[qs["date"] <= d]
    mm = qs["metrics"].map(_m)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(qs["date"], mm.map(lambda v: v["TVPI"]), color="#0E7C66", lw=1.6, label="TVPI")
    ax.plot(qs["date"], mm.map(lambda v: v["DPI"]), color="#d4a017", lw=1.4, label="DPI")
    ax.axhline(1, color="#bbb", lw=0.8, ls="--")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("基金 TVPI / DPI（LP 净）", fontsize=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    png = _png(fig)

    rows_p = "".join(f"<tr><td style='text-align:left'>{escape(str(r.company))}</td><td>{escape(r.status)}</td><td>{r.cost / 1e4:,.0f}</td><td>{r.fv / 1e4:,.0f}</td>"
                     f"<td style='text-align:left;font-size:11px'>{escape(r.method)}{('；' + escape(r.note)) if r.note else ''}</td></tr>" for r in held.itertuples())
    rows_c = "".join(f"<tr><td>{r.date:%Y-%m-%d}</td><td>{escape(r.type)}</td><td>{r.amount / 1e4:,.1f}</td>"
                     f"<td style='text-align:left;font-size:11px'>{escape('；'.join(f'{k} {v / 1e4:,.1f} 万' if isinstance(v, (int, float)) else f'{k}：{v}' for k, v in (_m(r.detail) or {}).items() if not isinstance(v, (int, float)) or abs(v) >= 1))}</td></tr>"
                     for r in mine.sort_values("date").tail(12).itertuples())
    comm_block = "".join(f"<h3>{t}</h3><p>{comm_html.get(k, '')}</p>" for k, t in (("portfolio", "投资组合进展"), ("performance", "基金业绩"), ("outlook", "下季度重点"))) \
        if comm else "<p>（未生成 GP 观点）</p>"
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{escape(cfg['name'])} {quarter} LP 季度报告</title>
<style>body{{font-family:-apple-system,"PingFang SC",sans-serif;color:#1b2430;max-width:820px;margin:24px auto;font-size:12.5px;line-height:1.6}}
h1{{font-size:19px;margin:0}}h2{{font-size:14px;border-bottom:2px solid #1b2430;padding-bottom:3px;margin-top:20px}}h3{{font-size:13px;margin:10px 0 2px}}
.sub{{color:#6b7280;font-size:11px}}.k{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:12px 0}}.k div{{border:1px solid #e6e8eb;border-radius:6px;padding:6px 8px}}
.k b{{display:block;font-size:16px}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border-bottom:1px solid #eee;padding:4px 6px;text-align:right}}
.fact{{background:#eef7f4;border-radius:3px;padding:0 2px}}</style></head><body>
<h1>{escape(cfg['name'])}</h1>
<div class="sub">LP 季度报告 · {quarter}（截至 {d:%Y-%m-%d}）· 致：{escape(lpc['name'])} · 管理人：{escape(cfg['manager'])} · 托管人：{escape(cfg['custodian'])}</div>
<div class="k"><div>TVPI<b>{m['TVPI']:.2f}x</b></div><div>DPI<b>{m['DPI']:.2f}x</b></div><div>净 IRR<b>{facts['irr'][1]}</b></div>
<div>KS-PME（沪深300）<b>{pme.get('沪深300', float('nan')):.2f}</b></div><div>实缴比例<b>{_pct(m['called_pct'], 1)}</b></div></div>
<img src="data:image/png;base64,{png}" style="width:100%">
<h2>贵方资本账户</h2>
<table><tr><th>认缴</th><th>累计实缴</th><th>未缴</th><th>累计分配</th><th>资本账户净值（假设清算）</th></tr>
<tr><td>{lpc['commitment'] / 1e4:,.0f} 万</td><td>{called / 1e4:,.0f} 万</td><td>{(lpc['commitment'] - called) / 1e4:,.0f} 万</td><td>{dist / 1e4:,.0f} 万</td><td>{my_nav / 1e4:,.0f} 万</td></tr></table>
<p class="sub">资本账户净值 = 按季末基金净值做假设清算、走完分配瀑布后贵方应得的金额（已扣除应计业绩报酬 {m['accrued_carry'] / 1e4:,.0f} 万元）。</p>
<h3>最近的缴款与分配</h3><table><tr><th>日期</th><th>类型</th><th>金额（万元）</th><th>明细</th></tr>{rows_c}</table>
<h2>投资组合（季末，万元）</h2><table><tr><th>项目</th><th>状态</th><th>剩余成本</th><th>公允价值</th><th>估值方法</th></tr>{rows_p}</table>
<p class="sub">估值依据中基协《私募投资基金非上市股权投资估值指引》：未上市项目交割后 12 个月内按最近融资价格，之后按申万行业指数校准；
上市限售股按《证券投资基金投资流通受限股票估值指引》的 AAP 模型扣除流动性折扣。项目的投资轮次、价格、锁定承诺来自招股说明书（尽调 Agent 抽取并逐一核对页码）。</p>
<h2>GP 观点</h2>{comm_block}
<p class="sub">本报告为模拟产品演示：组合公司真实、跟投行为与 LP 均为模拟，两个未上市项目为虚构；不构成投资建议。</p></body></html>"""
    out = OUTPUT_DIR / "pe" / code
    out.mkdir(parents=True, exist_ok=True)
    hp = out / f"{code}_{lp}_{quarter}.html"
    hp.write_text(html, encoding="utf-8")
    pdf = None
    if Path(CHROME).exists():
        pdf = out / f"{code}_{lp}_{quarter}.pdf"
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", hp.as_uri()], capture_output=True, timeout=120)
    return {"html": str(hp), "pdf": str(pdf) if pdf and pdf.exists() else None, "check": check, "commentary": comm}
