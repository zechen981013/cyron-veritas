"""投资者月报（HTML + PDF）：净值与基准、回撤、月度收益表、行业分布、前十大持仓、当月归因、经理观点。

经理观点由大模型撰写，沿用风控日报的「结构化引用」：只能写 {{键}} 占位符，数字由代码填入，占位符以外不许出现数字。
PDF 用本机 Chrome 无头打印（找不到 Chrome 时只输出 HTML）。
"""
from __future__ import annotations

import base64
import io
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from markupsafe import escape  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from cyron_veritas.config import OUTPUT_DIR, chat_model  # noqa: E402
from cyron_veritas.fund import marketdata as M  # noqa: E402
from cyron_veritas.fund import perf as PF  # noqa: E402
from cyron_veritas.fund import product as P  # noqa: E402
from cyron_veritas.fund import risk as R  # noqa: E402
from cyron_veritas.risk.report_graph import check_placeholders, fill  # noqa: E402

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _pct(x, nd=2):
    return f"{x * 100:.{nd}f}%"


class Commentary(BaseModel):
    market: str = Field(description="当月市场回顾，2-3 句，只用占位符引用数字")
    performance: str = Field(description="产品表现与归因，2-3 句：超额来自配置还是选股，超配 / 低配哪些行业")
    outlook: str = Field(description="下月关注与组合应对，2 句；不承诺收益、不给个股推荐")


POS_WORDS = ("正贡献", "贡献为正", "正向贡献", "跑赢", "领先基准", "增厚")
NEG_WORDS = ("拖累", "负贡献", "贡献为负", "跑输", "落后基准", "侵蚀")
EXCESS_WORDS = ("跑赢", "跑输", "领先基准", "落后基准")          # 没写「超额」但说的就是超额


def check_direction(text: str, signs: dict[str, float]) -> list[str]:
    """方向词校验：数字由占位符保证正确，但「配置效应拖累」这类定性判断可能与数字相反。
    按逗号句号切分句，句中出现某项名称（配置 / 选股 / 交互 / 超额）和方向词时，方向必须与该项数值的符号一致。"""
    errors = []
    for clause in re.split(r"[，。；,;！？\n]", text):
        pos, neg = any(w in clause for w in POS_WORDS), any(w in clause for w in NEG_WORDS)
        if pos == neg:          # 没有方向词，或正负词同句无法判断
            continue
        terms = [t for t in signs if t in clause] or (["超额"] if any(w in clause for w in EXCESS_WORDS) else [])
        for term in terms:
            v = signs[term]
            if v is not None and ((pos and v < 0) or (neg and v > 0)):
                errors.append(f"「{clause.strip()}」：{term} 实际为 {v:+.2%}")
    return errors


def build(code: str, month: str, with_llm: bool = True) -> dict:
    cfg = P.load(code)
    nav = R.nav_series(code)
    m_end = pd.Period(month, "M").end_time.normalize()
    nav = nav.loc[:m_end]
    d = nav.index[-1]
    met = PF.nav_metrics(code)
    # 截到报告月
    sub = nav["unit_nav"]
    bname = P.bench_name(cfg)
    b = M.index_close(cfg["benchmark"]).reindex(sub.index).ffill()
    m_start = pd.Period(month, "M").start_time
    prev = sub.loc[:m_start - pd.Timedelta(days=1)]
    p0 = prev.iloc[-1] if len(prev) else sub.iloc[0]
    b0 = b.loc[:m_start - pd.Timedelta(days=1)].iloc[-1] if len(prev) else b.iloc[0]
    m_ret, mb_ret = sub.iloc[-1] / p0 - 1, b.iloc[-1] / b0 - 1
    ytd0 = sub.loc[:pd.Timestamp(f"{d.year}-01-01") - pd.Timedelta(days=1)]
    ytd = sub.iloc[-1] / (ytd0.iloc[-1] if len(ytd0) else sub.iloc[0]) - 1
    snap = R.snapshot(code, d)
    att = PF.brinson(code)
    att_m = att[att["month"] == month]
    a = att_m.iloc[0] if len(att_m) else None
    h = R.holdings_on(code, d).sort_values("mv", ascending=False).head(10)

    facts = {
        "nav": ("月末单位净值", f"{sub.iloc[-1]:.4f}"), "m_ret": ("本月收益", _pct(m_ret)), "mb_ret": (f"本月{bname}", _pct(mb_ret)),
        "m_ex": ("本月超额", _pct(m_ret - mb_ret)), "ytd": ("今年以来收益", _pct(ytd)),
        "since": ("成立以来收益", _pct(sub.iloc[-1] - 1)), "mdd": ("成立以来最大回撤", _pct(1 - (sub / sub.cummax()).min())),
        "stock_w": ("股票仓位", _pct(snap["stock_weight"], 1)), "n": ("持股数量", str(snap["n_holdings"])),
    }
    if a is not None:
        facts.update({"alloc": ("当月配置效应", _pct(a["配置效应"])), "select": ("当月选股效应", _pct(a["选股效应"])),
                      "inter": ("当月交互效应", _pct(a["交互效应"])), "over": ("超配最多行业", a["超配最多"]), "under": ("低配最多行业", a["低配最多"])})
    for i, (k, v) in enumerate(list(snap["by_industry"].items())[:3], start=1):
        facts[f"ind{i}"] = (f"第 {i} 大行业", k)
        facts[f"ind{i}_w"] = (f"{k} 占净值", _pct(v, 1))
    table = "\n".join(f"{{{{{k}}}}} | {lab} | {val}" for k, (lab, val) in facts.items())
    names = " ".join(v for _, v in facts.values() if not any(ch.isdigit() for ch in v)) + " 沪深300 中证500 中证800"  # 指数名称里的数字不算数据

    signs = {"配置": a["配置效应"] if a is not None else None, "选股": a["选股效应"] if a is not None else None,
             "交互": a["交互效应"] if a is not None else None, "超额": m_ret - mb_ret}
    comm, check = None, None
    if with_llm:
        model = chat_model(0.3).with_structured_output(Commentary, method="function_calling")
        msgs = [("system", "你是私募基金经理，写投资者月报的「经理观点」。所有数字只能用事实表里的占位符 {{键}} 引用，不要写阿拉伯数字；"
                           "不承诺收益、不推荐个股、不出现买入卖出等措辞。"), ("user", f"产品：{cfg['name']}，报告月份：{month}\n【事实表】\n{table}")]
        for _ in range(2):
            comm = model.invoke(msgs).model_dump()
            text = "\n".join(comm.values())
            check = check_placeholders(text, facts, names)
            check["direction_errors"] = check_direction(text, signs)
            check["ok"] = check["ok"] and not check["direction_errors"]
            if check["ok"]:
                break
            msgs.append(("user", f"未通过校验：未知键 {check['unknown_keys']}，占位符外的数字 {check['raw_numbers']}，"
                                 f"方向与数据相反 {check['direction_errors']}。请重写。"))
    comm_html = {k: fill(v, facts, html=True) for k, v in (comm or {}).items()}

    # 图
    fig, ax = plt.subplots(figsize=(7.2, 3))
    ax.plot(sub.index, sub.values, color="#0E7C66", lw=1.5, label=cfg["name"][:12])
    ax.plot(b.index, (b / b.iloc[0]).values, color="#999", lw=1.1, label=f"{bname}（归一）")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("单位净值与基准", fontsize=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    nav_png = _png(fig)
    fig, ax = plt.subplots(figsize=(7.2, 1.6))
    dd = sub / sub.cummax() - 1
    ax.fill_between(dd.index, dd.values * 100, 0, color="#c0392b", alpha=0.3, lw=0)
    ax.set_title("回撤 %", fontsize=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    dd_png = _png(fig)

    mt = met["monthly"].loc[:m_end].tail(12)
    rows_m = "".join(f"<tr><td>{i:%Y-%m}</td><td>{_pct(r['组合'])}</td><td>{_pct(r['基准'])}</td><td>{_pct(r['超额'])}</td></tr>"
                     for i, r in mt.iterrows())
    rows_i = "".join(f"<tr><td>{escape(k)}</td><td>{_pct(v, 1)}</td></tr>" for k, v in list(snap["by_industry"].items())[:8])
    rows_h = "".join(f"<tr><td>{escape(r['name'])}</td><td>{r['code']}</td><td>{_pct(r['weight'], 1)}</td></tr>" for _, r in h.iterrows())
    att_html = ""
    if a is not None:
        att_html = (f"<table><tr><th>本月超额</th><th>配置效应</th><th>选股效应</th><th>交互效应</th><th>其他（交易与现金）</th></tr>"
                    f"<tr><td>{_pct(a['超额'])}</td><td>{_pct(a['配置效应'])}</td><td>{_pct(a['选股效应'])}</td><td>{_pct(a['交互效应'])}</td>"
                    f"<td>{_pct(a['其他（交易与现金）'])}</td></tr></table>")
    comm_block = "".join(f"<h3>{t}</h3><p>{comm_html.get(k, '')}</p>" for k, t in (("market", "市场回顾"), ("performance", "产品表现与归因"),
                                                                                   ("outlook", "下月关注"))) if comm else "<p>（未生成经理观点）</p>"
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{escape(cfg['name'])} {month} 月报</title>
<style>body{{font-family:-apple-system,"PingFang SC",sans-serif;color:#1b2430;max-width:820px;margin:24px auto;font-size:12.5px;line-height:1.6}}
h1{{font-size:19px;margin:0}}h2{{font-size:14px;border-bottom:2px solid #1b2430;padding-bottom:3px;margin-top:20px}}h3{{font-size:13px;margin:10px 0 2px}}
.sub{{color:#6b7280;font-size:11px}}.k{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:12px 0}}.k div{{border:1px solid #e6e8eb;border-radius:6px;padding:6px 8px}}
.k b{{display:block;font-size:16px}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border-bottom:1px solid #eee;padding:4px 6px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}th{{background:#f7f7f7}}.g{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}img{{max-width:100%}}
.fact{{border-bottom:1px dotted #0E7C66}}.brand{{color:#0E7C66;letter-spacing:.1em;font-weight:600;font-size:11px}}</style></head><body>
<div class="brand">CYRON VERITAS · 投资者月报</div><h1>{escape(cfg['name'])}</h1>
<div class="sub">报告期：{month} · 截至 {d:%Y-%m-%d} · 管理人：{escape(cfg['manager'])} · 托管人：{escape(cfg['custodian'])} · 业绩比较基准：{bname}</div>
<div class="k"><div>月末单位净值<b>{sub.iloc[-1]:.4f}</b></div><div>本月收益<b>{_pct(m_ret)}</b><span class="sub">基准 {_pct(mb_ret)}</span></div>
<div>今年以来<b>{_pct(ytd)}</b></div><div>成立以来<b>{_pct(sub.iloc[-1] - 1)}</b></div><div>最大回撤<b>{_pct(1 - (sub / sub.cummax()).min())}</b></div></div>
<h2>经理观点</h2>{comm_block}
<h2>净值表现</h2><img src="data:image/png;base64,{nav_png}"><img src="data:image/png;base64,{dd_png}">
<div class="g"><div><h3>近 12 个月收益</h3><table><tr><th>月份</th><th>产品</th><th>{bname}</th><th>超额</th></tr>{rows_m}</table></div>
<div><h3>行业分布（占净值）</h3><table><tr><th>申万一级</th><th>占比</th></tr>{rows_i}</table>
<h3>前十大持仓</h3><table><tr><th>名称</th><th>代码</th><th>占净值</th></tr>{rows_h}</table></div></div>
<h2>本月业绩归因（Brinson，申万一级行业）</h2>{att_html}
<p class="sub">风险指标：年化波动 {_pct(snap['vol_ann'], 1)}，1 日 95% VaR {_pct(snap['var95_1d'])}，Beta {snap['beta']:.2f}；
5 日内可变现比例 {_pct(snap['liquidity'][5], 1)}；距预警线 {_pct(snap['to_warning_line'], 1)}。</p>
<p class="sub">重要提示：本报告为系统演示用模拟产品，数据由真实行情按规则模拟生成。私募基金过往业绩不代表未来表现，不构成投资建议。
归因基准权重采用中证指数公司最新月末权重近似。经理观点中的数字由系统根据数据库自动填入。</p></body></html>"""
    out = OUTPUT_DIR / "fund" / code
    out.mkdir(parents=True, exist_ok=True)
    hp = out / f"{code}_月报_{month}.html"
    hp.write_text(html, encoding="utf-8")
    pdf = None
    if Path(CHROME).exists():
        pdf = out / f"{code}_月报_{month}.pdf"
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", hp.as_uri()],
                       capture_output=True, timeout=120)
    return {"html": str(hp), "pdf": str(pdf) if pdf and pdf.exists() else None, "check": check, "commentary": comm}
