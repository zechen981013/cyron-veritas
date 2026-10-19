"""报告图表（matplotlib → base64 PNG，直接嵌进 HTML，单文件可发送）。"""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
INK, ACCENT, RED, GREY = "#1f2937", "#0E7C66", "#c0392b", "#9ca3af"


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)


def barh(data: dict, title: str, limit: float | None = None) -> str:
    items = list(data.items())[::-1]
    fig, ax = plt.subplots(figsize=(5, 0.32 * len(items) + 0.8))
    vals = [v * 100 for _, v in items]
    colors = [RED if limit is not None and v / 100 > limit else ACCENT for v in vals]
    ax.barh([k for k, _ in items], vals, color=colors, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=7, color=INK)
    if limit is not None:
        ax.axvline(limit * 100, color=RED, lw=0.8, ls="--")
        ax.text(limit * 100, len(items) - 0.4, f" 上限 {limit * 100:.0f}%", color=RED, fontsize=7)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    _style(ax)
    return _b64(fig)


def nav_drawdown(port_returns) -> str:
    nav = (1 + port_returns.fillna(0)).cumprod()
    dd = nav / nav.cummax() - 1
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 3.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(nav.index, nav.values, color=ACCENT, lw=1)
    a1.set_title("当前持仓回放净值（起点 = 1）", fontsize=9, color=INK, loc="left")
    a2.fill_between(dd.index, dd.values * 100, 0, color=RED, alpha=0.3, lw=0)
    a2.set_ylabel("回撤 %", fontsize=7)
    for a in (a1, a2):
        _style(a)
    return _b64(fig)


def var_backtest(series) -> str:
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.bar(series.index, series["ret"] * 100, color=GREY, width=1.0)
    ax.plot(series.index, series["var"] * 100, color=ACCENT, lw=1, label="95% VaR（前 250 天历史法）")
    hits = series[series["hit"]]
    ax.scatter(hits.index, hits["ret"] * 100, color=RED, s=10, zorder=3, label=f"突破 {len(hits)} 次")
    ax.set_title("VaR 回测：日收益 vs 前一日预测的 VaR", fontsize=9, color=INK, loc="left")
    ax.set_ylabel("%", fontsize=7)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    return _b64(fig)
