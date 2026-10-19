"""数字溯源校验：大模型写的文字里，每个数字都必须能在「事实表」（代码算出来的结果）里找到。

做法很简单、可解释：
  1. 事实表先由代码格式化成文本（百分比保留 1 位、金额以万元为单位），大模型只能引用这张表；
  2. 从大模型输出里抽出所有数字，逐个去事实表的数字集合里找；
  3. 找不到的数字就是「无出处」，整段解读不通过，退回重写；重写还不通过就改用模板文字。
同时检查是否出现买卖建议类措辞（本工具只做风险分析，不给投资建议）。
"""
from __future__ import annotations

import re

NUM_RE = re.compile(r"(?<![\d.])\d+(?:,\d{3})*(?:\.\d+)?")
ADVICE_WORDS = ["买入", "卖出", "加仓", "减仓", "清仓", "建仓", "抄底", "止盈", "止损", "目标价", "推荐", "强烈"]
FREE_INTS = {str(i) for i in range(0, 11)}  # 「3 个持仓」「前 5 大」这类小整数不要求出处


def numbers(text: str) -> list[str]:
    return [m.group().replace(",", "") for m in NUM_RE.finditer(text)]


def _norm(tok: str) -> str:
    v = float(tok)
    return repr(round(v, 6))


def allowed_set(*texts: str) -> set[str]:
    return {_norm(t) for text in texts for t in numbers(text)} | {_norm(t) for t in FREE_INTS}


def check(text: str, allowed: set[str]) -> dict:
    unknown = sorted({t for t in numbers(text) if _norm(t) not in allowed}, key=float)
    advice = [w for w in ADVICE_WORDS if w in text]
    total = len(numbers(text))
    return {"ok": not unknown and not advice, "unknown_numbers": unknown, "advice_words": advice,
            "numbers_total": total, "numbers_traced": total - sum(numbers(text).count(u) for u in unknown)}


# ---- 统一的数字格式，事实表和报告都用这一套 ----
def pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:.{nd}f}%"


def wan(x: float, nd: int = 1) -> str:
    """金额：万元。"""
    return f"{x / 1e4:,.{nd}f} 万元"


def num(x: float, nd: int = 2) -> str:
    return f"{x:,.{nd}f}"
