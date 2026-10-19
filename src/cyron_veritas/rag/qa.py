"""年报问答：问题改写 + HyDE → 混合检索 → 带页码回答 → 引用与数字校验。

    问题 → 大模型改写成年报措辞的检索词 + 假设性答案段落（HyDE）
         → 三路查询分别做混合检索，RRF 合并 → 精排后取证据
         → 大模型只根据证据回答，每句标注 [p.页码]
         → 校验：① 引用的页码必须在证据范围内；② 回答里的数字必须出现在被检索到的原文里
         → 不通过带着错误重写一次；还不通过就照实标注「有未核实内容」
以事件流（生成器）输出，Web 界面用 SSE 逐步显示：检索到的证据 → 逐字回答 → 校验结果。
"""
from __future__ import annotations

import json
import re
from typing import Iterator

from pydantic import BaseModel, Field

from cyron_veritas import verify as V
from cyron_veritas.config import chat_model
from cyron_veritas.rag import index as I

CITE_RE = re.compile(r"\[p\.(\d+)(?:[-–](\d+))?\]")


class Rewrite(BaseModel):
    keywords: str = Field(description="把问题改写成年报里可能出现的措辞和科目名，空格分隔，10-20 个词")
    hyde: str = Field(description="假设年报里有一段话正好回答了这个问题，写出这段话（80-150 字，数字可以用 X 代替）")


def rewrite(question: str, company: str) -> Rewrite:
    try:
        m = chat_model(0).with_structured_output(Rewrite, method="function_calling")
        return m.invoke([("system", "你是熟悉中国上市公司年报结构和会计科目的研究员。"),
                         ("user", f"公司：{company}\n问题：{question}")])
    except Exception:
        return Rewrite(keywords=question, hyde="")


def retrieve(question: str, ticker: str, year: int, top: int = 5) -> tuple[list[dict], dict]:
    meta = I.manifest()[f"{ticker}_{year}"]
    rw = rewrite(question, meta["company"])
    queries = [q for q in (question, rw.keywords, rw.hyde) if q]
    pool: dict[str, dict] = {}
    rrf: dict[str, float] = {}
    for q in queries:
        for r, d in enumerate(I.search(q, ticker, year, top=10, rerank=False)):
            pool[d["id"]] = d
            rrf[d["id"]] = rrf.get(d["id"], 0) + 1 / (60 + r + 1)
    cands = [pool[c] for c in sorted(rrf, key=rrf.get, reverse=True)[:20]]
    scores = I.rerank_scores([(question, f"{d['subsection']}\n{d['text']}") for d in cands])
    for d, s in zip(cands, scores):
        d["score"] = float(s)
    cands.sort(key=lambda d: -d["score"])
    best = cands[0]["score"] if cands else 0
    hits = [d for d in cands[:top] if d["score"] >= best * 0.25] or cands[:1]
    return hits, {"queries": queries}


SYSTEM = """你是买方研究员，根据上市公司年报原文回答问题。
规则：
1. 只能使用【证据】里的内容，不要用你自己的知识补充；证据里没有就直接说「年报中未找到」；
2. 每句话末尾用 [p.页码] 标注出处（页码取证据标签里的页码，跨页写 [p.12-13]）；
3. 数字原样引用（包括单位：年报表格多为人民币千元，注意换算说明时也要写清原始数字）；
4. 先给结论，再列 2-4 个要点；不给投资建议。"""


def _context(hits: list[dict]) -> str:
    return "\n\n".join(f"【证据{i + 1}｜{d['section']} {d['subsection']}｜p.{d['page_start']}"
                       f"{'-' + str(d['page_end']) if d['page_end'] != d['page_start'] else ''}】\n{d['text']}"
                       for i, d in enumerate(hits))


def check_answer(answer: str, hits: list[dict], question: str) -> dict:
    pages = {p for d in hits for p in range(d["page_start"], d["page_end"] + 1)}
    cited = []
    for m in CITE_RE.finditer(answer):
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        cited.extend(range(a, b + 1))
    bad_pages = sorted({p for p in cited if p not in pages})
    body = CITE_RE.sub("", answer)
    nums = V.check(body, V.allowed_set(*(d["text"] for d in hits), question))
    ok = not bad_pages and nums["ok"] and bool(cited)
    return {"ok": ok, "cited_pages": sorted(set(cited)), "bad_pages": bad_pages, "has_citation": bool(cited),
            "unknown_numbers": nums["unknown_numbers"], "advice_words": nums["advice_words"],
            "numbers_total": nums["numbers_total"], "numbers_traced": nums["numbers_traced"]}


def answer_stream(question: str, ticker: str, year: int) -> Iterator[dict]:
    yield {"type": "status", "text": "改写问题并检索年报"}
    hits, info = retrieve(question, ticker, year)
    yield {"type": "evidence", "queries": info["queries"],
           "items": [{k: d[k] for k in ("id", "section", "subsection", "page_start", "page_end", "text", "score")} for d in hits]}
    msgs = [("system", SYSTEM), ("user", f"【证据】\n{_context(hits)}\n\n【问题】{question}")]
    model = chat_model(0.1)
    answer, check = "", {}
    for attempt in range(2):
        if attempt:
            yield {"type": "status", "text": "校验未通过，重写回答"}
            yield {"type": "reset"}
        answer = ""
        for part in model.stream(msgs):
            if part.content:
                answer += part.content
                yield {"type": "token", "text": part.content}
        check = check_answer(answer, hits, question)
        if check["ok"]:
            break
        problems = []
        if not check["has_citation"]:
            problems.append("没有标注页码")
        if check["bad_pages"]:
            problems.append(f"引用了证据之外的页码 {check['bad_pages']}")
        if check["unknown_numbers"]:
            problems.append(f"这些数字在证据原文里找不到：{', '.join(check['unknown_numbers'])}（不要自己换算）")
        if check["advice_words"]:
            problems.append("出现了投资建议措辞")
        msgs = msgs + [("assistant", answer), ("user", "回答没有通过校验：" + "；".join(problems) + "。请改正后重新回答。")]
    yield {"type": "done", "answer": answer, "check": check, "attempts": attempt + 1}


def ask(question: str, ticker: str, year: int) -> dict:
    """非流式版本（给研究 Agent 和评测用）。"""
    out = {}
    for ev in answer_stream(question, ticker, year):
        if ev["type"] in ("evidence", "done"):
            out[ev["type"]] = ev
    return {"answer": out["done"]["answer"], "check": out["done"]["check"], "evidence": out["evidence"]["items"],
            "queries": out["evidence"]["queries"]}
