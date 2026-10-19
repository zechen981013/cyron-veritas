"""年报问答评测：检索消融（命中率）+ 端到端问答（答案正确率、校验通过率、耗时）。

    uv run python -m cyron_veritas.rag.evaluate            # 只跑检索（不调大模型，约 1 分钟）
    uv run python -m cyron_veritas.rag.evaluate --qa       # 加上端到端问答（调大模型）
命中的定义：检索结果中任一切块的页码范围覆盖了人工标注的答案页。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

from cyron_veritas.config import ROOT
from cyron_veritas.rag import index as I
from cyron_veritas.rag import qa

EVAL = ROOT / "eval" / "annual_report_qa.json"


def _rank(hits: list[dict], gold: list[int]) -> int | None:
    for r, d in enumerate(hits, start=1):
        if any(d["page_start"] <= g <= d["page_end"] for g in gold):
            return r
    return None


VARIANTS = {
    "仅向量（bge-m3）": lambda q, t, y: I.search(q, t, y, top=5, rerank=False, mode="dense"),
    "仅 BM25": lambda q, t, y: I.search(q, t, y, top=5, rerank=False, mode="bm25"),
    "混合（RRF）": lambda q, t, y: I.search(q, t, y, top=5, rerank=False),
    "混合 + 重排": lambda q, t, y: I.search(q, t, y, top=5, rerank=True),
    "改写 + HyDE + 混合 + 重排": lambda q, t, y: qa.retrieve(q, t, y, top=5)[0],
}


def _items(data: dict) -> list[tuple[dict, dict]]:
    return [(s, it) for s in data["sets"] for it in s["items"]]


def retrieval(data: dict) -> dict:
    out = {}
    for name, fn in VARIANTS.items():
        ranks, rows, t0 = [], [], time.time()
        for st, it in _items(data):
            r = _rank(fn(it["q"], st["ticker"], st["year"]), it["gold_pages"])
            ranks.append(r)
            rows.append((st["company"], it.get("type", "数字"), r))
        n = len(ranks)
        out[name] = {"hit@1": sum(r == 1 for r in ranks) / n, "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
                     "hit@5": sum(r is not None for r in ranks) / n,
                     "mrr": sum(1 / r for r in ranks if r) / n, "sec_per_q": (time.time() - t0) / n,
                     "misses": [it["q"] for (_, it), r in zip(_items(data), ranks) if r is None],
                     "by_group": {g: sum(r is not None for c, tp, r in rows if g in (c, tp)) / max(1, sum(1 for c, tp, _ in rows if g in (c, tp)))
                                  for g in sorted({c for c, _, _ in rows} | {tp for _, tp, _ in rows})}}
    return out


def end_to_end(data: dict) -> dict:
    rows = []
    for st, it in _items(data):
        t0 = time.time()
        r = qa.ask(it["q"], st["ticker"], st["year"])
        ans = r["answer"].replace(" ", "")
        rows.append({"company": st["company"], "type": it.get("type", "数字"), "q": it["q"], "correct": any(e.replace(" ", "") in ans for e in it["expect"]), "check_ok": r["check"]["ok"],
                     "cited_pages": r["check"]["cited_pages"], "unknown_numbers": r["check"]["unknown_numbers"],
                     "seconds": round(time.time() - t0, 1), "answer": r["answer"]})
    n = len(rows)
    return {"accuracy": sum(r["correct"] for r in rows) / n, "check_pass": sum(r["check_ok"] for r in rows) / n,
            "avg_seconds": sum(r["seconds"] for r in rows) / n, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", action="store_true")
    a = ap.parse_args()
    data = json.loads(EVAL.read_text(encoding="utf-8"))
    res = {"date": str(dt.date.today()), "n": len(_items(data)), "retrieval": retrieval(data)}
    if a.qa:
        res["qa"] = end_to_end(data)
    companies = "、".join(s["company"] for s in data["sets"])
    groups = list(next(iter(res["retrieval"].values()))["by_group"])
    lines = [f"# 年报问答评测（{companies} 2025 年报，{res['n']} 题，{res['date']}）", "",
             "| 检索方案 | Hit@1 | Hit@3 | Hit@5 | MRR | 秒/题 | " + " | ".join(f"Hit@5·{g}" for g in groups) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(groups)]
    for k, v in res["retrieval"].items():
        lines.append(f"| {k} | {v['hit@1']:.0%} | {v['hit@3']:.0%} | {v['hit@5']:.0%} | {v['mrr']:.3f} | {v['sec_per_q']:.1f} | "
                     + " | ".join(f"{v['by_group'][g]:.0%}" for g in groups) + " |")
    if "qa" in res:
        q = res["qa"]
        lines += ["", f"端到端：答案正确率 {q['accuracy']:.0%}，引用与数字校验通过率 {q['check_pass']:.0%}，平均 {q['avg_seconds']:.1f} 秒/题", "",
                  "| 公司 | 类型 | 问题 | 正确 | 校验 | 引用页 | 未溯源数字 |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['company']} | {r['type']} | {r['q']} | {'✓' if r['correct'] else '✗'} | {'✓' if r['check_ok'] else '✗'} | {r['cited_pages']} | {', '.join(r['unknown_numbers'])} |"
                  for r in q["rows"]]
    (ROOT / "eval" / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "eval" / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
