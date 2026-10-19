"""业绩会纪要的运行入口：列记录、生成纪要（可与上一次对比）、录音转写后生成纪要。结果存 output/calls/。"""
from __future__ import annotations

import json
from pathlib import Path

from cyron_veritas.calls import analyze as A
from cyron_veritas.calls import source as S
from cyron_veritas.config import OUTPUT_DIR

OUT = OUTPUT_DIR / "calls"


def records(ticker: str) -> list[dict]:
    df = S.list_records(ticker)
    return df.to_dict("records") if len(df) else []


def run(ticker: str, record_id: str | None = None, compare: bool = True, say=None) -> dict:
    say = say or (lambda e: None)
    recs = records(ticker)
    if not recs:
        raise LookupError(f"{ticker} 没有找到投资者关系活动记录")
    i = next((k for k, r in enumerate(recs) if r["record_id"] == record_id), 0)
    say({"type": "status", "text": f"下载并解析 {recs[i]['title']}"})
    cur = S.fetch(recs[i])
    say({"type": "status", "text": f"大模型抽取纪要（{len(cur.qa)} 组问答）"})
    a = A.summarize(cur)
    cmp = None
    if compare and i + 1 < len(recs):
        prev = S.fetch(recs[i + 1])
        say({"type": "status", "text": f"抽取上一次（{prev.date}）并对比"})
        b = A.summarize(prev)
        cmp = A.compare(b, a, prev, cur)
    md = A.to_markdown(a, cmp)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{ticker}_{cur.date}"
    (OUT / f"{stem}.md").write_text(md, encoding="utf-8")
    (OUT / f"{stem}.json").write_text(json.dumps({"result": a, "compare": cmp}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"markdown": md, "result": a, "compare": cmp, "path": str(OUT / f"{stem}.md")}


def from_audio(path: str | Path, company: str = "", say=None) -> dict:
    say = say or (lambda e: None)
    say({"type": "status", "text": "Whisper 转写录音"})
    rec = S.transcribe(path)
    rec.company = company
    say({"type": "status", "text": f"转写完成（{len(rec.text)} 字），大模型抽取纪要"})
    a = A.summarize(rec)
    md = A.to_markdown(a)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"audio_{Path(path).stem}.md").write_text(md, encoding="utf-8")
    return {"markdown": md, "result": a, "transcript": rec.text}
