"""业绩会 / 投资者交流的原始材料：
  1. 文字：巨潮资讯网「投资者关系活动记录表」（A 股公司每次业绩说明会、路演、调研后发布的问答记录）
  2. 语音：用户提供的录音 → 本机 Whisper（mlx-whisper，large-v3-turbo）转写

记录表结构固定：活动类别（█ 勾选）、参与单位、时间、主要内容（编号问答）。这里解析成结构化问答，
后面的分析只基于这些原文，每条结论都要带原文引用。
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import pandas as pd
import pymupdf

from cyron_veritas.config import ROOT
from cyron_veritas.data.market import _cached

IR_DIR = ROOT / "data" / "ir"
CATS = ["特定对象调研", "分析师会议", "媒体采访", "业绩说明会", "新闻发布会", "路演活动", "现场参观", "其他"]


@dataclass
class Record:
    ticker: str
    company: str
    date: str
    title: str
    url: str
    categories: list[str] = field(default_factory=list)
    participants: str = ""
    qa: list[dict] = field(default_factory=list)  # [{"q": ..., "a": ...}]
    text: str = ""                                # 主要内容全文（引用校验用）
    source: str = "巨潮资讯网投资者关系活动记录表"

    def to_dict(self) -> dict:
        return asdict(self)


def list_records(ticker: str, start: str = "20240101", end: str = "20991231") -> pd.DataFrame:
    import akshare as ak

    df = _cached(f"ir_list_{ticker}_{start}", lambda: ak.stock_zh_a_disclosure_relation_cninfo(
        symbol=ticker, market="沪深京", start_date=start, end_date=end))
    if df.empty:
        return df
    df = df.copy()
    ids = df["公告链接"].str.extract(r"announcementId=(\d+)")[0]
    df["date"] = pd.to_datetime(df["公告时间"]).dt.strftime("%Y-%m-%d")
    df["pdf_url"] = "http://static.cninfo.com.cn/finalpage/" + df["date"] + "/" + ids + ".PDF"
    df["record_id"] = ticker + "_" + ids
    return df.rename(columns={"简称": "company", "公告标题": "title", "公告链接": "url"})[
        ["record_id", "company", "title", "date", "url", "pdf_url"]].sort_values("date", ascending=False).reset_index(drop=True)


def _clean(text: str) -> str:
    text = re.sub(r"^\s*\d+\s*/\s*\d+\s*$", "", text, flags=re.M)  # 页码「1 / 3」
    return re.sub(r"[ \t]+\n", "\n", text)


def _between(text: str, a: str, b: str) -> str:
    i = text.find(a)
    if i < 0:
        return ""
    j = text.find(b, i + len(a))
    return text[i + len(a): j if j > 0 else None]


def split_qa(content: str) -> list[dict]:
    """按「1. 问题？」编号切问答。问题可能跨行，到第一个问号为止。"""
    flat = content.replace("\r", "")
    starts = [m for m in re.finditer(r"(?m)^\s*(\d{1,2})\s*[\.、．]\s*(?=\S)", flat)]
    qa = []
    for k, m in enumerate(starts):
        seg = flat[m.end(): starts[k + 1].start() if k + 1 < len(starts) else len(flat)]
        q_end = re.search(r"[？?]", seg)
        if not q_end or q_end.start() > 200:  # 不是问答格式（例如正文里的编号列表）
            continue
        q = re.sub(r"\s+", "", seg[: q_end.end()])
        a = re.sub(r"\n(?!\s*[（(]?\d)", "", seg[q_end.end():]).strip()
        # 网上业绩说明会：一个问题常有多个问号，回答以「尊敬的投资者，您好」开头，以它为界把剩下的问句并回问题
        greet = re.search(r"尊敬的投资者|投资者您好|您好[，,。！]", a)
        if greet and 0 < greet.start() < 400 and re.search(r"[？?]\s*$", a[: greet.start()]):
            q += re.sub(r"\s+", "", a[: greet.start()])
            a = a[greet.start():]
        qa.append({"q": q, "a": re.sub(r"[ \t]+", " ", a)})
    return qa


def parse_pdf(path: Path, meta: dict) -> Record:
    raw = _clean("".join(p.get_text() for p in pymupdf.open(path)))
    cats = [c for c in CATS if re.search(rf"[█■☑√]\s*{c}", raw)]
    parts = _between(raw, "参与单位名称", "时间")
    content = _between(raw, "主要内容介", "附件清单") or _between(raw, "主要内容", "附件清单") or raw
    content = re.sub(r"^\s*绍\s*", "", content)
    rec = Record(ticker=meta["record_id"].split("_")[0], company=meta.get("company", ""), date=meta["date"],
                 title=meta["title"], url=meta["url"], categories=cats, participants=re.sub(r"^及人员姓名", "", re.sub(r"\s+", "", parts))[:400],
                 text=content.strip())
    rec.qa = split_qa(content)
    return rec


def fetch(meta: dict) -> Record:
    d = IR_DIR / meta["record_id"].split("_")[0]
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{meta['date']}_{meta['record_id']}.pdf"
    if not path.exists():
        r = httpx.get(meta["pdf_url"], timeout=60, follow_redirects=True)
        r.raise_for_status()
        path.write_bytes(r.content)
    return parse_pdf(path, meta)


def transcribe(audio: str | Path, language: str = "zh") -> Record:
    """录音 → 文字（本机 Whisper）。没有说话人分离：问答由后面的大模型根据语义切分。"""
    import mlx_whisper

    r = mlx_whisper.transcribe(str(audio), path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language=language,
                               initial_prompt="以下是上市公司业绩说明会的录音，使用简体中文。")
    text = r["text"].strip()
    p = Path(audio)
    return Record(ticker="", company="", date="", title=p.name, url=str(p), categories=["录音转写"], text=text,
                  source=f"录音转写（Whisper large-v3-turbo，{r['segments'][-1]['end'] if r['segments'] else 0:.0f} 秒）")
