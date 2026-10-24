"""招股说明书：巨潮资讯网下载 → 按页抽文本（去页眉页脚）→ 关键词检索页面。

尽调 Agent 的原始材料。用上市后的正式版招股书（含发行价格），按上市日前 90 天到上市日在「首发」类公告里找，
排除附录、摘要、提示性公告、英文版。页面文本缓存成 JSON，检索用 jieba 分词 + BM25（招股书 400–700 页，按页检索足够）。
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import httpx
import pandas as pd
import pymupdf

from cyron_veritas.config import ROOT
from cyron_veritas.rag.filings import _clean_pages

DIR = ROOT / "data" / "prospectus"


def listing_date(code: str) -> pd.Timestamp:
    """上市首日（腾讯日线第一根 K 线）。"""
    from cyron_veritas.pe import market

    return market.bars(code).index[0]


def find(code: str) -> dict:
    import akshare as ak

    ld = listing_date(code)
    df = ak.stock_zh_a_disclosure_report_cninfo(symbol=code, market="沪深京", category="首发",
                                                start_date=f"{ld - pd.Timedelta(days=90):%Y%m%d}", end_date=f"{ld:%Y%m%d}")
    cand = df[df["公告标题"].str.contains("招股说明书") & ~df["公告标题"].str.contains("附录|摘要|提示性|英文|意向书|更正|补充")]
    if cand.empty:
        raise LookupError(f"巨潮资讯网找不到 {code} 的招股说明书")
    row = cand.sort_values("公告时间").iloc[-1]
    m = re.search(r"announcementId=(\d+)", row["公告链接"])
    date = str(row["公告时间"])[:10]
    return {"code": code, "company": row["简称"], "title": row["公告标题"], "date": date, "listing_date": str(ld.date()),
            "pdf_url": f"http://static.cninfo.com.cn/finalpage/{date}/{m.group(1)}.PDF", "page_url": row["公告链接"]}


def download(code: str) -> tuple[Path, dict]:
    DIR.mkdir(parents=True, exist_ok=True)
    path, mp = DIR / f"{code}.pdf", DIR / f"{code}.meta.json"
    if mp.exists() and path.exists() and path.stat().st_size > 500_000:
        return path, json.loads(mp.read_text(encoding="utf-8"))
    meta = find(code)
    with httpx.stream("GET", meta["pdf_url"], timeout=300, follow_redirects=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for b in r.iter_bytes():
                f.write(b)
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return path, meta


@lru_cache(maxsize=16)
def pages(code: str) -> list[str]:
    """逐页文本（下标 0 = 第 1 页），缓存为 JSON。"""
    cp = DIR / f"{code}.pages.json"
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))
    path, _ = download(code)
    txt = ["\n".join(lines) for lines in _clean_pages(pymupdf.open(path))]
    cp.write_text(json.dumps(txt, ensure_ascii=False), encoding="utf-8")
    return txt


@lru_cache(maxsize=16)
def _bm25(code: str):
    import jieba
    from rank_bm25 import BM25Okapi

    toks = [[w for w in jieba.lcut(p) if w.strip()] for p in pages(code)]
    return BM25Okapi([t or ["空"] for t in toks])


def search(code: str, query: str, top: int = 6, must: tuple[str, ...] = ()) -> list[dict]:
    """BM25 检索页面；must 里的词必须全部出现在页面上。返回 [{page, score, text}]（page 从 1 开始）。"""
    import jieba

    ps = pages(code)
    scores = _bm25(code).get_scores([w for w in jieba.lcut(query) if w.strip()])
    order = sorted(range(len(ps)), key=lambda i: -scores[i])
    out = []
    for i in order:
        if must and not all(m in ps[i] for m in must):
            continue
        out.append({"page": i + 1, "score": float(scores[i]), "text": ps[i]})
        if len(out) >= top:
            break
    return out


def norm(s: str) -> str:
    """校验用：去掉空白和千分位逗号。"""
    return re.sub(r"[\s,，]", "", str(s))


def on_page(code: str, page: int, value) -> bool:
    """value（数字或文字）是否出现在第 page 页（允许 ±1 页，表格常跨页）。"""
    ps = pages(code)
    v = norm(value)
    if not v:
        return False
    cands = {v}
    try:
        x = float(v)
        for y in (x, x * 10000, x / 10000):          # 万元 / 元、万股 / 股 两种写法都认
            cands |= {f"{y:.2f}", f"{y:.4f}".rstrip("0").rstrip("."), f"{y:.0f}" if abs(y - round(y)) < 1e-6 else "", f"{y:g}"}
        cands.discard("")
        cands = {c for c in cands if len(c.replace(".", "").lstrip("0")) >= 2}
        if len(f"{x:g}".replace(".", "").lstrip("0")) < 2:
            return True                                  # 1 元、0 这类数字在页面上到处都是，核对没有意义，不算不通过
    except ValueError:
        pass
    for p in (page - 1, page, page + 1):
        if 1 <= p <= len(ps) and any(c and c in norm(ps[p - 1]) for c in cands):
            return True
    return False
