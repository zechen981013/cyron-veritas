"""年报获取与解析：巨潮资讯网（cninfo）下载 PDF → 按页抽取文本 → 识别章节 → 切块（每块带页码）。

切块策略（年报的特点：长、结构规整、表格多）：
  - 先按页抽文本，去掉页眉页脚和页码，跳过目录页；
  - 用「第X节」识别一级章节，用「一、」「（一）」识别二级标题，章节信息跟着页往后传；
  - 同一章节内按段落拼接到约 600 字，块之间重叠约 100 字，跨页时记录起止页；
  - 每块带 metadata：公司、年份、章节、小标题、起止页 → 回答时能精确引用到页。
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import pymupdf

from cyron_veritas.config import ROOT

REPORT_DIR = ROOT / "data" / "reports"
SECTION_RE = re.compile(r"^\s*第([一二三四五六七八九十]+)节\s*(\S.*?)\s*$")
SUB_RE = re.compile(r"^\s*(?:[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）)\s*\S{2,40}\s*$")
TOC_RE = re.compile(r"\.{6,}|…{3,}")
CHUNK, OVERLAP = 600, 100
PARSER_VERSION = 3  # 改了切块逻辑就加 1，已入库的年报会重新入库


@dataclass
class Chunk:
    id: str
    ticker: str
    company: str
    year: int
    section: str
    subsection: str
    page_start: int
    page_end: int
    text: str


def _join_lines(text: str) -> str:
    """PDF 的硬换行：正文被折行时上一行接近满行（≥ 30 字）且不以句末标点结尾，拼回去；
    表格单元格（短行、数字）保留换行，否则两列数字会粘在一起。"""
    lines = text.split("\n")
    out = [lines[0]] if lines else []
    for prev_raw, ln in zip(lines, lines[1:]):
        # 用「原始」上一行判断是否被折行（不能用已经拼接过的行，否则后面的短单元格会被一路拼上去）
        wrapped = len(prev_raw) >= 30 and not re.search(r"[。；：！？:;]$", prev_raw)
        is_cell = re.fullmatch(r"[\d,.\-()%（）\s]+", ln) is not None
        if wrapped and not is_cell and not re.match(r"^(（|[一二三四五六七八九十]+、|\d+[、.])", ln):
            out[-1] = out[-1] + ln
        else:
            out.append(ln)
    return "\n".join(out)


def find_report(ticker: str, year: int) -> dict:
    """在巨潮资讯网找某年年报（排除摘要、英文版、更正前版本）。"""
    import akshare as ak

    df = ak.stock_zh_a_disclosure_report_cninfo(symbol=ticker, market="沪深京", category="年报",
                                                start_date=f"{year + 1}0101", end_date=f"{year + 1}1231")
    title = f"{year}年度报告"  # 标题有「2025年年度报告」和「XX公司2025年度报告」两种写法
    cand = df[df["公告标题"].str.contains(rf"{year}年(?:年)?度报告") & ~df["公告标题"].str.contains("摘要|英文|已取消|半年度")]
    if cand.empty:
        raise LookupError(f"巨潮资讯网找不到 {ticker} 的{title}")
    row = cand.sort_values("公告时间").iloc[-1]  # 有更正版时取最新
    m = re.search(r"announcementId=(\d+)", row["公告链接"])
    date = str(row["公告时间"])[:10]
    return {"ticker": ticker, "company": row["简称"], "year": year, "title": row["公告标题"], "date": date,
            "pdf_url": f"http://static.cninfo.com.cn/finalpage/{date}/{m.group(1)}.PDF", "page_url": row["公告链接"]}


def download(ticker: str, year: int) -> tuple[Path, dict]:
    meta = find_report(ticker, year)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{ticker}_{year}.pdf"
    if not path.exists() or path.stat().st_size < 100_000:
        with httpx.stream("GET", meta["pdf_url"], timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for b in r.iter_bytes():
                    f.write(b)
    return path, meta


def _clean_pages(doc) -> list[list[str]]:
    pages = [[ln.strip() for ln in p.get_text().splitlines() if ln.strip()] for p in doc]
    # 页眉页脚：在超过 30% 的页面里重复出现的短行
    freq = Counter(ln for lines in pages for ln in set(lines[:2] + lines[-2:]) if len(ln) < 40)
    noise = {ln for ln, c in freq.items() if c >= max(3, 0.3 * len(pages))}  # 至少在 3 页重复出现，短文档不会误删
    out = []
    for lines in pages:
        lines = [ln for ln in lines if ln not in noise and not re.fullmatch(r"[\d\s/\-第页共]+", ln)]
        out.append(lines)
    return out


def parse(path: Path, ticker: str, company: str, year: int) -> list[Chunk]:
    doc = pymupdf.open(path)
    pages = _clean_pages(doc)
    section, sub = "封面与提示", ""
    units: list[tuple[int, str, str, str]] = []  # (页码, 章节, 小标题, 段落)
    for pno, lines in enumerate(pages, start=1):
        if sum(bool(TOC_RE.search(ln)) for ln in lines) >= 3:  # 目录页
            continue
        buf = []
        for ln in lines:
            m = SECTION_RE.match(ln)
            if m and not TOC_RE.search(ln):
                if buf:
                    units.append((pno, section, sub, "\n".join(buf)))
                    buf = []
                section, sub = f"第{m.group(1)}节 {m.group(2)}", ""
                continue
            if SUB_RE.match(ln):
                if buf:
                    units.append((pno, section, sub, "\n".join(buf)))
                    buf = []
                sub = ln
            buf.append(ln)
        if buf:
            units.append((pno, section, sub, "\n".join(buf)))

    chunks: list[Chunk] = []
    cur, start, cur_sec, cur_sub, end = "", 0, None, "", 0

    def flush():
        nonlocal cur
        if cur.strip():
            chunks.append(Chunk(id=f"{ticker}_{year}_{len(chunks):05d}", ticker=ticker, company=company, year=year,
                                section=cur_sec, subsection=cur_sub, page_start=start, page_end=end, text=_join_lines(cur.strip())))

    for pno, sec, sb, text in units:
        if cur_sec is not None and (sec != cur_sec or len(cur) + len(text) > CHUNK):
            flush()
            tail = cur[-OVERLAP:] if sec == cur_sec else ""
            cur, start = tail, (end if tail else pno)
        if not cur:
            start = pno
        cur_sec, cur_sub, end = sec, sb or cur_sub, pno
        cur = (cur + "\n" + text) if cur else text
        while len(cur) > CHUNK * 2:  # 超长段落（大表格）强制切开
            head, cur = cur[:CHUNK], cur[CHUNK - OVERLAP:]
            chunks.append(Chunk(id=f"{ticker}_{year}_{len(chunks):05d}", ticker=ticker, company=company, year=year,
                                section=cur_sec, subsection=cur_sub, page_start=start, page_end=pno, text=_join_lines(head)))
            start = pno
    flush()
    return chunks


def page_text(path: Path, page: int) -> str:
    """取原文某一页（引用校验和前端「查看原文」用）。"""
    return pymupdf.open(path)[page - 1].get_text()


def chunk_dict(c: Chunk) -> dict:
    return asdict(c)
