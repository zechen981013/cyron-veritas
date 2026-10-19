"""年报 RAG 的纯逻辑部分：切块、换行拼接、引用与数字校验（不加载模型、不联网）。"""
import pymupdf
import pytest

from cyron_veritas.rag import filings as F
from cyron_veritas.rag import qa


def test_join_lines_keeps_table_cells_apart():
    prose = "公司坚持科技领先、用户直达、数智驱动、全球突破四大战略主轴，推进ToC与\nToB并重发展。"
    assert F._join_lines(prose) == prose.replace("\n", "")
    table = "2025 年12 月31 日  2024 年12 月31 日固定收益类产品(a)\n1,427,134\n20,652,188"
    assert "1,427,134\n20,652,188" in F._join_lines(table)          # 两列数字不能粘在一起


def test_parse_tracks_sections_and_pages(tmp_path):
    doc = pymupdf.open()
    pages = ["目录\n第一节 重要提示 .......... 1\n第二节 公司简介 .......... 2\n第三节 管理层讨论与分析 .......... 3",
             "第二节 公司简介和主要财务指标\n营业收入 4,585 亿元",
             "第三节 管理层讨论与分析\n一、报告期内公司从事的主要业务\n海外收入 1,959 亿元，占比 42.93%。"]
    for t in pages:
        p = doc.new_page()
        p.insert_text((50, 72), t, fontname="china-s", fontsize=10)
    path = tmp_path / "r.pdf"
    doc.save(path)
    chunks = F.parse(path, "000001", "测试公司", 2025)
    secs = {c.section: c for c in chunks}
    assert "第三节 管理层讨论与分析" in secs and secs["第三节 管理层讨论与分析"].page_start == 3
    assert all("......" not in c.text for c in chunks)               # 目录页被跳过
    assert any("42.93%" in c.text for c in chunks)


HITS = [{"page_start": 49, "page_end": 50, "text": "海外 195,947,693 42.93% 169,034,383 41.52%"}]


def test_check_answer_passes_with_valid_citation():
    r = qa.check_answer("海外收入 195,947,693 千元，占比 42.93% [p.49-50]。", HITS, "海外收入多少")
    assert r["ok"] and r["cited_pages"] == [49, 50]


def test_check_answer_flags_wrong_page_and_made_up_number():
    r = qa.check_answer("海外收入约 1,960 亿元 [p.12]。", HITS, "海外收入多少")
    assert not r["ok"] and r["bad_pages"] == [12] and "1960" in r["unknown_numbers"]


def test_check_answer_requires_citation():
    assert not qa.check_answer("海外收入占比 42.93%。", HITS, "q")["ok"]
