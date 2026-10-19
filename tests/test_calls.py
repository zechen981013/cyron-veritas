"""业绩会纪要：问答切分、原文引用校验（不联网、不调模型）。"""
from cyron_veritas.calls import analyze as A
from cyron_veritas.calls import source as S

CONTENT = """1. 公司回购的进度如何？
本年度回购金额合计为82.06 亿元。
2. 美的现在的商业及工业解决方案收入占比是25.64，未来几年发展预期是占比到多少？有什么机会和困难呢？尊敬的投资者，您好。商业及工业解决方案业务2025 年收入规模突破1,200 亿元。
3. 2B 业务通常现金流较差，如何规避这个风险？
尊敬的投资者，您好。公司持续做好汇率对冲与成本管控。"""


def test_split_qa_keeps_multi_question_together():
    qa = S.split_qa(CONTENT)
    assert len(qa) == 3
    assert qa[1]["q"].endswith("有什么机会和困难呢？")           # 多个问号的问题并在一起
    assert qa[1]["a"].startswith("尊敬的投资者")
    assert "82.06" in qa[0]["a"]


def test_quote_ok_tolerates_whitespace_and_punctuation():
    src = "商业及工业解决方案业务2025 年收入规模突破1,200 亿元。"
    assert A.quote_ok("商业及工业解决方案业务2025年收入规模突破1200亿元", src)
    assert not A.quote_ok("收入规模突破1,300 亿元", src)            # 改了数字就不算原文


def test_check_flags_bad_quotes_and_numbers():
    src = S.split_qa(CONTENT)
    text = "\n".join(f"{x['q']}{x['a']}" for x in src)
    summary = {"overview": "回购 82.06 亿元，B 端收入 1,200 亿元，增长 30%。",
               "guidance": [{"metric": "回购", "value": "82.06 亿元", "quote": "本年度回购金额合计为82.06 亿元"}],
               "kpis": [{"name": "B 端收入", "value": "1,200 亿元", "quote": "收入规模突破1,500 亿元"}],
               "themes": [], "risks": [], "new_info": []}
    c = A.check(summary, text)
    assert c["items"] == 2 and c["quotes_ok"] == 1 and not c["ok"]
    assert "30" in c["unknown_numbers"]                               # 原文里没有 30%
    assert summary["kpis"][0]["quote_ok"] is False
