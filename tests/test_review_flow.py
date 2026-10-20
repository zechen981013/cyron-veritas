"""估值假设人工审批（LangGraph interrupt）：暂停 → 批准 / 驳回 → 定稿。用假 Agent 和假数据，不联网、不调模型。"""
import pytest
from langchain_core.messages import AIMessage

from cyron_veritas.research import agent as A
from cyron_veritas.valuation import dcf as D


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "financials_summary", lambda t: {"company": "测试公司", "latest_year": 2025, "is_financial": False, "market": "A"})
    import cyron_veritas.rag.index as I
    monkeypatch.setattr(I, "manifest", lambda: {"000001_2025": {}})

    class FakeAgent:
        def __init__(self, ledger):
            self.ledger = ledger

        def invoke(self, *_a, **_k):
            x = D.DCFInputs(base_year=2025, base_revenue=100, base_nwc=0, revenue_growth=[0.05] * 5, ebit_margin=[0.2] * 5, tax_rate=0.25,
                            da_pct=0.05, capex_pct=0.05, nwc_pct=0, wacc=0.09, terminal_growth=0.02, debt=10, cash=5, minority=0, shares=10,
                            notes={"wacc": "CAPM [E1]"})
            self.ledger.add("run_dcf", {"scenario": "base"}, {"per_share": 12.3, "price": 10.0})
            self.ledger.dcf_runs.append({"scenario": "base", "inputs": x, "result": {"price": 10.0, "per_share": 12.3}, "per_share": 12.3})
            return {"messages": [AIMessage(content="# 测试公司研究备忘录\n基准每股 12.3 元 [E1]。")]}

    monkeypatch.setattr(A, "build", lambda ledger, model=None, financial=False, market="A": FakeAgent(ledger))
    monkeypatch.setattr(A, "build_workbook", lambda *a, path=None, **k: path)
    import cyron_veritas.data.financials as F
    monkeypatch.setattr(F, "annual", lambda t: {"table": None})
    return tmp_path


def test_pause_then_approve(fake):
    r = A.start("000001", review=True, out_root=fake)
    assert r["status"] == "review"
    assert r["proposal"]["assumptions"]["wacc"] == 0.09 and r["proposal"]["per_share"] == 12.3
    done = A.resume(r["run_id"], {"action": "approve", "reviewer": "张三"})
    assert done["status"] == "done" and "已批准" in done["memo"] and "张三" in done["memo"]
    assert "excel" in done["paths"]


def test_reject_skips_excel(fake):
    r = A.start("000001", review=True, out_root=fake)
    done = A.resume(r["run_id"], {"action": "reject", "comment": "WACC 偏低"})
    assert "已驳回" in done["memo"] and "WACC 偏低" in done["memo"] and "excel" not in done["paths"]


def test_review_event_sent_once(fake):
    events = []
    r = A.start("000001", review=True, out_root=fake, on_event=events.append)
    A.resume(r["run_id"], {"action": "approve"})
    assert sum(e["type"] == "review" for e in events) == 1      # 恢复时节点会重跑，事件不能重复


def test_without_review_marks_unreviewed(fake):
    done = A.run("000001", out_root=fake)
    assert done["status"] == "done" and "未开启人工审批" in done["memo"] and "附录：估值假设审批" in done["memo"]
