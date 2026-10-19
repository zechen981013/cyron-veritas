"""日报流程的路由：校验通过 → 出报告；不通过 → 重写一次；还不通过或模型挂了 → 模板。"""
from cyron_veritas.risk import report_graph as G


def test_routes():
    assert G.route_after_verify({"check": {"ok": True}, "attempts": 1}) == "render"
    assert G.route_after_verify({"check": {"ok": False}, "attempts": 1}) == "narrate"
    assert G.route_after_verify({"check": {"ok": False}, "attempts": 2}) == "fallback"
    assert G.route_after_verify({"check": {"ok": False}, "attempts": 1, "llm_error": "timeout"}) == "fallback"


def _state():
    from tests.test_verify_limits import _facts
    from cyron_veritas.risk import limits as L
    f = _facts(total_value=1e6, n_positions=3, accounts=["券商A"], fx={"USD": 7.1, "HKD": 0.91}, vol_ann=0.15, beta_csi300=0.8,
               var95_amount=2e4, var95_param=0.021, var99_hist=0.03, var99_amount=3e4, cvar95_hist=0.025, cvar95_amount=2.5e4,
               max_drawdown=0.2, mdd_peak="2025-01-01", mdd_trough="2025-03-01",
               concentration={"top1": 0.30, "top1_name": "A", "top5": 0.60, "effective_n": 3.2},
               var_backtest={"days": 250, "exceptions": 10, "expected": 12.5, "p_value": 0.45},
               top_corr_pairs=[("A", "B", 0.58)], stress={"2020 新冠暴跌": {"type": "历史", "period": "", "pnl_pct": -0.2, "pnl_amount": -2e5}},
               risk_contribution={"A": 0.6, "B": 0.4})
    lim = L.check(f)
    m = G.fact_map(f, lim)
    return {"facts": f, "limits": lim, "fact_map": m, "facts_table": G.keyed_table(m, lim),
            "names_text": " ".join(list(f["stress"]) + [r["label"] for r in lim] + list(f["weights"]))}


def test_fallback_text_passes_structured_check():
    out = G.node_fallback(_state())
    assert out["check"]["ok"], out["check"]


def test_placeholders_bind_numbers_to_objects():
    st = _state()
    text = "A 权重 {{w.A}}，贡献 {{rc.A}} 的波动；2020 新冠暴跌情景 {{stress.2020 新冠暴跌}}。"
    assert G.check_placeholders(text, st["fact_map"], st["names_text"])["ok"]
    assert G.fill("A 权重 {{w.A}}", st["fact_map"]) == "A 权重 30.0%"
    bad = G.check_placeholders("A 权重 30%，B {{w.不存在}}", st["fact_map"], st["names_text"])
    assert not bad["ok"] and bad["raw_numbers"] == ["30"] and bad["unknown_keys"] == ["w.不存在"]
    html = G.fill("A {{w.A}} <b>", st["fact_map"], html=True)
    assert 'title="w.A：A 权重"' in html and "&lt;b&gt;" in html          # 悬停说明 + 文本转义
