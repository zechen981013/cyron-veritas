"""日报流程的路由：校验通过 → 出报告；不通过 → 重写一次；还不通过或模型挂了 → 模板。"""
from analyst_desk.risk import report_graph as G


def test_routes():
    assert G.route_after_verify({"check": {"ok": True}, "attempts": 1}) == "render"
    assert G.route_after_verify({"check": {"ok": False}, "attempts": 1}) == "narrate"
    assert G.route_after_verify({"check": {"ok": False}, "attempts": 2}) == "fallback"
    assert G.route_after_verify({"check": {"ok": False}, "attempts": 1, "llm_error": "timeout"}) == "fallback"


def test_fallback_text_is_fully_traced():
    from tests.test_verify_limits import _facts
    from analyst_desk.risk import limits as L
    f = _facts(total_value=1e6, var95_amount=2e4, stress={"x": {"type": "假设", "period": "", "pnl_pct": -0.1, "pnl_amount": -1e5}},
               risk_contribution={"A": 0.6, "B": 0.4})
    lim = L.check(f)
    table = "\n".join([f"总市值 {G.V.wan(1e6)} VaR {G.V.pct(0.02, 2)} 回撤 {G.V.pct(0.05)}", "A 60.0% 压力 -10.0%"]
                      + [f"{r['label']} {G.V.pct(r['value'])}" for r in lim])
    out = G.node_fallback({"facts": f, "limits": lim, "facts_table": table})
    assert out["check"]["ok"], out["check"]
