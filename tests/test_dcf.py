import pytest

from analyst_desk.valuation import dcf as D


def _inputs(**kw):
    base = dict(base_year=2025, base_revenue=100.0, base_nwc=0.0, revenue_growth=[0.0] * 5, ebit_margin=[0.2] * 5,
                tax_rate=0.25, da_pct=0.05, capex_pct=0.05, nwc_pct=0.0, wacc=0.10, terminal_growth=0.0,
                debt=20.0, cash=10.0, minority=0.0, shares=10.0)
    base.update(kw)
    return D.DCFInputs(**base)


def test_zero_growth_is_perpetuity():
    # 零增长、D&A = 资本开支、营运资本不变 → FCFF = EBIT×(1−t) = 15，企业价值 = 15 / 10% = 150
    v = D.value(_inputs())
    assert v["enterprise_value"] == pytest.approx(150.0)
    assert v["per_share"] == pytest.approx((150 - 20 + 10) / 10)


def test_nwc_change_reduces_fcff():
    p = D.project(_inputs(revenue_growth=[0.1] * 5, nwc_pct=0.2, base_nwc=20.0))
    assert p["d_nwc"].iloc[0] == pytest.approx(110 * 0.2 - 20)


def test_validation_rejects_bad_terminal_growth():
    assert any("永续增长率" in e for e in _inputs(terminal_growth=0.095).validate())
    assert _inputs().validate() == []


def test_sensitivity_center_equals_base():
    x = _inputs(terminal_growth=0.02)
    s = D.sensitivity(x)
    assert s.loc[0.10, 0.02] == pytest.approx(D.value(x)["per_share"])


def test_wacc_weights():
    w = D.wacc(0.02, 1.0, 0.06, 0.04, 0.25, 900, 100)
    assert w["wacc"] == pytest.approx(0.9 * 0.08 + 0.1 * 0.03)
