"""交叉验证：自研的估值、风险、利润质量计算 vs 业界开源库（FinanceToolkit、empyrical），同样的输入必须得到同样的结果。"""
import numpy as np
import pandas as pd
import pytest

from cyron_veritas.risk import metrics as M
from cyron_veritas.valuation import dcf as D
from cyron_veritas.valuation import fin as FV
from cyron_veritas.valuation import quality as Q

ftk_intrinsic = pytest.importorskip("financetoolkit.models.intrinsic_model")
from financetoolkit.models import altman_model, dupont_model, piotroski_model  # noqa: E402
from financetoolkit.performance import performance_model  # noqa: E402

empyrical = pytest.importorskip("empyrical")


# ---------------- 估值 ----------------
def test_dcf_matches_financetoolkit_intrinsic_value():
    # 营运资本为 0 时，我的 FCFF = 收入 × (EBIT 率 ×(1−t) + 折旧率 − 资本开支率)，按收入增速增长，和 FinanceToolkit 的「现金流按固定增速增长」等价
    g, wacc, tg = 0.06, 0.09, 0.02
    x = D.DCFInputs(base_year=2025, base_revenue=1000.0, base_nwc=0.0, revenue_growth=[g] * 5, ebit_margin=[0.15] * 5, tax_rate=0.2,
                    da_pct=0.03, capex_pct=0.04, nwc_pct=0.0, wacc=wacc, terminal_growth=tg, debt=300.0, cash=120.0, minority=0.0, shares=50.0)
    base_cf = 1000.0 * (0.15 * 0.8 + 0.03 - 0.04)
    ftk = ftk_intrinsic.get_intrinsic_value(cash_flow=base_cf, growth_rate=g, perpetual_growth_rate=tg, weighted_average_cost_of_capital=wacc,
                                            cash_and_cash_equivalents=120.0, total_debt=300.0, shares_outstanding=50.0, periods=5)
    ftk_val = float(ftk.loc["Intrinsic Value"].iloc[0])
    assert D.value(x)["per_share"] == pytest.approx(ftk_val, rel=1e-9)


def test_ddm_matches_financetoolkit_two_stage():
    # 第一阶段派息率取稳态值（1 − g/ROE）时，我的两阶段 DDM 与 FinanceToolkit 的两阶段股利模型一致
    roe, g = 0.13, 0.03
    p = 1 - g / roe
    x = FV.FinInputs(base_year=2025, bvps=40.0, eps=5.2, roe=roe, ke=0.10, g=g, g1=0.06, payout1=p)
    ftk = ftk_intrinsic.get_two_stage_dividend_discount_model(dividends_per_share=5.2 * p, rate_of_return=0.10, high_growth_rate=0.06,
                                                              stable_growth_rate=g, high_growth_periods=5)
    assert FV.ddm(x)["per_share"] == pytest.approx(float(ftk.loc["Intrinsic Value"].iloc[0]), rel=1e-9)


def test_pb_roe_vs_gordon_convention():
    # PB-ROE 用「当期净资产 × ROE」作下一年利润，戈登模型用「本期股利 × (1+g)」：股利取 E0·p/(1+g) 时两者相等
    x = FV.FinInputs(base_year=2025, bvps=40.0, eps=5.2, roe=0.13, ke=0.10, g=0.03, g1=0.03, payout1=0.3)
    e1 = x.bvps * x.roe
    gordon = ftk_intrinsic.get_gorden_growth_model(dividends_per_share=e1 * (1 - x.g / x.roe) / (1 + x.g), rate_of_return=x.ke, growth_rate=x.g)
    assert FV.pb_roe(x)["per_share"] == pytest.approx(gordon, rel=1e-9)


def test_capm_matches_financetoolkit():
    rf, beta, mkt = 0.0165, 0.87, 0.0765  # 市场回报 = rf + ERP 6%
    ours = D.wacc(rf, beta, mkt - rf, 0.03, 0.2, 900, 100)["cost_of_equity"]
    ftk = performance_model.get_capital_asset_pricing_model(risk_free_rate=rf, beta=beta, benchmark_returns=mkt)
    assert ours == pytest.approx(float(ftk), rel=1e-12)


def test_reverse_dcf_recovers_growth():
    x = D.DCFInputs(base_year=2025, base_revenue=1000.0, base_nwc=0.0, revenue_growth=[0.07] * 5, ebit_margin=[0.12] * 5, tax_rate=0.2,
                    da_pct=0.03, capex_pct=0.04, nwc_pct=0.05, wacc=0.09, terminal_growth=0.02, debt=100, cash=50, minority=0, shares=40)
    price = D.value(x)["per_share"]
    assert D.implied_growth(x, price) == pytest.approx(0.07, abs=1e-5)


# ---------------- 风险 ----------------
def _rets(n=750, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    m = pd.Series(rng.normal(0.0003, 0.012, n), index=idx)
    r = 0.8 * m + pd.Series(rng.normal(0, 0.008, n), index=idx)
    return r, m


def test_risk_metrics_match_empyrical():
    r, m = _rets()
    assert M.historical_var(r, 0.95) == pytest.approx(-empyrical.value_at_risk(r, cutoff=0.05), rel=1e-9)
    assert M.historical_cvar(r, 0.95) == pytest.approx(-empyrical.conditional_value_at_risk(r, cutoff=0.05), rel=1e-9)
    assert M.max_drawdown(r)["max_drawdown"] == pytest.approx(-empyrical.max_drawdown(r), rel=1e-9)
    assert M.annualized_vol(r) == pytest.approx(empyrical.annual_volatility(r), rel=1e-9)
    assert M.beta(r, m) == pytest.approx(empyrical.beta(r, m), rel=1e-9)


# ---------------- 利润质量 ----------------
def _table():
    years = [2021, 2022, 2023, 2024, 2025]
    t = pd.DataFrame(index=years)
    t["revenue"] = [100, 110, 125, 130, 150.0]
    t["cogs"] = [70, 76, 85, 90, 100.0]
    t["gross_profit"] = t["revenue"] - t["cogs"]
    t["ebit"] = [12, 13, 16, 15, 19.0]
    t["pretax_profit"] = [11, 12.5, 15, 14, 18.0]
    t["net_profit"] = [8.5, 9.5, 11.5, 10.5, 14.0]
    t["cfo"] = [10, 9, 14, 12, 16.0]
    t["total_assets"] = [200, 215, 230, 240, 260.0]
    t["total_equity"] = [90, 98, 106, 112, 122.0]
    t["total_liab"] = t["total_assets"] - t["total_equity"]
    t["current_assets"] = [90, 95, 104, 108, 118.0]
    t["current_liab"] = [70, 72, 78, 82, 86.0]
    t["long_term_debt"] = [30, 32, 31, 29, 28.0]
    t["retained_earnings"] = [40, 47, 55, 62, 71.0]
    t["shares"] = [10, 10, 10, 10.5, 10.5]
    t["receivables"] = [20, 23, 25, 28, 30.0]
    t["fixed_assets"] = [60, 62, 65, 66, 70.0]
    t["da"] = [5, 5.5, 6, 6.2, 6.8]
    t["sga"] = [10, 11, 12, 13, 14.0]
    return t


def _wide(s: pd.Series) -> pd.DataFrame:
    """FinanceToolkit 的口径：行 = 股票，列 = 期间。"""
    return pd.DataFrame([s.values], index=["X"], columns=s.index)


def test_dupont_matches_financetoolkit():
    t = _table()
    ours = Q.dupont(t)
    avg_a, avg_e = (t.total_assets + t.total_assets.shift()) / 2, (t.total_equity + t.total_equity.shift()) / 2
    ftk3 = dupont_model.get_dupont_analysis(t.net_profit, t.revenue, avg_a, avg_e)
    ftk5 = dupont_model.get_extended_dupont_analysis(t.ebit, t.pretax_profit, t.net_profit, t.revenue, avg_a, avg_e)
    roe3 = ftk3.loc["Return on Equity"] if "Return on Equity" in ftk3.index else ftk3["Return on Equity"]
    roe5 = ftk5.loc["Return on Equity"] if "Return on Equity" in ftk5.index else ftk5["Return on Equity"]
    pd.testing.assert_series_equal(ours["roe_3step"].dropna(), pd.Series(roe3, dtype=float).dropna(), check_names=False, rtol=1e-9)
    pd.testing.assert_series_equal(ours["roe_5step"].dropna(), pd.Series(roe5, dtype=float).dropna(), check_names=False, rtol=1e-9)


def test_altman_matches_financetoolkit():
    t = _table()
    mcap = 300.0
    ours = Q.altman(t, market_cap=mcap)
    last = t.index[-1]
    r = t.loc[last]
    ftk = altman_model.get_altman_z_score(
        altman_model.get_working_capital_to_total_assets_ratio(r.current_assets - r.current_liab, r.total_assets),
        altman_model.get_retained_earnings_to_total_assets_ratio(r.retained_earnings, r.total_assets),
        altman_model.get_earnings_before_interest_and_taxes_to_total_assets_ratio(r.ebit, r.total_assets),
        altman_model.get_market_value_of_equity_to_book_value_of_total_liabilities_ratio(mcap, r.total_liab),
        altman_model.get_sales_to_total_assets_ratio(r.revenue, r.total_assets))
    assert ours.loc[last, "z"] == pytest.approx(float(ftk), rel=1e-12)


def test_piotroski_criteria_match_financetoolkit():
    t = _table()
    ours = Q.piotroski(t)
    w = {c: _wide(t[c]) for c in t.columns}
    avg_a = (w["total_assets"] + w["total_assets"].shift(1, axis=1)) / 2
    ftk = {
        "roa_positive": piotroski_model.get_return_on_assets_criteria(w["net_profit"], avg_a),
        "accruals": piotroski_model.get_accruals_criteria(w["net_profit"], avg_a, w["cfo"], w["total_assets"]),
        "leverage_falling": piotroski_model.get_change_in_leverage_criteria(w["long_term_debt"], avg_a),
        "current_ratio_rising": piotroski_model.get_change_in_current_ratio_criteria(w["current_assets"], w["current_liab"]),
        "gross_margin_rising": piotroski_model.get_gross_margin_criteria(w["revenue"], w["cogs"]),
        "turnover_rising": piotroski_model.get_asset_turnover_ratio_criteria(w["revenue"], avg_a),
    }
    for k, f in ftk.items():
        assert list(ours[k].iloc[2:].astype(int)) == list(f.iloc[0, 2:].astype(int)), k   # 前两年因平均资产缺失不比较


def test_credibility_grade_rules():
    assert Q.credibility_grade(7, -2.5, 0.02, 1.2)[0] == "A"
    assert Q.credibility_grade(7, -1.5, 0.02, 1.2)[0] == "C"          # Beneish 红旗扣 2 级
    assert Q.credibility_grade(2, -1.5, 0.15, 0.5)[0] == "D"
