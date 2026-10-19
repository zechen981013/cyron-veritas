"""Excel 公式模型和 Python 引擎必须算出同一个每股价值。"""
import pytest

from cyron_veritas.valuation import dcf as D
from cyron_veritas.valuation.excel import build_workbook

formulas = pytest.importorskip("formulas")


def test_excel_matches_python(tmp_path):
    x = D.DCFInputs(base_year=2025, base_revenue=4585.0, base_nwc=-146.7, revenue_growth=[0.07, 0.06, 0.05, 0.045, 0.04],
                    ebit_margin=[0.10, 0.10, 0.101, 0.102, 0.102], tax_rate=0.164, da_pct=0.02, capex_pct=0.02, nwc_pct=-0.032,
                    wacc=0.0645, terminal_growth=0.025, debt=674.8, cash=859.6, minority=132.0, shares=75.97,
                    extra_cash=2000.0, notes={"extra_cash": "年报 p.208 一年内到期固定收益类产品"})
    path = build_workbook(x, "测试公司", 86.54, path=tmp_path / "m.xlsx")
    sol = formulas.ExcelModel().loads(str(path)).finish().calculate()
    get = lambda ref: next(v.value[0][0] for k, v in sol.items() if k.upper().endswith(ref))
    py = D.value(x)
    assert get("估值'!B11") == pytest.approx(py["per_share"], rel=1e-9)
    assert get("估值'!B6") == pytest.approx(py["enterprise_value"], rel=1e-9)
    assert get("敏感性'!D7") == pytest.approx(py["per_share"], rel=1e-9)   # 敏感性表中心格
    assert get("敏感性'!B5") == pytest.approx(D.value(x, 0.0545, 0.015)["per_share"], rel=1e-9)


def test_fin_excel_matches_python(tmp_path):
    from cyron_veritas.valuation import fin as V
    from cyron_veritas.valuation.excel import build_fin_workbook

    x = V.FinInputs(base_year=2025, bvps=43.43, eps=5.95, roe=0.13, ke=0.085, g=0.03, g1=0.04, payout1=0.35)
    path = build_fin_workbook(x, "测试银行", 43.0, path=tmp_path / "f.xlsx")
    sol = formulas.ExcelModel().loads(str(path)).finish().calculate()
    get = lambda ref: next(v.value[0][0] for k, v in sol.items() if k.upper().endswith(ref))
    assert get("估值'!B4") == pytest.approx(V.pb_roe(x)["per_share"], rel=1e-9)
    assert get("估值'!B5") == pytest.approx(V.ddm(x)["per_share"], rel=1e-9)
    assert get("敏感性'!D7") == pytest.approx(V.pb_roe(x)["per_share"], rel=1e-9)
