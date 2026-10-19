"""Excel 公式模型和 Python 引擎必须算出同一个每股价值。"""
import pytest

from analyst_desk.valuation import dcf as D
from analyst_desk.valuation.excel import build_workbook

formulas = pytest.importorskip("formulas")


def test_excel_matches_python(tmp_path):
    x = D.DCFInputs(base_year=2025, base_revenue=4585.0, base_nwc=-146.7, revenue_growth=[0.07, 0.06, 0.05, 0.045, 0.04],
                    ebit_margin=[0.10, 0.10, 0.101, 0.102, 0.102], tax_rate=0.164, da_pct=0.02, capex_pct=0.02, nwc_pct=-0.032,
                    wacc=0.0645, terminal_growth=0.025, debt=674.8, cash=859.6, minority=132.0, shares=75.97)
    path = build_workbook(x, "测试公司", 86.54, path=tmp_path / "m.xlsx")
    sol = formulas.ExcelModel().loads(str(path)).finish().calculate()
    get = lambda ref: next(v.value[0][0] for k, v in sol.items() if k.upper().endswith(ref))
    py = D.value(x)
    assert get("估值'!B11") == pytest.approx(py["per_share"], rel=1e-9)
    assert get("估值'!B6") == pytest.approx(py["enterprise_value"], rel=1e-9)
    assert get("敏感性'!D7") == pytest.approx(py["per_share"], rel=1e-9)   # 敏感性表中心格
    assert get("敏感性'!B5") == pytest.approx(D.value(x, 0.0545, 0.015)["per_share"], rel=1e-9)
