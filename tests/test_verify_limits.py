from analyst_desk import verify as V
from analyst_desk.risk import limits as L


def test_numbers_must_come_from_facts():
    allowed = V.allowed_set("英伟达 权重 26.7%，VaR 1.53%，总市值 215.8 万元")
    assert V.check("英伟达占 26.7%，VaR 为 1.53%。", allowed)["ok"]
    bad = V.check("英伟达占 27%，前 3 大持仓。", allowed)
    assert not bad["ok"] and bad["unknown_numbers"] == ["27"]          # 3 是小整数，不要求出处


def test_advice_words_blocked():
    allowed = V.allowed_set("26.7%")
    assert V.check("建议减仓英伟达", allowed)["advice_words"] == ["减仓"]


def _facts(**kw):
    f = {"weights": {"A": 0.30, "B": 0.20, "C": 0.10, "现金": 0.40}, "pnl_pct": {"A": 0.1, "B": 0.1, "C": 0.1},
         "by_sector": {"科技": 0.50, "金融": 0.10, "现金": 0.40}, "by_market": {"US": 0.5, "A": 0.1, "CASH": 0.4},
         "by_currency": {"USD": 0.5, "CNY": 0.5}, "concentration": {"top1": 0.30, "top5": 0.60},
         "var95_hist": 0.02, "current_drawdown": 0.05, "cash_weight": 0.40}
    f.update(kw)
    return f


def test_limits_list_every_breaching_name():
    res = {r["key"]: r for r in L.check(_facts())}
    assert res["single_name_max"]["status"] == L.BREACH
    assert res["single_name_max"]["note"] == "A、B"
    assert res["sector_max"]["status"] == L.BREACH
    assert res["var95_1d_max"]["status"] == L.OK
    assert res["top5_max"]["status"] == L.WARN          # 60% > 上限 60% × 预警比例 0.9


def test_cash_min_direction():
    res = {r["key"]: r for r in L.check(_facts(cash_weight=0.01))}
    assert res["cash_min"]["status"] == L.BREACH
