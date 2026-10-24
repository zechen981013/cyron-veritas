"""PE：分配瀑布、XIRR / PME / AAP 折扣、投后估值与稀释、员工持股轮次拦截、原文核对、压力测试。全部离线。"""
import math

import numpy as np
import pandas as pd
import pytest

from cyron_veritas.pe import deals as D
from cyron_veritas.pe import metrics as X
from cyron_veritas.pe import prospectus as PR
from cyron_veritas.pe import stress as S
from cyron_veritas.pe.waterfall import Waterfall

T0 = pd.Timestamp("2020-01-01")


def test_waterfall_tiers_textbook_case():
    w = Waterfall()
    w.contribute(T0, 100.0)
    r = w.distribute(T0 + pd.Timedelta(days=365), 200.0)
    assert r["返还实缴"] == pytest.approx(100)
    assert r["门槛收益"] == pytest.approx(8)
    assert r["GP 追补"] == pytest.approx(2)                     # 追补到 GP 拿利润的 20%：8 × 0.25
    assert r["超额收益（LP 80%）"] == pytest.approx(72) and r["超额收益（GP 20%）"] == pytest.approx(18)
    assert r["GP 合计"] == pytest.approx(0.2 * 100)             # GP 合计 = 全部利润 100 的 20%


def test_waterfall_below_hurdle_and_pref_compounds():
    w = Waterfall()
    w.contribute(T0, 100.0)
    r = w.distribute(T0 + pd.Timedelta(days=730), 50.0)
    assert r["返还实缴"] == 50 and r["GP 合计"] == 0
    r = w.distribute(T0 + pd.Timedelta(days=730), 60.0)        # 剩余本金 50 + 两年复利门槛 100 × (1.08² − 1) = 16.64
    assert r["返还实缴"] == pytest.approx(50) and r["门槛收益"] == pytest.approx(10)
    assert w.pref_out == pytest.approx(6.64, abs=1e-6) and r["GP 合计"] == 0


def test_waterfall_hypothetical_does_not_mutate():
    w = Waterfall()
    w.contribute(T0, 100.0)
    h = w.hypothetical(T0 + pd.Timedelta(days=365), 300.0)
    assert h["lp_nav"] + h["accrued_carry"] == pytest.approx(300)
    assert w.capital_out == 100 and w.gp_carry == 0


def test_xirr_pme_and_multiples():
    assert X.xirr([(T0, -100), (T0 + pd.Timedelta(days=365), 110)]) == pytest.approx(0.10, abs=1e-6)
    assert math.isnan(X.xirr([(T0, -100)]))
    flat = pd.Series(1.0, index=pd.date_range("2019-12-01", "2021-12-31"))
    pme = X.ks_pme([(T0, 100)], [(T0 + pd.Timedelta(days=365), 150)], 20, flat, T0 + pd.Timedelta(days=400))
    assert pme == pytest.approx(1.7)                             # 指数不动时 PME = TVPI
    assert X.multiples(100, 150, 20) == {"TVPI": 1.7, "DPI": 1.5, "RVPI": 0.2}


def test_aap_discount():
    assert X.aap_discount(0.6, 0) == 0 and X.aap_discount(0, 1) == 0
    assert X.aap_discount(0.6, 1.0) == pytest.approx(0.2358, abs=1e-4)      # N(0.3) − N(−0.3)
    assert X.aap_discount(0.6, 0.5) < X.aap_discount(0.6, 1.0) < X.aap_discount(0.9, 1.0)


def test_post_money_three_ways():
    assert D.post_money({"valuation": 220.49, "valuation_type": "投后"}) == pytest.approx(220.49e8)
    r = {"price": 78.31, "capital_before": 10854.5139, "capital_after": 12891.3313, "valuation": 85.0, "valuation_type": "投前"}
    assert D.post_money(r) == pytest.approx(85e8 + 78.31 * 2036.8174e4)     # 投前 + 新增资金 ≈ 100.95 亿
    assert D.post_money({"price": 11.21, "capital_after": 202433.8091}) == pytest.approx(226.93e8, rel=1e-4)


def _fake_dd(rounds, pre=45_000_000, lockup=None):
    return {"company": "测试公司", "listing_date": "2022-12-21", "unverified": [],
            "items": {"rounds": {"rounds": rounds}, "ipo": {"pre_ipo_shares": {"value": pre, "page": 3}, "issue_price": {"value": 100.0, "page": 3},
                                                            "issue_shares": {"value": 1500, "page": 3}}}}


def test_terms_same_event_not_dilution_and_later_round_dilutes(monkeypatch):
    rounds = [{"date": "2020-09", "kind": "增资", "price": 66.56, "capital_before": 2884.6411, "capital_after": 3004.8345, "valuation": None,
               "valuation_type": None, "investors": "哈勃投资、国开基金", "note": "", "page": 56},
              {"date": "2020-09", "kind": "增资", "price": 22.19, "capital_before": 2884.6411, "capital_after": 3004.8345, "valuation": None,
               "valuation_type": None, "investors": "欣芯聚源", "note": "", "page": 56},                  # 同一次增资的另一档价格
              {"date": "2020-12", "kind": "股改", "capital_before": 3004.8345, "capital_after": 4500.0, "investors": "", "note": "", "page": 49},
              {"date": "2021-03", "kind": "增资", "price": 30.0, "capital_before": 4500.0, "capital_after": 5000.0, "investors": "某基金", "note": "", "page": 60}]
    monkeypatch.setattr(D.dd, "load", lambda code: _fake_dd(rounds, pre=50_000_000))
    monkeypatch.setattr(D.dd, "lockup_for", lambda code, inv: {"found": True, "months_from_listing": 12, "months_from_acquisition": 36, "page": 9})
    t = D.terms({"code": "688498", "sector": "通信", "round": {"date": "2020-09", "kind": "增资", "price": 66.56}, "amount": 20_000_000,
                 "close": "2020-09-30", "lead": "哈勃投资"})
    assert len(t["dilution"]) == 1 and t["dilution"][0]["factor"] == pytest.approx(0.9)       # 只有 2021-03 那次稀释
    assert t["ownership_pre_ipo"] == pytest.approx(20_000_000 / (66.56 * 3004.8345e4) * 0.9)
    assert t["shares"] == pytest.approx(t["ownership_pre_ipo"] * 50_000_000)
    assert t["unlock_date"] == "2023-12-21"                        # max(上市 + 12 个月, 取得 + 36 个月 = 2023-09-30)
    assert t["cross_check"]["confirmed"] is False                  # 原文没写估值，只能用价格 × 股本（不算交叉确认）


def test_esop_round_rejected(monkeypatch):
    rounds = [{"date": "2020-09", "kind": "增资", "price": 5.55, "unit": "元/股", "capital_before": 32000.0, "capital_after": 37647.08,
               "investors": "宁波捷芯", "note": "员工持股平台增资", "page": 78}]
    monkeypatch.setattr(D.dd, "load", lambda code: _fake_dd(rounds))
    with pytest.raises(ValueError, match="员工持股"):
        D.terms({"code": "688220", "sector": "电子", "round": {"date": "2020-09", "kind": "增资", "price": 5.55}, "amount": 1, "close": "2020-09-30", "lead": "x"})


def test_on_page_units_and_short_numbers(monkeypatch):
    monkeypatch.setattr(PR, "pages", lambda code: ["无关", "发行前总股本为 386,739,939 股，发行价格 82.79 元/股", "无关", "无关"])
    assert PR.on_page("x", 2, 38673.9939)                         # 万股写法也认
    assert PR.on_page("x", 3, 82.79)                              # 允许相邻一页
    assert not PR.on_page("x", 4, 82.79)
    assert PR.on_page("x", 4, 1.0)                                # 1 这类数字不核对
    assert not PR.on_page("x", 2, 99.5)


def test_stress_is_deterministic_and_monotonic(monkeypatch):
    monkeypatch.setattr(S, "deal_moic", lambda code: pd.DataFrame(
        {"invested": [100.0, 100.0, 50.0], "moic": [5.0, 1.0, 0.0], "synthetic": [False, False, True]}, index=["A", "B", "SYN"]))
    r = S.run(n=2000)
    assert r[0]["mean"] == pytest.approx((500 + 100 + 0) / 250)     # 上市成功率 100% = 实际结果
    assert r[0]["mean"] > r[1]["mean"] > r[2]["mean"]
    assert S.run(n=2000) == r
    assert np.isclose(sum(S.FAIL_P), 1)
