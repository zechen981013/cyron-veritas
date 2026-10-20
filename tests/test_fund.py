"""资管中台：估值表解析、估值核对、合同限额、交易前检查、风控整改、事件合并、月报方向词。全部离线，用临时数据库。"""
import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from cyron_veritas.fund import db
from cyron_veritas.fund import marketdata as M
from cyron_veritas.fund import ops
from cyron_veritas.fund import product as P
from cyron_veritas.fund import report
from cyron_veritas.fund import risk as R
from cyron_veritas.fund import sheet as S
from cyron_veritas.fund import simulate as SIM

CFG = P.load("CV-VG1")
D0, D1 = dt.date(2026, 3, 2), dt.date(2026, 3, 3)


@pytest.fixture
def tmpdb(tmp_path, monkeypatch):
    monkeypatch.setenv("PLATFORM_DB_URL", f"sqlite:///{tmp_path / 'p.db'}")
    monkeypatch.setattr(db, "_engine", None)
    yield
    db._engine = None


def _pos(price):
    return [{"code": "600030", "name": "中信证券", "exchange": "SH", "qty": 10_000, "cost": 250_000.0, "price": price, "suspended": False},
            {"code": "000001", "name": "平安银行", "exchange": "SZ", "qty": 20_000, "cost": 220_000.0, "price": 11.0, "suspended": False}]


def test_sheet_roundtrip(tmp_path):
    f = tmp_path / "s.xlsx"
    summ = S.write(f, CFG, pd.Timestamp(D0), _pos(26.0), 900_000, 100_000, [{"code": "000001", "name": "平安银行", "amount": 5_000}],
                   3_000, 100, 1_000_000)
    r = S.parse(f)
    assert r["date"] == D0
    assert {h["code"]: h["exchange"] for h in r["holdings"]} == {"600030": "SH", "000001": "SZ"}
    assert r["summary"]["stock_mv"] == pytest.approx(10_000 * 26 + 20_000 * 11)
    assert r["summary"]["net_assets"] == pytest.approx(summ["net_assets"])
    assert r["summary"]["net_assets"] == pytest.approx(900_000 + 100_000 + 480_000 + 5_000 - 3_100)
    assert r["receivable"][0]["amount"] == pytest.approx(5_000)


def _seed_two_days(tmp_path, tamper=0.0):
    """两张估值表：第二天价格上涨，托管净值 = 自估净值 + tamper。"""
    net0 = 1_000_000 + 10_000 * 26 + 20_000 * 11
    accrual = net0 * (CFG["fees"]["management"] + CFG["fees"]["custody"]) / 365
    rows = []
    for d, px, fee in ((D0, 26.0, 0.0), (D1, 27.0, accrual)):
        mv = 10_000 * px + 20_000 * 11
        net = 1_000_000 + mv - fee + (tamper if d == D1 else 0)
        rows.append({"product": "CV-VG1", "date": d, "unit_nav": net / 1e6, "net_assets": net, "cash": 1_000_000.0, "receivable": 0.0,
                     "fee_payable": fee, "stock_mv": mv})
        db.add(db.holdings, [{"product": "CV-VG1", "date": d, "code": p["code"], "qty": p["qty"], "price": p["price"],
                              "mv": p["qty"] * p["price"]} for p in _pos(px)])
    db.add(db.nav, rows)


def _fake_market(monkeypatch):
    bars = pd.DataFrame({"date": pd.to_datetime([D0, D0, D1, D1]), "code": ["600030", "000001"] * 2, "close": [26.0, 11.0, 27.0, 11.0]})
    monkeypatch.setattr(M, "bars", lambda codes: bars)
    monkeypatch.setattr(M, "corporate_actions", lambda codes: pd.DataFrame(
        {"code": pd.Series(dtype=str), "ex_date": pd.Series(dtype="datetime64[ns]"), "cash_per_share": [], "bonus_per_share": []}))


def test_reconcile_matches_and_flags_tampered_sheet(tmp_path, tmpdb, monkeypatch):
    _fake_market(monkeypatch)
    _seed_two_days(tmp_path)
    r = ops.reconcile("CV-VG1")
    assert r["status"].tolist() == ["一致"] and abs(r["diff"].iloc[0]) < 1e-6

    db.replace_rows(db.nav, [], db.nav.c.product == "CV-VG1")
    db.replace_rows(db.holdings, [], db.holdings.c.product == "CV-VG1")
    _seed_two_days(tmp_path, tamper=5_000)                         # 托管估值表多出 5000 元，无法解释
    r = ops.reconcile("CV-VG1")
    assert r["status"].tolist() == ["待核查"]
    assert r["explained"].iloc[0]["残差"] == pytest.approx(5_000, abs=1)


def test_check_limits_statuses():
    snap = {"weights": {"600030": 0.12, "000002": 0.02}, "top1": 0.12, "top1_name": "中信证券", "by_industry": {"非银金融": 0.28},
            "cash_weight": 0.021, "liquidity": {1: 1.0, 5: 0.7, 20: 1.0}, "unit_nav": 0.90}
    st = {c["rule"]: c["status"] for c in R.check_limits("CV-VG1", snap)}
    assert st["single_name_max"] == R.BREACH          # 12% > 10%
    assert st["industry_max"] == R.WARN               # 28% 超过上限的 90%
    assert st["cash_min"] == R.WARN                   # 2.1% 高于下限但在 2% / 0.9 以内
    assert st["liquidity_5d_min"] == R.BREACH
    assert st["warning_line"] == R.WARN and st["stop_line"] == R.OK    # 0.90 < 0.85 / 0.9；> 0.80 / 0.9
    assert st["banned"] == R.BREACH                   # 持有禁投的 000002


def test_pretrade_rules(tmpdb, monkeypatch):
    d = pd.Timestamp(D1)
    close = pd.DataFrame({"600030": [27.0], "000002": [3.5], "600519": [1300.0], "601318": [50.0]}, index=[d])
    monkeypatch.setattr(R, "market_view", lambda: {"close": close})
    monkeypatch.setattr(R, "nav_series", lambda code: pd.DataFrame({"unit_nav": [1.2], "cash": [300_000.0], "net_assets": [3_000_000.0]}, index=[d]))
    monkeypatch.setattr(R, "holdings_on", lambda code, dd: pd.DataFrame({"code": ["600030"], "qty": [10_000.0]}))
    monkeypatch.setattr(R, "core_pool", lambda code, dd: {"600030", "601318"})
    monkeypatch.setattr(M, "sw_industry", lambda: {"600030": "非银金融", "601318": "非银金融", "000002": "房地产", "600519": "食品饮料"})
    res = R.pretrade("CV-VG1", d, [{"code": "000002", "side": "buy", "qty": 100}, {"code": "600519", "side": "buy", "qty": 100},
                                   {"code": "600030", "side": "sell", "qty": 20_000}, {"code": "601318", "side": "buy", "qty": 150}])
    by = {o["code"]: o for o in res["orders"]}
    assert any("禁投池" in x for x in by["000002"]["reasons"])
    assert by["600519"]["reasons"] == ["不在核心池（需先经投委会批准入池）"]
    assert any("超过持仓" in x for x in by["600030"]["reasons"])
    assert by["601318"]["reasons"] == ["买入须为 100 股整数倍"]
    assert not res["pass"]

    ok = R.pretrade("CV-VG1", d, [{"code": "601318", "side": "buy", "qty": 1_000}])
    assert ok["orders"][0]["pass"]
    # 成交后：中信 27 万（9%）+ 平安 5 万 = 非银 10.7%，现金 25 万（8.3%），全部在限额内
    assert ok["pass"], ok["portfolio_checks"]

    big = R.pretrade("CV-VG1", d, [{"code": "601318", "side": "buy", "qty": 5_000}])
    assert big["orders"][0]["pass"] and not big["pass"]
    assert any("现金" in x for x in big["portfolio_checks"])      # 买 25 万后现金 5 万 = 1.7% < 2%


def _mkt(prices: dict, industry: dict):
    return SimpleNamespace(last_price=lambda c, d: prices[c], industry=industry)


def test_breaches_and_fix_targets_only_sell_what_is_needed():
    lim, st = CFG["contract_limits"], CFG["strategy"]
    mkt = _mkt({"A": 10.0, "B": 10.0, "C": 10.0, "D": 10.0},
               {"A": "非银金融", "B": "非银金融", "C": "非银金融", "D": "电子"})
    pos = {"A": {"qty": 12_000}, "B": {"qty": 11_000}, "C": {"qty": 10_000}, "D": {"qty": 9_000}}   # 120k / 110k / 100k / 90k
    cash = 580_000.0                                                                                  # 净值 1,000,000
    fix = SIM._breaches(mkt, pos, cash, {}, 0.0, None, lim)
    assert set(fix) == {"单票", "行业"} and set(fix["单票"]) == {"A", "B"}
    assert fix["行业"]["非银金融"] == pytest.approx(0.33)
    t = SIM._fix_targets(mkt, pos, None, 1_000_000, fix, st, lim)
    # A、B 压到单票目标 8% 后非银合计 26%，已回到 30% 以内：C、D 都不动，行业这一步不能反过来加仓
    assert t == pytest.approx({"A": st["max_weight"], "B": st["max_weight"]})

    pos2 = {"A": {"qty": 9_000}, "B": {"qty": 9_000}, "C": {"qty": 9_000}, "D": {"qty": 9_000}}   # 非银 27%……再加 E
    mkt.industry["E"] = "非银金融"
    mkt2 = _mkt({**{k: 10.0 for k in "ABCDE"}}, mkt.industry)
    pos2["E"] = {"qty": 6_000}                                                                       # 非银 33%，单票都没超
    fix2 = SIM._breaches(mkt2, pos2, 580_000.0, {}, 0.0, None, lim)
    assert set(fix2) == {"行业"}
    t2 = SIM._fix_targets(mkt2, pos2, None, 1_000_000, fix2, st, lim)
    assert set(t2) == {"A", "B", "C", "E"} and sum(t2.values()) == pytest.approx(lim["industry_max"] * 0.95)

    t3 = SIM._fix_targets(mkt, {"D": {"qty": 99_000}}, None, 1_000_000, {"现金": 0.01}, st, lim)
    assert t3["D"] == pytest.approx(1 - st["cash_target"])         # 现金不足：全组合等比减到 cash_target


def test_events_merge_into_episodes(tmpdb):
    days = pd.bdate_range("2026-03-02", periods=6).date
    db.add(db.nav, [{"product": "CV-VG1", "date": d, "net_assets": 1.0} for d in days])
    ev = lambda d, rule, v, status="预警": {"product": "CV-VG1", "date": d, "rule": rule, "value": v, "limit": 0.3, "status": status, "detail": rule}
    db.add(db.limit_events, [ev(days[0], "industry_max", 0.28), ev(days[1], "industry_max", 0.31, "超限"), ev(days[2], "industry_max", 0.29),
                             ev(days[4], "industry_max", 0.28), ev(days[1], "cash_min", 0.019, "超限")])
    eps = R.episodes("CV-VG1")
    ind = sorted((e for e in eps if e["rule"] == "industry_max"), key=lambda e: e["start"])
    assert [e["days"] for e in ind] == [3, 1]                 # 第 4 天没触发，断开成两段
    assert ind[0]["status"] == "超限" and ind[0]["peak"] == pytest.approx(0.31)
    assert R.handle_events("CV-VG1", ind[0]["ids"], "风控岗", "已整改") == 3
    assert len(R.episodes("CV-VG1")) == 2


def test_commentary_direction_check():
    signs = {"配置": 0.0076, "选股": -0.013, "交互": -0.005, "超额": -0.007}
    assert report.check_direction("超额主要受选股与结构交互拖累，配置层面仍有正贡献。", signs) == []
    errs = report.check_direction("配置效应拖累了组合表现，选股贡献为正。本月跑赢基准。", signs)
    assert len(errs) == 3
