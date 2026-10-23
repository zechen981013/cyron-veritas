"""量化与账簿：成交量单位、账务规则、因子中性化、合成权重无前视、优化器约束、指增合同限额。全部离线、合成数据。"""
import math

import numpy as np
import pandas as pd
import pytest

from cyron_veritas.fund import marketdata as M
from cyron_veritas.fund import product as P
from cyron_veritas.fund import risk as R
from cyron_veritas.fund.ledger import Ledger
from cyron_veritas.quant import factors as F
from cyron_veritas.quant import optimizer as O
from cyron_veritas.quant import research as Q
from cyron_veritas.quant import riskmodel as RMOD

QE = P.load("CV-QE1")


def test_volume_units_normalized_per_stock():
    rows = []
    for code, unit in (("A", 1), ("B", 0.01), ("C", 100)):            # 股 / 手 / 股×100
        for i in range(5):
            shares = 1_000_000 * (i + 1)
            rows.append({"date": pd.Timestamp("2026-01-05") + pd.Timedelta(days=i), "code": code, "open": 10, "close": 10.0, "high": 10.2,
                         "low": 9.8, "volume": shares * unit, "amount": shares * 10.0, "turnover": 0.01})
    b = M._volume_in_shares(pd.DataFrame(rows))
    est = b["amount"] / 10.0
    assert np.allclose(b["volume"], est, rtol=0.05)


class _Mkt:
    def __init__(self, prices, actions=None):
        self.prices = prices                          # {date: {code: px}}
        self.actions = actions or {}
        self.names, self.exchange = {"600000": "浦发银行"}, {"600000": "SH"}

    def last_price(self, c, d):
        return self.prices[pd.Timestamp(d)][c]

    def tradable(self, c, d):
        return True


def test_ledger_dividend_bonus_flows_and_fees(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "sheet_dir", lambda code: tmp_path)
    cal = pd.DatetimeIndex(["2026-03-30", "2026-03-31", "2026-04-01"])
    ex = pd.DataFrame({"code": ["600000"], "cash_per_share": [0.5], "bonus_per_share": [0.2]})
    mkt = _Mkt({d: {"600000": 10.0} for d in cal}, {cal[1]: ex})
    cfg = {**QE, "initial_capital": 1_000_000, "flows": [{"date": cal[2].date(), "type": "subscribe", "amount": 100_000}]}
    L = Ledger("T", cfg, mkt, cal, write_sheets=False)
    L.start_day(cal[0])
    L.buy(cal[0], "600000", 10_000, 10.0, "建仓")
    s0 = L.end_day(cal[0])
    L.start_day(cal[1])                                                # 除权除息：派 0.5 记应收，送 0.2 加股数
    assert L.recv["600000"]["amount"] == pytest.approx(5_000) and L.qty("600000") == 12_000
    assert L.pos["600000"]["cost"] == pytest.approx(100_000 + L.trade_rows[0]["fee"])   # 送转不改成本
    L.end_day(cal[1])                                                  # 3 月 31 日是季末：计提的费用当天付掉
    assert L.payable() == 0
    cash_before = L.cash
    L.start_day(cal[2])                                                # 股利到账；申购按上一日净值确认
    assert L.cash == pytest.approx(cash_before + 5_000 + 100_000)
    assert L.flow_rows[0]["nav"] == L.prev_unit and L.flow_rows[0]["shares"] == pytest.approx(100_000 / L.prev_unit)
    assert s0["net_assets"] < 1_000_000                                # 佣金 + 当日计提


def test_ledger_quarter_end_fee_payment(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "sheet_dir", lambda code: tmp_path)
    cal = pd.DatetimeIndex(["2026-03-30", "2026-03-31", "2026-04-01"])
    mkt = _Mkt({d: {"600000": 10.0} for d in cal})
    L = Ledger("T", {**QE, "initial_capital": 1_000_000, "flows": []}, mkt, cal, write_sheets=False)
    for d in cal[:2]:
        L.start_day(d)
        L.end_day(d)
    days = (cal[1] - cal[0]).days + 1
    paid = 1_000_000 * (QE["fees"]["management"] + QE["fees"]["custody"]) * days / 365
    assert L.payable() == 0 and L.cash == pytest.approx(1_000_000 - paid, rel=1e-4)


def test_neutralize_removes_industry_and_size():
    rng = np.random.default_rng(0)
    n = 300
    ind = pd.Series(rng.choice(["银行", "电子", "医药"], n))
    size = pd.Series(rng.normal(size=n))
    z = pd.Series(rng.normal(size=n)) + 2 * size + ind.map({"银行": 1.0, "电子": -1.0, "医药": 0.0})
    r = F.neutralize(z, ind, size)
    assert abs(np.corrcoef(r, size)[0, 1]) < 1e-8
    assert r.groupby(ind).mean().abs().max() < 1e-8


def test_composite_weights_have_no_lookahead():
    ric = pd.DataFrame(0.05, index=range(20), columns=["EP", "MOM"])
    ric["MOM"] = -0.05
    w = Q.weights_at(ric, 10)
    assert w["EP"] == pytest.approx(1.0) and w["MOM"] == 0               # 负 ICIR 不翻转方向，权重为 0
    spiked = ric.copy()
    spiked.loc[9, "MOM"] = 50.0                                           # 第 m−1 期在 e_m 还不知道：改它不能影响权重
    assert Q.weights_at(spiked, 10).equals(w)
    few = Q.weights_at(ric, Q.MIN_OBS)                                    # 已知期数不足 MIN_OBS：全部为 0
    assert few.abs().sum() == 0


def _synthetic_model(n=80, seed=1):
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}" for i in range(n)]
    inds = pd.get_dummies(pd.Series(rng.choice(["银行", "电子", "医药", "机械"], n), index=codes)).astype(float)
    sty = pd.DataFrame(rng.normal(size=(n, len(RMOD.STYLES))), index=codes, columns=RMOD.STYLES)
    X = pd.concat([inds, sty], axis=1)
    k = X.shape[1]
    A = rng.normal(size=(k, k)) * 0.03
    Fm = pd.DataFrame(A @ A.T + np.eye(k) * 1e-4, index=X.columns, columns=X.columns)
    D = pd.Series(rng.uniform(0.04, 0.12, n), index=codes)
    wb = pd.Series(rng.uniform(0.5, 1.5, n), index=codes)
    wb.iloc[60:] = 0                                                      # 后 20 只不是成分股
    wb /= wb.sum()
    return {"X": X, "F": Fm, "D": D}, wb, pd.Series(rng.normal(size=n), index=codes)


def test_optimizer_respects_contract_with_buffer():
    model, wb, z = _synthetic_model()
    cfg = {**QE, "strategy": {**QE["strategy"], "style_exposure_max": 0.5, "size_exposure_max": 0.3}}
    r = O.optimize(z, model, wb, pd.Series(dtype=float), cfg, initial=True)
    assert r["status"].startswith("optimal")
    w = r["weights"].reindex(model["X"].index).fillna(0)
    lim, st = cfg["contract_limits"], cfg["strategy"]
    invest = 1 - st["cash_target"]
    assert w.sum() == pytest.approx(invest, abs=1e-6)
    a = w - wb * invest
    buf = st["limit_buffer"] if "limit_buffer" in st else 0.8
    assert a.abs().max() <= lim["active_single_max"] * buf + 1e-6
    ind_cols = [c for c in model["X"].columns if c not in RMOD.STYLES]
    assert (model["X"][ind_cols].T @ a).abs().max() <= lim["active_industry_max"] * buf + 1e-6
    assert w[wb > 0].sum() / invest >= lim["bench_constituent_min"] - 1e-6
    assert r["ex_ante_te"] <= lim["te_max"] + 1e-6


def test_optimizer_frozen_and_turnover():
    model, wb, z = _synthetic_model(seed=2)
    cfg = {**QE, "strategy": {**QE["strategy"], "style_exposure_max": 0.5, "size_exposure_max": 0.3}}
    w0 = O.optimize(z, model, wb, pd.Series(dtype=float), cfg, initial=True)["weights"]
    frozen = {w0.index[0]}
    r = O.optimize(-z, model, wb, w0, cfg, frozen=frozen)                 # 信号反过来，逼它大幅调仓
    w = r["weights"].reindex(model["X"].index).fillna(0)
    assert w[w0.index[0]] == pytest.approx(w0.iloc[0], abs=1e-6)          # 停牌股不动
    if not r["relaxed"]:
        assert r["turnover"] <= cfg["strategy"]["turnover_max"] + 1e-4


def test_active_risk_decomposition_adds_up():
    model, wb, _ = _synthetic_model(seed=3)
    a = pd.Series(np.random.default_rng(3).normal(0, 0.005, len(wb)), index=wb.index)
    r = RMOD.active_risk(model, a)
    assert r["te"] ** 2 == pytest.approx(r["factor_var"] + r["specific_var"])


def test_quant_limits_checked_from_contract(monkeypatch):
    snap = {"weights": {"600000": 0.01}, "top1": 0.01, "top1_name": "", "by_industry": {"银行": 0.2}, "cash_weight": 0.02,
            "liquidity": {1: 1.0, 5: 1.0, 20: 1.0}, "unit_nav": 1.1, "active_single": 0.025, "active_single_name": "600000",
            "active_industry": 0.028, "active_industry_name": "银行", "bench_member_weight": 0.78, "te_60d": 0.03}
    st = {c["rule"]: c["status"] for c in R.check_limits("CV-QE1", snap)}
    assert "single_name_max" not in st and "warning_line" not in st          # 合同里没有的条款不查
    assert st["active_single_max"] == R.BREACH and st["active_industry_max"] == R.WARN
    assert st["bench_constituent_min"] == R.BREACH and st["te_max"] == R.OK
    assert math.isclose(len(st), 6)
