import numpy as np
import pandas as pd
import pytest

from cyron_veritas.risk import metrics as M


def _rets(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n), index=pd.bdate_range("2020-01-01", periods=n))


def test_historical_var_matches_quantile():
    r = _rets()
    assert M.historical_var(r, 0.95) == pytest.approx(-np.quantile(r, 0.05))
    assert M.historical_cvar(r, 0.95) >= M.historical_var(r, 0.95)


def test_parametric_var_close_to_normal():
    r = _rets(20000)
    assert M.parametric_var(r, 0.95) == pytest.approx(1.645 * 0.01 - 0.0005, rel=0.03)


def test_max_drawdown_simple_path():
    r = pd.Series([0.10, -0.50, 0.20], index=pd.bdate_range("2024-01-01", periods=3))
    dd = M.max_drawdown(r)
    assert dd["max_drawdown"] == pytest.approx(0.5)
    assert dd["current_drawdown"] == pytest.approx(1 - 0.6)


def test_risk_contribution_sums_to_one():
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.01, (500, 3)), columns=list("abc"))
    rc = M.risk_contribution(pd.Series([0.5, 0.3, 0.2], index=list("abc")), rets)
    assert rc.sum() == pytest.approx(1.0)


def test_kupiec_expected_exceptions_not_rejected():
    assert M.kupiec_pof(12, 250, 0.95)["p_value"] > 0.5      # 12.5 次是理论值
    assert M.kupiec_pof(30, 250, 0.95)["p_value"] < 0.01     # 明显太多


def test_beta_of_scaled_series():
    b = _rets()
    assert M.beta(2 * b, b) == pytest.approx(2.0)
