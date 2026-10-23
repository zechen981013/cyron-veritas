"""组合优化（cvxpy + Clarabel）。

  max  αᵀw − λ · 主动风险² − 交易成本ᵀ|w − w₀|
  s.t. Σw = 1 − 现金目标；w ≥ 0；
       |w_i − w_b,i| ≤ 单票偏离上限；|行业主动权重| ≤ 行业偏离上限；|风格主动暴露| ≤ 风格上限（规模更严）；
       成分股权重 ≥ 下限 ×（1 − 现金目标）；单边换手 ≤ 上限；主动风险 ≤ 跟踪误差上限；
       不能交易的股票（停牌、一字板）权重固定为当前值。
α 按 Grinold 公式把合成因子 z 分数换成预期超额：α_i = IC × σ_特异,i × z_i（IC 取 0.03，比样本内估计保守）。
限额都按 limit_buffer 打折（单票 1.6%、行业 2.4%、成分股 ≥ 90%），给两次调仓之间的漂移留空间。
约束不可行时依次放宽：先去掉换手上限，再把跟踪误差上限放宽 50%，并在结果里记录放宽了什么。
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from cyron_veritas.quant import riskmodel as RMOD

IC_ASSUMED = 0.03


def optimize(alpha_z: pd.Series, model: dict, wb: pd.Series, w0: pd.Series, cfg: dict, frozen: set[str] = frozenset(),
             bench_members: set[str] | None = None, initial: bool = False, te_scale: float = 1.0) -> dict:
    """initial=True 表示从现金建仓，不加换手约束；te_scale < 1 表示事后跟踪误差偏高，事前上限收紧。
    合同限额都打一个缓冲（cfg.strategy.limit_buffer，默认 0.8）：组合在两次调仓之间会漂移，贴着限额优化第二天就会超。"""
    st, lim = cfg["strategy"], cfg["contract_limits"]
    X, Fm, D = model["X"], model["F"], model["D"]
    codes = list(X.index)
    n = len(codes)
    z = alpha_z.reindex(codes).fillna(0).values
    alpha = IC_ASSUMED * np.sqrt(D.values) * z
    b = wb.reindex(codes).fillna(0).values
    x0 = w0.reindex(codes).fillna(0).values
    invest = 1 - st["cash_target"]
    b_scaled = b * invest / b.sum()                                     # 基准按股票仓位缩放后比较主动权重
    Xv = X.values
    Lf = np.linalg.cholesky(Fm.values + 1e-10 * np.eye(len(Fm)))
    ind_cols = [i for i, c in enumerate(X.columns) if c not in RMOD.STYLES]
    sty = {c: X.columns.get_loc(c) for c in RMOD.STYLES}
    cost = st.get("tcost", 0.003) * np.ones(n)
    buf = st.get("limit_buffer", 0.8)
    single_cap, ind_cap = lim["active_single_max"] * buf, lim["active_industry_max"] * buf
    member_min = min(lim["bench_constituent_min"] + (1 - buf) / 2, 0.95)          # 成分股下限往上留缓冲（0.80 → 0.90）
    te_cap0 = lim["te_max"] * te_scale
    member = np.array([c in (bench_members if bench_members is not None else set(wb.index[wb > 0])) for c in codes], dtype=float)
    frozen_idx = [i for i, c in enumerate(codes) if c in frozen]

    def solve(turnover_cap, te_cap):
        w = cp.Variable(n)
        a = w - b_scaled
        y = Xv.T @ a
        risk = cp.sum_squares(Lf.T @ y) + cp.sum(cp.multiply(D.values, cp.square(a)))
        cons = [cp.sum(w) == invest, w >= 0, cp.abs(a) <= single_cap, cp.abs(y[ind_cols]) <= ind_cap,
                member @ w >= member_min * invest]
        for c, k in sty.items():
            cons.append(cp.abs(y[k]) <= (st["size_exposure_max"] if c == "规模" else st["style_exposure_max"]))
        if turnover_cap is not None:
            cons.append(cp.sum(cp.abs(w - x0)) <= 2 * turnover_cap)
        if te_cap is not None:
            cons.append(risk <= te_cap ** 2)
        if frozen_idx:
            cons.append(w[frozen_idx] == x0[frozen_idx])
        obj = cp.Maximize(alpha @ w - st["risk_aversion"] * risk - cost @ cp.abs(w - x0))
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.CLARABEL)
        return prob, w

    relaxed = []
    prob, w = solve(None if initial else st["turnover_max"], te_cap0)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        relaxed.append("取消换手上限")
        prob, w = solve(None, te_cap0)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        relaxed.append("跟踪误差上限放宽 50%")
        prob, w = solve(None, te_cap0 * 1.5)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {"status": prob.status, "relaxed": relaxed, "weights": w0}
    wt = pd.Series(np.clip(w.value, 0, None), index=codes)
    wt[wt < 1e-4] = 0                                                    # 碎股权重清零
    wt = wt * invest / wt.sum()
    act = wt - pd.Series(b_scaled, index=codes)
    ra = RMOD.active_risk(model, act)
    return {"status": prob.status, "relaxed": relaxed, "weights": wt[wt > 0], "ex_ante_te": ra["te"],
            "alpha": float(alpha @ wt.values - alpha @ b_scaled), "turnover": float(np.abs(wt.values - x0).sum() / 2),
            "n_holdings": int((wt > 0).sum()), "member_weight": float((wt * member).sum() / invest),
            "exposure": ra["exposure"], "risk": {k: ra[k] for k in ("factor_var", "specific_var", "industry_var", "style_var")}}
