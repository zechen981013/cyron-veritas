"""利润质量审查（估值之前先做）：杜邦分析、Piotroski F 分、Altman Z 分、Beneish M 分、应计比率 → A–D 利润可信度评级。

公式与 FinanceToolkit（JerBouma/FinanceToolkit）的 models 模块一致，tests/test_cross_validation.py 用同样的输入逐项核对。
输入是 data.financials.annual() 的年度表（单位亿，行 = 年份）。

  杜邦三步：ROE = 净利率 × 资产周转率 × 权益乘数（资产、权益取期初期末平均）
  杜邦五步：ROE = 税负（净利 / 税前）× 利息负担（税前 / EBIT）× EBIT 利润率 × 资产周转率 × 权益乘数
  Piotroski（9 项，每项 0/1）：ROA>0、经营现金流>0、ROA 上升、经营现金流/总资产 > ROA（应计）、长期负债率下降、
          流动比率上升、没有增发（股本不增加）、毛利率上升、资产周转率上升
  Altman Z（制造业原版）：1.2·营运资本/总资产 + 1.4·留存收益/总资产 + 3.3·EBIT/总资产 + 0.6·股权市值/总负债 + 1.0·收入/总资产
          安全 > 2.99，灰色 1.81–2.99，困境 < 1.81；非制造业另给 Z''（新兴市场版，用账面权益）
  Beneish M（8 变量）：-4.84 + 0.92 DSRI + 0.528 GMI + 0.404 AQI + 0.892 SGI + 0.115 DEPI − 0.172 SGAI + 4.679 TATA − 0.327 LVGI
          > -1.78 提示可能存在利润操纵
金融企业（银行、保险）的资产负债结构不同，这些模型不适用，直接返回说明。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _avg(s: pd.Series) -> pd.Series:
    return (s + s.shift()) / 2


def dupont(t: pd.DataFrame) -> pd.DataFrame:
    avg_assets, avg_equity = _avg(t["total_assets"]), _avg(t["total_equity"])
    ebt = t["pretax_profit"]
    out = pd.DataFrame(index=t.index)
    out["net_margin"] = t["net_profit"] / t["revenue"]
    out["asset_turnover"] = t["revenue"] / avg_assets
    out["equity_multiplier"] = avg_assets / avg_equity
    out["roe_3step"] = out["net_margin"] * out["asset_turnover"] * out["equity_multiplier"]
    out["tax_burden"] = t["net_profit"] / ebt
    out["interest_burden"] = ebt / t["ebit"]
    out["ebit_margin"] = t["ebit"] / t["revenue"]
    out["roe_5step"] = out["tax_burden"] * out["interest_burden"] * out["ebit_margin"] * out["asset_turnover"] * out["equity_multiplier"]
    return out


def piotroski(t: pd.DataFrame) -> pd.DataFrame:
    avg_assets = _avg(t["total_assets"])
    roa = t["net_profit"] / avg_assets
    lev = t["long_term_debt"] / avg_assets
    cur = t["current_assets"] / t["current_liab"]
    gm = (t["revenue"] - t["cogs"]) / t["revenue"]
    turnover = t["revenue"] / avg_assets
    c = pd.DataFrame(index=t.index)
    c["roa_positive"] = roa > 0
    c["cfo_positive"] = t["cfo"] > 0
    c["roa_improving"] = roa > roa.shift()
    c["accruals"] = t["cfo"] / t["total_assets"] > roa
    c["leverage_falling"] = lev < lev.shift()
    c["current_ratio_rising"] = cur > cur.shift()
    c["no_dilution"] = t["shares"] <= t["shares"].shift() * 1.001
    c["gross_margin_rising"] = gm > gm.shift()
    c["turnover_rising"] = turnover > turnover.shift()
    c = c.astype(int)
    c.iloc[0] = np.nan  # 第一年没有上年对比
    c["score"] = c.sum(axis=1, min_count=9)
    return c


def altman(t: pd.DataFrame, market_cap: float | None = None) -> pd.DataFrame:
    ta = t["total_assets"]
    out = pd.DataFrame(index=t.index)
    out["x1_wc_ta"] = (t["current_assets"] - t["current_liab"]) / ta
    out["x2_re_ta"] = t["retained_earnings"] / ta
    out["x3_ebit_ta"] = t["ebit"] / ta
    mve = pd.Series(np.nan, index=t.index)
    if market_cap is not None:
        mve.iloc[-1] = market_cap  # 只有最新一年有市值；历史年份用账面权益的 Z''
    out["x4_mve_tl"] = mve / t["total_liab"]
    out["x5_sales_ta"] = t["revenue"] / ta
    out["z"] = 1.2 * out.x1_wc_ta + 1.4 * out.x2_re_ta + 3.3 * out.x3_ebit_ta + 0.6 * out.x4_mve_tl + 1.0 * out.x5_sales_ta
    out["z_double_prime"] = (6.56 * out.x1_wc_ta + 3.26 * out.x2_re_ta + 6.72 * out.x3_ebit_ta
                             + 1.05 * t["total_equity"] / t["total_liab"])
    out["zone"] = np.where(out["z"].isna(), "—", np.where(out["z"] > 2.99, "安全", np.where(out["z"] >= 1.81, "灰色", "困境")))
    return out


def beneish(t: pd.DataFrame) -> pd.DataFrame:
    s, p = t, t.shift()
    rec_s = s["receivables"] / s["revenue"]
    gm = (s["revenue"] - s["cogs"]) / s["revenue"]
    aq = 1 - (s["current_assets"] + s["fixed_assets"]) / s["total_assets"]
    dep = s["da"] / (s["da"] + s["fixed_assets"])
    sga = s["sga"] / s["revenue"]
    lev = (s["current_liab"] + s["long_term_debt"]) / s["total_assets"]
    out = pd.DataFrame(index=t.index)
    out["DSRI"] = rec_s / rec_s.shift()
    out["GMI"] = gm.shift() / gm
    out["AQI"] = aq / aq.shift()
    out["SGI"] = s["revenue"] / p["revenue"]
    out["DEPI"] = dep.shift() / dep
    out["SGAI"] = sga / sga.shift()
    out["LVGI"] = lev / lev.shift()
    out["TATA"] = (s["net_profit"] - s["cfo"]) / s["total_assets"]
    out["m"] = (-4.84 + 0.92 * out.DSRI + 0.528 * out.GMI + 0.404 * out.AQI + 0.892 * out.SGI + 0.115 * out.DEPI
                - 0.172 * out.SGAI + 4.679 * out.TATA - 0.327 * out.LVGI)
    out["flag"] = np.where(out["m"].isna(), "—", np.where(out["m"] > -1.78, "可能操纵", "正常"))
    return out


def credibility_grade(pio: float | None, m: float | None, accrual_ratio: float | None, cfo_to_np: float | None) -> tuple[str, list[str]]:
    """A–D 利润可信度评级（规则，不经大模型）：从 A 开始按红旗扣级。C 级结论只能「观望」，D 级不应给出正面结论。"""
    level, reasons = 0, []
    if m is not None and not np.isnan(m) and m > -1.78:
        level += 2
        reasons.append(f"Beneish M 分 {m:.2f} > -1.78，提示可能存在利润操纵")
    if accrual_ratio is not None and not np.isnan(accrual_ratio) and accrual_ratio > 0.10:
        level += 1
        reasons.append(f"应计比率 {accrual_ratio:.1%} 偏高（利润明显多于经营现金流）")
    if cfo_to_np is not None and not np.isnan(cfo_to_np) and cfo_to_np < 0.8:
        level += 1
        reasons.append(f"经营现金流 / 净利润 {cfo_to_np:.2f} < 0.8")
    if pio is not None and not np.isnan(pio) and pio <= 3:
        level += 1
        reasons.append(f"Piotroski F 分 {pio:.0f} ≤ 3，基本面在恶化")
    grade = "ABCD"[min(level, 3)]
    return grade, reasons or ["未发现红旗"]


def review(t: pd.DataFrame, market_cap: float | None = None, is_financial: bool = False) -> dict:
    if is_financial:
        return {"applicable": False, "note": "金融企业的资产负债结构不同，杜邦以外的 Piotroski / Altman / Beneish 不适用；看 ROE、资产质量和资本充足率"}
    d, pio, alt, ben = dupont(t), piotroski(t), altman(t, market_cap), beneish(t)
    last = t.index[-1]
    accrual = float((t.loc[last, "net_profit"] - t.loc[last, "cfo"]) / t.loc[last, "total_assets"])
    cfo_np = float(t.loc[last, "cfo"] / t.loc[last, "net_profit"]) if t.loc[last, "net_profit"] else np.nan
    grade, reasons = credibility_grade(pio.loc[last, "score"], ben.loc[last, "m"], accrual, cfo_np)
    r = lambda x, n=3: None if pd.isna(x) else round(float(x), n)
    return {
        "applicable": True,
        "dupont": {int(y): {k: r(v) for k, v in row.items()} for y, row in d.iterrows()},
        "piotroski": {int(y): {k: (None if pd.isna(v) else int(v)) for k, v in row.items()} for y, row in pio.iterrows()},
        "altman": {"latest_z": r(alt.loc[last, "z"], 2), "latest_zone": alt.loc[last, "zone"],
                   "z_double_prime_by_year": {int(y): r(v, 2) for y, v in alt["z_double_prime"].items()},
                   "components_latest": {k: r(alt.loc[last, k]) for k in ("x1_wc_ta", "x2_re_ta", "x3_ebit_ta", "x4_mve_tl", "x5_sales_ta")},
                   "note": "Z 原版适用于制造业（> 2.99 安全，< 1.81 困境）；非制造业参考 Z''（> 2.6 安全，< 1.1 困境）"},
        "beneish": {int(y): {k: (r(v) if not isinstance(v, str) else v) for k, v in row.items()} for y, row in ben.iterrows()},
        "accrual_ratio": r(accrual), "cfo_to_np": r(cfo_np, 2),
        "credibility_grade": grade, "grade_reasons": reasons,
        "grade_rule": "从 A 开始扣级：Beneish > -1.78 扣 2 级；应计比率 > 10%、经营现金流/净利 < 0.8、Piotroski ≤ 3 各扣 1 级；C 级结论只能观望，D 级不应给出正面结论",
    }
