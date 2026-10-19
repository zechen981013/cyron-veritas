"""A 股年报三张表（东方财富，经 akshare）→ 统一口径的年度财务表（单位：亿元）。

口径说明（面试时要能讲清楚）：
  EBIT      = 利润总额 + 利息费用 − 利息收入        （剔除财务收支，现金和负债在估值时单独加减）
  税率       = 所得税 / 利润总额
  D&A       = 固定资产折旧 + 无形资产摊销 + 长期待摊费用摊销 + 使用权资产折旧
  资本开支    = 购建固定资产、无形资产和其他长期资产支付的现金
  经营性营运资本 = (流动资产 − 货币资金 − 交易性金融资产) − (流动负债 − 短期借款 − 一年内到期的非流动负债)
  有息负债    = 短期借款 + 长期借款 + 应付债券 + 一年内到期的非流动负债 + 租赁负债
  现金       = 货币资金 + 交易性金融资产
"""
from __future__ import annotations

import pandas as pd

from cyron_veritas.data.market import _cached

YI = 1e8
FIN_KEYWORDS = ("银行", "保险", "证券", "信托", "期货")


def _symbol(ticker: str) -> str:
    return ("SH" if ticker.startswith(("6", "9")) else "SZ") + ticker


def _get(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df.columns else pd.Series(0.0, index=df.index)


def annual(ticker: str, years: int = 6) -> dict:
    """返回 {'name', 'table'(DataFrame, 行=年份), 'source'}。"""
    import akshare as ak

    sym = _symbol(ticker)
    inc = _cached(f"inc_{ticker}", lambda: ak.stock_profit_sheet_by_yearly_em(symbol=sym))
    bal = _cached(f"bal_{ticker}", lambda: ak.stock_balance_sheet_by_yearly_em(symbol=sym))
    cf = _cached(f"cf_{ticker}", lambda: ak.stock_cash_flow_sheet_by_yearly_em(symbol=sym))
    name = str(inc["SECURITY_NAME_ABBR"].iloc[0])

    def by_year(df):
        df = df.copy()
        df["year"] = pd.to_datetime(df["REPORT_DATE"]).dt.year
        return df.drop_duplicates("year").set_index("year").sort_index()

    inc, bal, cf = by_year(inc), by_year(bal), by_year(cf)
    yrs = sorted(set(inc.index) & set(bal.index) & set(cf.index))[-years:]
    inc, bal, cf = inc.loc[yrs], bal.loc[yrs], cf.loc[yrs]

    t = pd.DataFrame(index=yrs)
    t["revenue"] = _get(inc, "TOTAL_OPERATE_INCOME") / YI
    t["gross_profit"] = (_get(inc, "OPERATE_INCOME") - _get(inc, "OPERATE_COST")) / YI
    t["pretax_profit"] = _get(inc, "TOTAL_PROFIT") / YI
    t["interest_expense"] = _get(inc, "FE_INTEREST_EXPENSE") / YI
    t["interest_income"] = _get(inc, "FE_INTEREST_INCOME") / YI
    t["ebit"] = t["pretax_profit"] + t["interest_expense"] - t["interest_income"]
    t["income_tax"] = _get(inc, "INCOME_TAX") / YI
    t["net_profit_parent"] = _get(inc, "PARENT_NETPROFIT") / YI
    t["da"] = (_get(cf, "FA_IR_DEPR") + _get(cf, "IA_AMORTIZE") + _get(cf, "LPE_AMORTIZE") + _get(cf, "USERIGHT_ASSET_AMORTIZE")) / YI
    t["capex"] = _get(cf, "CONSTRUCT_LONG_ASSET") / YI
    t["cfo"] = _get(cf, "NETCASH_OPERATE") / YI
    cash = _get(bal, "MONETARYFUNDS") + _get(bal, "TRADE_FINASSET_NOTFVTPL")
    st_debt = _get(bal, "SHORT_LOAN") + _get(bal, "NONCURRENT_LIAB_1YEAR")
    t["cash"] = cash / YI
    t["debt"] = (st_debt + _get(bal, "LONG_LOAN") + _get(bal, "BOND_PAYABLE") + _get(bal, "LEASE_LIAB")) / YI
    t["nwc"] = ((_get(bal, "TOTAL_CURRENT_ASSETS") - cash) - (_get(bal, "TOTAL_CURRENT_LIAB") - st_debt)) / YI
    t["equity_parent"] = _get(bal, "TOTAL_PARENT_EQUITY") / YI
    t["minority"] = _get(bal, "MINORITY_EQUITY") / YI
    t["shares"] = _get(bal, "SHARE_CAPITAL") / YI  # A 股面值 1 元：股本（元）= 股数，这里单位亿股
    t["total_assets"] = _get(bal, "TOTAL_ASSETS") / YI
    t["receivables"] = _get(bal, "ACCOUNTS_RECE") / YI
    t["inventory"] = _get(bal, "INVENTORY") / YI

    # 常用比率
    t["revenue_growth"] = t["revenue"].pct_change()
    t["gross_margin"] = t["gross_profit"] / t["revenue"]
    t["ebit_margin"] = t["ebit"] / t["revenue"]
    t["tax_rate"] = t["income_tax"] / t["pretax_profit"]
    t["da_pct"] = t["da"] / t["revenue"]
    t["capex_pct"] = t["capex"] / t["revenue"]
    t["nwc_pct"] = t["nwc"] / t["revenue"]
    t["roe"] = t["net_profit_parent"] / t["equity_parent"]
    t["cfo_to_np"] = t["cfo"] / t["net_profit_parent"]  # 利润含金量（经营现金流 / 归母净利润）
    t["fcff"] = t["ebit"] * (1 - t["tax_rate"]) + t["da"] - t["capex"] - t["nwc"].diff()
    t["receivable_days"] = t["receivables"] / t["revenue"] * 365
    t["inventory_days"] = t["inventory"] / (t["revenue"] - t["gross_profit"]) * 365

    return {"name": name, "ticker": ticker, "is_financial": any(k in name for k in FIN_KEYWORDS), "table": t,
            "source": "akshare（东方财富年报：利润表、资产负债表、现金流量表）", "unit": "亿元"}
