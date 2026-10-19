"""A 股年报三张表（东方财富，经 akshare）→ 统一口径的年度财务表（单位：亿元）。

口径说明（面试时要能讲清楚）：
  EBIT      = 利润总额 + 利息费用 − 利息收入        （剔除财务收支，现金和负债在估值时单独加减）
  税率       = 所得税 / 利润总额
  D&A       = 固定资产折旧 + 无形资产摊销 + 长期待摊费用摊销 + 使用权资产折旧
  资本开支    = 购建固定资产、无形资产和其他长期资产支付的现金
  经营性营运资本 = (应收账款 + 应收票据 + 应收款项融资 + 预付款项 + 存货 + 合同资产)
                − (应付账款 + 应付票据 + 合同负债 + 预收款项 + 应付职工薪酬 + 应交税费)
               只用经营性科目：「其他流动资产」「一年内到期的非流动资产」里常有定期存款和理财，混进来会让营运资本失真
  其他类现金资产（候选）= 一年内到期的非流动资产 + 其他流动资产 + 其他非流动资产 + 其他非流动金融资产
               只作提示，不自动计入现金；是否属于存款 / 理财要到年报附注里核实（DCF 工具的 extra_cash 参数）
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


def market_of(ticker: str) -> tuple[str, str]:
    """识别市场并规范代码：6 位数字 → A 股；1-5 位数字 → 港股（补零到 5 位）；字母 → 美股。"""
    t = ticker.strip().upper()
    if t.isdigit() and len(t) == 6:
        return "A", t
    if t.isdigit() and len(t) <= 5:
        return "HK", t.zfill(5)
    if t.replace(".", "").replace("-", "").isalpha():
        return "US", t
    raise ValueError(f"无法识别的代码：{ticker}")


def annual(ticker: str, years: int = 6) -> dict:
    """统一入口：返回 {'name', 'market', 'currency'（报表币种）, 'table'(行=年份，单位亿), 'is_financial', 'source'}。"""
    mkt, t = market_of(ticker)
    if mkt == "HK":
        return _annual_hk(t, years)
    if mkt == "US":
        return _annual_us(t, years)
    out = _annual_a(t, years)
    out.update(market="A", currency="CNY")
    return out


def _ratios(t: pd.DataFrame) -> pd.DataFrame:
    """三个市场共用的比率（输入列口径一致）。"""
    t["revenue_growth"] = t["revenue"].pct_change()
    t["gross_margin"] = t["gross_profit"] / t["revenue"]
    t["ebit_margin"] = t["ebit"] / t["revenue"]
    t["tax_rate"] = t["income_tax"] / t["pretax_profit"]
    t["da_pct"] = t["da"] / t["revenue"]
    t["capex_pct"] = t["capex"] / t["revenue"]
    t["nwc_pct"] = t["nwc"] / t["revenue"]
    t["roe"] = t["net_profit_parent"] / t["equity_parent"]
    t["cfo_to_np"] = t["cfo"] / t["net_profit_parent"]
    t["fcff"] = t["ebit"] * (1 - t["tax_rate"]) + t["da"] - t["capex"] - t["nwc"].diff()
    t["receivable_days"] = t["receivables"] / t["revenue"] * 365
    t["inventory_days"] = t["inventory"] / (t["revenue"] - t["gross_profit"]) * 365
    return t


def _long_to_wide(df: pd.DataFrame, name_col: str, years: int) -> pd.DataFrame:
    d = df.copy()
    d["year"] = pd.to_datetime(d["REPORT_DATE"]).dt.year
    w = d.pivot_table(index="year", columns=name_col, values="AMOUNT", aggfunc="first").sort_index()
    return w.iloc[-years:]


def _col(w: pd.DataFrame, *names: str) -> pd.Series:
    out = pd.Series(0.0, index=w.index)
    for n in names:
        if n in w.columns:
            out = out + pd.to_numeric(w[n], errors="coerce").fillna(0)
    return out


def _annual_hk(ticker: str, years: int) -> dict:
    """港股（东方财富港股财报，长表 → 宽表）。EBIT 直接用「经营溢利」；股本取已发行股本；报表币种按市值和市净率反推。"""
    import akshare as ak

    get = lambda sym: _cached(f"hk_{sym}_{ticker}", lambda: ak.stock_financial_hk_report_em(stock=ticker, symbol=sym, indicator="年度"))
    inc, bal, cf = get("利润表"), get("资产负债表"), get("现金流量表")
    name = str(inc["SECURITY_NAME_ABBR"].iloc[0])
    wi, wb, wc = (_long_to_wide(x, "STD_ITEM_NAME", years + 1) for x in (inc, bal, cf))
    yrs = sorted(set(wi.index) & set(wb.index) & set(wc.index))[-years:]
    wi, wb, wc = wi.loc[yrs], wb.loc[yrs], wc.loc[yrs]
    t = pd.DataFrame(index=yrs)
    t["revenue"] = _col(wi, "营业额") / YI
    t["gross_profit"] = _col(wi, "毛利") / YI
    t["pretax_profit"] = _col(wi, "除税前溢利") / YI
    t["interest_expense"] = _col(wi, "融资成本") / YI
    t["interest_income"] = _col(wi, "利息收入") / YI
    t["ebit"] = _col(wi, "经营溢利") / YI
    t["income_tax"] = _col(wi, "税项") / YI
    t["net_profit_parent"] = _col(wi, "股东应占溢利") / YI
    t["da"] = _col(wc, "加:折旧及摊销") / YI
    t["capex"] = (_col(wc, "购建固定资产").abs() + _col(wc, "购建无形资产及其他资产").abs()) / YI
    t["cfo"] = _col(wc, "经营业务现金净额") / YI
    t["cash"] = _col(wb, "现金及等价物", "短期存款") / YI
    t["cash_like_other"] = _col(wb, "中长期存款", "交易性金融资产(流动)", "其他金融资产(流动)", "指定以公允价值记账之金融资产(流动)") / YI
    t["debt"] = _col(wb, "短期贷款", "长期贷款", "应付票据", "应付票据(非流动)", "融资租赁负债(流动)", "融资租赁负债(非流动)") / YI
    t["nwc"] = (_col(wb, "应收帐款", "存货", "预付款项", "预付款按金及其他应收款")
                - _col(wb, "应付帐款", "其他应付款及应计费用", "递延收入(流动)", "应付税项")) / YI
    t["equity_parent"] = _col(wb, "股东权益") / YI
    t["minority"] = _col(wb, "少数股东权益") / YI
    t["total_assets"] = _col(wb, "总资产") / YI
    t["receivables"] = _col(wb, "应收帐款") / YI
    t["inventory"] = _col(wb, "存货") / YI

    ind = _cached(f"hkind_{ticker}", lambda: ak.stock_hk_financial_indicator_em(symbol=ticker))
    row = ind.iloc[0]
    shares = float(row["已发行股本(股)"]) / YI
    t["shares"] = shares
    # 报表币种：每股市价 / 市净率 = 以港元计的每股净资产；和报表上的每股净资产之比就是汇率
    price_hkd = float(row["总市值(港元)"]) / float(row["已发行股本(股)"])
    ratio = price_hkd / float(row["市净率"]) / float(row["每股净资产(元)"])
    from cyron_veritas.data.market import fx_history
    fx = fx_history().iloc[-1]
    cands = {"HKD": 1.0, "CNY": 1 / fx["HKD"], "USD": fx["USD"] / fx["HKD"]}  # 1 单位该币种 = 多少港元
    currency = min(cands, key=lambda c: abs(ratio / cands[c] - 1))
    t = _ratios(t)
    return {"name": name, "ticker": ticker, "market": "HK", "currency": currency, "table": t,
            "is_financial": any(k in name for k in FIN_KEYWORDS + ("友邦", "人寿", "保诚")),
            "source": "akshare（东方财富港股年报：利润表、资产负债表、现金流量表；财务指标）", "unit": f"亿{currency}",
            "currency_note": f"报表币种按市值 / 市净率反推（比值 {ratio:.3f}）"}


def _annual_us(ticker: str, years: int) -> dict:
    """美股（东方财富美股财报）。EBIT 用「营业利润」；股本用基本加权平均股数。暂只支持以美元报告的公司（不含中概 ADR）。"""
    import akshare as ak

    get = lambda sym: _cached(f"us_{sym}_{ticker}", lambda: ak.stock_financial_us_report_em(stock=ticker, symbol=sym, indicator="年报"))
    inc, bal, cf = get("综合损益表"), get("资产负债表"), get("现金流量表")
    name = str(inc["SECURITY_NAME_ABBR"].iloc[0])
    wi, wb, wc = (_long_to_wide(x, "ITEM_NAME", years + 1) for x in (inc, bal, cf))
    yrs = sorted(set(wi.index) & set(wb.index) & set(wc.index))[-years:]
    wi, wb, wc = wi.loc[yrs], wb.loc[yrs], wc.loc[yrs]
    t = pd.DataFrame(index=yrs)
    rev = _col(wi, "营业收入")
    t["revenue"] = (rev.where(rev != 0, _col(wi, "主营收入"))) / YI
    t["gross_profit"] = _col(wi, "毛利") / YI
    t["pretax_profit"] = _col(wi, "持续经营税前利润") / YI
    t["interest_expense"] = 0.0
    t["interest_income"] = _col(wi, "利息收入") / YI
    t["ebit"] = _col(wi, "营业利润") / YI
    t["income_tax"] = _col(wi, "所得税") / YI
    np_ = _col(wi, "归属于母公司股东净利润")
    t["net_profit_parent"] = np_.where(np_ != 0, _col(wi, "归属于普通股股东净利润")) / YI
    t["da"] = _col(wc, "折旧及摊销") / YI
    t["capex"] = (_col(wc, "购买固定资产").abs() + _col(wc, "购建无形资产及其他资产").abs()) / YI
    t["cfo"] = _col(wc, "经营活动产生的现金流量净额") / YI
    t["cash"] = _col(wb, "现金及现金等价物", "短期投资") / YI
    t["cash_like_other"] = _col(wb, "有价证券投资(流动)", "有价证券投资(非流动)") / YI
    t["debt"] = _col(wb, "短期债务", "长期负债", "长期负债(本期部分)", "应付票据(流动)") / YI
    t["nwc"] = (_col(wb, "应收账款", "存货", "其他应收款") - _col(wb, "应付账款", "预收及预提费用", "递延收入(流动)")) / YI
    t["equity_parent"] = _col(wb, "归属于母公司股东权益") / YI
    t["minority"] = (_col(wb, "股东权益合计") - _col(wb, "归属于母公司股东权益")).clip(lower=0) / YI
    t["total_assets"] = _col(wb, "总资产") / YI
    t["receivables"] = _col(wb, "应收账款") / YI
    t["inventory"] = _col(wb, "存货") / YI
    t["shares"] = _col(wi, "基本加权平均股数-普通股") / YI
    t = _ratios(t)
    return {"name": name, "ticker": ticker, "market": "US", "currency": "USD", "table": t,
            "is_financial": any(k in name for k in FIN_KEYWORDS + ("摩根", "高盛", "富国", "花旗", "伯克希尔")),
            "source": "akshare（东方财富美股年报）", "unit": "亿USD",
            "currency_note": "按美元报告；中概 ADR（人民币报表、ADS 比例）暂不支持"}


def _annual_a(ticker: str, years: int = 6) -> dict:
    """A 股：返回 {'name', 'table'(DataFrame, 行=年份), 'source'}。"""
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
    op_assets = sum(_get(bal, c) for c in ("ACCOUNTS_RECE", "NOTE_RECE", "FINANCE_RECE", "PREPAYMENT", "INVENTORY", "CONTRACT_ASSET"))
    op_liabs = sum(_get(bal, c) for c in ("ACCOUNTS_PAYABLE", "NOTE_PAYABLE", "CONTRACT_LIAB", "ADVANCE_RECEIVABLES",
                                         "STAFF_SALARY_PAYABLE", "TAX_PAYABLE"))
    t["nwc"] = (op_assets - op_liabs) / YI
    t["cash_like_other"] = sum(_get(bal, c) for c in ("NONCURRENT_ASSET_1YEAR", "OTHER_CURRENT_ASSET", "OTHER_NONCURRENT_ASSET",
                                                     "OTHER_NONCURRENT_FINASSET")) / YI
    t["equity_parent"] = _get(bal, "TOTAL_PARENT_EQUITY") / YI
    t["minority"] = _get(bal, "MINORITY_EQUITY") / YI
    t["shares"] = _get(bal, "SHARE_CAPITAL") / YI  # A 股面值 1 元：股本（元）= 股数，这里单位亿股
    t["total_assets"] = _get(bal, "TOTAL_ASSETS") / YI
    t["receivables"] = _get(bal, "ACCOUNTS_RECE") / YI
    t["inventory"] = _get(bal, "INVENTORY") / YI

    t = _ratios(t)

    return {"name": name, "ticker": ticker, "is_financial": any(k in name for k in FIN_KEYWORDS), "table": t,
            "source": "akshare（东方财富年报：利润表、资产负债表、现金流量表）", "unit": "亿元"}


def financial_institution(ticker: str, years: int = 6) -> dict:
    """银行 / 保险 / 券商的关键指标（亿元，比率为小数）。普通股净资产 = 归母权益 − 其他权益工具（优先股、永续债）。"""
    import akshare as ak

    sym = _symbol(ticker)
    inc = _cached(f"inc_{ticker}", lambda: ak.stock_profit_sheet_by_yearly_em(symbol=sym))
    bal = _cached(f"bal_{ticker}", lambda: ak.stock_balance_sheet_by_yearly_em(symbol=sym))
    name = str(inc["SECURITY_NAME_ABBR"].iloc[0])

    def by_year(df):
        df = df.copy()
        df["year"] = pd.to_datetime(df["REPORT_DATE"]).dt.year
        return df.drop_duplicates("year").set_index("year").sort_index()

    inc, bal = by_year(inc), by_year(bal)
    yrs = sorted(set(inc.index) & set(bal.index))[-years:]
    inc, bal = inc.loc[yrs], bal.loc[yrs]
    t = pd.DataFrame(index=yrs)
    t["operating_income"] = _get(inc, "OPERATE_INCOME") / YI
    t["net_interest_income"] = _get(inc, "INTEREST_NI") / YI
    t["fee_income"] = _get(inc, "FEE_COMMISSION_NI") / YI
    t["credit_impairment"] = _get(inc, "CREDIT_IMPAIRMENT_LOSS") / YI
    t["net_profit_parent"] = _get(inc, "PARENT_NETPROFIT") / YI
    t["total_assets"] = _get(bal, "TOTAL_ASSETS") / YI
    t["loans"] = _get(bal, "LOAN_ADVANCE") / YI
    t["deposits"] = _get(bal, "ACCEPT_DEPOSIT") / YI
    t["other_equity_tools"] = _get(bal, "OTHER_EQUITY_TOOL") / YI
    t["common_equity"] = _get(bal, "TOTAL_PARENT_EQUITY") / YI - t["other_equity_tools"]
    t["shares"] = _get(bal, "SHARE_CAPITAL") / YI
    t["bvps"] = t["common_equity"] / t["shares"]
    t["eps"] = t["net_profit_parent"] / t["shares"]
    t["roe"] = t["net_profit_parent"] / ((t["common_equity"] + t["common_equity"].shift()) / 2)
    t["roa"] = t["net_profit_parent"] / ((t["total_assets"] + t["total_assets"].shift()) / 2)
    t["nii_share"] = t["net_interest_income"] / t["operating_income"]
    t["credit_cost"] = t["credit_impairment"] / t["loans"].replace(0, float("nan"))
    t["loan_to_deposit"] = t["loans"] / t["deposits"].replace(0, float("nan"))
    t["leverage"] = t["total_assets"] / t["common_equity"]
    t["profit_growth"] = t["net_profit_parent"].pct_change()
    t["bvps_growth"] = t["bvps"].pct_change()

    dps = {}
    try:
        dv = _cached(f"div_{ticker}", lambda: ak.stock_history_dividend_detail(symbol=ticker, indicator="分红"))
        dv = dv[dv["进度"] == "实施"].copy()
        dv["year"] = pd.to_datetime(dv["除权除息日"]).dt.year
        dps = (pd.to_numeric(dv["派息"], errors="coerce") / 10).groupby(dv["year"]).sum().to_dict()
    except Exception:
        pass
    t["dps_paid"] = [dps.get(y, float("nan")) for y in t.index]  # 当年实际派发（含上年末期 + 当年中期）
    t["payout"] = t["dps_paid"] / t["eps"].shift()                # 近似：当年派发 / 上年 EPS
    return {"name": name, "ticker": ticker, "table": t, "unit": "亿元（每股为元，比率为小数）",
            "source": "akshare（东方财富年报、分红送配）",
            "note": "ROE 用平均普通股净资产；普通股净资产已扣除优先股和永续债；EPS 未扣除永续债利息，略有高估"}
