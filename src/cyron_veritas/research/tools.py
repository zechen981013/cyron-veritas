"""研究 Agent 的工具。所有数字都由这些工具用代码算出，每次调用记进「证据账本」并返回证据编号（E1、E2…）。

Agent 写备忘录时只能引用工具结果里的数字；生成后再做一次数字溯源，统计有多少数字能在账本里找到。
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field

import httpx
import numpy as np
import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from cyron_veritas.data import financials as F
from cyron_veritas.data import market
from cyron_veritas.risk import metrics as M
from cyron_veritas.valuation import dcf as D
from cyron_veritas.valuation import fin as FV

DEFAULT_ERP = 0.06  # A 股股权风险溢价假设（常见取值 5%-7%）
ERP_BY_MARKET = {"A": 0.06, "HK": 0.065, "US": 0.05}
PRICE_CCY = {"A": "CNY", "HK": "HKD", "US": "USD"}
BENCH = {"A": ("CSI300", "沪深300"), "HK": ("HSI", "恒生指数"), "US": ("SPX", "标普500")}


@dataclass
class Ledger:
    items: list = field(default_factory=list)
    dcf_runs: list = field(default_factory=list)

    def add(self, tool: str, args: dict, result: dict) -> dict:
        eid = f"E{len(self.items) + 1}"
        self.items.append({"id": eid, "tool": tool, "args": args, "result": result,
                           "time": dt.datetime.now().isoformat(timespec="seconds")})
        return {"evidence_id": eid, **result}


def _r(x, nd=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), nd)


# ---------- 计算函数（不依赖大模型，可单测） ----------
def financials_summary(ticker: str) -> dict:
    ticker = F.market_of(ticker)[1]
    fin = F.annual(ticker)
    t = fin["table"]
    cols = {"revenue": 1, "revenue_growth": 3, "gross_margin": 3, "ebit": 1, "ebit_margin": 3, "tax_rate": 3,
            "net_profit_parent": 1, "roe": 3, "cfo": 1, "cfo_to_np": 2, "da_pct": 3, "capex_pct": 3, "nwc_pct": 3,
            "fcff": 1, "cash": 1, "cash_like_other": 1, "debt": 1, "minority": 1, "shares": 2, "receivable_days": 0, "inventory_days": 0}
    years = {int(y): {k: _r(t.loc[y, k], nd) for k, nd in cols.items()} for y in t.index}
    last3 = t.iloc[-3:]
    avg = {f"avg3_{k}": _r(last3[k].mean(), 3) for k in ["revenue_growth", "ebit_margin", "tax_rate", "da_pct", "capex_pct", "nwc_pct", "roe"]}
    return {"company": fin["name"], "ticker": ticker, "market": fin.get("market", "A"), "is_financial": fin["is_financial"],
            "unit": f"亿{fin.get('currency', 'CNY')}（报表币种；比率为小数）", "currency_note": fin.get("currency_note", ""),
            "latest_year": int(t.index[-1]), "years": years, **avg, "source": fin["source"]}


def market_snapshot(ticker: str) -> dict:
    mkt, ticker = F.market_of(ticker)
    px = market.price_history(ticker, mkt)
    bench = market.price_history(BENCH[mkt][0], "INDEX")
    a = F.annual(ticker)
    fin = a["table"].iloc[-1].copy()
    if a["is_financial"]:  # 金融企业：PB 用普通股净资产（扣除优先股、永续债）
        fi = F.financial_institution(ticker)["table"].iloc[-1]
        fin["equity_parent"], fin["shares"], fin["net_profit_parent"] = fi["common_equity"], fi["shares"], fi["net_profit_parent"]
    last, date = float(px.iloc[-1]), str(px.index[-1].date())
    one_year = px[px.index >= px.index[-1] - pd.Timedelta(days=365)]
    def reg(freq: str, n: int) -> tuple[float, float]:
        a = px.resample(freq).last().pct_change().dropna().iloc[-n:]
        b = bench.resample(freq).last().pct_change().dropna()
        r2 = pd.concat([a, b], axis=1).dropna().corr().iloc[0, 1] ** 2
        return M.beta(a, b), float(r2)

    b_w, r2_w = reg("W-FRI", 104)   # 近 2 年周线
    b_m, r2_m = reg("ME", 60)        # 近 5 年月线
    use_long = r2_w < 0.1            # 回归解释力太低时，近两年的 Beta 不可靠
    raw_beta = b_m if use_long else b_w
    price_ccy, stmt_ccy = PRICE_CCY[mkt], a.get("currency", "CNY")
    fx = market.fx_rate(stmt_ccy, price_ccy) if stmt_ccy != price_ccy else 1.0
    mcap = last * fin["shares"] / fx  # 换算成报表币种，和利润、净资产同一币种
    return {"ticker": ticker, "market": mkt, "price_currency": price_ccy, "statement_currency": stmt_ccy,
            "fx_statement_to_price": _r(fx, 4), "price": _r(last, 2), "price_date": date, "high_52w": _r(one_year.max(), 2),
            "low_52w": _r(one_year.min(), 2), "return_1y": _r(last / one_year.iloc[0] - 1, 3),
            "market_cap": _r(mcap, 1), "pe_ttm_approx": _r(mcap / fin["net_profit_parent"], 1),
            "pb": _r(mcap / fin["equity_parent"], 2),
            "beta_2y_weekly": _r(b_w, 2), "r2_2y_weekly": _r(r2_w, 3), "beta_5y_monthly": _r(b_m, 2), "r2_5y_monthly": _r(r2_m, 3),
            "beta_used": "5 年月线" if use_long else "2 年周线", "beta_adjusted": _r(D.blume_beta(raw_beta), 2), "benchmark": BENCH[mkt][1],
            "note": f"市值以报表币种（{stmt_ccy}）计；PE、PB 用最新年报利润和净资产近似；Beta 对{BENCH[mkt][1]}回归，近 2 年周线 R² 低于 0.1 时改用近 5 年月线；"
                    "Blume 调整 = 0.67×原始 + 0.33",
            "source": "akshare（日线前复权、基准指数）"}


def wacc_inputs(ticker: str, erp: float | None = None) -> dict:
    mkt, ticker = F.market_of(ticker)
    erp = ERP_BY_MARKET[mkt] if erp is None else erp
    mk = market_snapshot(ticker)
    t = F.annual(ticker)["table"]
    last = t.iloc[-1]
    avg_debt = t["debt"].iloc[-2:].mean()
    kd = float(np.clip(last["interest_expense"] / avg_debt, 0.02, 0.08)) if avg_debt > 0 else 0.04
    rf = market.risk_free_rate(mkt)
    w = D.wacc(rf["rate"], mk["beta_adjusted"], erp, kd, float(last["tax_rate"]), mk["market_cap"], float(last["debt"]))
    return {**{k: _r(v, 4) for k, v in w.items()}, "rf_date": rf["date"], "equity_value": mk["market_cap"],
            "note": f"rf = {'中国' if mkt == 'A' else '美国'} 10 年期国债收益率；β 为 Blume 调整后；ERP 取 {erp:.0%}（假设）；债务成本 = 利息费用 / 平均有息负债，限制在 2%-8%",
            "source": rf["source"]}


def dcf_run(ticker: str, scenario: str, revenue_growth: list[float], ebit_margin: list[float], tax_rate: float,
            da_pct: float, capex_pct: float, nwc_pct: float, wacc: float, terminal_growth: float,
            notes: dict | None = None, extra_cash: float = 0.0) -> tuple[dict, D.DCFInputs | None]:
    t = F.annual(ticker)["table"]
    last = t.iloc[-1]
    snap = market_snapshot(ticker)
    x = D.DCFInputs(base_year=int(t.index[-1]), base_revenue=float(last["revenue"]), fx_to_price=float(snap["fx_statement_to_price"]),
                    base_nwc=float(last["revenue"]) * nwc_pct,  # 基期营运资本按假设比例标准化
                    revenue_growth=list(revenue_growth), ebit_margin=list(ebit_margin), tax_rate=tax_rate, da_pct=da_pct,
                    capex_pct=capex_pct, nwc_pct=nwc_pct, wacc=wacc, terminal_growth=terminal_growth,
                    debt=float(last["debt"]), cash=float(last["cash"]), minority=float(last["minority"]),
                    shares=float(last["shares"]), extra_cash=float(extra_cash), notes={**(notes or {}),
                    "base_nwc": "基期营运资本按「营运资本/收入」假设标准化，避免单年科目重分类造成的跳变"})
    errs = x.validate()
    if errs:
        return {"ok": False, "errors": errs}, None
    v = D.value(x)
    price = snap["price"]
    sens = D.sensitivity(x)
    proj = v["projection"]
    return {"ok": True, "scenario": scenario, "per_share": _r(v["per_share"], 2), "price": price,
            "currency": snap["price_currency"], "statement_currency": snap["statement_currency"], "fx_statement_to_price": snap["fx_statement_to_price"],
            "vs_price": _r(v["per_share"] / price - 1, 3), "enterprise_value": _r(v["enterprise_value"], 1),
            "equity_value": _r(v["equity_value"], 1), "pv_fcff_sum": _r(v["pv_fcff_sum"], 1),
            "pv_terminal": _r(v["pv_terminal"], 1), "tv_share_of_ev": _r(v["tv_share_of_ev"], 3),
            "fcff_by_year": {int(y): _r(f, 1) for y, f in proj["fcff"].items()},
            "revenue_by_year": {int(y): _r(r, 1) for y, r in proj["revenue"].items()},
            "sensitivity_per_share": {f"wacc={w:.3f}": {f"g={g:.3f}": _r(val, 1) for g, val in row.items()}
                                      for w, row in sens.iterrows()},
            "net_debt": _r(x.debt - x.cash, 1), "extra_cash": _r(x.extra_cash, 1), "minority": _r(x.minority, 1), "shares": _r(x.shares, 2),
            "assumptions": {"revenue_growth": revenue_growth, "ebit_margin": ebit_margin, "tax_rate": tax_rate,
                            "da_pct": da_pct, "capex_pct": capex_pct, "nwc_pct": nwc_pct, "wacc": wacc,
                            "terminal_growth": terminal_growth, "extra_cash": extra_cash}}, x


def fin_summary(ticker: str) -> dict:
    d = F.financial_institution(ticker)
    t = d["table"]
    pct_cols = {"roe", "roa", "nii_share", "credit_cost", "loan_to_deposit", "profit_growth", "bvps_growth", "payout"}
    years = {int(y): {k: _r(t.loc[y, k], 4 if k in pct_cols else 2) for k in t.columns} for y in t.index}
    last3 = t.iloc[-3:]
    return {"company": d["name"], "ticker": ticker, "unit": d["unit"], "latest_year": int(t.index[-1]), "years": years,
            "avg3_roe": _r(last3["roe"].mean(), 4), "avg3_payout": _r(last3["payout"].mean(), 4),
            "avg3_profit_growth": _r(last3["profit_growth"].mean(), 4), "note": d["note"], "source": d["source"]}


def fin_run(ticker: str, scenario: str, roe: float, ke: float, g: float, g1: float, payout1: float,
            notes: dict | None = None) -> tuple[dict, FV.FinInputs | None]:
    t = F.financial_institution(ticker)["table"]
    last = t.iloc[-1]
    x = FV.FinInputs(base_year=int(t.index[-1]), bvps=float(last["bvps"]), eps=float(last["eps"]), roe=roe, ke=ke, g=g,
                     g1=g1, payout1=payout1, notes=notes or {})
    errs = x.validate()
    if errs:
        return {"ok": False, "errors": errs}, None
    pb, dd = FV.pb_roe(x), FV.ddm(x)
    price = market_snapshot(ticker)["price"]
    avg = (pb["per_share"] + dd["per_share"]) / 2
    sens = FV.sensitivity(x)
    cur_pb = price / x.bvps
    implied_ke = g + (roe - g) / cur_pb  # 按现价 P/B 和假设 ROE 倒推的股权成本：和模型 Ke 差很多时说明假设与市场分歧大
    return {"ok": True, "scenario": scenario, "bvps": _r(x.bvps, 2), "eps": _r(x.eps, 2), "price": price,
            "current_pb": _r(cur_pb, 2), "justified_pb": _r(pb["justified_pb"], 2), "market_implied_ke": _r(implied_ke, 4),
            "pb_roe_per_share": _r(pb["per_share"], 2), "ddm_per_share": _r(dd["per_share"], 2), "average_per_share": _r(avg, 2),
            "vs_price": _r(avg / price - 1, 3), "ddm_terminal_payout": _r(dd["terminal_payout"], 3), "ddm_tv_share": _r(dd["tv_share"], 3),
            "sensitivity_pb_roe": {f"ke={k:.3f}": {f"g={g_:.3f}": _r(v, 1) for g_, v in row.items()} for k, row in sens.iterrows()},
            "assumptions": {"roe": roe, "ke": ke, "g": g, "g1": g1, "payout1": payout1}}, x


def web_search(query: str, max_results: int = 5) -> dict:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return {"results": [], "error": "未配置 TAVILY_API_KEY"}
    r = httpx.post("https://api.tavily.com/search", timeout=40,
                   json={"api_key": key, "query": query, "max_results": max_results, "search_depth": "basic", "topic": "news", "days": 365})
    r.raise_for_status()
    res = [{"title": x.get("title"), "url": x.get("url"), "date": x.get("published_date"),
            "snippet": (x.get("content") or "")[:400]} for x in r.json().get("results", [])]
    return {"query": query, "results": res, "source": "Tavily 新闻搜索（近一年）"}


def annual_report_search(ticker: str, query: str) -> dict:
    """在已入库的最新年报里检索（混合检索 + 重排），返回原文片段和页码。"""
    from cyron_veritas.rag import index as I

    years = sorted(int(k.split("_")[1]) for k in I.manifest() if k.startswith(ticker + "_"))
    if not years:
        return {"ok": False, "error": f"{ticker} 的年报还没有入库"}
    y = years[-1]
    meta = I.manifest()[f"{ticker}_{y}"]
    hits = I.search(query, ticker, y, top=4)
    return {"ok": True, "report": f"{meta['company']} {y} 年年度报告", "url": meta["page_url"],
            "unit_note": "年报表格金额单位多为人民币千元",
            "passages": [{"page": f"p.{d['page_start']}" + (f"-{d['page_end']}" if d["page_end"] != d["page_start"] else ""),
                          "section": d["section"], "subsection": d["subsection"], "text": d["text"][:900]} for d in hits]}


# ---------- 包成 LangChain 工具（带账本） ----------
class TickerArgs(BaseModel):
    ticker: str = Field(description="股票代码：A 股 6 位（000333）、港股 5 位（00700）、美股字母（AAPL）")


class WaccArgs(TickerArgs):
    erp: float | None = Field(None, description="股权风险溢价，小数；默认 A 股 6%、港股 6.5%、美股 5%")


class DCFArgs(TickerArgs):
    scenario: str = Field(description="情景名称：base / bull / bear")
    revenue_growth: list[float] = Field(description="未来 5 年每年收入增速，小数，例如 [0.07,0.06,0.05,0.045,0.04]")
    ebit_margin: list[float] = Field(description="未来 5 年每年 EBIT 利润率，小数，长度与 revenue_growth 相同")
    tax_rate: float
    da_pct: float = Field(description="折旧摊销 / 收入")
    capex_pct: float = Field(description="资本开支 / 收入")
    nwc_pct: float = Field(description="经营性营运资本 / 收入（可为负）")
    wacc: float
    terminal_growth: float
    notes: dict[str, str] = Field(default_factory=dict, description="每个假设的依据，键为假设名，值写清锚定的历史数据或理由")
    extra_cash: float = Field(0.0, description="其他类现金资产（亿元）：年报附注证实计入其他科目的定期存款 / 理财。默认 0；填写时 notes['extra_cash'] 必须写明科目、金额和年报页码")


class ReportArgs(TickerArgs):
    query: str = Field(description="要在年报里查找的内容，用年报措辞，例如「其他流动资产 定期存款」「分地区收入 海外」「可能面对的风险」")


class FinValArgs(TickerArgs):
    scenario: str = Field(description="情景名称：base / bull / bear")
    roe: float = Field(description="长期可持续 ROE，小数")
    ke: float = Field(description="股权成本，小数（参考 get_wacc_inputs 的 cost_of_equity）")
    g: float = Field(description="永续增长率，小数")
    g1: float = Field(description="未来 5 年 EPS 增速，小数")
    payout1: float = Field(description="未来 5 年派息率，小数")
    notes: dict[str, str] = Field(default_factory=dict, description="每个假设的依据")


class SearchArgs(BaseModel):
    query: str = Field(description="搜索关键词，例如「美的集团 2026 海外收入 关税」")


def make_tools(ledger: Ledger) -> dict[str, StructuredTool]:
    def wrap(name, fn, schema, desc, post=None):
        def run(**kw):
            try:
                result = fn(**kw)
            except Exception as e:  # 工具失败返回错误信息，让 Agent 自己决定怎么办
                result = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            if post:
                result = post(kw, result)
            out = ledger.add(name, kw, result)
            return json.dumps(out, ensure_ascii=False, default=str)[:6000]
        return StructuredTool.from_function(run, name=name, description=desc, args_schema=schema)

    def dcf_post(kw, res):
        if isinstance(res, tuple):
            res, x = res
            if x is not None:
                ledger.dcf_runs.append({"scenario": kw["scenario"], "inputs": x, "result": res,
                                        "per_share": res.get("per_share") or res.get("average_per_share")})
        return res

    return {
        "get_financials": wrap("get_financials", financials_summary, TickerArgs,
                               "取 A 股公司近 6 年年报关键财务数据和比率（收入、利润率、ROE、现金流、营运资本、资本开支、负债），以及近 3 年均值。"),
        "get_market_data": wrap("get_market_data", market_snapshot, TickerArgs,
                                "取最新股价、52 周高低、近一年涨跌、市值、PE/PB（近似）、Beta（近两年周线，含 Blume 调整）。"),
        "get_wacc_inputs": wrap("get_wacc_inputs", wacc_inputs, WaccArgs,
                                "用 CAPM 计算参考 WACC：无风险利率、Beta、ERP、债务成本、资本结构权重。"),
        "run_dcf": wrap("run_dcf", dcf_run, DCFArgs,
                        "按给定假设运行 5 年 FCFF 折现模型，返回每股价值、相对现价、终值占比、敏感性表。假设不合理会返回错误让你修改。",
                        post=dcf_post),
        "search_news": wrap("search_news", web_search, SearchArgs, "搜索近一年的公司和行业新闻，返回标题、链接、日期、摘要。"),
        "get_fin_metrics": wrap("get_fin_metrics", fin_summary, TickerArgs,
                                "金融企业（银行、保险、券商）近 6 年指标：归母净利润、普通股净资产、BVPS、EPS、ROE、ROA、净利息收入占比、信用成本、贷存比、权益乘数、分红与派息率。"),
        "run_fin_valuation": wrap("run_fin_valuation", fin_run, FinValArgs,
                                  "金融企业估值：PB-ROE（合理 P/B =（ROE−g）/（Ke−g））和两阶段 DDM，返回每股价值、相对现价、敏感性表。假设不合理会返回错误。",
                                  post=dcf_post),
        "search_annual_report": wrap("search_annual_report", annual_report_search, ReportArgs,
                                     "在公司最新年报原文里检索（混合检索 + 重排），返回原文片段、章节和页码。用于核实业务结构、科目明细、风险披露。"),
    }
