"""资管中台的市场数据（全部来自 akshare 的公开接口，按文件缓存，增量更新）。

  - 不复权日线（腾讯）：收盘价、成交量 → 估值表行情、流动性（20 日均量）
  - 分红送转（东方财富）：除权除息日、每股派息、每股送转股
  - 申万一级行业成分 → 行业映射（行业限额、Brinson 归因）
  - 指数日线（沪深300、中证500）→ 业绩基准；沪深300 成分权重 → Brinson 基准权重
  - 业绩报表（全市场，按报告期）→ 基本面选股（只在披露截止日之后使用，避免前视）
"""
from __future__ import annotations

import datetime as dt
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from cyron_veritas.config import ROOT

warnings.filterwarnings("ignore")
MD = ROOT / "data" / "fund_md"
START = "20230101"


def _p(name: str):
    MD.mkdir(parents=True, exist_ok=True)
    return MD / f"{name}.parquet"


def _retry(fn, n=3):
    for i in range(n):
        try:
            return fn()
        except Exception:
            if i == n - 1:
                raise
            time.sleep(2 * (i + 1))


def prefix(code: str) -> str:
    return "sh" if code.startswith(("6", "9", "5")) else ("bj" if code.startswith(("4", "8")) else "sz")


def universe() -> pd.DataFrame:
    """中证 800 当前成分股（代码、名称、交易所）。"""
    p = _p("universe_csi800")
    if not p.exists():
        import akshare as ak
        c = _retry(lambda: ak.index_stock_cons_csindex(symbol="000906"))
        pd.DataFrame({"code": c["成分券代码"].astype(str).str.zfill(6), "name": c["成分券名称"],
                      "exchange": c["交易所"]}).to_parquet(p)
    return pd.read_parquet(p)


def _bars(code: str) -> pd.DataFrame | None:
    import akshare as ak
    try:
        df = _retry(lambda: ak.stock_zh_a_hist_tx(symbol=prefix(code) + code, start_date=START, end_date="20991231", adjust="", timeout=15))
    except Exception:
        return None
    df = df.rename(columns={"date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = code
    # turnover 是换手率（小数）；成交量单位不统一，见 _volume_in_shares
    df["turnover"] = pd.to_numeric(df.get("turnover"), errors="coerce")
    return df[["date", "code", "open", "close", "high", "low", "volume", "amount", "turnover"]]


def bars(codes: list[str], refresh: bool = False) -> pd.DataFrame:
    """不复权日线长表（date, code, open, close, high, low, volume[股], amount[元], turnover[换手率]）。"""
    p = _p("bars_raw")
    have = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["date", "code"])
    have_codes = set(have["code"].unique())  # 只算一次（写在推导式里会对 70 多万行重复遍历 800 次）
    todo = [c for c in codes if refresh or c not in have_codes]
    if todo:
        got = []
        with ThreadPoolExecutor(4) as ex:
            futs = {ex.submit(_bars, c): c for c in todo}
            for k, f in enumerate(as_completed(futs)):
                df = f.result()
                if df is not None:
                    got.append(df)
                if k % 100 == 0:
                    print(f"  日线 {k}/{len(todo)}", flush=True)
        if got:
            new = pd.concat(got)
            have = pd.concat([have[~have["code"].isin(new["code"].unique())], new]).sort_values(["code", "date"])
            have.to_parquet(p)
    return _volume_in_shares(have[have["code"].isin(codes)])


def _volume_in_shares(b: pd.DataFrame) -> pd.DataFrame:
    """腾讯接口的成交量单位因股票而异：实测中证 800 里 706 只是「股」、93 只是「手」、1 只是「股 × 100」（同一只股票内一致）。
    以成交额（元）为准：成交额 ÷ 典型价 ≈ 成交股数，按每只股票的数量级众数把 volume 统一换算成股。"""
    b = b.copy()
    est = b["amount"] / ((b["high"] + b["low"] + b["close"]) / 3)
    ok = (b["volume"] > 0) & (est > 0)
    mag = np.log10(b.loc[ok, "volume"] / est[ok]).round()
    scale = mag.groupby(b.loc[ok, "code"]).agg(lambda s: s.mode().iloc[0])
    b["volume"] = b["volume"] / (10.0 ** b["code"].map(scale).fillna(0))
    return b


def wide(b: pd.DataFrame, field: str) -> pd.DataFrame:
    return b.pivot(index="date", columns="code", values=field).sort_index()


def _actions_one(code: str) -> pd.DataFrame | None:
    import akshare as ak
    try:
        d = _retry(lambda: ak.stock_history_dividend_detail(symbol=code, indicator="分红"))
    except Exception:
        return None
    d = d[d["进度"] == "实施"].copy()
    if d.empty:
        return None
    d["ex_date"] = pd.to_datetime(d["除权除息日"], errors="coerce")
    d = d.dropna(subset=["ex_date"])
    return pd.DataFrame({"code": code, "ex_date": d["ex_date"],
                         "cash_per_share": pd.to_numeric(d["派息"], errors="coerce").fillna(0) / 10,
                         "bonus_per_share": (pd.to_numeric(d["送股"], errors="coerce").fillna(0)
                                             + pd.to_numeric(d["转增"], errors="coerce").fillna(0)) / 10})


def corporate_actions(codes: list[str]) -> pd.DataFrame:
    """现金分红（每股，税前）与送转（每股送转股数），按除权除息日。"""
    p = _p("corporate_actions")
    have = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["code", "ex_date", "cash_per_share", "bonus_per_share"])
    done = set(pd.read_parquet(_p("ca_done"))["code"]) if _p("ca_done").exists() else set()
    todo = [c for c in codes if c not in done]
    if todo:
        got = []
        with ThreadPoolExecutor(4) as ex:
            for f in as_completed([ex.submit(_actions_one, c) for c in todo]):
                if (df := f.result()) is not None:
                    got.append(df)
        if got:
            have = pd.concat([have, *got]).drop_duplicates(["code", "ex_date"])
            have.to_parquet(p)
        pd.DataFrame({"code": sorted(done | set(todo))}).to_parquet(_p("ca_done"))
    return have[have["code"].isin(codes)]


def sw_industry() -> pd.Series:
    """代码 → 申万一级行业名称。"""
    p = _p("sw_l1")
    if not p.exists():
        import akshare as ak
        info = _retry(lambda: ak.sw_index_first_info())
        rows = []
        for code, name in zip(info["行业代码"], info["行业名称"]):
            comp = _retry(lambda: ak.index_component_sw(symbol=code.split(".")[0]))
            rows += [(str(c).zfill(6), name) for c in comp["证券代码"]]
        pd.DataFrame(rows, columns=["code", "industry"]).drop_duplicates("code").to_parquet(p)
    s = pd.read_parquet(p)
    return s.set_index("code")["industry"]


def index_close(symbol: str) -> pd.Series:
    """指数日收盘：sh000300 沪深300、sh000905 中证500。"""
    p = _p(f"idx_{symbol}")
    if not p.exists() or dt.date.fromtimestamp(p.stat().st_mtime) < dt.date.today():
        import akshare as ak
        df = _retry(lambda: ak.stock_zh_index_daily(symbol=symbol))
        pd.DataFrame({"close": df["close"].values}, index=pd.to_datetime(df["date"])).to_parquet(p)
    return pd.read_parquet(p)["close"]


def index_weights(symbol: str = "000300") -> pd.Series:
    """指数成分权重（中证指数公司最新月末权重，小数）。历史权重需要付费数据，Brinson 用最新权重近似（报告中注明）。"""
    p = _p(f"w_{symbol}")
    if not p.exists():
        import akshare as ak
        w = _retry(lambda: ak.index_stock_cons_weight_csindex(symbol=symbol))
        pd.DataFrame({"code": w["成分券代码"].astype(str).str.zfill(6), "weight": w["权重"].astype(float) / 100}).to_parquet(p)
    s = pd.read_parquet(p)
    return s.set_index("code")["weight"]


def index_weights_hist(symbol: str = "000300", start: str = "20231201") -> pd.DataFrame | None:
    """指数历史成分权重（Tushare index_weight，月度，需要 2000 积分）。没有 TUSHARE_TOKEN 时返回 None，调用方退回近似算法。
    直接调 Tushare HTTP 接口，不依赖 tushare 包。返回列：date, code, weight（小数，每期合计约 1）。"""
    import os

    token = os.getenv("TUSHARE_TOKEN")
    p = _p(f"wh_{symbol}")
    if not token and not p.exists():
        return None
    if not p.exists() or dt.date.fromtimestamp(p.stat().st_mtime) < dt.date.today() - dt.timedelta(days=7):
        import requests

        ts_code = f"{symbol}.SH" if symbol.startswith("000") else f"{symbol}.SZ"
        rows, end = [], dt.date.today()
        cur = pd.Timestamp(start)
        while cur.date() <= end:                     # 按年分段拉，单次返回有行数上限
            nxt = min(cur + pd.DateOffset(years=1) - pd.Timedelta(days=1), pd.Timestamp(end))
            r = requests.post("http://api.tushare.pro", json={"api_name": "index_weight", "token": token, "fields": "",
                                                               "params": {"index_code": ts_code, "start_date": f"{cur:%Y%m%d}", "end_date": f"{nxt:%Y%m%d}"}},
                              timeout=30).json()
            if r.get("code"):
                raise RuntimeError(f"Tushare index_weight：{r.get('msg')}")
            rows += [dict(zip(r["data"]["fields"], it)) for it in r["data"]["items"]]
            cur = nxt + pd.Timedelta(days=1)
        w = pd.DataFrame(rows)
        pd.DataFrame({"date": pd.to_datetime(w["trade_date"]), "code": w["con_code"].str[:6],
                      "weight": w["weight"].astype(float) / 100}).to_parquet(p)
    return pd.read_parquet(p)


# ---- 基本面（业绩报表）：披露截止日之后才能用 ----
DEADLINE = {"0331": "0430", "0630": "0831", "0930": "1031", "1231": "0430"}


def available_period(on: pd.Timestamp) -> str:
    """在日期 on 时，最新一个已过披露截止日的报告期。"""
    cands = []
    for y in range(on.year - 2, on.year + 1):
        for p, d in DEADLINE.items():
            dy = y + 1 if p == "1231" else y
            if pd.Timestamp(f"{dy}{d}") < on:
                cands.append(f"{y}{p}")
    return max(cands)


def fundamentals(period: str) -> pd.DataFrame:
    p = _p(f"yjbb_{period}")
    if not p.exists():
        import akshare as ak
        d = _retry(lambda: ak.stock_yjbb_em(date=period))
        out = pd.DataFrame({"code": d["股票代码"].astype(str).str.zfill(6), "eps": pd.to_numeric(d["每股收益"], errors="coerce"),
                            "bvps": pd.to_numeric(d["每股净资产"], errors="coerce"), "roe": pd.to_numeric(d["净资产收益率"], errors="coerce"),
                            "rev_yoy": pd.to_numeric(d["营业总收入-同比增长"], errors="coerce"),
                            "np_yoy": pd.to_numeric(d["净利润-同比增长"], errors="coerce"),
                            "gross_margin": pd.to_numeric(d["销售毛利率"], errors="coerce"),
                            "ocfps": pd.to_numeric(d["每股经营现金流量"], errors="coerce")})
        out.drop_duplicates("code").to_parquet(p)
    return pd.read_parquet(p).set_index("code")


def sw_index_close() -> pd.DataFrame:
    """申万一级行业指数日收盘（列 = 行业名称）。行业收益用指数本身，避免用「当前成分股」倒推历史的幸存者偏差。"""
    p = _p("sw_index_close")
    if not p.exists() or dt.date.fromtimestamp(p.stat().st_mtime) < dt.date.today():
        import akshare as ak
        info = _retry(lambda: ak.sw_index_first_info())
        cols = {}
        for code, name in zip(info["行业代码"], info["行业名称"]):
            d = _retry(lambda: ak.index_hist_sw(symbol=code.split(".")[0], period="day"))
            cols[name] = pd.Series(pd.to_numeric(d["收盘"], errors="coerce").values, index=pd.to_datetime(d["日期"]))
        pd.DataFrame(cols).sort_index().to_parquet(p)
    return pd.read_parquet(p)
