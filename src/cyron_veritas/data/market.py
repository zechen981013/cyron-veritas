"""行情、汇率、无风险利率。数据来自 akshare（公开数据），按天缓存到 data/cache，避免重复请求。"""
from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd

from cyron_veritas.config import CACHE_DIR

warnings.filterwarnings("ignore")

# 各市场的基准指数（算 Beta 用）
BENCHMARKS = {"A": "CSI300", "HK": "HSI", "US": "SPX"}


def _cached(key: str, fetch) -> pd.DataFrame:
    """同一天内同一个 key 只请求一次。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}_{dt.date.today():%Y%m%d}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    df = fetch()
    df.to_pickle(path)
    return df


def _to_series(df: pd.DataFrame, date_col: str, close_col: str) -> pd.Series:
    s = pd.Series(pd.to_numeric(df[close_col], errors="coerce").values, index=pd.to_datetime(df[date_col]))
    return s.sort_index().dropna()


def price_history(ticker: str, market: str, start: str = "20190101") -> pd.Series:
    """日收盘价（前复权，本币）。market: A / HK / US / INDEX。"""
    import akshare as ak

    def fetch():
        if market == "A":
            try:
                df = ak.stock_zh_a_hist(symbol=ticker, period="daily", start_date=start, adjust="qfq")
                return pd.DataFrame({"date": df["日期"], "close": df["收盘"]})
            except Exception:  # 东方财富接口失败时退回新浪
                prefix = "sh" if ticker.startswith(("6", "9", "5")) else "sz"
                df = ak.stock_zh_a_daily(symbol=prefix + ticker, start_date=start, adjust="qfq")
                return df[["date", "close"]]
        if market == "HK":
            return ak.stock_hk_daily(symbol=ticker, adjust="qfq")[["date", "close"]]
        if market == "US":
            return ak.stock_us_daily(symbol=ticker, adjust="qfq")[["date", "close"]]
        if market == "INDEX":
            if ticker == "CSI300":
                return ak.stock_zh_index_daily(symbol="sh000300")[["date", "close"]]
            if ticker == "HSI":
                return ak.stock_hk_index_daily_sina(symbol="HSI")[["date", "close"]]
            if ticker == "SPX":
                return ak.index_us_stock_sina(symbol=".INX")[["date", "close"]]
        raise ValueError(f"不支持的市场或代码: {market} {ticker}")

    s = _to_series(_cached(f"px_{market}_{ticker}", fetch), "date", "close")
    return s[s.index >= pd.Timestamp(start)]


def fx_history() -> pd.DataFrame:
    """人民币中间价：每 1 单位外币兑多少人民币。列：USD、HKD。"""
    import akshare as ak

    def fetch():
        df = ak.currency_boc_safe()
        return pd.DataFrame({"date": df["日期"], "USD": df["美元"], "HKD": df["港元"]})

    df = _cached("fx_boc", fetch)
    out = pd.DataFrame({"USD": pd.to_numeric(df["USD"], errors="coerce").values / 100,
                        "HKD": pd.to_numeric(df["HKD"], errors="coerce").values / 100}, index=pd.to_datetime(df["date"]).values)
    out["CNY"] = 1.0
    return out.sort_index().dropna()


def risk_free_rate(market: str = "A") -> dict:
    """10 年期国债收益率（最新一天，小数）：A 股用中国国债；港股（港元联系汇率）和美股用美国国债。"""
    import akshare as ak

    start = (dt.date.today() - dt.timedelta(days=60)).strftime("%Y%m%d")
    df = _cached("cn_bond", lambda: ak.bond_zh_us_rate(start_date=start))
    col = "中国国债收益率10年" if market == "A" else "美国国债收益率10年"
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return {"rate": float(s.iloc[-1]) / 100, "date": str(pd.to_datetime(df["日期"]).iloc[s.index[-1]].date()),
            "source": f"akshare bond_zh_us_rate（{col}）"}


def fx_rate(from_ccy: str, to_ccy: str) -> float:
    """最新中间价换算：1 单位 from_ccy = ? to_ccy。"""
    last = fx_history().iloc[-1]
    return float(last[from_ccy] / last[to_ccy])
