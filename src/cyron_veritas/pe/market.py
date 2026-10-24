"""PE 组合公司上市后的行情与公司行为（腾讯日线，不复权，2020 年起；和二级市场产品的中证 800 缓存分开存）。"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from cyron_veritas.config import ROOT
from cyron_veritas.fund import marketdata as M

DIR = ROOT / "data" / "pe"


@lru_cache(maxsize=64)
def bars(code: str) -> pd.DataFrame:
    """日线（index = 日期；close, high, low, volume[股], amount, turnover）。"""
    DIR.mkdir(parents=True, exist_ok=True)
    p = DIR / f"bars_{code}.parquet"
    if not p.exists():
        import akshare as ak

        df = M._retry(lambda: ak.stock_zh_a_hist_tx(symbol=M.prefix(code) + code, start_date="20200101", end_date="20991231", adjust="", timeout=30))
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = code
        df["turnover"] = pd.to_numeric(df.get("turnover"), errors="coerce")
        df = M._volume_in_shares(df[["date", "code", "open", "close", "high", "low", "volume", "amount", "turnover"]])
        df.to_parquet(p)
    return pd.read_parquet(p).set_index("date").sort_index()


def refresh(code: str) -> None:
    (DIR / f"bars_{code}.parquet").unlink(missing_ok=True)
    bars.cache_clear()


@lru_cache(maxsize=64)
def actions(code: str) -> pd.DataFrame:
    """已实施的分红送转（ex_date, cash_per_share, bonus_per_share）。"""
    return M.corporate_actions([code])
