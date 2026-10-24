"""后见之明压力测试。

组合里的真实项目都是「后来上市成功」的公司——投资时并不知道。这里假设每个真实项目只有 p 的概率按实际路径上市退出，
否则按未上市项目的典型结局处理：40% 核销（0x）、40% 对赌回购但只收回 70% 左右（0.7x）、20% 被并购或老股转让（1.2x）。
虚构项目保持原样。按投资额加权得到组合毛回报倍数（Gross MOIC）的分布，1 万次模拟，固定随机种子。
这些失败结局的比例是假设，不是统计数据，用来说明结论对「上市成功率」有多敏感。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cyron_veritas.fund import db

FAIL = np.array([0.0, 0.7, 1.2])
FAIL_P = np.array([0.4, 0.4, 0.2])


def deal_moic(code: str = "CV-PE1") -> pd.DataFrame:
    tx = db.df("select deal, type, amount from pe_tx where product = :p and deal != 'FUND'", p=code)
    marks = db.df("select deal, fv from pe_marks where product = :p and date = (select max(date) from pe_marks where product = :p)", p=code)
    inv = -tx[tx["type"] == "投资"].groupby("deal")["amount"].sum()
    out = tx[tx["amount"] > 0].groupby("deal")["amount"].sum().reindex(inv.index).fillna(0)
    fv = marks.groupby("deal")["fv"].sum().reindex(inv.index).fillna(0)
    return pd.DataFrame({"invested": inv, "moic": (out + fv) / inv, "synthetic": inv.index.str.startswith("SYN")})


def run(code: str = "CV-PE1", probs=(1.0, 0.5, 0.3), n: int = 10_000, seed: int = 7) -> list[dict]:
    d = deal_moic(code)
    rng = np.random.default_rng(seed)
    real = d[~d["synthetic"]]
    syn = d[d["synthetic"]]
    w_tot = d["invested"].sum()
    rows = []
    for p in probs:
        listed = rng.random((n, len(real))) < p
        fail = rng.choice(FAIL, size=(n, len(real)), p=FAIL_P)
        m = np.where(listed, real["moic"].values, fail)
        port = (m * real["invested"].values).sum(axis=1) + (syn["moic"] * syn["invested"]).sum()
        port /= w_tot
        rows.append({"ipo_rate": p, "mean": float(port.mean()), "p10": float(np.percentile(port, 10)), "p50": float(np.median(port)),
                     "p90": float(np.percentile(port, 90)), "prob_below_1": float((port < 1).mean())})
    return rows
