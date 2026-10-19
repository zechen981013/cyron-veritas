"""金融企业（银行、保险、券商）估值：PB-ROE 与两阶段 DDM。

为什么不用 DCF：金融企业的负债（存款、保单）是经营原料而不是融资，「自由现金流」和「营运资本」没有意义，
行业惯例是看净资产回报和分红。

  PB-ROE（戈登增长推导）：合理 P/B = (ROE − g) / (Ke − g)，每股价值 = 合理 P/B × 每股普通股净资产
  两阶段 DDM：第一阶段 N 年每股收益按 g1 增长、按派息率 p1 分红；
             第二阶段永续增长 g2，派息率 = 1 − g2 / ROE（可持续增长：留存的利润按 ROE 再投资）
             每股价值 = Σ DPS_t / (1+Ke)^t + [EPS_N × (1+g2) × (1 − g2/ROE)] / (Ke − g2) / (1+Ke)^N
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass
class FinInputs:
    base_year: int
    bvps: float              # 每股普通股净资产（元）
    eps: float               # 基期每股收益（元，归母净利润 / 总股本，已扣其他权益工具利息的近似）
    roe: float               # 长期可持续 ROE
    ke: float                # 股权成本
    g: float                 # 永续增长率
    g1: float                # 第一阶段 EPS 增速
    payout1: float           # 第一阶段派息率
    years: int = 5
    notes: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        errs = []
        if not (0.04 <= self.ke <= 0.18):
            errs.append("股权成本应在 4%-18% 之间")
        if self.g >= self.ke - 0.01:
            errs.append("永续增长率必须明显低于股权成本（至少低 1 个百分点）")
        if not (-0.02 <= self.g <= 0.04):
            errs.append("永续增长率应在 -2% 到 4% 之间")
        if not (0 < self.roe <= 0.30):
            errs.append("ROE 应在 0-30% 之间")
        if self.g >= self.roe:
            errs.append("永续增长率不能超过 ROE（否则可持续派息率为负）")
        if not (0 <= self.payout1 <= 1):
            errs.append("派息率应在 0-100% 之间")
        if not (3 <= self.years <= 10):
            errs.append("第一阶段应为 3-10 年")
        return errs


def pb_roe(x: FinInputs, ke: float | None = None, g: float | None = None) -> dict:
    ke = x.ke if ke is None else ke
    g = x.g if g is None else g
    pb = (x.roe - g) / (ke - g)
    return {"justified_pb": pb, "per_share": pb * x.bvps}


def ddm(x: FinInputs, ke: float | None = None, g: float | None = None) -> dict:
    ke = x.ke if ke is None else ke
    g = x.g if g is None else g
    rows, eps = [], x.eps
    for t in range(1, x.years + 1):
        eps = eps * (1 + x.g1)
        dps = eps * x.payout1
        rows.append({"year": x.base_year + t, "t": t, "eps": eps, "dps": dps, "pv": dps / (1 + ke) ** t})
    p = pd.DataFrame(rows).set_index("year")
    payout2 = 1 - g / x.roe
    tv = eps * (1 + g) * payout2 / (ke - g)
    pv_tv = tv / (1 + ke) ** x.years
    value = p["pv"].sum() + pv_tv
    return {"projection": p, "terminal_payout": payout2, "terminal_value": tv, "pv_terminal": pv_tv,
            "per_share": value, "tv_share": pv_tv / value}


def sensitivity(x: FinInputs, method: str = "pb_roe", step: float = 0.005, n: int = 2) -> pd.DataFrame:
    fn = pb_roe if method == "pb_roe" else ddm
    kes = [round(x.ke + i * step, 4) for i in range(-n, n + 1)]
    gs = [round(x.g + j * step, 4) for j in range(-n, n + 1)]
    return pd.DataFrame([[fn(x, k, g)["per_share"] if g < k - 0.005 and g < x.roe else float("nan") for g in gs] for k in kes],
                        index=kes, columns=gs)


def to_dict(x: FinInputs) -> dict:
    return asdict(x)
