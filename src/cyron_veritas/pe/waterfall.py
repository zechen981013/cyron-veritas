"""欧式（整体）收益分配瀑布，按事件逐笔计算，可做「假设清算」得出 LP 净值和应计业绩报酬。

顺序（只对 LP 类别；GP 自己的认缴部分不收业绩报酬，按比例直接分）：
  1. 返还实缴：LP 累计实缴（含管理费、基金费用）全部返还
  2. 门槛收益：未返还本金 + 未付门槛收益，按年 8% 复利计提，付清为止
  3. GP 追补：100% 给 GP，直到 GP 累计拿到全部已分配利润的 20%
  4. 超额：80% 给 LP、20% 给 GP
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Waterfall:
    hurdle: float = 0.08
    carry: float = 0.20
    catch_up: float = 1.0
    capital_out: float = 0.0          # 未返还的实缴
    pref_out: float = 0.0             # 已计提未支付的门槛收益
    last: pd.Timestamp | None = None
    paid: dict = field(default_factory=lambda: {"返还实缴": 0.0, "门槛收益": 0.0, "GP 追补": 0.0, "超额收益（LP 80%）": 0.0, "超额收益（GP 20%）": 0.0})

    def _accrue(self, d: pd.Timestamp) -> None:
        if self.last is not None and d > self.last:
            yrs = (d - self.last).days / 365
            self.pref_out += (self.capital_out + self.pref_out) * ((1 + self.hurdle) ** yrs - 1)
        self.last = d if self.last is None or d > self.last else self.last

    def contribute(self, d, amount: float) -> None:
        self._accrue(pd.Timestamp(d))
        self.capital_out += amount

    @property
    def lp_profit(self) -> float:
        return self.paid["门槛收益"] + self.paid["超额收益（LP 80%）"]

    @property
    def gp_carry(self) -> float:
        return self.paid["GP 追补"] + self.paid["超额收益（GP 20%）"]

    def distribute(self, d, amount: float) -> dict:
        """分配 amount 给 LP 类别 + GP 业绩报酬，返回本次各档金额。"""
        self._accrue(pd.Timestamp(d))
        x, out = amount, {k: 0.0 for k in self.paid}
        a = min(x, self.capital_out)
        self.capital_out -= a
        out["返还实缴"], x = a, x - a
        b = min(x, self.pref_out)
        self.pref_out -= b
        out["门槛收益"], x = b, x - b
        # 追补：GP 拿到「已分配利润」的 carry 比例为止。target 满足 gp = carry × (lp_profit + gp)
        lp_profit = self.lp_profit + b
        target = self.carry / (1 - self.carry) * lp_profit - self.gp_carry
        c = min(x, max(target, 0.0) / self.catch_up) if self.catch_up > 0 else 0.0
        out["GP 追补"], x = c * self.catch_up, x - c
        out["超额收益（LP 80%）"] = x * (1 - self.carry) + c * (1 - self.catch_up)
        out["超额收益（GP 20%）"] = x * self.carry
        for k, v in out.items():
            self.paid[k] += v
        out["LP 合计"] = out["返还实缴"] + out["门槛收益"] + out["超额收益（LP 80%）"]
        out["GP 合计"] = out["GP 追补"] + out["超额收益（GP 20%）"]
        return out

    def hypothetical(self, d, nav: float) -> dict:
        """假设在 d 日按 nav 清算：LP 应得多少、GP 应计业绩报酬多少（不改变当前状态）。"""
        w = copy.deepcopy(self)
        r = w.distribute(d, max(nav, 0.0))
        return {"lp_nav": r["LP 合计"], "accrued_carry": r["GP 合计"], "pref_outstanding": w.pref_out}
