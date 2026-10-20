"""每日盘后流水线：估值表入库 → 估值核对 → 风险快照与限额检查。

    uv run veritas fund daily CV-VG1               # 处理最新一张估值表
    uv run veritas fund backfill CV-VG1            # 从成立日起全部重算
定时：交易日 17:30（托管行估值表一般在收盘后发出），launchd / cron 配置见 README。
"""
from __future__ import annotations

import datetime as dt

from cyron_veritas.fund import db, ops
from cyron_veritas.fund import risk as R


def daily(code: str, date: str | None = None, say=print) -> dict:
    last = db.df("select max(date) as d from nav where product = :p", p=code)["d"].iloc[0]
    since = dt.date.fromisoformat(str(last)) if last else None
    n = ops.load_sheets(code, since=since)
    say(f"新估值表 {n} 张")
    rec = ops.reconcile(code)
    d = date or str(db.df("select max(date) as d from nav where product = :p", p=code)["d"].iloc[0])
    out = R.daily(code, d)
    bad = [c for c in out["checks"] if c["status"] != R.OK]
    say(f"{d} 单位净值 {out['snapshot']['unit_nav']:.4f}；估值核对 {rec.iloc[-1]['status'] if len(rec) else '—'}；限额异常 {len(bad)} 项")
    return {"date": d, "new_sheets": n, "checks": out["checks"], "snapshot": out["snapshot"]}


def backfill(code: str, say=print) -> dict:
    n = ops.load_sheets(code)
    rec = ops.reconcile(code)
    dates = db.df("select date from nav where product = :p order by date", p=code)["date"].tolist()
    events = 0
    for i, d in enumerate(dates):
        if i < 20:   # 前 20 个交易日历史太短，不计算风险快照
            continue
        out = R.daily(code, d)
        events += sum(c["status"] != R.OK for c in out["checks"])
        if i % 100 == 0:
            say(f"  {d} 已处理 {i}/{len(dates)}")
    return {"sheets": n, "reconciled": len(rec), "days": len(dates), "limit_events": events}
