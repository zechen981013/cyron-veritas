"""风险限额检查：纯规则，不经大模型。输出每条限额的实际值、状态（正常 / 预警 / 超限）。"""
from __future__ import annotations

from pathlib import Path

import yaml

from cyron_veritas.config import CONFIG_DIR

OK, WARN, BREACH = "正常", "预警", "超限"


def load_limits(path: Path | None = None) -> dict:
    return yaml.safe_load((path or CONFIG_DIR / "limits.yaml").read_text(encoding="utf-8"))


def _worst(items: dict, limit: float, skip: set) -> tuple[float, str]:
    """取最大值；说明里列出所有超过限额的项目（不只最大的那个）。"""
    items = {k: v for k, v in items.items() if k not in skip}
    top, val = max(items.items(), key=lambda kv: kv[1])
    over = [k for k, v in sorted(items.items(), key=lambda kv: -kv[1]) if v > limit]
    return val, "、".join(over) if len(over) > 1 else top


def _actuals(f: dict, cfg: dict) -> dict[str, tuple[float, str]]:
    """每条限额对应的实际值和说明（说明里写清是哪个持仓 / 行业 / 市场）。"""
    lim = {k: v["value"] for k, v in cfg["limits"].items()}
    names = {k: v for k, v in f["weights"].items() if k in f["pnl_pct"]}  # 只看非现金持仓
    foreign = sum(v for k, v in f["by_currency"].items() if k != "CNY")
    return {
        "single_name_max": _worst(names, lim["single_name_max"], set()),
        "top5_max": (f["concentration"]["top5"], "前五大持仓"),
        "sector_max": _worst(f["by_sector"], lim["sector_max"], {"现金"}),
        "market_max": _worst(f["by_market"], lim["market_max"], {"CASH"}),
        "foreign_ccy_max": (foreign, "美元 + 港元资产"),
        "var95_1d_max": (f["var95_hist"], "历史模拟法"),
        "drawdown_alert": (f["current_drawdown"], "距近两年高点"),
        "cash_min": (f["cash_weight"], "现金"),
    }


def check(facts: dict, cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_limits()
    warn = cfg.get("warn_ratio", 0.9)
    actual = _actuals(facts, cfg)
    out = []
    for key, spec in cfg["limits"].items():
        value, note = actual[key]
        limit = spec["value"]
        if spec.get("direction") == "min":
            status = BREACH if value < limit else (WARN if value < limit / warn else OK)
        else:
            status = BREACH if value > limit else (WARN if value > limit * warn else OK)
        out.append({"key": key, "label": spec["label"], "limit": limit, "value": value, "note": note,
                    "status": status, "direction": spec.get("direction", "max")})
    return out
