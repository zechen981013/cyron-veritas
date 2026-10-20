"""产品与合同配置（config/products/*.yaml）。"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml

from cyron_veritas.config import ROOT

PRODUCT_DIR = ROOT / "config" / "products"


def load(code: str) -> dict:
    cfg = yaml.safe_load((PRODUCT_DIR / f"{code}.yaml").read_text(encoding="utf-8"))
    for f in cfg.get("flows", []):
        f["date"] = _d(f["date"])
    cfg["inception"] = _d(cfg["inception"])
    return cfg


def all_codes() -> list[str]:
    return sorted(p.stem for p in PRODUCT_DIR.glob("*.yaml"))


def _d(x) -> dt.date:
    return x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x))


def sheet_dir(code: str) -> Path:
    p = ROOT / "data" / "valuation_sheets" / code
    p.mkdir(parents=True, exist_ok=True)
    return p
