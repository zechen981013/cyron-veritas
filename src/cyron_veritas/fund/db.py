"""中台数据库（SQLAlchemy Core，默认 SQLite：data/platform.db；设置 PLATFORM_DB_URL 可换 Postgres）。"""
from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import (JSON, Column, Date, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine, delete, func,
                        insert, select)

from cyron_veritas.config import ROOT

meta = MetaData()

products = Table("products", meta, Column("code", String(32), primary_key=True), Column("name", String(128)), Column("type", String(32)),
                 Column("config", JSON))
# 解析后的估值表：汇总行
nav = Table("nav", meta, Column("product", String(32), primary_key=True), Column("date", Date, primary_key=True),
            Column("unit_nav", Float), Column("acc_nav", Float), Column("net_assets", Float), Column("shares", Float),
            Column("total_assets", Float), Column("total_liab", Float), Column("cash", Float), Column("stock_mv", Float),
            Column("receivable", Float), Column("fee_payable", Float), Column("sheet_file", String(256)), Column("loaded_at", DateTime))
# 解析后的估值表：股票明细
holdings = Table("holdings", meta, Column("product", String(32), primary_key=True), Column("date", Date, primary_key=True),
                 Column("code", String(12), primary_key=True), Column("name", String(32)), Column("exchange", String(8)),
                 Column("qty", Float), Column("unit_cost", Float), Column("cost", Float), Column("price", Float), Column("mv", Float),
                 Column("weight", Float), Column("unrealized", Float), Column("suspended", String(16)))
trades = Table("trades", meta, Column("id", Integer, primary_key=True, autoincrement=True), Column("product", String(32)),
               Column("date", Date), Column("code", String(12)), Column("side", String(4)), Column("qty", Float), Column("price", Float),
               Column("amount", Float), Column("fee", Float), Column("reason", String(64)))
flows = Table("flows", meta, Column("id", Integer, primary_key=True, autoincrement=True), Column("product", String(32)), Column("date", Date),
              Column("type", String(16)), Column("amount", Float), Column("shares", Float), Column("nav", Float))
risk = Table("risk_snapshots", meta, Column("product", String(32), primary_key=True), Column("date", Date, primary_key=True),
             Column("metrics", JSON))
limit_events = Table("limit_events", meta, Column("id", Integer, primary_key=True, autoincrement=True), Column("product", String(32)),
                     Column("date", Date), Column("rule", String(64)), Column("value", Float), Column("limit", Float), Column("status", String(8)),
                     Column("detail", Text), Column("handled_by", String(32)), Column("handled_at", DateTime), Column("note", Text))
reconcile = Table("reconciliation", meta, Column("product", String(32), primary_key=True), Column("date", Date, primary_key=True),
                  Column("sheet_nav", Float), Column("est_nav", Float), Column("diff", Float), Column("explained", JSON), Column("status", String(8)))
pool = Table("stock_pool", meta, Column("id", Integer, primary_key=True, autoincrement=True), Column("product", String(32)),
             Column("code", String(12)), Column("name", String(32)), Column("pool", String(8)), Column("since", Date), Column("until", Date),
             Column("reason", Text), Column("decision_id", Integer))
ic = Table("ic_decisions", meta, Column("id", Integer, primary_key=True, autoincrement=True), Column("product", String(32)),
           Column("code", String(12)), Column("name", String(32)), Column("request", String(16)), Column("requested_by", String(32)),
           Column("requested_at", Date), Column("material", Text), Column("decision", String(16)), Column("votes", JSON),
           Column("decided_at", Date), Column("note", Text))
orders = Table("pretrade_checks", meta, Column("id", Integer, primary_key=True, autoincrement=True), Column("product", String(32)),
               Column("date", Date), Column("orders", JSON), Column("result", JSON), Column("created_at", DateTime, server_default=func.now()))

# 量化产品每次调仓：优化结果与执行统计
quant_rebalances = Table("quant_rebalances", meta, Column("id", Integer, primary_key=True, autoincrement=True), Column("product", String(32)),
                         Column("date", Date), Column("exec_start", Date), Column("exec_end", Date), Column("status", String(32)),
                         Column("relaxed", JSON), Column("ex_ante_te", Float), Column("exp_alpha", Float), Column("turnover", Float),
                         Column("n_holdings", Integer), Column("member_weight", Float), Column("bench_source", String(32)), Column("exposure", JSON),
                         Column("fill_rate", Float), Column("unfilled_value", Float), Column("traded_value", Float), Column("impact_bp", Float),
                         Column("created_at", DateTime))

_engine = None


def engine():
    global _engine
    if _engine is None:
        url = os.getenv("PLATFORM_DB_URL") or f"sqlite:///{ROOT / 'data' / 'platform.db'}"
        (ROOT / "data").mkdir(exist_ok=True)
        _engine = create_engine(url, future=True)
        meta.create_all(_engine)
    return _engine


def df(sql_or_stmt, **params) -> pd.DataFrame:
    with engine().connect() as c:
        return pd.read_sql(sql_or_stmt, c, params=params or None)


def replace_rows(table: Table, rows: list[dict], where) -> None:
    with engine().begin() as c:
        c.execute(delete(table).where(where))
        if rows:
            c.execute(insert(table), rows)


def add(table: Table, rows: list[dict] | dict) -> None:
    rows = [rows] if isinstance(rows, dict) else rows
    if rows:
        with engine().begin() as c:
            c.execute(insert(table), rows)


__all__ = ["engine", "df", "replace_rows", "add", "select", "delete", "products", "nav", "holdings", "trades", "flows", "risk",
           "limit_events", "reconcile", "quant_rebalances", "pool", "ic", "orders"]
