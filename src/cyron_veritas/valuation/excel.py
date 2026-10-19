"""把 DCF 导出成带公式的 Excel 模型：改「假设」页的蓝色单元格，预测、估值、敏感性会自动重算。

投行惯例：蓝色字 = 手工输入的假设，黑色字 = 公式。
测试里会用 formulas 库重算这个工作簿，确认 Excel 公式算出的每股价值和 Python 引擎一致。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as col

from cyron_veritas.valuation.dcf import DCFInputs

BLUE = Font(color="1F4E9E")
BOLD = Font(bold=True)
HEAD = PatternFill("solid", fgColor="0E7C66")
HEAD_FONT = Font(bold=True, color="FFFFFF")
THIN = Border(bottom=Side(style="thin", color="BBBBBB"))
PCT, NUM, NUM2 = "0.0%", "#,##0.0", "#,##0.00"

# 假设页单元格地址（公式里引用）
A = {"company": "B3", "base_year": "B4", "base_revenue": "B5", "base_nwc": "B6", "tax_rate": "B7", "da_pct": "B8",
     "capex_pct": "B9", "nwc_pct": "B10", "wacc": "B11", "terminal_growth": "B12", "debt": "B13", "cash": "B14",
     "minority": "B15", "shares": "B16", "price": "B17"}


def _header(ws, row: int, values: list):
    for j, v in enumerate(values, start=1):
        c = ws.cell(row=row, column=j, value=v)
        c.fill, c.font = HEAD, HEAD_FONT
        c.alignment = Alignment(horizontal="center")


def build_workbook(x: DCFInputs, company: str, price: float, history: pd.DataFrame | None = None,
                   wacc_detail: dict | None = None, path: Path | None = None) -> Path:
    wb = Workbook()
    n = len(x.revenue_growth)
    last = col(1 + n)  # 预测最后一年所在列

    # ---- 假设 ----
    ws = wb.active
    ws.title = "假设"
    ws["A1"], ws["A1"].font = f"{company} DCF 模型（单位：亿元；蓝色为输入假设）", Font(bold=True, size=13)
    rows = [("company", "公司", company, None), ("base_year", "基期年份", x.base_year, None),
            ("base_revenue", "基期营业总收入", x.base_revenue, NUM), ("base_nwc", "基期经营性营运资本", x.base_nwc, NUM),
            ("tax_rate", "有效税率", x.tax_rate, PCT), ("da_pct", "折旧摊销 / 收入", x.da_pct, PCT),
            ("capex_pct", "资本开支 / 收入", x.capex_pct, PCT), ("nwc_pct", "经营性营运资本 / 收入", x.nwc_pct, PCT),
            ("wacc", "WACC", x.wacc, PCT), ("terminal_growth", "永续增长率", x.terminal_growth, PCT),
            ("debt", "有息负债", x.debt, NUM), ("cash", "现金及交易性金融资产", x.cash, NUM),
            ("minority", "少数股东权益", x.minority, NUM), ("shares", "总股本（亿股）", x.shares, NUM2),
            ("price", "当前股价（元）", price, NUM2)]
    _header(ws, 2, ["项目", "数值", "依据"])
    for key, label, val, fmt in rows:
        r = int(A[key][1:])
        ws.cell(row=r, column=1, value=label)
        c = ws.cell(row=r, column=2, value=val)
        c.font = BLUE
        if fmt:
            c.number_format = fmt
        ws.cell(row=r, column=3, value=x.notes.get(key, ""))
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width, ws.column_dimensions["C"].width = 24, 14, 70

    # ---- 预测 ----
    p = wb.create_sheet("预测")
    p["A1"], p["A1"].font = "自由现金流预测（FCFF）", Font(bold=True, size=13)
    _header(p, 2, ["年份"] + [x.base_year + i for i in range(1, n + 1)])
    labels = {3: "期数 t", 4: "收入增速", 5: "EBIT 利润率", 6: "营业总收入", 7: "EBIT", 8: "NOPAT = EBIT×(1−税率)",
              9: "折旧摊销", 10: "资本开支", 11: "经营性营运资本", 12: "营运资本变动 ΔNWC", 13: "FCFF", 14: "折现因子",
              15: "FCFF 现值"}
    for r, lab in labels.items():
        p.cell(row=r, column=1, value=lab).font = BOLD if r in (13, 15) else Font()
    for i in range(n):
        c = col(2 + i)
        prev = col(1 + i)
        p[f"{c}3"] = i + 1
        p[f"{c}4"], p[f"{c}5"] = x.revenue_growth[i], x.ebit_margin[i]
        p[f"{c}4"].font = p[f"{c}5"].font = BLUE
        p[f"{c}6"] = f"=假设!$B$5*(1+{c}4)" if i == 0 else f"={prev}6*(1+{c}4)"
        p[f"{c}7"] = f"={c}6*{c}5"
        p[f"{c}8"] = f"={c}7*(1-假设!$B$7)"
        p[f"{c}9"] = f"={c}6*假设!$B$8"
        p[f"{c}10"] = f"={c}6*假设!$B$9"
        p[f"{c}11"] = f"={c}6*假设!$B$10"
        p[f"{c}12"] = f"={c}11-假设!$B$6" if i == 0 else f"={c}11-{prev}11"
        p[f"{c}13"] = f"={c}8+{c}9-{c}10-{c}12"
        p[f"{c}14"] = f"=1/(1+假设!$B$11)^{c}3"
        p[f"{c}15"] = f"={c}13*{c}14"
        for r in (4, 5):
            p[f"{c}{r}"].number_format = PCT
        for r in range(6, 16):
            p[f"{c}{r}"].number_format = NUM if r != 14 else "0.0000"
        p.column_dimensions[c].width = 12
    p.column_dimensions["A"].width = 26

    # ---- 估值 ----
    v = wb.create_sheet("估值")
    v["A1"], v["A1"].font = "估值汇总", Font(bold=True, size=13)
    items = [("预测期 FCFF 现值合计", f"=SUM(预测!B15:{last}15)", NUM),
             ("终值（戈登增长）", f"=预测!{last}13*(1+假设!B12)/(假设!B11-假设!B12)", NUM),
             ("终值现值", f"=B4/(1+假设!B11)^{n}", NUM),
             ("企业价值 EV", "=B3+B5", NUM),
             ("减：有息负债", "=假设!B13", NUM),
             ("加：现金", "=假设!B14", NUM),
             ("减：少数股东权益", "=假设!B15", NUM),
             ("股权价值", "=B6-B7+B8-B9", NUM),
             ("每股价值（元）", "=B10/假设!B16", NUM2),
             ("当前股价（元）", "=假设!B17", NUM2),
             ("相对现价", "=B11/B12-1", PCT),
             ("终值占 EV 比重", "=B5/B6", PCT)]
    for i, (lab, f, fmt) in enumerate(items, start=3):
        v.cell(row=i, column=1, value=lab).font = BOLD if lab.startswith(("每股", "企业", "股权")) else Font()
        c = v.cell(row=i, column=2, value=f)
        c.number_format = fmt
    v.column_dimensions["A"].width, v.column_dimensions["B"].width = 24, 14

    # ---- 敏感性 ----
    s = wb.create_sheet("敏感性")
    s["A1"], s["A1"].font = "每股价值敏感性（行：WACC，列：永续增长率）", Font(bold=True, size=13)
    waccs = [round(x.wacc + k * 0.005, 4) for k in range(-2, 3)]
    gs = [round(x.terminal_growth + k * 0.005, 4) for k in range(-2, 3)]
    s["A4"] = "WACC \\ g"
    for j, g in enumerate(gs):
        c = s.cell(row=4, column=2 + j, value=g)
        c.number_format, c.font = PCT, BLUE
    for i, w in enumerate(waccs):
        r = 5 + i
        c = s.cell(row=r, column=1, value=w)
        c.number_format, c.font = PCT, BLUE
        for j in range(len(gs)):
            gc = f"{col(2 + j)}$4"
            f = (f"=(SUMPRODUCT(预测!$B$13:${last}$13/(1+$A{r})^预测!$B$3:${last}$3)"
                 f"+预测!${last}$13*(1+{gc})/($A{r}-{gc})/(1+$A{r})^{n}"
                 f"-假设!$B$13+假设!$B$14-假设!$B$15)/假设!$B$16")
            cell = s.cell(row=r, column=2 + j, value=f)
            cell.number_format = NUM2
    s.column_dimensions["A"].width = 12

    # ---- WACC ----
    if wacc_detail:
        w = wb.create_sheet("WACC")
        w["A1"], w["A1"].font = "WACC 参考计算（CAPM）", Font(bold=True, size=13)
        rows = [("无风险利率 rf", wacc_detail["rf"], PCT), ("Beta（Blume 调整后）", wacc_detail["beta"], NUM2),
                ("股权风险溢价 ERP", wacc_detail["erp"], PCT), ("税前债务成本", wacc_detail["cost_of_debt"], PCT),
                ("税率", x.tax_rate, PCT), ("股权市值（亿元）", wacc_detail["equity_value"], NUM),
                ("有息负债（亿元）", x.debt, NUM)]
        for i, (lab, val, fmt) in enumerate(rows, start=3):
            w.cell(row=i, column=1, value=lab)
            c = w.cell(row=i, column=2, value=val)
            c.font, c.number_format = BLUE, fmt
        calc = [("股权成本 = rf + β×ERP", "=B3+B4*B5", PCT), ("税后债务成本", "=B6*(1-B7)", PCT),
                ("股权权重", "=B8/(B8+B9)", PCT), ("债务权重", "=B9/(B8+B9)", PCT), ("WACC", "=B13*B10+B14*B11", PCT)]
        for i, (lab, f, fmt) in enumerate(calc, start=10):
            w.cell(row=i, column=1, value=lab).font = BOLD if lab == "WACC" else Font()
            w.cell(row=i, column=2, value=f).number_format = fmt
        w.column_dimensions["A"].width = 26

    # ---- 历史 ----
    if history is not None:
        h = wb.create_sheet("历史")
        h["A1"], h["A1"].font = "历史财务（亿元，来源：akshare / 东方财富年报）", Font(bold=True, size=13)
        show = {"revenue": "营业总收入", "revenue_growth": "收入增速", "gross_margin": "毛利率", "ebit": "EBIT",
                "ebit_margin": "EBIT 利润率", "tax_rate": "有效税率", "net_profit_parent": "归母净利润", "roe": "ROE",
                "cfo": "经营现金流", "cfo_to_np": "经营现金流 / 归母净利润", "da_pct": "折旧摊销 / 收入",
                "capex_pct": "资本开支 / 收入", "nwc_pct": "营运资本 / 收入", "fcff": "FCFF", "cash": "现金", "debt": "有息负债"}
        _header(h, 2, ["项目"] + [int(y) for y in history.index])
        for i, (k, lab) in enumerate(show.items(), start=3):
            h.cell(row=i, column=1, value=lab)
            for j, yv in enumerate(history[k].tolist(), start=2):
                c = h.cell(row=i, column=j, value=None if pd.isna(yv) else float(yv))
                c.number_format = PCT if k.endswith(("growth", "margin", "rate", "pct", "roe", "_np")) or k == "roe" else NUM
        h.column_dimensions["A"].width = 26

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
