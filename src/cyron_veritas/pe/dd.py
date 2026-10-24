"""尽调 Agent：按清单从招股说明书抽取结构化信息，每个数字带页码并逐一核对原文。

清单（每项一次检索 + 一次结构化抽取）：
  1. 股本演变：历次增资 / 股权转让 / 股改 / 转增的时间、价格、单位、注册资本（股本）变化、对应估值、投资方
  2. 发行条款：发行前总股本、发行价格、发行股数
  3. 股份锁定：最近一年（申报前 12 个月）新增股东的锁定承诺
  4. 财务摘要：最近三年营业收入、归母净利润、扣非归母净利润
  5. 客户集中度：最近一年前五大客户收入占比
  6. 主要风险：5 条最重要的风险因素
大模型只做「读和抄」，所有计算（持股比例、稀释、每股成本）在 pe/deals.py 里由代码完成。
核对规则：抽出来的每个数字必须出现在它引用的那一页（允许相邻一页，表格常跨页）；核对不通过的字段标出来交给分析师复核，不自动采用。
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from cyron_veritas.config import ROOT, chat_model
from cyron_veritas.pe import prospectus as PR

OUT = ROOT / "data" / "pe" / "dd"


class Round(BaseModel):
    date: str = Field(description="工商变更时间，YYYY-MM")
    kind: str = Field(description="增资 / 股权转让 / 股改 / 转增 / 其他")
    price: float | None = Field(None, description="每单位价格（元），原文数字")
    unit: str | None = Field(None, description="价格单位：元/注册资本 或 元/股 或 元/出资额")
    capital_before: float | None = Field(None, description="本次之前注册资本或股本（万元或万股），原文数字；股权转让不填")
    capital_after: float | None = Field(None, description="本次之后注册资本或股本（万元或万股），原文数字；股权转让不填")
    valuation: float | None = Field(None, description="原文写的估值（亿元），原文数字")
    valuation_type: str | None = Field(None, description="投前 / 投后 / 公司估值")
    investors: str = Field("", description="投资方或受让方，原文简称，多个用顿号")
    note: str = Field("", description="原文对本次的说明，如员工持股平台、代持还原")
    page: int = Field(description="出处页码（PDF 页）")


class Rounds(BaseModel):
    rounds: list[Round]


class Cited(BaseModel):
    value: float | None
    page: int


class IpoTerms(BaseModel):
    pre_ipo_shares: Cited = Field(description="本次发行前总股本（股）")
    issue_price: Cited = Field(description="每股发行价格（元）")
    issue_shares: Cited = Field(description="本次发行股数（万股）")


class Lockup(BaseModel):
    months: int = Field(description="锁定月数")
    start: str = Field(description="起算点：取得股份之日 或 上市之日")
    quote: str = Field(description="原文承诺的一句话（逐字）")
    page: int


class Lockups(BaseModel):
    new_shareholders_last_year: Lockup = Field(description="最近一年（申报前 12 个月内）新增股东的锁定承诺")
    other_shareholders: Lockup | None = Field(None, description="其他股东（非控股股东、非最近一年新增）的锁定承诺")


class YearFin(BaseModel):
    year: str
    revenue: float | None = Field(None, description="营业收入（万元）")
    net_profit: float | None = Field(None, description="归属于母公司股东的净利润（万元）")
    net_profit_deducted: float | None = Field(None, description="扣除非经常性损益后归属于母公司股东的净利润（万元）")
    page: int


class Financials(BaseModel):
    years: list[YearFin]


class Customers(BaseModel):
    year: str
    top5_share: float = Field(description="前五大客户收入占营业收入比例（%）")
    top1_share: float | None = Field(None, description="第一大客户占比（%）")
    page: int


class Risk(BaseModel):
    title: str
    summary: str = Field(description="一句话概括（不超过 60 字）")
    page: int


class Risks(BaseModel):
    risks: list[Risk]


class Business(BaseModel):
    summary: str = Field(description="主营业务一句话（不超过 60 字）")
    industry: str = Field(description="所属行业（招股书原文的行业分类）")
    page: int


CHECKLIST = {
    "rounds": (Rounds, "股本演变 历次增资 股权转让 注册资本 对应估值 元/注册资本 元/股 投资者 股改 整体变更 资本公积转增", ("注册资本",), 8,
               "列出发行人设立以来全部增资、股权转让、整体变更（股改）和转增事件，按时间排序。价格、注册资本、估值都抄原文数字，不要换算。"),
    "ipo": (IpoTerms, "本次发行概况 每股发行价格 发行股数 发行前总股本 发行后总股本", ("发行",), 5,
            "抄出发行前总股本（股）、每股发行价格（元）、本次发行股数（万股），并给出页码。"),
    "lockup": (Lockups, "股份流通限制 自愿锁定 承诺 自取得发行人股份之日起 三十六个月 36个月 自发行人股票上市之日起 十二个月 不转让或者委托他人管理 最近一年新增股东",
               ("不转让",), 8,
               "抄出最近一年新增股东（申报前 12 个月内增资或受让入股）的股份锁定承诺，以及其他非控股股东的锁定承诺；quote 逐字抄原文。"),
    "financials": (Financials, "主要财务数据 营业收入 归属于母公司 净利润 扣除非经常性损益", ("营业收入",), 5,
                   "抄出报告期最近三个完整年度的营业收入、归母净利润、扣非归母净利润（万元；原文是元则换成万元并保留两位小数）。"),
    "customers": (Customers, "前五名客户 销售收入 占营业收入比例", ("前五",), 4,
                  "抄出报告期最近一个完整年度前五大客户合计占营业收入比例（%）和第一大客户占比。"),
    "risks": (Risks, "风险因素 技术风险 经营风险 财务风险 内控风险", ("风险",), 6,
              "挑出 5 条对投资决策最重要的风险，标题用原文标题，摘要用自己的话。"),
    "business": (Business, "主营业务 主要产品 所属行业 证监会 行业分类", ("主营业务",), 4,
                 "一句话概括发行人主营业务，并抄出所属行业。"),
}


def _context(code: str, query: str, must: tuple, top: int) -> str:
    hits = PR.search(code, query, top=top, must=must)
    hits.sort(key=lambda h: h["page"])
    return "\n\n".join(f"【第 {h['page']} 页】\n{h['text'][:3500]}" for h in hits)


def extract(code: str, item: str) -> dict:
    schema, query, must, top, instr = CHECKLIST[item]
    model = chat_model(0.0, max_tokens=8000).with_structured_output(schema, method="function_calling")
    ctx = _context(code, query, must, top)
    msgs = [("system", "你是 PE 基金的尽调分析师，从招股说明书原文里抽取信息。只抄原文，不推测；原文没有的字段留空。"
                       "页码用【第 N 页】标记里的 N。"), ("user", f"{instr}\n\n招股说明书节选：\n{ctx}")]
    return model.invoke(msgs).model_dump()


def verify(code: str, item: str, data: dict) -> list[dict]:
    """逐字段核对：数字（和锁定承诺原文）必须出现在引用页。返回未通过的字段。"""
    bad = []

    def chk(path, value, page):
        if value is None or page is None:
            return
        if not PR.on_page(code, int(page), value):
            bad.append({"field": path, "value": value, "page": page})

    if item == "rounds":
        for i, r in enumerate(data["rounds"]):
            for k in ("price", "capital_before", "capital_after", "valuation"):
                chk(f"rounds[{i}].{k}", r.get(k), r["page"])
    elif item == "ipo":
        for k, v in data.items():
            chk(f"ipo.{k}", v["value"], v["page"])
    elif item == "lockup":
        for k, v in data.items():
            if v and (not v["quote"] or not v["months"]):
                bad.append({"field": f"lockup.{k}", "value": "未找到", "page": v.get("page")})
            elif v:
                chk(f"lockup.{k}.quote", v["quote"][:30], v["page"])
    elif item == "financials":
        for i, y in enumerate(data["years"]):
            for k in ("revenue", "net_profit", "net_profit_deducted"):
                chk(f"financials[{i}].{k}", y.get(k), y["page"])
    elif item == "customers":
        chk("customers.top5_share", data["top5_share"], data["page"])
    return bad


def run(code: str, refresh: bool = False, say=print) -> dict:
    """完整尽调：逐项抽取 + 核对，结果存 data/pe/dd/<code>.json。"""
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{code}.json"
    if p.exists() and not refresh:
        return json.loads(p.read_text(encoding="utf-8"))
    _, meta = PR.download(code)
    out = {"code": code, "company": meta["company"], "listing_date": meta["listing_date"], "prospectus": meta, "items": {}, "unverified": []}
    for item in CHECKLIST:
        data = extract(code, item)
        bad = verify(code, item, data)
        out["items"][item] = data
        out["unverified"] += bad
        say(f"  {meta['company']} {item}：{'核对通过' if not bad else f'{len(bad)} 个字段待复核'}")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


class InvestorLockup(BaseModel):
    found: bool = Field(description="原文里有没有这家股东的锁定承诺")
    months_from_listing: int | None = Field(None, description="自上市之日起锁定的月数")
    months_from_acquisition: int | None = Field(None, description="自取得股份（增资工商变更登记）之日起锁定的月数")
    rule: str = Field("", description="两个期限的关系：孰长 / 仅上市日 / 仅取得日")
    quote: str = Field("", description="承诺原文的关键一句（逐字）")
    page: int | None = None


def lockup_for(code: str, investor: str, refresh: bool = False) -> dict:
    """某一家股东自己的锁定承诺（锁定承诺是逐家股东作出的，同一轮的投资人可能不一样）。结果缓存在尽调 JSON 里。"""
    import re

    data = load(code)
    cache = data["items"].setdefault("lockup_by_investor", {})
    if investor in cache and not refresh:
        return cache[investor]
    ps = PR.pages(code)
    flat = [re.sub(r"\s+", "", t) for t in ps]
    hits = [i for i, t in enumerate(flat) if investor in t and "不转让" in t and "个月" in t]
    if not hits:
        res = {"found": False, "months_from_listing": None, "months_from_acquisition": None, "rule": "", "quote": "", "page": None}
    else:
        ctx = "\n\n".join(f"【第 {i + 1} 页】\n{ps[i][:4000]}" for i in hits[:4])
        model = chat_model(0.0, max_tokens=2000).with_structured_output(InvestorLockup, method="function_calling")
        res = model.invoke([("system", "你是 PE 基金的尽调分析师。只抄原文，不推测。"),
                            ("user", f"找出股东「{investor}」本人（或其所在的一组股东）作出的股份锁定承诺：自上市之日起几个月、自取得股份之日起几个月、"
                                     f"两者关系。quote 逐字抄关键一句。\n\n{ctx}")]).model_dump()
        if res.get("page") and res.get("quote") and not PR.on_page(code, res["page"], res["quote"][:20]):
            res["unverified"] = True
    cache[investor] = res
    (OUT / f"{code}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def reverify(code: str) -> dict:
    """核对规则改了以后，对已存的抽取结果重新核对（不重新调大模型）。"""
    data = load(code)
    data["unverified"] = [b for item in CHECKLIST for b in verify(code, item, data["items"][item])]
    (OUT / f"{code}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def load(code: str) -> dict:
    return json.loads((OUT / f"{code}.json").read_text(encoding="utf-8"))
