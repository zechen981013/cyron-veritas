"""业绩会纪要：结构化抽取 + 原文引用校验 + 环比对比。

  抽取（大模型）：一句话概览、管理层指引、关键经营数据、分析师关注点（问什么、怎么答）、风险提示、新披露信息、语气
  校验（代码）：每条都必须带 quote（原文摘录）；quote 去掉空白和标点后必须是原文的子串，否则标记「引用未找到」；
              概览和字段里的数字必须出现在原文里
  对比（大模型 + 代码）：两次会议的抽取结果 → 新话题、消失的话题、指引变化、语气变化；变化项同样要带两边的原文引用
研究员最需要的不是「这次说了什么」，而是「这次和上次说法哪里不一样」。
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from cyron_veritas import verify as V
from cyron_veritas.calls.source import Record
from cyron_veritas.config import chat_model


class Guidance(BaseModel):
    metric: str = Field(description="指标或事项，如「海外收入增速」「回购金额」「新产能投产时间」")
    stance: Literal["上调", "下调", "维持", "新增", "定性表述"] = Field(description="相对以往的表述方向；没有比较基础时用「新增」或「定性表述」")
    value: str = Field(description="具体数值或区间，没有就写「无」")
    period: str = Field(description="对应期间，如「2026 年全年」「未来三年」，不确定写「未说明」")
    quote: str = Field(description="原文摘录（必须逐字摘自原文，20-80 字）")


class KPI(BaseModel):
    name: str
    value: str
    quote: str = Field(description="原文摘录，逐字")


class Theme(BaseModel):
    topic: str = Field(description="分析师 / 投资者关注的话题，如「2B 业务现金流」")
    concern: str = Field(description="提问的核心关切，一句话")
    answer: str = Field(description="管理层回答要点，一句话；回避了问题要写明「未正面回答」")
    quote: str = Field(description="回答的原文摘录，逐字")


class Item(BaseModel):
    text: str
    quote: str = Field(description="原文摘录，逐字")


class CallSummary(BaseModel):
    overview: str = Field(description="3-4 句话概览：这次交流最重要的信息")
    guidance: list[Guidance]
    kpis: list[KPI] = Field(description="管理层提到的关键经营数据")
    themes: list[Theme] = Field(description="3-8 个投资者关注话题")
    risks: list[Item] = Field(description="管理层承认或提示的风险、压力")
    new_info: list[Item] = Field(description="以前可能没披露过的新信息（新订单、新产能、新业务进展）")
    tone: float = Field(description="管理层整体语气，-1（悲观、防御）到 1（乐观、进攻）")
    tone_reason: str = Field(description="语气判断依据，一句话")
    evasive: list[str] = Field(description="管理层回避或答非所问的问题（如有）")


class Change(BaseModel):
    kind: Literal["新话题", "消失的话题", "指引变化", "数据变化", "语气变化"]
    description: str = Field(description="一句话说明变化")
    quote_before: str = Field(description="上次的原文摘录（新话题写「无」）")
    quote_after: str = Field(description="这次的原文摘录（消失的话题写「无」）")
    importance: Literal["高", "中", "低"]


class Comparison(BaseModel):
    summary: str = Field(description="2-3 句话：这次相比上次最重要的变化")
    changes: list[Change]
    tone_shift: float = Field(description="这次语气减上次语气，-2 到 2")


SYSTEM = """你是买方研究员，负责把上市公司的业绩说明会 / 投资者交流记录整理成纪要。
规则：
1. 只根据原文，不补充外部信息；
2. 每一条的 quote 必须是原文的逐字摘录（可以截取一句或半句，但不能改写、不能合并不相邻的句子）；
3. 数字原样引用，不要换算；
4. 关注「指引」（对未来的表态）和「回避」（答非所问），这是研究员最在意的；
5. 不给投资建议。"""


def _norm(s: str) -> str:
    return re.sub(r"[\s，。、；：！？,.;:!?“”\"'（）()【】\[\]《》…—\-]", "", s or "")


def quote_ok(quote: str, source: str) -> bool:
    q = _norm(quote)
    return bool(q) and (q in _norm(source) or q in {"无"})


def _record_text(rec: Record) -> str:
    if rec.qa:
        return "\n\n".join(f"问{i + 1}：{x['q']}\n答：{x['a']}" for i, x in enumerate(rec.qa))
    return rec.text


def check(summary: dict, source: str) -> dict:
    """引用校验 + 数字校验。返回每条的结果和总体通过率。"""
    items, bad = 0, []
    for sec in ("guidance", "kpis", "themes", "risks", "new_info"):
        for it in summary.get(sec, []):
            items += 1
            it["quote_ok"] = quote_ok(it.get("quote", ""), source)
            if not it["quote_ok"]:
                bad.append(f"{sec}: {it.get('quote', '')[:30]}")
    text = " ".join([summary.get("overview", "")] + [f"{g.get('value', '')}" for g in summary.get("guidance", [])]
                    + [k.get("value", "") for k in summary.get("kpis", [])])
    nums = V.check(text, V.allowed_set(source))
    return {"items": items, "quotes_ok": items - len(bad), "bad_quotes": bad, "unknown_numbers": nums["unknown_numbers"],
            "numbers_total": nums["numbers_total"], "numbers_traced": nums["numbers_traced"],
            "ok": not bad and not nums["unknown_numbers"]}


def summarize(rec: Record, retries: int = 1) -> dict:
    source = _record_text(rec)
    head = f"公司：{rec.company or '未知'}；日期：{rec.date or '未知'}；类别：{'、'.join(rec.categories) or '未知'}；参与方：{rec.participants[:200] or '未知'}"
    model = chat_model(0.1).with_structured_output(CallSummary, method="function_calling")
    msgs = [("system", SYSTEM), ("user", f"{head}\n\n【原文】\n{source[:24000]}")]
    out, chk = None, None
    for _ in range(retries + 1):
        out = model.invoke(msgs).model_dump()
        chk = check(out, source)
        if chk["ok"]:
            break
        msgs = msgs + [("assistant", str(out)[:4000]),
                       ("user", f"校验未通过：引用在原文中找不到 {chk['bad_quotes'][:8]}；数字找不到 {chk['unknown_numbers'][:8]}。"
                                "请重新输出，quote 必须逐字摘自原文。")]
    return {"record": {k: v for k, v in rec.to_dict().items() if k != "text"}, "summary": out, "check": chk}


def compare(prev: dict, cur: dict, prev_rec: Record, cur_rec: Record) -> dict:
    model = chat_model(0.1).with_structured_output(Comparison, method="function_calling")
    p_src, c_src = _record_text(prev_rec), _record_text(cur_rec)
    msg = (f"【上次 {prev_rec.date}】纪要：{prev['summary']}\n\n【这次 {cur_rec.date}】纪要：{cur['summary']}\n\n"
           f"【上次原文】\n{p_src[:10000]}\n\n【这次原文】\n{c_src[:10000]}\n\n"
           "请找出这次相比上次的变化，quote_before 逐字摘自上次原文，quote_after 逐字摘自这次原文。")
    out = model.invoke([("system", SYSTEM), ("user", msg)]).model_dump()
    for c in out["changes"]:
        c["before_ok"] = c["quote_before"] in ("无", "") or quote_ok(c["quote_before"], p_src)
        c["after_ok"] = c["quote_after"] in ("无", "") or quote_ok(c["quote_after"], c_src)
    ok = sum(c["before_ok"] and c["after_ok"] for c in out["changes"])
    return {"prev": prev_rec.date, "cur": cur_rec.date, **out, "check": {"changes": len(out["changes"]), "quotes_ok": ok}}


def to_markdown(res: dict, cmp: dict | None = None) -> str:
    r, s, c = res["record"], res["summary"], res["check"]
    mark = lambda it: "" if it.get("quote_ok", True) else " ⚠️引用未在原文找到"
    L = [f"# {r['company']} {r['date']} 交流纪要（{'、'.join(r['categories']) or '—'}）", "",
         f"来源：{r['source']}" + (f" · [原文]({r['url']})" if str(r['url']).startswith("http") else "")
         + f" · 参与方：{r['participants'][:120] or '—'}", "",
         "## 概览", s["overview"], "", f"管理层语气：**{s['tone']:+.2f}**（{s['tone_reason']}）", ""]
    if s["guidance"]:
        L += ["## 管理层指引", "| 指标 | 表态 | 数值 | 期间 | 原文 |", "|---|---|---|---|---|"]
        L += [f"| {g['metric']} | {g['stance']} | {g['value']} | {g['period']} | 「{g['quote']}」{mark(g)} |" for g in s["guidance"]]
        L.append("")
    if s["kpis"]:
        L += ["## 关键经营数据"] + [f"- {k['name']}：{k['value']} —「{k['quote']}」{mark(k)}" for k in s["kpis"]] + [""]
    L += ["## 投资者关注点"] + [f"- **{t['topic']}**：{t['concern']} → {t['answer']}「{t['quote']}」{mark(t)}" for t in s["themes"]] + [""]
    if s["new_info"]:
        L += ["## 新披露信息"] + [f"- {i['text']}「{i['quote']}」{mark(i)}" for i in s["new_info"]] + [""]
    if s["risks"]:
        L += ["## 风险与压力"] + [f"- {i['text']}「{i['quote']}」{mark(i)}" for i in s["risks"]] + [""]
    if s["evasive"]:
        L += ["## 回避或未正面回答"] + [f"- {e}" for e in s["evasive"]] + [""]
    if cmp:
        L += [f"## 与上次（{cmp['prev']}）相比", cmp["summary"], "", f"语气变化：{cmp['tone_shift']:+.2f}", "",
              "| 类型 | 重要性 | 变化 | 上次原文 | 这次原文 |", "|---|---|---|---|---|"]
        L += [f"| {x['kind']} | {x['importance']} | {x['description']} | {x['quote_before']} | {x['quote_after']} |" for x in cmp["changes"]]
        L.append("")
    L += ["---", f"引用校验：{c['quotes_ok']}/{c['items']} 条原文引用可在原文中找到；数字 {c['numbers_traced']}/{c['numbers_total']} 可溯源"
          + (f"；对比引用 {cmp['check']['quotes_ok']}/{cmp['check']['changes']}" if cmp else "") + "。"]
    return "\n".join(L)
