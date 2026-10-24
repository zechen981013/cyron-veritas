"""Web 后端（FastAPI）。

    uv run veritas serve            → http://127.0.0.1:8100

长任务（风控日报、研究 Agent、年报入库）放后台线程跑，前端通过 SSE 订阅进度事件；年报问答直接 SSE 流式输出。
模型在第一次用到时才加载，启动很快。
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from cyron_veritas.config import OUTPUT_DIR, PORTFOLIO_DIR, ROOT

app = FastAPI(title="Cyron Veritas", version="0.2")
POOL = ThreadPoolExecutor(max_workers=2)
FRONTEND = ROOT / "frontend" / "dist"


# ---------------- 后台任务 ----------------
@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "running"  # running / done / error
    events: list = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    started: float = field(default_factory=time.time)

    def emit(self, ev: dict):
        self.events.append({**ev, "t": round(time.time() - self.started, 1)})


JOBS: dict[str, Job] = {}
_lock = threading.Lock()


def start_job(kind: str, params: dict, fn) -> Job:
    job = Job(id=uuid.uuid4().hex[:10], kind=kind, params=params)
    with _lock:
        JOBS[job.id] = job

    def run():
        try:
            job.result = fn(job)
            if job.status == "review":
                job.emit({"type": "awaiting_review"})
                return
            job.status = "done"
            job.emit({"type": "done"})
        except BaseException as e:  # SystemExit（金融企业不适用 DCF）也要报给前端
            job.status, job.error = "error", str(e) or type(e).__name__
            job.emit({"type": "error", "text": job.error})
            traceback.print_exc()

    POOL.submit(run)
    return job


def _job_view(j: Job) -> dict:
    return {"id": j.id, "kind": j.kind, "params": j.params, "status": j.status, "error": j.error,
            "result": j.result, "events": j.events, "elapsed": round(time.time() - j.started, 1)}


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "任务不存在")
    return _job_view(JOBS[jid])


@app.get("/api/jobs/{jid}/events")
async def job_events(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "任务不存在")
    job = JOBS[jid]

    async def gen():
        i = 0
        while True:
            while i < len(job.events):
                yield {"data": json.dumps(job.events[i], ensure_ascii=False, default=str)}
                i += 1
            if job.status in ("done", "error") and i >= len(job.events):
                break
            await asyncio.sleep(0.4)

    return EventSourceResponse(gen())


# ---------------- 组合风控 ----------------
def _url(path: str | Path) -> str:
    return "/outputs/" + str(Path(path).resolve().relative_to(OUTPUT_DIR.resolve())).replace("\\", "/")


@app.get("/api/portfolios")
def portfolios():
    out = []
    for p in sorted(PORTFOLIO_DIR.glob("*.csv")):
        df = pd.read_csv(p, dtype={"ticker": str})
        out.append({"name": p.name, "positions": int((df["market"] != "CASH").sum()), "accounts": sorted(df["account"].unique().tolist()),
                    "private": p.name.startswith("private_")})
    return out


class RiskReq(BaseModel):
    portfolio: str = "sample_portfolio.csv"


@app.post("/api/risk/run")
def risk_run(req: RiskReq):
    path = PORTFOLIO_DIR / req.portfolio
    if not path.exists() or path.parent != PORTFOLIO_DIR:
        raise HTTPException(400, "持仓文件不存在")

    def work(job: Job):
        from cyron_veritas.risk import report_graph as G

        labels = {"compute": "计算风险指标与限额", "narrate": "大模型写解读", "verify": "数字溯源校验",
                  "fallback": "改用模板文字", "render": "生成报告"}
        state = {}
        for upd in G.build_graph().stream({"holdings_path": str(path)}, stream_mode="updates"):
            for node, val in upd.items():
                state.update(val or {})
                msg = labels.get(node, node)
                if node == "verify":
                    c = val.get("check", {})
                    msg += "：通过" if c.get("ok") else f"：未通过（{val.get('feedback', '')[:80]}）"
                job.emit({"type": "step", "node": node, "text": msg})
        f = state["facts"]
        keep = ["as_of", "total_value", "n_positions", "accounts", "by_sector", "by_market", "by_currency", "concentration",
                "cash_weight", "vol_ann", "var95_hist", "var99_hist", "cvar95_hist", "var95_amount", "max_drawdown",
                "current_drawdown", "beta_csi300", "risk_contribution", "stress", "var_backtest", "weights", "pnl_pct"]
        return {"facts": {k: f[k] for k in keep}, "limits": state["limits"], "narrative": state["narrative"],
                "check": state["check"], "source": state["source"], "report_url": _url(state["outputs"]["html"])}

    return {"job_id": start_job("risk", req.model_dump(), work).id}


# ---------------- 研究 Agent ----------------
class ResearchReq(BaseModel):
    ticker: str
    review: bool = True


@app.post("/api/research/start")
def research_start(req: ResearchReq):
    from cyron_veritas.data.financials import market_of

    try:
        t = market_of(req.ticker)[1]
    except ValueError:
        raise HTTPException(400, "请输入 A 股 6 位、港股 5 位数字或美股字母代码")

    def work(job: Job):
        from cyron_veritas.research import agent

        res = agent.start(t, review=req.review, on_event=job.emit, run_id=job.id)
        return _research_result(job, res)

    return {"job_id": start_job("research", {"ticker": t, "review": req.review}, work).id}


def _research_result(job: Job, res: dict) -> dict:
    if res["status"] == "review":  # 停在审批节点：任务保持「待审批」，等 /review 接口恢复
        job.status = "review"
        return {"proposal": res["proposal"]}
    return {"memo": (Path(res["paths"]["memo"]).read_text(encoding="utf-8")), "stats": res["stats"],
            "files": {k: _url(v) for k, v in res["paths"].items()}}


class ReviewReq(BaseModel):
    action: str  # approve / edit / reject
    assumptions: dict = {}
    comment: str = ""
    reviewer: str = "分析师"


@app.post("/api/research/{jid}/review")
def research_review(jid: str, req: ReviewReq):
    job = JOBS.get(jid)
    if not job or job.status != "review":
        raise HTTPException(400, "这个任务不在待审批状态")
    if req.action not in ("approve", "edit", "reject"):
        raise HTTPException(400, "action 只能是 approve / edit / reject")
    job.status = "running"
    job.emit({"type": "status", "text": {"approve": "分析师已批准，定稿中", "edit": "分析师修改了假设，重新计算估值",
                                          "reject": "分析师驳回了估值假设"}[req.action]})

    def run():
        from cyron_veritas.research import agent

        try:
            job.result = _research_result(job, agent.resume(jid, req.model_dump()))
            if job.status == "running":
                job.status = "done"
                job.emit({"type": "done"})
        except BaseException as e:
            job.status, job.error = "error", str(e)
            job.emit({"type": "error", "text": job.error})
            traceback.print_exc()

    POOL.submit(run)
    return {"ok": True}


@app.get("/api/research/history")
def research_history():
    root = OUTPUT_DIR / "research"
    out = []
    for d in sorted(root.glob("*_*"), reverse=True) if root.exists() else []:
        stats = json.loads((d / "run_stats.json").read_text(encoding="utf-8")) if (d / "run_stats.json").exists() else {}
        xlsx = next(d.glob("*.xlsx"), None)
        out.append({"id": d.name, "ticker": d.name.split("_")[0], "date": d.name.split("_")[1], "dcf_runs": stats.get("dcf_runs"),
                    "seconds": stats.get("seconds"), "trace": stats.get("trace", {}).get("numbers_traced"),
                    "numbers": stats.get("trace", {}).get("numbers_total"),
                    "memo_url": _url(d / "memo.md"), "excel_url": _url(xlsx) if xlsx else None})
    return out


@app.get("/api/research/memo/{rid}")
def research_memo(rid: str):
    p = OUTPUT_DIR / "research" / rid / "memo.md"
    if not p.exists() or ".." in rid:
        raise HTTPException(404)
    return {"memo": p.read_text(encoding="utf-8")}


# ---------------- 年报 ----------------
@app.get("/api/filings")
def filings():
    from cyron_veritas.rag import index as I

    return [{"key": k, **{x: v[x] for x in ("ticker", "company", "year", "title", "date", "n_chunks", "page_url", "pdf_path")}}
            for k, v in sorted(I.manifest().items())]


class IngestReq(BaseModel):
    ticker: str
    year: int


@app.post("/api/filings/ingest")
def filings_ingest(req: IngestReq):
    def work(job: Job):
        from cyron_veritas.rag import index as I

        return I.ingest(req.ticker, req.year, progress=lambda m: job.emit({"type": "status", "text": m}))

    return {"job_id": start_job("ingest", req.model_dump(), work).id}


@app.get("/api/filings/{ticker}/{year}/page/{page}")
def filing_page(ticker: str, year: int, page: int):
    from cyron_veritas.rag import index as I

    try:
        return {"page": page, "text": I.page_text(ticker, year, page)}
    except Exception as e:
        raise HTTPException(404, str(e))


@app.get("/api/filings/{ticker}/{year}/pdf")
def filing_pdf(ticker: str, year: int):
    from cyron_veritas.rag import index as I

    m = I.manifest().get(f"{ticker}_{year}")
    if not m:
        raise HTTPException(404)
    return FileResponse(ROOT / m["pdf_path"], media_type="application/pdf")


class QAReq(BaseModel):
    ticker: str
    year: int
    question: str


@app.post("/api/qa")
async def qa_stream(req: QAReq):
    from starlette.concurrency import iterate_in_threadpool

    from cyron_veritas.rag import qa

    async def gen():
        try:
            async for ev in iterate_in_threadpool(qa.answer_stream(req.question, req.ticker, req.year)):
                yield {"data": json.dumps(ev, ensure_ascii=False, default=str)}
        except Exception as e:
            yield {"data": json.dumps({"type": "error", "text": str(e)}, ensure_ascii=False)}

    return EventSourceResponse(gen())


# ---------- 业绩会纪要 ----------
@app.get("/api/calls/{ticker}/records")
def call_records(ticker: str):
    from cyron_veritas.calls import run as R

    try:
        return R.records(ticker)
    except Exception as e:
        raise HTTPException(400, str(e))


class CallReq(BaseModel):
    ticker: str
    record_id: str | None = None
    compare: bool = True


@app.post("/api/calls/run")
def call_run(req: CallReq):
    def work(job: Job):
        from cyron_veritas.calls import run as R

        res = R.run(req.ticker, req.record_id, req.compare, say=job.emit)
        return {"markdown": res["markdown"], "check": res["result"]["check"], "summary": res["result"]["summary"],
                "record": res["result"]["record"], "compare": res["compare"]}

    return {"job_id": start_job("calls", req.model_dump(), work).id}


@app.post("/api/calls/audio")
async def call_audio(file: UploadFile = File(...), company: str = Form("")):
    import tempfile

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp = Path(tempfile.mkdtemp()) / f"upload{suffix}"
    tmp.write_bytes(await file.read())

    def work(job: Job):
        from cyron_veritas.calls import run as R

        res = R.from_audio(tmp, company, say=job.emit)
        return {"markdown": res["markdown"], "check": res["result"]["check"], "transcript": res["transcript"]}

    return {"job_id": start_job("audio", {"file": file.filename}, work).id}


# ---------- 资管中台 ----------
def _jsonable(x):
    """转成可 JSON 序列化的结构；NaN / ±inf（如没有正现金流的项目 IRR）变成 null。"""
    def clean(o):
        if isinstance(o, float):
            return o if o == o and o not in (float("inf"), float("-inf")) else None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    return clean(json.loads(json.dumps(x, ensure_ascii=False, default=str)))


@app.get("/api/fund/products")
def fund_products():
    from cyron_veritas.fund import db as FDB

    return _jsonable(FDB.df("select code, name, type from products").to_dict("records"))


@app.get("/api/fund/{code}/overview")
def fund_overview(code: str):
    from cyron_veritas.fund import db as FDB
    from cyron_veritas.fund import marketdata as FM
    from cyron_veritas.fund import perf as PF
    from cyron_veritas.fund import product as FP

    cfg = FP.load(code)
    nav = FDB.df("select date, unit_nav, net_assets, shares, cash, stock_mv from nav where product = :p order by date", p=code)
    nav["date"] = pd.to_datetime(nav["date"])
    b = FM.index_close(cfg["benchmark"]).reindex(nav["date"]).ffill()
    nav["bench"] = (b / b.iloc[0]).values
    m = PF.nav_metrics(code)
    snap = FDB.df("select date, metrics from risk_snapshots where product = :p order by date desc limit 1", p=code)
    rec = FDB.df("select date, status, diff, explained from reconciliation where product = :p order by date desc limit 10", p=code)
    flows = FDB.df("select date, type, amount, shares, nav from flows where product = :p order by date", p=code)
    from cyron_veritas.fund import risk as FR
    open_events = len(FR.episodes(code))
    metrics = snap["metrics"].iloc[0] if len(snap) else {}
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    return _jsonable({"product": {**{k: cfg[k] for k in ("code", "name", "type", "manager", "custodian", "inception", "benchmark", "fees", "contract_limits")},
                                  "bench_name": FP.bench_name(cfg)},
                      "nav": nav[["date", "unit_nav", "bench", "net_assets"]].assign(date=nav["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
                      "metrics": {k: v for k, v in m.items() if k != "monthly"},
                      "monthly": m["monthly"].reset_index().rename(columns={"index": "month", "date": "month"}).assign(
                          month=lambda x: pd.to_datetime(x.iloc[:, 0]).dt.strftime("%Y-%m")).to_dict("records"),
                      "risk": metrics, "reconciliation": rec.to_dict("records"), "flows": flows.to_dict("records"), "open_events": int(open_events)})


@app.get("/api/fund/{code}/events")
def fund_events(code: str, only_open: bool = True):
    """风控事件：同一规则连续触发的逐日记录合并为一个区间。"""
    from cyron_veritas.fund import risk as FR

    return _jsonable(FR.episodes(code, only_open))


class HandleReq(BaseModel):
    handled_by: str
    note: str
    ids: list[int] = []


@app.post("/api/fund/{code}/events/handle")
def fund_events_handle(code: str, req: HandleReq):
    from cyron_veritas.fund import risk as FR

    return {"ok": True, "n": FR.handle_events(code, req.ids, req.handled_by, req.note)}


@app.post("/api/fund/{code}/events/{eid}/handle")
def fund_event_handle(code: str, eid: int, req: HandleReq):
    import datetime as _dt

    from cyron_veritas.fund import db as FDB

    with FDB.engine().begin() as c:
        c.execute(FDB.limit_events.update().where((FDB.limit_events.c.id == eid) & (FDB.limit_events.c.product == code))
                  .values(handled_by=req.handled_by, handled_at=_dt.datetime.now(), note=req.note))
    return {"ok": True}


@app.get("/api/fund/{code}/holdings")
def fund_holdings(code: str, date: str | None = None):
    from cyron_veritas.fund import db as FDB
    from cyron_veritas.fund import marketdata as FM
    from cyron_veritas.fund import risk as FR

    d = date or str(FDB.df("select max(date) as d from nav where product = :p", p=code)["d"].iloc[0])
    h = FR.holdings_on(code, d)
    h["industry"] = h["code"].map(FM.sw_industry()).fillna("未分类")
    return _jsonable({"date": d, "holdings": h.sort_values("mv", ascending=False).to_dict("records")})


@app.get("/api/fund/{code}/attribution")
def fund_attribution(code: str):
    from cyron_veritas.fund import perf as PF

    b = PF.brinson(code)
    s = PF.style_exposure(code)
    return _jsonable({"brinson": b.to_dict("records"), "brinson_total": b.drop(columns=["month", "超配最多", "低配最多", "基准权重来源"]).sum().to_dict(),
                      "style": s.reset_index().rename(columns={"index": "factor"}).to_dict("records")})


@app.get("/api/quant/research")
def quant_research():
    """因子检验结果：汇总表、累计 RankIC、最新合成权重、因子相关性。"""
    from cyron_veritas.quant import factors as QF
    from cyron_veritas.quant import research as QR

    summ = QR.load("summary")
    ric = QR.load("rank_ic").dropna(how="all")
    wt = QR.load("weights")
    corr = QR.load("corr")
    cum = ric.fillna(0).cumsum()
    return _jsonable({"summary": summ.reset_index(names="key").to_dict("records"),
                      "cum_rank_ic": {"dates": [d.strftime("%Y-%m") for d in cum.index], **{c: cum[c].round(4).tolist() for c in cum.columns}},
                      "weights": {"date": str(wt.index[-1].date()), **wt.iloc[-1].round(4).to_dict()},
                      "corr": {"factors": [QF.NAME[c] for c in corr.columns], "values": corr.round(2).values.tolist()}})


@app.get("/api/quant/{code}/rebalances")
def quant_rebalances(code: str):
    """每次调仓的优化与执行记录，以及最近一次目标组合的主动权重和主动暴露。"""
    from cyron_veritas.config import ROOT as _ROOT
    from cyron_veritas.fund import db as FDB
    from cyron_veritas.fund import marketdata as FM

    reb = FDB.df("select * from quant_rebalances where product = :p order by date", p=code)
    for c in ("relaxed", "exposure"):
        reb[c] = reb[c].map(lambda x: json.loads(x) if isinstance(x, str) else x)
    t = pd.read_parquet(_ROOT / "data" / "quant" / f"targets_{code}.parquet")
    from cyron_veritas.fund import product as FP
    from cyron_veritas.fund import risk as FR

    d = pd.Timestamp(t["date"].max())
    held = t[t["date"] == d].set_index("code")["weight"]
    wb, _ = FR.bench_weights(FP.bench_index(FP.load(code)), d)               # 全部成分股（含没持有的，才能看到低配）
    allc = held.index.union(wb.index)
    last = pd.DataFrame({"code": allc, "weight": held.reindex(allc).fillna(0).values, "bench": wb.reindex(allc).fillna(0).values})
    names = FM.universe().set_index("code")["name"]
    ind = FM.sw_industry()
    last["active"] = last["weight"] - last["bench"] * held.sum()            # 基准按股票仓位缩放
    last["name"] = last["code"].map(names)
    last["industry"] = last["code"].map(ind)
    top = pd.concat([last.nlargest(10, "active"), last.nsmallest(5, "active")])
    exp = reb["exposure"].iloc[-1] if len(reb) else {}
    from cyron_veritas.quant.riskmodel import STYLES
    return _jsonable({"rebalances": reb.drop(columns=["exposure"]).to_dict("records"),
                      "latest": {"date": str(d.date()), "n": int((last["weight"] > 0).sum()),
                                 "top_active": top[["code", "name", "industry", "weight", "bench", "active"]].to_dict("records"),
                                 "style_exposure": {k: exp.get(k, 0) for k in STYLES},
                                 "industry_exposure": dict(sorted(((k, v) for k, v in exp.items() if k not in STYLES), key=lambda kv: kv[1]))}})


@app.get("/api/quant/{code}/attribution")
def quant_attribution(code: str):
    """超额收益逐日分解（按月汇总）、全期年化、风险模型偏差检验。"""
    from cyron_veritas.quant import attribution as QA

    d = QA.decompose_cached(code)
    mo = QA.monthly(d)
    cum = d[["超额"] + QA.COLS].cumsum()
    return _jsonable({"summary": QA.summary(d), "cols": QA.COLS,
                      "monthly": mo[["产品", "指数", "超额"] + QA.COLS + ["复利差"]].reset_index(names="month").to_dict("records"),
                      "cum": {"dates": [x.strftime("%Y-%m-%d") for x in cum.index], **{c: cum[c].round(5).tolist() for c in cum.columns}},
                      "styles": {c.split("·")[1]: float(d[c].sum() * 252 / len(d)) for c in d.columns if c.startswith("风格·")},
                      "bias": QA.bias_test(code).to_dict("records")})


class PretradeReq(BaseModel):
    orders: list[dict]
    date: str | None = None


@app.post("/api/fund/{code}/pretrade")
def fund_pretrade(code: str, req: PretradeReq):
    from cyron_veritas.fund import risk as FR

    d = req.date or str(FR.nav_series(code).index[-1].date())
    return _jsonable(FR.pretrade(code, d, req.orders))


@app.get("/api/fund/{code}/pool")
def fund_pool(code: str):
    from cyron_veritas.fund import db as FDB

    p = FDB.df("select * from stock_pool where product = :p order by since desc, id desc", p=code)
    latest = p.sort_values(["since", "id"]).groupby("code").tail(1)
    ic = FDB.df("select * from ic_decisions where product = :p order by decided_at desc, id desc limit 100", p=code)
    from cyron_veritas.fund import product as FP
    return _jsonable({"pools": latest.sort_values(["pool", "code"]).to_dict("records"), "ic": ic.to_dict("records"),
                      "banned": FP.load(code).get("banned", [])})


class ICReq(BaseModel):
    code: str
    name: str = ""
    request: str = "入核心池"
    requested_by: str = "研究员"
    material: str = ""
    decision: str          # 批准 / 否决 / 补充材料
    votes: dict = {}
    note: str = ""


@app.post("/api/fund/{code}/ic")
def fund_ic(code: str, req: ICReq):
    import datetime as _dt

    from cyron_veritas.fund import db as FDB

    today = _dt.date.today()
    FDB.add(FDB.ic, {"product": code, "code": req.code.zfill(6), "name": req.name, "request": req.request, "requested_by": req.requested_by,
                     "requested_at": today, "material": req.material, "decision": req.decision, "votes": req.votes, "decided_at": today, "note": req.note})
    if req.decision == "批准":
        pool_name = {"入核心池": "核心池", "调出核心池": "观察池", "入禁投池": "禁投池"}.get(req.request, "观察池")
        FDB.add(FDB.pool, {"product": code, "code": req.code.zfill(6), "name": req.name, "pool": pool_name, "since": today, "until": None,
                           "reason": f"投委会批准：{req.request}", "decision_id": None})
    return {"ok": True}


@app.get("/api/fund/{code}/sheets")
def fund_sheets(code: str, limit: int = 30):
    from cyron_veritas.fund import product as FP

    files = sorted(FP.sheet_dir(code).glob("*.xlsx"), reverse=True)[:limit]
    return [{"name": f.name, "url": f"/sheets/{code}/{f.name}"} for f in files]


# ---------------- PE / VC ----------------
def _pe_m(x):
    return x if isinstance(x, dict) else json.loads(x)


@app.get("/api/pe/{code}/overview")
def pe_overview(code: str):
    """基金条款、最新指标、季度序列、项目汇总、后见之明压力测试。"""
    from cyron_veritas.fund import db as FDB
    from cyron_veritas.fund import product as FP
    from cyron_veritas.pe import metrics as PX
    from cyron_veritas.pe import stress as PS

    cfg = FP.load(code)
    qs = FDB.df("select date, metrics from pe_quarters where product = :p order by date", p=code)
    ms = [_pe_m(m) for m in qs["metrics"]]
    inception = pd.Timestamp(cfg["inception"])
    series = [{"date": str(d), **{k: m.get(k) for k in ("TVPI", "DPI", "RVPI", "lp_paid_in", "lp_distributed", "lp_nav", "cum_net_cf", "accrued_carry", "carry_paid")},
               "net_irr": m.get("net_irr") if (pd.Timestamp(d) - inception).days >= 730 else None} for d, m in zip(qs["date"], ms)]
    tx = FDB.df("select * from pe_tx where product = :p", p=code)
    tx["date"] = pd.to_datetime(tx["date"])
    pipe = FDB.df("select deal, company, sector, status, synthetic from pe_pipeline where product = :p and status != '未投资'", p=code).set_index("deal")
    last_marks = FDB.df("select deal, fv, method from pe_marks where product = :p and date = (select max(date) from pe_marks where product = :p)", p=code).set_index("deal")
    deals = []
    for deal, g in tx[tx["deal"] != "FUND"].groupby("deal"):
        inv = -g.loc[g["type"] == "投资", "amount"].sum()
        real = g.loc[g["amount"] > 0, "amount"].sum()
        fv = float(last_marks["fv"].get(deal, 0) or 0)
        flows = [(r.date, r.amount) for r in g.itertuples() if r.amount] + ([(tx["date"].max(), fv)] if fv else [])
        ev = {t: g.loc[g["type"] == t, "date"].min() for t in ("投资", "上市")}
        exits = g.loc[g["type"].isin(["减持", "回购退出", "核销"]), "date"]
        deals.append({"deal": deal, "company": pipe["company"].get(deal, deal), "sector": pipe["sector"].get(deal, ""), "status": pipe["status"].get(deal, ""),
                      "synthetic": bool(pipe["synthetic"].get(deal, 0)), "invested": inv, "realized": real, "fv": fv, "moic": (real + fv) / inv if inv else None,
                      "irr": PX.xirr(flows), "close": str(ev["投资"].date()) if pd.notna(ev["投资"]) else None,
                      "listing": str(ev["上市"].date()) if pd.notna(ev["上市"]) else None, "exit_end": str(exits.max().date()) if len(exits) else None})
    fees = tx[tx["deal"] == "FUND"].groupby("type")["amount"].sum().to_dict()
    return _jsonable({"product": {k: cfg[k] for k in ("code", "name", "type", "manager", "custodian", "inception", "commitment", "terms", "lps", "exit_policy")},
                      "latest": {**ms[-1], "date": str(qs["date"].iloc[-1])}, "series": series, "deals": sorted(deals, key=lambda x: x["close"] or ""),
                      "fees": fees, "stress": PS.run(code)})


@app.get("/api/pe/{code}/pipeline")
def pe_pipeline(code: str):
    from cyron_veritas.fund import db as FDB

    p = FDB.df("select deal, company, sector, stage, status, events, reason, synthetic from pe_pipeline where product = :p", p=code)
    p["events"] = p["events"].map(_pe_m)
    return _jsonable(p.to_dict("records"))


@app.get("/api/pe/{code}/deal/{deal}")
def pe_deal(code: str, deal: str):
    """单个项目：交易条款（招股书出处）、季度估值、交易流水。"""
    from cyron_veritas.fund import db as FDB
    from cyron_veritas.fund import product as FP

    marks = FDB.df("select date, status, method, qty, price, discount, fv, cost, note from pe_marks where product = :p and deal = :d order by date", p=code, d=deal)
    tx = FDB.df("select date, type, qty, price, amount, note from pe_tx where product = :p and deal = :d order by date, id", p=code, d=deal)
    sells = tx[tx["type"] == "减持"]
    tx_small = pd.concat([tx[tx["type"] != "减持"], sells.groupby(pd.to_datetime(sells["date"]).dt.to_period("M")).agg(
        date=("date", "max"), qty=("qty", "sum"), amount=("amount", "sum"), price=("price", "mean")).assign(type="减持（按月汇总）", note="")]).sort_values("date")
    terms = None
    if not deal.startswith("SYN"):
        from cyron_veritas.pe import deals as PD

        spec = next(x for x in FP.load(code)["deals"] if x["code"] == deal)
        terms = PD.terms(spec)
    return _jsonable({"marks": marks.to_dict("records"), "tx": tx_small.to_dict("records"), "terms": terms})


@app.get("/api/pe/{code}/capital")
def pe_capital(code: str):
    """缴款与分配（按日期汇总）、各 LP 资本账户、瀑布各档累计。"""
    from cyron_veritas.fund import db as FDB
    from cyron_veritas.fund import product as FP

    cfg = FP.load(code)
    cap = FDB.df("select date, notice_date, lp, type, amount, detail from pe_capital where product = :p order by date", p=code)
    cap["detail"] = cap["detail"].map(_pe_m)
    events = []
    for (d, t), g in cap.groupby(["date", "type"], sort=True):
        purpose = next((x.get("用途") for x in g["detail"] if x and x.get("用途")), "")
        tiers: dict = {}
        for x in g["detail"]:
            for k, v in (x or {}).items():
                if isinstance(v, (int, float)):
                    tiers[k] = tiers.get(k, 0) + v
        events.append({"date": str(d), "notice_date": str(g["notice_date"].iloc[0]), "type": t, "amount": float(g["amount"].sum()), "purpose": purpose, "tiers": tiers})
    m = _pe_m(FDB.df("select metrics from pe_quarters where product = :p order by date desc limit 1", p=code)["metrics"].iloc[0])
    lp_commit = sum(x["commitment"] for x in cfg["lps"] if x["type"] != "GP")
    accounts = []
    for x in cfg["lps"]:
        g = cap[cap["lp"] == x["id"]]
        called, dist = g.loc[g["type"] == "缴款", "amount"].sum(), g.loc[g["type"] == "分配", "amount"].sum()
        nav = m["lp_nav"] * x["commitment"] / lp_commit if x["type"] != "GP" else None
        accounts.append({**x, "called": called, "distributed": dist, "carry": g.loc[g["type"] == "业绩报酬", "amount"].sum(), "nav": nav,
                         "tvpi": (dist + (nav or 0)) / called if called else None})
    return _jsonable({"events": events, "accounts": accounts, "waterfall": m.get("waterfall_paid", {}), "accrued_carry": m["accrued_carry"]})


@app.get("/api/pe/dd/{deal}")
def pe_dd(deal: str):
    from cyron_veritas.pe import dd as PDD

    try:
        return _jsonable(PDD.load(deal))
    except FileNotFoundError:
        raise HTTPException(404, "还没有做尽调")


@app.get("/prospectus/{deal}.pdf")
def pe_prospectus_pdf(deal: str):
    from cyron_veritas.pe import prospectus as PPR

    f = PPR.DIR / f"{deal}.pdf"
    if not f.exists() or not deal.isdigit():
        raise HTTPException(404)
    return FileResponse(f, media_type="application/pdf")


class PEReportReq(BaseModel):
    quarter: str
    lp: str = "LP02"


@app.post("/api/pe/{code}/report")
def pe_report(code: str, req: PEReportReq):
    def work(job: Job):
        from cyron_veritas.pe import report as PRP

        job.emit({"type": "status", "text": "汇总资本账户、组合估值，生成 GP 观点"})
        r = PRP.build(code, req.quarter, req.lp)
        return {"html": _url(r["html"]), "pdf": _url(r["pdf"]) if r["pdf"] else None, "check": r["check"]}

    return {"job_id": start_job("pe_report", {"code": code, **req.model_dump()}, work).id}

@app.get("/sheets/{code}/{name}")
def fund_sheet_file(code: str, name: str):
    from cyron_veritas.fund import product as FP

    f = FP.sheet_dir(code) / name
    if not f.exists() or ".." in name:
        raise HTTPException(404)
    return FileResponse(f, filename=name)


class ReportReq(BaseModel):
    month: str


@app.post("/api/fund/{code}/report")
def fund_report(code: str, req: ReportReq):
    def work(job: Job):
        from cyron_veritas.fund import report as FRP

        job.emit({"type": "status", "text": "计算业绩、归因、风险，生成经理观点"})
        r = FRP.build(code, req.month)
        return {"html": _url(r["html"]), "pdf": _url(r["pdf"]) if r["pdf"] else None, "check": r["check"]}

    return {"job_id": start_job("report", {"code": code, "month": req.month}, work).id}


@app.get("/api/health")
def health():
    return {"ok": True}


# ---------------- 静态文件 ----------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
if FRONTEND.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = FRONTEND / path
        return FileResponse(f if path and f.is_file() else FRONTEND / "index.html")


def serve(host: str = "127.0.0.1", port: int = 8100):
    import uvicorn

    uvicorn.run(app, host=host, port=port)
