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
