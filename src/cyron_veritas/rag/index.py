"""年报索引与混合检索。

    入库：PDF → 切块 → bge-m3 向量 → Milvus Lite（本地单文件，不需要 Docker）
                     └→ 切块 JSON（BM25 和「查看原文」用）
    检索：问题 ─┬─ 稠密检索 Top20（Milvus，按公司 / 年份 / 章节过滤）
               └─ BM25 Top20（jieba 分词，擅长精确匹配数字、科目名、专有名词）
             → RRF 融合 → bge-reranker-v2-m3 精排 → 分数断崖截断（最多 5 块）

模型路径默认用本机 ~/models 下的 bge-m3 和 bge-reranker-v2-m3；没有的话从 HuggingFace 下载（EMBED_MODEL / RERANK_MODEL 可覆盖）。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from cyron_veritas.config import ROOT
from cyron_veritas.rag import filings as F

INDEX_DIR = ROOT / "data" / "index"
COLLECTION = "annual_reports"
DIM = 1024
_lock = threading.Lock()
# 模型推理锁：MPS（苹果 GPU）不支持多线程同时提交计算，多个子 Agent 并行检索时会触发 Metal 断言直接崩溃，所以串行化
_infer_lock = threading.Lock()
_load_lock = threading.Lock()  # 两个线程同时第一次用到模型时，避免各加载一份（每份约 2GB）


def _model_path(env: str, local: str, hub: str) -> str:
    if os.getenv(env):
        return os.path.expanduser(os.environ[env])
    p = Path.home() / "models" / local
    return str(p) if p.exists() else hub


def _device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_model_path("EMBED_MODEL", "bge-m3", "BAAI/bge-m3"), device=_device())


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(_model_path("RERANK_MODEL", "bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3"), device=_device(), max_length=512)


@lru_cache(maxsize=1)
def client():
    from pymilvus import DataType, MilvusClient

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    c = MilvusClient(str(INDEX_DIR / "milvus.db"))
    if not c.has_collection(COLLECTION):
        schema = c.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("ticker", DataType.VARCHAR, max_length=16)
        schema.add_field("year", DataType.INT64)
        schema.add_field("section", DataType.VARCHAR, max_length=128)
        schema.add_field("page_start", DataType.INT64)
        schema.add_field("page_end", DataType.INT64)
        idx = c.prepare_index_params()
        idx.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        c.create_collection(COLLECTION, schema=schema, index_params=idx)
    c.load_collection(COLLECTION)  # Milvus Lite 重新打开后集合是 released 状态，要先加载
    return c


def embed(texts: list[str], batch: int = 16) -> np.ndarray:
    with _load_lock:
        model = embedder()
    with _infer_lock:
        return model.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False)


def rerank_scores(pairs: list[tuple[str, str]]) -> list[float]:
    with _load_lock:
        model = reranker()
    with _infer_lock:
        return [float(x) for x in model.predict(pairs, batch_size=8, show_progress_bar=False)]


# ---------------- 入库 ----------------
def manifest() -> dict:
    p = INDEX_DIR / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_manifest(m: dict):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest(ticker: str, year: int, progress=None) -> dict:
    """下载、解析、向量化一份年报。同一份 PDF（sha256 相同）已经入库就跳过；内容变了就整份替换。"""
    say = progress or (lambda msg: None)
    say("查找并下载年报")
    path, meta = F.download(ticker, year)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    key = f"{ticker}_{year}"
    m = manifest()
    if m.get(key, {}).get("sha256") == sha and m[key].get("parser_version") == F.PARSER_VERSION:
        say("已入库，跳过")
        return m[key]
    say("解析 PDF、按章节切块")
    chunks = F.parse(path, ticker, meta["company"], year)
    say(f"向量化 {len(chunks)} 个切块（bge-m3）")
    vecs = embed([f"{c.section} {c.subsection}\n{c.text}" for c in chunks])
    with _lock:
        c = client()
        c.delete(COLLECTION, filter=f'ticker == "{ticker}" and year == {year}')
        rows = [{"id": ch.id, "vector": v.tolist(), "ticker": ticker, "year": year, "section": ch.section[:120],
                 "page_start": ch.page_start, "page_end": ch.page_end} for ch, v in zip(chunks, vecs)]
        for i in range(0, len(rows), 200):
            c.insert(COLLECTION, rows[i:i + 200])
        (INDEX_DIR / f"chunks_{key}.json").write_text(json.dumps([F.chunk_dict(x) for x in chunks], ensure_ascii=False), encoding="utf-8")
        m = manifest()
        m[key] = {**meta, "sha256": sha, "parser_version": F.PARSER_VERSION, "n_chunks": len(chunks), "pdf_path": str(path.relative_to(ROOT)),
                  "sections": sorted({x.section for x in chunks})}
        _save_manifest(m)
    _bm25.cache_clear()
    _chunks.cache_clear()
    say("完成")
    return m[key]


# ---------------- 检索 ----------------
@lru_cache(maxsize=16)
def _chunks(key: str) -> dict[str, dict]:
    data = json.loads((INDEX_DIR / f"chunks_{key}.json").read_text(encoding="utf-8"))
    return {d["id"]: d for d in data}


def _tok(text: str) -> list[str]:
    import jieba
    return [t for t in jieba.lcut(text.lower()) if t.strip() and t not in "，。、；：（）“”《》 \n"]


@lru_cache(maxsize=16)
def _bm25(key: str):
    from rank_bm25 import BM25Okapi
    docs = list(_chunks(key).values())
    return BM25Okapi([_tok(f"{d['section']} {d['subsection']} {d['text']}") for d in docs]), [d["id"] for d in docs]


def search(query: str, ticker: str, year: int, top: int = 5, k: int = 20, rerank: bool = True, mode: str = "hybrid") -> list[dict]:
    """mode: hybrid（默认）/ dense / bm25（评测消融用）。"""
    key = f"{ticker}_{year}"
    if key not in manifest():
        raise LookupError(f"{ticker} 的 {year} 年年报还没有入库")
    chunks = _chunks(key)

    dense, sparse = [], []
    if mode in ("hybrid", "dense"):
        qv = embed([query])[0]
        hits = client().search(COLLECTION, data=[qv.tolist()], limit=k, filter=f'ticker == "{ticker}" and year == {year}',
                               output_fields=["id"], search_params={"metric_type": "COSINE"})[0]
        dense = [h["id"] for h in hits]
    if mode in ("hybrid", "bm25"):
        bm, ids = _bm25(key)
        scores = bm.get_scores(_tok(query))
        sparse = [ids[i] for i in np.argsort(scores)[::-1][:k] if scores[i] > 0]

    rrf: dict[str, float] = {}
    for ranking in (dense, sparse):
        for r, cid in enumerate(ranking):
            rrf[cid] = rrf.get(cid, 0) + 1 / (60 + r + 1)
    cands = sorted(rrf, key=rrf.get, reverse=True)[:k]
    out = [{**chunks[c], "rrf": rrf[c], "dense_rank": dense.index(c) + 1 if c in dense else None,
            "bm25_rank": sparse.index(c) + 1 if c in sparse else None} for c in cands]
    if rerank and out:
        s = rerank_scores([(query, f"{d['subsection']}\n{d['text']}") for d in out])
        for d, v in zip(out, s):
            d["score"] = float(v)
        out.sort(key=lambda d: -d["score"])
        best = out[0]["score"]
        # 分数断崖截断：比最高分低太多的不要
        out = [d for d in out[:top] if d["score"] >= best * 0.25] or out[:1]
    else:
        out = out[:top]
    return out


def page_text(ticker: str, year: int, page: int) -> str:
    return F.page_text(ROOT / manifest()[f"{ticker}_{year}"]["pdf_path"], page)
