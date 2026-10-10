"""Pipeline 1 — RAG: embed the question, top-k chunk similarity search in TigerVector, one generation."""
from __future__ import annotations

from .. import graph_tools as gt, tracing
from .common import Result, answer_from_context, run_traced

TOP_K = 8


def _run(question: str, res: Result):
    with tracing.agent("similarity_search"):
        chunks = gt.search_chunks(question, k=TOP_K)
    res.retrieved_doc_ids = [c["doc_id"] for c in chunks]
    context = "\n\n".join(f"[doc_id={c['doc_id']}] {c['text']}" for c in chunks)
    with tracing.agent("answer_generator"):
        out = answer_from_context(question, context)
    res.answer, res.citations = out["answer"], out["citations"]


def run(q: dict) -> Result:
    return run_traced("rag", q, _run)
