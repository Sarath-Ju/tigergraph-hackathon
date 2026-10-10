"""Pipeline 2 — GraphRAG: a fixed, single-pass graph retrieval.

1. Seed entities by TigerVector search over Event and Venue embeddings (plus a few chunks).
2. Expand the seeds one hop over the typed edges (games, venue, medallists, previous/next edition).
3. Serialise the subgraph (entity records + triples) and the chunks, and generate once.

No planning and no loop: the retrieval path is identical for every question.
"""
from __future__ import annotations

from .. import graph_tools as gt, llm, tracing
from .common import Result, answer_from_context, run_traced

K_EVENTS, K_VENUES, K_CHUNKS = 6, 2, 3


def _run(question: str, res: Result):
    qv = llm.embed_one(question)
    with tracing.agent("similarity_search"):
        ev = gt._vset(gt.gsql("vs_event", {"qv": qv, "k": K_EVENTS}), "hits")
        vn = gt._vset(gt.gsql("vs_venue", {"qv": qv, "k": K_VENUES}), "hits")
        chunks = gt._vset(gt.gsql("vs_chunks", {"qv": qv, "k": K_CHUNKS}), "chunks")
    seeds = [(e["id"], "Event") for e in ev] + [(v["id"], "Venue") for v in vn]
    with tracing.agent("graph_traversal"):
        rows = gt.gsql("expand_seeds", {"seeds": seeds})
    triples = gt._val(rows, "triples") or []
    events = {e["id"]: e for e in gt._vset(rows, "events")}
    for e in ev:  # seed events carry fewer attributes; neighbours from the expansion have the full record
        events.setdefault(e["id"], e)

    res.retrieved_doc_ids = list(events) + [c["doc_id"] for c in chunks]
    lines = ["# Entity records (doc_id = event id)"]
    lines += [f"[doc_id={i}] " + "; ".join(f"{k}={v}" for k, v in e.items() if k != "id") for i, e in events.items()]
    lines += ["", "# Graph triples", *triples[:120], "", "# Text chunks"]
    lines += [f"[doc_id={c['doc_id']}] {c['text']}" for c in chunks]
    with tracing.agent("answer_generator"):
        out = answer_from_context(question, "\n".join(lines))
    res.answer, res.citations = out["answer"], out["citations"]


def run(q: dict) -> Result:
    return run_traced("graphrag", q, _run)
