"""TigerGraph's own GraphRAG service (github.com/tigergraph/graphrag) as benchmark pipelines.

The service runs in its own containers against a graph built by its standard pipeline
(chunking -> LLM entity/relationship extraction -> communities). We call it exactly as
shipped, with its own retriever routing, and record what it reports:

  tg_classic   POST /{graph}/query  mode=classic  rag_method=auto     (fixed pipeline; auto-selected retriever)
  tg_agentic   POST /{graph}/query  mode=agentic  rag_method=planned  (planner -> executor DAG -> synthesis)

Token usage comes from the service's ``query_sources.token_usage`` (LLM input/output only;
the service does not report embedding tokens). Its plan/steps are mapped onto our trace events.
"""
from __future__ import annotations

import os
import re
import time

import requests

from .. import config, tracing
from .common import Result, run_traced

TG_SERVICE = os.getenv("TG_GRAPHRAG_HOST", "http://localhost:8000")
TG_SERVICE_GRAPH = os.getenv("TG_GRAPHRAG_GRAPH", "corpus5")
_DOC_ID = re.compile(r"\b(q\d+)(?:_chunk_\d+)?\b", re.I)


def _doc_ids(obj) -> list[str]:
    """Collect corpus doc ids (the service lowercases them, and chunk ids are '<doc>_chunk_<n>')."""
    found: list[str] = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            found.extend(m.group(1).upper() for m in _DOC_ID.finditer(x))
    walk(obj)
    return list(dict.fromkeys(found))


def _call(question: str, mode: str, rag_method: str) -> dict:
    url = f"{TG_SERVICE}/{TG_SERVICE_GRAPH}/query"
    body = {"query": question, "mode": mode, "rag_method": rag_method, "include_fields": ["all"]}
    for attempt in range(4):
        try:
            r = requests.post(url, json=body, auth=(config.TG_USERNAME, config.TG_PASSWORD), timeout=600)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    return {}


def _make(pipeline: str, mode: str, rag_method: str):
    def _run(question: str, res: Result):
        tr = tracing.current()
        with tracing.agent(f"tg_{mode}"), tracing.op("llm", f"graphrag_service_{mode}", model="service") as o:
            out = _call(question, mode, rag_method)
            qs = out.get("query_sources") or {}
            usage = qs.get("token_usage") or {}
            o.input_tokens = int(usage.get("input_tokens") or 0)
            o.output_tokens = int(usage.get("output_tokens") or 0)
            o.detail.update(engine=qs.get("engine"), cost_usd=usage.get("cost"),
                            answered=out.get("answered_question"), response_type=out.get("response_type"))
        # map the service's own plan / steps onto trace events so all pipelines are inspectable the same way
        steps = qs.get("steps") or qs.get("agent_steps") or []
        if qs.get("plan"):
            tr.note("plan", actions=[s.get("tool") for s in (qs["plan"].get("steps") if isinstance(qs["plan"], dict) else qs["plan"]) or []
                                     if isinstance(s, dict)], thought=str(qs["plan"])[:500])
        for s in steps:
            if isinstance(s, dict):
                tr.note("action", tool=s.get("tool") or s.get("node") or s.get("step_id") or "step",
                        agent=f"tg_{mode}", method=s.get("kind", ""), args=s.get("args", {}),
                        reason=str(s.get("rationale") or s.get("summary") or s.get("output") or "")[:300])
        tr.note("stop", reason="service_returned", turn=len(steps), why=qs.get("engine", mode), evidence_items=len(steps))
        tr.note("service_raw", query_sources={k: v for k, v in qs.items() if k not in ("error_traceback",)})
        res.answer = (out.get("natural_language_response") or "").strip()
        res.citations = _doc_ids(qs.get("citations") or [])
        res.retrieved_doc_ids = _doc_ids({k: v for k, v in qs.items() if k != "token_usage"})
        if not out.get("answered_question", True) and "error_traceback" in qs:
            res.error = "service error: " + res.answer[:200]

    return lambda q: run_traced(pipeline, q, _run)


run_classic = _make("tg_classic", "classic", "auto")
run_agentic = _make("tg_agentic", "agentic", "planned")
