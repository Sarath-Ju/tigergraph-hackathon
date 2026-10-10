"""The approaches we compare.

All of them use TigerGraph GraphRAG's own retrievers and agent over the same graph. For rag and graphrag
we call the retriever directly and write the answer with one LLM call, using the same prompt for both,
so the only difference between them is what was retrieved. That also gives us exact token counts.

  rag       TigerGraph GraphRAG's similarity retriever finds the closest chunks, then a single LLM
            call writes the answer.
  graphrag  TigerGraph GraphRAG's hybrid retriever finds starting chunks by vector search and
            expands the graph two hops from there, then a single LLM call writes the answer.
  agentic   The planning agent decides which tools it needs (reading the schema, writing graph
            queries, vector and hybrid search), runs them, plans again if something is missing,
            and writes a cited answer.
  reactive  The same agent tools, but it decides one step at a time.

Each run gives back one record with the answer, the documents it cited and retrieved, token usage,
time taken, and the agent's own steps as the trace.
"""
from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import asdict, dataclass, field

from . import config, tg

ANSWER_STYLE = ("Answer from the corpus only. Reply with the short final answer only, "
                "such as a name, a number, a title or a date, without explanation.")

PIPELINES = {
    "rag": ("retriever", "similarity",
            {"index": "DocumentChunk", "top_k": config.TOP_K, "withHyDE": False}),
    # chunk_only keeps the expansion to text chunks, as the service's own hybrid search does; with
    # every neighbour included the context grows to about 20,000 tokens per question
    "graphrag": ("retriever", "hybrid",
                 {"indices": ["DocumentChunk"], "top_k": config.TOP_K, "num_hops": config.NUM_HOPS, "num_seen_min": 2,
                  "chunk_only": True}),
    "agentic": ("agentic", "planned", None),
    "reactive": ("agentic", "reactive", None),
}

ANSWER_PROMPT = """You answer questions about a document collection using only the passages given.
Each passage starts with its id in square brackets.
Reply with JSON: {"answer": "<short final answer: a name, number, title or date, no explanation>",
"sources": ["<ids of the passages the answer comes from>"]}.
If the passages do not contain the answer, reply {"answer": "UNKNOWN", "sources": []}."""
MAX_CONTEXT_CHARS = 60000
_llm = None


@dataclass
class Result:
    qid: str
    question: str
    pipeline: str
    qtype: str = ""
    answer: str = ""
    raw_answer: str = ""
    citations: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    error: str = ""
    trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------- doc-id mapping

class DocIds:
    """GraphRAG stores ids in lower case ("q303623", and chunks as "q303623_chunk_4"). This turns
    whatever ids appear in its response back into the corpus's own document ids."""
    _map: dict[str, str] | None = None

    @classmethod
    def load(cls, corpus_ids: list[str]):
        from .records.blocks import vertex_id
        cls._map = {vertex_id(i): i for i in corpus_ids}

    @classmethod
    def find(cls, obj) -> list[str]:
        assert cls._map is not None, "DocIds.load() first"
        found: list[str] = []

        def walk(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    walk(k)
                    walk(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    walk(v)
            elif isinstance(x, str):
                for tok in re.findall(r"[A-Za-z0-9_\-\.]+", x):
                    t = re.sub(r"_chunk_\d+$", "", tok.lower())
                    if t in cls._map:
                        found.append(cls._map[t])
        walk(obj)
        return list(dict.fromkeys(found))


# --------------------------------------------------------------------------------------- run

def _usage(qs: dict) -> dict:
    u = qs.get("token_usage") or {}
    i, o = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
    return {"llm_input_tokens": i, "llm_output_tokens": o, "total_tokens": i + o,
            "llm_calls": int(u.get("llm_calls") or u.get("calls") or 0), "cost_usd": u.get("cost")}


def _steps(qs: dict) -> list[dict]:
    steps = qs.get("steps") or qs.get("agent_steps") or []
    if not steps and isinstance(qs.get("plan"), dict):
        steps = qs["plan"].get("steps") or []
    out = []
    for s in steps if isinstance(steps, list) else []:
        if isinstance(s, dict):
            out.append({"tool": s.get("tool") or s.get("method") or s.get("node") or s.get("step_id") or "step",
                        "args": s.get("args") or s.get("tool_args") or {},
                        "reason": str(s.get("rationale") or s.get("reason") or s.get("summary") or "")[:300]})
    return out


def clean_answer(text: str) -> str:
    """The service answers in prose with markdown links for citations. For scoring we keep just the
    answer itself, without the links or formatting."""
    t = re.sub(r"\[[^\]]*\]\([^)]*\)", "", text or "")       # markdown links / citations
    t = re.sub(r"\*\*|__|`", "", t).strip()
    return t.split("\n\n")[0].strip()


def _context(found: dict) -> str:
    parts = []
    for cid, text in (found.get("final_retrieval") or {}).items():
        text = "\n".join(text) if isinstance(text, list) else str(text)
        parts.append(f"[{cid}]\n{text.strip()}")
    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


def _retrieve_and_answer(question: str, method: str, params: dict) -> dict:
    """One retriever call to the service, then one LLM call. Returns the same shape as tg.query."""
    global _llm
    from openai import OpenAI
    found = tg.search(question, method, params)
    context = _context(found)
    _llm = _llm or OpenAI()
    r = _llm.chat.completions.create(
        model=config.LLM_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": ANSWER_PROMPT},
                  {"role": "user", "content": f"Passages:\n\n{context}\n\nQuestion: {question}"}])
    try:
        out = json.loads(r.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        out = {"answer": r.choices[0].message.content or ""}
    u = r.usage
    return {"natural_language_response": str(out.get("answer", "")),
            "answered_question": True,
            "query_sources": {
                "engine": "retriever", "citations": out.get("sources") or [],
                "retrieved": list((found.get("final_retrieval") or {}).keys()),
                "token_usage": {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens,
                                "llm_calls": 1},
                "steps": [{"tool": f"{method}_search", "args": params,
                           "summary": f"{len(found.get('final_retrieval') or {})} chunks, "
                                      f"{len(context)} characters of context"},
                          {"tool": "generate_answer", "args": {"model": config.LLM_MODEL}}]}}


def run(pipeline: str, q: dict) -> Result:
    mode, method, params = PIPELINES[pipeline]
    res = Result(qid=q["qid"], question=q["question"], pipeline=pipeline, qtype=q.get("qtype", ""))
    t0 = time.perf_counter()
    out: dict = {}
    try:
        if mode == "retriever":
            out = _retrieve_and_answer(q["question"], method, params)
        else:
            out = tg.query(f"{q['question']}\n\n{ANSWER_STYLE}", mode, method)
    except Exception as e:  # one failed question should not stop the benchmark; we keep the error
        res.error = f"{type(e).__name__}: {e}"
        res.trace = {"error_tb": traceback.format_exc(limit=3)}
    wall_ms = round((time.perf_counter() - t0) * 1000, 1)
    qs = out.get("query_sources") or {}
    res.raw_answer = (out.get("natural_language_response") or "").strip()
    res.answer = clean_answer(res.raw_answer)
    res.citations = DocIds.find(qs.get("citations") or qs.get("selected") or out.get("natural_language_response"))
    # rows returned by the agent's graph queries are evidence too, and for counting questions they are the
    # main evidence, so we cite the documents those records came from
    structural = (qs.get("result") or {}).get("structural") if isinstance(qs.get("result"), dict) else None
    res.citations = list(dict.fromkeys(res.citations + DocIds.find([r.get("result") for r in structural or []
                                                                     if isinstance(r, dict)])))
    res.retrieved_doc_ids = DocIds.find({k: v for k, v in qs.items() if k not in ("token_usage", "error_traceback")})
    if not out.get("answered_question", True) and qs.get("error_traceback"):
        res.error = "service error: " + res.raw_answer[:200]
    steps = _steps(qs)
    res.trace = {**res.trace, "engine": qs.get("engine", mode), "style": method,
                 "totals": {**_usage(qs), "wall_ms": wall_ms, "tool_calls": len(steps)},
                 "steps": steps, "service": {k: v for k, v in qs.items() if k != "error_traceback"}}
    return res
