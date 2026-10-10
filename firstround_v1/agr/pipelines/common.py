"""Pieces shared by all pipelines: result record, answer-format rules, context rendering."""
from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, field
from typing import Callable

from .. import llm, tracing

ANSWER_RULES = """Answer rules:
- The provided corpus/evidence is the only source of truth, even if it disagrees with what you remember.
- Give ONLY the short final answer: a name, a number, or (for "which event" questions) the full event
  article title such as "Athletics at the 2008 Summer Olympics – Men's marathon".
- Copy names exactly as written in the evidence (including accents and any run-together team names).
- If the evidence does not contain the answer, answer "UNKNOWN"."""


@dataclass
class Result:
    qid: str
    question: str
    pipeline: str
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    qtype: str = ""
    error: str = ""
    trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def run_traced(pipeline: str, q: dict, fn: Callable[[str, "Result"], None]) -> Result:
    """Run ``fn(question, result)`` inside a fresh trace and attach the trace to the result."""
    res = Result(qid=q["qid"], question=q["question"], pipeline=pipeline, qtype=q.get("qtype", ""))
    tr = tracing.Trace(pipeline=pipeline, qid=q["qid"], question=q["question"])
    with tracing.tracing(tr):
        try:
            fn(q["question"], res)
        except Exception as e:  # keep the benchmark running; the error is part of the record
            res.error = f"{type(e).__name__}: {e}"
            tr.note("error", error=res.error, tb=traceback.format_exc(limit=3))
    res.trace = tr.to_dict()
    res.retrieved_doc_ids = sorted(set(res.retrieved_doc_ids))
    return res


def answer_from_context(question: str, context: str, *, name: str = "answer_generator") -> dict:
    """Single grounded generation step used by RAG and GraphRAG."""
    msgs = [
        {"role": "system", "content": "You answer questions about Olympic events using only the given context.\n"
                                      + ANSWER_RULES +
                                      '\nReturn JSON: {"answer": str, "citations": [doc_id, ...]} '
                                      "citing the doc_ids of the context items you used."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    out = llm.chat_json(msgs, name=name, max_tokens=200, context_tokens=llm.count_tokens(context))
    return {"answer": str(out.get("answer", "")).strip(), "citations": [str(c) for c in out.get("citations", [])]}


def compact(obj, max_tokens: int = 2500) -> str:
    """Serialise a tool result for the LLM, truncating to a token budget."""
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    toks = llm._enc.encode(s, disallowed_special=())
    if len(toks) <= max_tokens:
        return s
    return llm._enc.decode(toks[:max_tokens]) + f"...[truncated {len(toks) - max_tokens} tokens]"
