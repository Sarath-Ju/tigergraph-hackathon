"""Per-question trace of every operation: tokens, latency, agent, tool, detail.

A ``Trace`` is bound to the current context (contextvars), so LLM / embedding /
graph calls made anywhere during a pipeline run are attributed automatically.
"""
from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

_current: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar("trace", default=None)


@dataclass
class Op:
    step: int
    kind: str                 # llm | embed | graph | tool | harness
    name: str                 # e.g. orchestrator, entity_linker, vs_chunks
    agent: str = ""
    start_ms: float = 0.0
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    embed_tokens: int = 0
    context_tokens: int = 0   # retrieved evidence tokens handed to an LLM
    detail: dict = field(default_factory=dict)


@dataclass
class Trace:
    pipeline: str
    qid: str
    question: str
    ops: list[Op] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)  # agentic decisions: plan, strategy changes, stop reason
    t0: float = field(default_factory=time.perf_counter)
    _agent_stack: list[str] = field(default_factory=list)

    @property
    def agent(self) -> str:
        return self._agent_stack[-1] if self._agent_stack else ""

    def add(self, op: Op) -> Op:
        op.step = len(self.ops) + 1
        if not op.agent:
            op.agent = self.agent
        self.ops.append(op)
        return op

    def note(self, kind: str, **data):
        self.events.append({"kind": kind, "t_ms": round((time.perf_counter() - self.t0) * 1000, 1), **data})

    def totals(self) -> dict:
        s = lambda k: sum(getattr(o, k) for o in self.ops)
        llm_ops = [o for o in self.ops if o.kind == "llm"]
        return {
            "llm_calls": len(llm_ops),
            "llm_input_tokens": s("input_tokens"),
            "llm_output_tokens": s("output_tokens"),
            "embed_tokens": s("embed_tokens"),
            "context_tokens": s("context_tokens"),
            "total_tokens": s("input_tokens") + s("output_tokens") + s("embed_tokens"),
            "wall_ms": round((time.perf_counter() - self.t0) * 1000, 1),
        }

    def to_dict(self) -> dict:
        return {
            "pipeline": self.pipeline,
            "qid": self.qid,
            "totals": self.totals(),
            "events": self.events,
            "ops": [asdict(o) for o in self.ops],
        }


def current() -> Trace | None:
    return _current.get()


@contextmanager
def tracing(trace: Trace):
    token = _current.set(trace)
    try:
        yield trace
    finally:
        _current.reset(token)


@contextmanager
def agent(name: str):
    """Attribute all ops inside the block to a named (specialised) agent."""
    tr = current()
    if tr is None:
        yield
        return
    tr._agent_stack.append(name)
    try:
        yield
    finally:
        tr._agent_stack.pop()


@contextmanager
def op(kind: str, name: str, **detail):
    """Time a block and record it as an op; yields the Op so callers can add tokens/detail."""
    tr = current()
    o = Op(step=0, kind=kind, name=name, detail=dict(detail))
    t = time.perf_counter()
    if tr is not None:
        o.start_ms = round((t - tr.t0) * 1000, 1)
    try:
        yield o
    finally:
        o.duration_ms = round((time.perf_counter() - t) * 1000, 1)
        if tr is not None:
            tr.add(o)
