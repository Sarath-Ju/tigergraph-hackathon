"""Pipeline 3 — Agentic GraphRAG.

An orchestrator LLM plans the investigation and picks the next action from the
question, the graph schema, and the evidence gathered so far.  Each tool is
served by a specialised agent:

  entity_linker      resolve free-text mentions to Games / Sport / Event / Venue / Competitor vertices
  graph_traversal    event profiles, venue -> events, competitor -> medals, previous/next edition
  temporal_reasoner  Games chronology (NEXT_GAMES / PREV_GAMES)
  aggregator         in-graph filtering, counting and arg-max over every event of a Games + sport
  similarity_search  TigerVector search over document chunks
  doc_retriever      full document text for grounding and citations
  evidence_evaluator gate on ``finish``: deterministic grounding check first, LLM judge only if that fails

The harness owns state (messages, evidence ledger), the tool registry, context
budgets, loop guards and stopping criteria, and writes every decision to the trace.
Deterministic agents cost no LLM tokens; the LLM is used where judgement is needed.
"""
from __future__ import annotations

import json
import re

from .. import graph_tools as gt, llm, tracing
from ..parse import norm
from .common import ANSWER_RULES, Result, compact, run_traced

MAX_TURNS = 6          # orchestrator LLM calls before a forced finish
TOKEN_BUDGET = 60_000  # total LLM tokens per question before a forced finish
TOOL_RESULT_TOKENS = 2500

SYSTEM = f"""You are the orchestrator of an investigation over a TigerGraph knowledge graph built from a corpus of
Wikipedia articles (mostly Olympic event articles 1988–2022, plus some films and people).

Graph schema:
  Games(year, season) -NEXT_GAMES-> Games
  Event(title, sport, games, date, venue, competitors, nations, gold) -AT_GAMES-> Games, -HELD_AT-> Venue, -OF_SPORT-> Sport
  Competitor -WON_MEDAL(medal)-> Event ; Event -NEXT_EDITION-> Event (same event at the next Games)
  Every Event is described by a Document with the same id (cite it). All document text is chunked and vector-indexed.

Work like an investigator:
1. Decide what facts are needed. Call the tool(s) that retrieve them; independent calls can run in parallel.
2. Read each result. If a lookup fails, is ambiguous, or looks wrong, change strategy (re-link with other wording,
   try similarity search, read the document) instead of guessing.
3. Call `finish` as soon as the evidence answers the question. Do not answer from memory.

Routing hints: counting / "which event had the most ..." -> aggregate_events (scans every event in the graph; never
estimate from text). "Games held immediately before/after YEAR" -> neighbor_games first, then look up the event at that
Games. "event held at VENUE on DATE" -> events_at_venue. A named event -> lookup_event.
{ANSWER_RULES}"""

_REASON = {"reason": {"type": "string", "description": "One short sentence: why this action now."}}


def _fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {**props, **_REASON}, "required": required + ["reason"]}}}


_S = {"type": "string"}
_I = {"type": "integer"}
_SEASON = {"type": "string", "enum": ["Summer", "Winter"]}
TOOLS = [
    _fn("lookup_event", "Link a named Olympic event (sport + Games + event name) and return its full profile: date, "
        "venue, competitors, nations, medallists, previous/next edition. If a previous lookup listed other_candidates, "
        "pass the chosen candidate's event_id instead.",
        {"sport": _S, "year": _I, "season": _SEASON, "event_name": {"type": "string", "description": "e.g. Men's 20 kilometres walk"},
         "event_id": {"type": "string", "description": "optional: exact event id from other_candidates"}},
        ["sport", "year", "season", "event_name"]),
    _fn("events_at_venue", "Events held at a venue, optionally filtered by date text and/or year, with gold medallists.",
        {"venue": _S, "date": _S, "year": _I}, ["venue"]),
    _fn("neighbor_games", "The Games of the same season held immediately before and after a given Games.",
        {"year": _I, "season": _SEASON}, ["year", "season"]),
    _fn("aggregate_events", "Scan every event of one sport at one Games in the graph and compute an aggregate over "
        "competitors or nations: count_greater_than / count_at_least / count_less_than (needs threshold), argmax, argmin, list.",
        {"sport": _S, "year": _I, "season": _SEASON, "metric": {"type": "string", "enum": ["competitors", "nations"]},
         "operation": {"type": "string", "enum": ["count_greater_than", "count_at_least", "count_less_than", "argmax", "argmin", "list"]},
         "threshold": _I}, ["sport", "year", "season", "metric", "operation"]),
    _fn("competitor_medals", "All medals a competitor won, per event.", {"name": _S}, ["name"]),
    _fn("link_entities", "Resolve a free-text mention against sports, venues, competitors and events when you are unsure "
        "what it refers to. Returns ranked candidates.", {"mention": _S}, ["mention"]),
    _fn("similarity_search", "Vector search over document chunks. Use for facts not in the structured graph.",
        {"query": _S, "k": _I}, ["query"]),
    _fn("read_document", "Read a document's text (first ~3000 characters) by doc_id.", {"doc_id": _S}, ["doc_id"]),
    _fn("finish", "Submit the final answer. It is checked against the evidence before being accepted.",
        {"answer": _S, "citations": {"type": "array", "items": _S, "description": "doc_ids supporting the answer"},
         "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}, ["answer", "citations", "confidence"]),
]

AGENT_OF = {"lookup_event": "graph_traversal", "events_at_venue": "graph_traversal", "competitor_medals": "graph_traversal",
            "neighbor_games": "temporal_reasoner", "aggregate_events": "aggregator", "link_entities": "entity_linker",
            "similarity_search": "similarity_search", "read_document": "doc_retriever", "finish": "evidence_evaluator"}
METHOD_OF = {"lookup_event": "graph", "events_at_venue": "graph", "competitor_medals": "graph", "neighbor_games": "graph",
             "aggregate_events": "graph_aggregation", "link_entities": "entity_linking",
             "similarity_search": "vector", "read_document": "document"}


class ToolError(Exception):
    pass


# --------------------------------------------------------------------------- specialised agents
def _games(year, season) -> str:
    g = gt.link_games(year, season)
    if not g:
        raise ToolError(f"No {season} Games in {year} in the graph.")
    return g[0]


def _sport(sport: str) -> str:
    with tracing.agent("entity_linker"):
        hits = gt.link_sport(sport)
    if not hits or hits[0]["score"] < 80:
        raise ToolError(f"Sport '{sport}' not found. Known sports: {', '.join(gt.sport_names())}")
    return hits[0]["name"]


def t_lookup_event(sport, year, season, event_name, event_id=None, **_):
    if event_id:
        prof = gt.event_profile(event_id)
        if not prof["event"]:
            raise ToolError(f"No event with id {event_id}.")
        return "ok", prof, [event_id]
    games, sport_name = _games(year, season), _sport(sport)
    with tracing.agent("entity_linker"):
        cands = gt.find_event(games, sport_name, event_name)
    if not cands:
        raise ToolError(f"No {sport_name} events at {games}.")
    status = "ok" if cands[0]["score"] >= 70 else "ambiguous"
    prof = gt.event_profile(cands[0]["id"])
    prof["linked_as"] = cands[0]
    if len(cands) > 1 and cands[0]["score"] - cands[1]["score"] < 3:
        status = "ambiguous"
    if status == "ambiguous":
        prof["other_candidates"] = cands[1:]
    return status, prof, [cands[0]["id"]]


def t_events_at_venue(venue, date="", year=None, **_):
    with tracing.agent("entity_linker"):
        cands = gt.link_venue(venue)
    use = [c for c in cands if (c["score"] or 0) >= 95] or cands[:1]
    events, names = [], []
    for c in use:
        r = gt.venue_events(c["id"], date or "", year)
        names.append(r["venue"])
        events += r["events"]
    exact = [e for e in events if e.get("date_match") == "exact"]
    if exact:
        events = exact + [e for e in events if e.get("date_match") != "exact"]
    status = "ok" if events else "empty"
    if date and len(exact) != 1 and len(events) > 1:
        status = "ambiguous"
    return status, {"venues_matched": names, "events": events[:40],
                    "venue_candidates": cands if status != "ok" else []}, [e["id"] for e in events[:40]]


def t_neighbor_games(year, season, **_):
    return "ok", gt.adjacent_games(_games(year, season)), []


_OPS = {"count_greater_than": lambda v, t: v > t, "count_at_least": lambda v, t: v >= t,
        "count_less_than": lambda v, t: 0 <= v < t}


def t_aggregate_events(sport, year, season, metric, operation, threshold=None, **_):
    games, sport_name = _games(year, season), _sport(sport)
    data = gt.events_of(games, sport_name, -1)
    evs = [{"id": e["id"], "title": e["title"], metric: e[metric]} for e in data["events"]]
    known = [e for e in evs if e[metric] >= 0]
    out = {"games": games, "sport": sport_name, "metric": metric, "operation": operation,
           "n_events_scanned": len(evs), "n_missing_metric": len(evs) - len(known)}
    if not evs:
        raise ToolError(f"No {sport_name} events at {games}.")
    if operation in _OPS:
        if threshold is None:
            raise ToolError("threshold is required for count operations")
        hit = [e for e in known if _OPS[operation](e[metric], threshold)]
        out.update(threshold=threshold, result=len(hit), matching_events=[e["title"] for e in hit])
    elif operation in ("argmax", "argmin"):
        best = (max if operation == "argmax" else min)(e[metric] for e in known)
        out.update(result=[e["title"] for e in known if e[metric] == best], value=best)
    out["all_events"] = sorted(([e["title"].split(" – ", 1)[-1], e[metric]] for e in evs), key=lambda x: -x[1])
    return "ok", out, [e["id"] for e in evs]


def t_competitor_medals(name, **_):
    with tracing.agent("entity_linker"):
        cands = gt.link_competitor(name)
    if not cands or cands[0]["score"] < 80:
        return "empty", {"candidates": cands}, []
    r = gt.competitor_medals(cands[0]["id"])
    return "ok", {**r, "linked_as": cands[0]}, [m["id"] for m in r["medals"]]


def t_link_entities(mention, **_):
    idx = gt.EntityIndex.get()
    out = {"sports": gt.link_sport(mention)[:2], "venues": gt.link_venue(mention)[:3],
           "competitors": gt.link_competitor(mention)[:3],
           "events": [{"id": e["id"], "title": e["title"]} for e in gt.search_events(mention, 5)]}
    out["games_in_graph"] = sorted(idx.games)
    return "ok", out, [e["id"] for e in out["events"]]


def t_similarity_search(query, k=5, **_):
    hits = gt.search_chunks(query, k=min(int(k or 5), 10))
    return ("ok" if hits else "empty"), hits, [h["doc_id"] for h in hits]


def t_read_document(doc_id, **_):
    d = gt.document(doc_id, max_chars=3000)
    return ("error" if "error" in d else "ok"), d, [doc_id]


EXEC = {"lookup_event": t_lookup_event, "events_at_venue": t_events_at_venue, "neighbor_games": t_neighbor_games,
        "aggregate_events": t_aggregate_events, "competitor_medals": t_competitor_medals,
        "link_entities": t_link_entities, "similarity_search": t_similarity_search, "read_document": t_read_document}


def _answer_type_issue(question: str, answer: str) -> str:
    """Cheap check that the answer has the shape the question asks for."""
    q, a = question.lower().strip(), answer.strip()
    is_title = bool(re.search(r"\bat the \d{4} (summer|winter) olympics\b", a.lower()))
    if q.startswith("who") and is_title:
        return "The question asks WHO (a person or team) but the answer is an event title; give the gold medallist name."
    if q.startswith("how many") and not re.fullmatch(r"\d[\d,]*", a):
        return "The question asks HOW MANY; the answer must be a single number."
    if re.match(r".*\bwhich\b.*\bevent\b", q) and not is_title:
        return "The question asks WHICH EVENT; answer with the full event article title."
    return ""


# --------------------------------------------------------------------------- harness
class Harness:
    def __init__(self, question: str, res: Result):
        self.q, self.res = question, res
        self.tr = tracing.current()
        self.ledger: list[dict] = []           # evidence items: id, tool, args, status, doc_ids, text
        self.seen_calls: dict[str, str] = {}   # call signature -> evidence id (loop guard)
        self.last_status: str | None = None
        self.last_method: str | None = None
        self.rejections = 0

    # ---- tool execution ----------------------------------------------------------------------
    def call(self, name: str, args: dict) -> str:
        reason = args.pop("reason", "")
        sig = f"{name}:{json.dumps(args, sort_keys=True)}"
        agent = AGENT_OF.get(name, "?")
        self.tr.note("action", tool=name, agent=agent, method=METHOD_OF.get(name), args=args, reason=reason)
        if sig in self.seen_calls:
            self.tr.note("loop_guard", tool=name, duplicate_of=self.seen_calls[sig])
            self.last_status = "duplicate"
            return json.dumps({"status": "duplicate", "note": f"Identical call already made; its result is "
                               f"{self.seen_calls[sig]}. Repeating it cannot help: pick a candidate event_id, use another "
                               "tool (similarity_search, read_document, link_entities) or finish."})
        method = METHOD_OF.get(name)
        if self.last_status in ("empty", "error", "ambiguous", "duplicate") and method != self.last_method:
            self.tr.note("strategy_change", from_method=self.last_method, to_method=method,
                         why=f"previous {self.last_method} result was {self.last_status}")
        with tracing.agent(agent), tracing.op("tool", name, args=args) as o:
            try:
                status, payload, doc_ids = EXEC[name](**args)
            except ToolError as e:
                status, payload, doc_ids = "error", {"error": str(e)}, []
            except Exception as e:  # tool bug or DB hiccup: surface it to the orchestrator, don't crash
                status, payload, doc_ids = "error", {"error": f"{type(e).__name__}: {e}"}, []
            o.detail.update(status=status, n_docs=len(doc_ids))
        eid = f"E{len(self.ledger) + 1}"
        text = compact(payload, TOOL_RESULT_TOKENS)
        self.ledger.append({"id": eid, "tool": name, "args": args, "status": status, "doc_ids": doc_ids, "text": text,
                            "payload": payload})
        self.seen_calls[sig] = eid
        self.res.retrieved_doc_ids += doc_ids
        self.last_status, self.last_method = status, method
        return f'{{"evidence_id":"{eid}","status":"{status}","result":{text}}}'

    # ---- evidence evaluator ------------------------------------------------------------------
    def _grounding(self, answer: str) -> list[dict]:
        a = norm(answer)
        if not a:
            return []
        if a.isdigit():
            pat = re.compile(rf'(?<![\d.])"?{a}"?(?![\d.])')
            return [e for e in self.ledger if pat.search(e["text"])]
        return [e for e in self.ledger if a in norm(e["text"])]

    def evaluate(self, answer: str, citations: list[str], confidence: str, turns_left: bool) -> tuple[bool, str]:
        with tracing.agent("evidence_evaluator"), tracing.op("tool", "evaluate_evidence") as o:
            support = self._grounding(answer)
            known = {d for e in self.ledger for d in e["doc_ids"]}
            valid = [c for c in citations if c in known]
            o.detail.update(grounded=bool(support), citations_valid=len(valid), citations_given=len(citations))
            type_issue = _answer_type_issue(self.q, answer)
            if answer.strip().upper() == "UNKNOWN" or not answer.strip():
                verdict, why = (not turns_left), "answer is UNKNOWN"
            elif type_issue and turns_left:
                verdict, why = False, type_issue
            elif support and confidence != "low":
                verdict, why = True, f"answer string found in {', '.join(e['id'] for e in support)}"
            else:
                verdict, why = self._llm_judge(answer)
            o.detail.update(accepted=verdict, type_issue=type_issue,
                            method="deterministic" if (support and confidence != "low") or type_issue else "llm")
        if verdict:
            self.res.citations = self._cite(answer, support, valid)
        return verdict, why

    def _cite(self, answer: str, support: list[dict], valid: list[str]) -> list[str]:
        """Cite the records that actually contain the answer.

        Aggregates are supported by every event scanned; for anything else we cite the event record(s)
        whose fields contain the answer, falling back to the orchestrator's (validated) citations.
        """
        if any(e["tool"] == "aggregate_events" for e in support):
            return sorted({d for e in support if e["tool"] == "aggregate_events" for d in e["doc_ids"]})
        a, hits = norm(answer), []

        def walk(x, owner=None):
            if isinstance(x, dict):
                own = x.get("id") if isinstance(x.get("id"), str) and x.get("id", "").startswith("Q") else owner
                if own and any(isinstance(v, str) and a and a in norm(v) for v in x.values()):
                    hits.append(own)
                for v in x.values():
                    walk(v, own)
            elif isinstance(x, list):
                for v in x:
                    walk(v, owner)
        for e in support:
            walk(e["payload"], e["doc_ids"][0] if len(e["doc_ids"]) == 1 else None)
        return list(dict.fromkeys(hits)) or valid or sorted({d for e in support for d in e["doc_ids"]})[:50]

    def _llm_judge(self, answer: str) -> tuple[bool, str]:
        ev = "\n".join(f"{e['id']} {e['tool']} {e['text'][:1500]}" for e in self.ledger[-6:])
        out = llm.chat_json([
            {"role": "system", "content": "You check whether an answer is supported by evidence. Return JSON "
                                          '{"supported": bool, "issue": str, "suggestion": str} (issue/suggestion short).'},
            {"role": "user", "content": f"Question: {self.q}\nProposed answer: {answer}\n\nEvidence:\n{ev}"}],
            name="evidence_evaluator", max_tokens=150, context_tokens=llm.count_tokens(ev))
        ok = bool(out.get("supported"))
        return ok, (out.get("issue", "") + (" Suggestion: " + out["suggestion"] if out.get("suggestion") else "")).strip()

    # ---- main loop ---------------------------------------------------------------------------
    def run(self):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Question: {self.q}"}]
        for turn in range(1, MAX_TURNS + 2):
            used = self.tr.totals()["total_tokens"]
            forced = turn > MAX_TURNS or used > TOKEN_BUDGET
            if forced:
                self.tr.note("budget", reason="max_turns" if turn > MAX_TURNS else "token_budget", tokens=used)
            ctx = sum(llm.count_tokens(m["content"]) for m in msgs if m.get("role") == "tool")
            with tracing.agent("orchestrator"):
                msg = llm.chat(msgs, name="orchestrator", tools=TOOLS, max_tokens=400, context_tokens=ctx,
                               tool_choice={"type": "function", "function": {"name": "finish"}} if forced else None)
            calls = msg.tool_calls or []
            msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in calls]} if calls else {"role": "assistant", "content": msg.content or ""})
            if turn == 1:
                self.tr.note("plan", actions=[c.function.name for c in calls], thought=msg.content or "")
            if not calls:  # answered in prose: treat as a finish attempt
                calls_finish = [("text", {"answer": (msg.content or "").strip(), "citations": [], "confidence": "medium"})]
            else:
                calls_finish = []
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if c.function.name == "finish":
                    calls_finish.append((c.id, args))
                    continue
                if c.function.name not in EXEC:
                    out = json.dumps({"status": "error", "error": f"unknown tool {c.function.name}"})
                else:
                    out = self.call(c.function.name, args)
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
            for cid, args in calls_finish:
                answer = str(args.get("answer", "")).strip()
                ok, why = self.evaluate(answer, [str(x) for x in args.get("citations", [])],
                                        args.get("confidence", "medium"), turns_left=not forced)
                self.tr.note("evaluation", answer=answer, accepted=ok, why=why)
                if ok or forced:
                    self.res.answer = answer
                    self.tr.note("stop", reason="evidence_sufficient" if ok and not forced else "budget_exhausted",
                                 turn=turn, why=why, evidence_items=len(self.ledger))
                    return
                self.rejections += 1
                self.tr.note("strategy_change", from_method="finish", to_method="continue_investigation",
                             why=f"evaluator rejected answer: {why}")
                fb = json.dumps({"status": "rejected", "feedback": why or "answer not supported by evidence",
                                 "instruction": "Gather the missing evidence, then call finish again."})
                if cid == "text":
                    msgs.append({"role": "user", "content": fb})
                else:
                    msgs.append({"role": "tool", "tool_call_id": cid, "content": fb})


def _run(question: str, res: Result):
    Harness(question, res).run()


def run(q: dict) -> Result:
    return run_traced("agentic", q, _run)
