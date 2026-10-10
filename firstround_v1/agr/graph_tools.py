"""Graph-side retrieval primitives shared by the GraphRAG and Agentic pipelines.

Everything here is deterministic (GSQL queries + fuzzy matching); LLM calls live
in the pipelines.  Each function returns plain JSON-able dicts and records a
``graph`` op on the active trace, so the benchmark sees every database call.
"""
from __future__ import annotations

import re
import threading
from functools import lru_cache

from rapidfuzz import fuzz, process

from . import llm, tracing
from .parse import gender_of, norm
from .tg import get_conn, run_query

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}


def gsql(name: str, params: dict | None = None) -> list[dict]:
    with tracing.op("graph", name, params={k: v for k, v in (params or {}).items() if k != "qv"}) as o:
        res = run_query(name, params)
        o.detail["rows"] = sum(len(v) if isinstance(v, list) else 1 for r in res for v in r.values())
    return res


def _vset(rows: list[dict], key: str) -> list[dict]:
    for r in rows:
        if key in r:
            # GSQL prints projected attributes as "<alias>.<attr>"; drop the alias
            return [{"id": v["v_id"], **{k.split(".", 1)[-1]: x for k, x in v["attributes"].items()}}
                    for v in r[key]]
    return []


def _val(rows: list[dict], key: str):
    for r in rows:
        if key in r:
            return r[key]
    return None


# --------------------------------------------------------------------------- entity index
class EntityIndex:
    """In-memory catalogue of the small vertex types, loaded once from TigerGraph."""

    _lock = threading.Lock()
    _inst: "EntityIndex | None" = None

    def __init__(self):
        conn = get_conn()
        self.games = {v["v_id"] for v in conn.getVertices("Games")}
        self.sports = {v["v_id"]: v["attributes"]["name"] for v in conn.getVertices("Sport")}
        self.venues = {v["v_id"]: v["attributes"]["name"] for v in conn.getVertices("Venue", select="name")}
        self.competitors = {v["v_id"]: v["attributes"]["name"] for v in conn.getVertices("Competitor", select="name")}

    @classmethod
    def get(cls) -> "EntityIndex":
        with cls._lock:
            if cls._inst is None:
                cls._inst = EntityIndex()
            return cls._inst


def _fuzzy(query: str, choices: dict[str, str], limit: int = 3) -> list[dict]:
    hits = process.extract(norm(query), {k: norm(v) for k, v in choices.items()}, scorer=fuzz.WRatio, limit=limit)
    return [{"id": k, "name": choices[k], "score": round(s, 1)} for _, s, k in hits]


def link_games(year: int | None, season: str | None) -> list[str]:
    idx = EntityIndex.get()
    if not year:
        return []
    seasons = [season.capitalize()] if season else ["Summer", "Winter"]
    return [g for g in (f"{year} {s}" for s in seasons) if g in idx.games]


def link_sport(sport: str) -> list[dict]:
    return _fuzzy(sport, EntityIndex.get().sports)


def link_venue(venue: str) -> list[dict]:
    """Fuzzy name match; falls back to TigerVector similarity when the name is garbled."""
    hits = _fuzzy(venue, EntityIndex.get().venues)
    if not hits or hits[0]["score"] < 85:
        rows = gsql("vs_venue", {"qv": llm.embed_one(venue, "embed_mention"), "k": 3})
        seen = {h["id"] for h in hits}
        hits += [{"id": v["id"], "name": v["name"], "score": None, "via": "vector"}
                 for v in _vset(rows, "hits") if v["id"] not in seen]
    return hits


def link_competitor(name: str) -> list[dict]:
    return _fuzzy(name, EntityIndex.get().competitors)


# --------------------------------------------------------------------------- traversals
def events_of(games_id: str, sport: str = "", min_competitors: int = -1) -> dict:
    rows = gsql("games_events", {"g": (games_id,), "sport": sport, "min_competitors": min_competitors})
    return {
        "games": games_id, "sport": sport,
        "n_events": _val(rows, "n_events"),
        "n_above_threshold": _val(rows, "n_above_threshold"),
        "max_competitors": _val(rows, "max_competitors"),
        "events": _vset(rows, "events"),
    }


def _ev_norm(s: str) -> str:
    # keep weight-class signs that norm() would drop: "+80 kg" and "80 kg" are different events
    return norm(s.replace("+", " plus ").replace("−", " minus ").replace("-", " minus ") if re.search(r"[+−-]\s*\d", s) else s)


def find_event(games_id: str, sport: str, event_name: str) -> list[dict]:
    """Resolve an event mention inside one Games + sport; gender-aware fuzzy ranking, exact name wins."""
    evs = events_of(games_id, sport)["events"]
    want = gender_of(event_name)
    q = _ev_norm(re.sub(r"\b(event|athletics|swimming)\b", "", event_name, flags=re.I))
    scored = []
    for e in evs:
        name = _ev_norm(e["title"].split(" – ", 1)[-1])
        g = gender_of(name)
        penalty = 0 if (g == want or "open" in (g, want)) else 40
        score = 100.0 if q == name else fuzz.token_set_ratio(q, name) * 0.3 + fuzz.ratio(q, name) * 0.7 - penalty
        scored.append((min(score, 99.0) if q != name else score, e))
    scored.sort(key=lambda t: -t[0])
    return [{"id": e["id"], "title": e["title"], "score": round(s, 1)} for s, e in scored[:3]]


def event_profile(event_id: str) -> dict:
    rows = gsql("event_profile", {"e": (event_id,)})
    ev = (_vset(rows, "event") or [{}])[0]
    return {
        "event": ev,
        "medalists": [{"name": m["name"], "medals": m["@medals"]} for m in _vset(rows, "medalists")],
        "previous_edition": _vset(rows, "previous_edition"),
        "next_edition": _vset(rows, "next_edition"),
    }


def _date_match(event_date: str, day: int | None, month: int | None) -> bool:
    d = event_date.lower()
    if month and not any(name in d or name[:3] + " " in d for name, i in MONTHS.items() if i == month):
        return False
    if day:
        # day may be the start of a range ("11–19 August") or a single day ("August 12, 2008")
        nums = [int(n) for n in re.findall(r"\b(\d{1,2})\b", d)]
        if day not in nums and not any(a <= day <= b for a, b in zip(nums, nums[1:]) if a < b):
            return False
    return True


def parse_date(text: str) -> tuple[int | None, int | None, int | None]:
    t = text.lower()
    month = next((i for m, i in MONTHS.items() if m in t), None)
    year = next((int(y) for y in re.findall(r"\b(1[89]\d\d|20\d\d)\b", t)), None)
    days = [int(x) for x in re.findall(r"\b(\d{1,2})\b", re.sub(r"\b(1[89]\d\d|20\d\d)\b", "", t))]
    return (days[0] if days else None), month, year


def venue_events(venue_id: str, date: str = "", year: int | None = None) -> dict:
    rows = gsql("venue_events", {"v": (venue_id,)})
    evs = _vset(rows, "events")
    day, month, y2 = parse_date(date) if date else (None, None, None)
    year = year or y2
    in_year = [e for e in evs if not year or e["games"].startswith(str(year))]
    # question dates are usually copied verbatim from the infobox, so an exact string match ranks first
    exact = [dict(e, date_match="exact") for e in in_year if date and norm(date) == norm(e.get("date", ""))]
    partial = [dict(e, date_match="partial") for e in in_year
               if e["id"] not in {x["id"] for x in exact} and _date_match(e.get("date", ""), day, month)]
    events = exact + partial if date else in_year
    return {"venue": (_vset(rows, "venue") or [{}])[0].get("name", venue_id), "n_events_total": len(evs),
            "filter": {"date": date, "year": year}, "n_exact_date_matches": len(exact), "events": events or evs[:40],
            "note": "" if events else "no event matched the filter; showing unfiltered list"}


def adjacent_games(games_id: str) -> dict:
    rows = gsql("adjacent_games", {"g": (games_id,)})
    fmt = lambda xs: [f"{x['year']} {x['season']}" for x in xs]
    return {"games": games_id, "previous": fmt(_vset(rows, "previous")), "next": fmt(_vset(rows, "next"))}


def competitor_medals(competitor_id: str) -> dict:
    rows = gsql("competitor_medals", {"c": (competitor_id,)})
    return {"competitor": (_vset(rows, "competitor") or [{}])[0].get("name"), "medals": _vset(rows, "medals")}


def search_chunks(query: str, k: int = 6) -> list[dict]:
    rows = gsql("vs_chunks", {"qv": llm.embed_one(query), "k": k})
    dist = _val(rows, "dist") or {}
    out = [{"chunk_id": c["id"], "doc_id": c["doc_id"], "doc_title": c["doc_title"], "text": c["text"],
            "distance": round(float(dist.get(c["id"], 0)), 4)} for c in _vset(rows, "chunks")]
    return sorted(out, key=lambda c: c["distance"])


def search_events(query: str, k: int = 5) -> list[dict]:
    rows = gsql("vs_event", {"qv": llm.embed_one(query), "k": k})
    return _vset(rows, "hits")


def document(doc_id: str, max_chars: int = 4000) -> dict:
    with tracing.op("graph", "get_document", doc_id=doc_id):
        v = get_conn().getVerticesById("Document", doc_id)
    if not v:
        return {"doc_id": doc_id, "error": "not found"}
    a = v[0]["attributes"]
    return {"doc_id": doc_id, "title": a["title"], "url": a["url"], "text": a["text"][:max_chars],
            "truncated": len(a["text"]) > max_chars}


@lru_cache(maxsize=1)
def sport_names() -> list[str]:
    return sorted(EntityIndex.get().sports.values())
