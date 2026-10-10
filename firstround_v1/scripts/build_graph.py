"""Build the OlympicsKG graph in TigerGraph from corpus.jsonl.

Steps (idempotent; upserts):
  1. create schema (skipped if the graph exists)
  2. parse infoboxes -> Document / Event / Games / Sport / Venue / Competitor / NOC + edges
  3. link Games chronologically and each Event to the same event at the previous/next Games
  4. chunk every document, embed chunks and entity names, upsert vectors (TigerVector)

Usage:  python scripts/build_graph.py [--skip-schema] [--skip-vectors]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz import fuzz  # noqa: E402

from agr import config, llm  # noqa: E402
from agr.chunking import chunk_text  # noqa: E402
from agr.parse import load_docs, norm, to_event  # noqa: E402
from agr.tg import get_conn  # noqa: E402

GSQL_DIR = Path(__file__).resolve().parent.parent / "gsql"


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def create_schema(conn):
    graphs = conn.gsql("SHOW GRAPH *")
    if f"Graph {config.TG_GRAPH}(" in graphs:
        print(f"graph {config.TG_GRAPH} exists, skipping schema")
        return
    out = conn.gsql((GSQL_DIR / "schema.gsql").read_text())
    print(out[-400:])


def upsert_v(conn, vtype, rows, size=400):
    n = 0
    for part in batched(rows, size):
        n += conn.upsertVertices(vtype, part)
    print(f"  {vtype}: {n}")


def upsert_e(conn, st, et, tt, rows, size=2000):
    n = 0
    for part in batched(rows, size):
        n += conn.upsertEdges(st, et, tt, part)
    print(f"  {et}: {n}")


def games_chain(games_ids: set[str]) -> list[tuple[str, str]]:
    """Link consecutive Games of the same season (only real 2/4-year gaps, not 1900 -> 1988)."""
    by_season = defaultdict(list)
    for g in games_ids:
        y, s = g.split()
        by_season[s].append(int(y))
    links = []
    for s, years in by_season.items():
        years.sort()
        for a, b in zip(years, years[1:]):
            if b - a <= 4:
                links.append((f"{a} {s}", f"{b} {s}"))
    return links


def edition_links(events) -> list[tuple[str, str]]:
    """Connect each event to the same event at the next Games (fuzzy on event name within sport)."""
    by_games_sport = defaultdict(list)
    for e in events:
        by_games_sport[(e.games_id, e.sport)].append(e)
    next_of = dict(games_chain({e.games_id for e in events}))
    links = []
    for e in events:
        nxt = next_of.get(e.games_id)
        if not nxt:
            continue
        cands = by_games_sport.get((nxt, e.sport), [])
        best, score = None, 0
        for c in cands:
            if c.gender != e.gender and "open" not in (c.gender, e.gender):
                continue
            s = fuzz.ratio(norm(e.event_name), norm(c.event_name))
            if s > score:
                best, score = c, s
        if best and score >= 88:
            links.append((e.event_id, best.event_id))
    return links


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-schema", action="store_true")
    ap.add_argument("--skip-vectors", action="store_true")
    args = ap.parse_args()

    conn = get_conn("")
    if not args.skip_schema:
        create_schema(conn)
    conn = get_conn()
    t0 = time.time()

    docs = load_docs(config.CORPUS_PATH)
    events = [e for d in docs if (e := to_event(d))]
    print(f"{len(docs)} documents, {len(events)} Olympic events")

    # ---- vertices -------------------------------------------------------
    upsert_v(conn, "Document", [(d.doc_id, {"title": d.title, "url": d.url, "infobox_kind": d.infobox_kind or "",
                                            "approx_tokens": d.approx_tokens, "text": d.text}) for d in docs], 100)
    games = {e.games_id for e in events}
    upsert_v(conn, "Games", [(g, {"year": int(g.split()[0]), "season": g.split()[1]}) for g in games])
    sports = {e.sport for e in events}
    upsert_v(conn, "Sport", [(norm(s), {"name": s}) for s in sports])
    upsert_v(conn, "Event", [(e.event_id, {
        "title": e.title, "event_name": e.event_name, "sport": e.sport, "games": e.games_id, "year": e.year,
        "season": e.season, "gender": e.gender, "competitors": e.competitors, "nations": e.nations,
        "date": e.date, "venue": e.venue, "win_value": e.win_value,
        "gold": " | ".join(w for m, w, _ in e.medals if m == "gold")}) for e in events])
    venues = {norm(e.venue): e.venue for e in events if e.venue}
    upsert_v(conn, "Venue", [(k, {"name": v}) for k, v in venues.items()])
    comps = {norm(w): w for e in events for _, w, _ in e.medals}
    upsert_v(conn, "Competitor", [(k, {"name": v}) for k, v in comps.items()])
    nocs = {n for e in events for _, _, n in e.medals if n}
    upsert_v(conn, "NOC", [(n, {}) for n in nocs])

    # ---- edges ----------------------------------------------------------
    upsert_e(conn, "Document", "DESCRIBES", "Event", [(e.event_id, e.event_id, {}) for e in events])
    upsert_e(conn, "Event", "AT_GAMES", "Games", [(e.event_id, e.games_id, {}) for e in events])
    upsert_e(conn, "Event", "OF_SPORT", "Sport", [(e.event_id, norm(e.sport), {}) for e in events])
    upsert_e(conn, "Event", "HELD_AT", "Venue", [(e.event_id, norm(e.venue), {}) for e in events if e.venue])
    upsert_e(conn, "Competitor", "WON_MEDAL", "Event",
             [(norm(w), e.event_id, {"medal": m}) for e in events for m, w, _ in e.medals])
    upsert_e(conn, "Competitor", "REPRESENTS", "NOC",
             list({(norm(w), n): (norm(w), n, {}) for e in events for _, w, n in e.medals if n}.values()))
    upsert_e(conn, "Games", "NEXT_GAMES", "Games", [(a, b, {}) for a, b in games_chain(games)])
    links = edition_links(events)
    upsert_e(conn, "Event", "NEXT_EDITION", "Event", [(a, b, {}) for a, b in links])
    print(f"structured graph loaded in {time.time() - t0:.0f}s")

    if args.skip_vectors:
        return

    # ---- chunks + vectors ----------------------------------------------
    chunk_rows, chunk_texts, edges = [], [], []
    for d in docs:
        for i, c in enumerate(chunk_text(d.title, d.text)):
            cid = f"{d.doc_id}#{i}"
            chunk_rows.append((cid, {"doc_id": d.doc_id, "doc_title": d.title, "seq": i, "text": c}))
            chunk_texts.append(c)
            edges.append((d.doc_id, cid, {}))
    print(f"{len(chunk_rows)} chunks; embedding (cached) ...")
    for part_rows, part_texts in zip(batched(chunk_rows, 1024), batched(chunk_texts, 1024)):
        vecs = llm.embed(part_texts, name="index_chunks")
        upsert_v(conn, "Chunk", [(cid, {**a, "emb": v.tolist()}) for (cid, a), v in zip(part_rows, vecs)], 200)
    upsert_e(conn, "Document", "HAS_CHUNK", "Chunk", edges)

    ev_texts = [f"{e.title}. Venue: {e.venue}. Date: {e.date}." for e in events]
    vecs = llm.embed(ev_texts, name="index_events")
    upsert_v(conn, "Event", [(e.event_id, {"emb": v.tolist()}) for e, v in zip(events, vecs)], 200)
    vk = list(venues)
    vecs = llm.embed([venues[k] for k in vk], name="index_venues")
    upsert_v(conn, "Venue", [(k, {"emb": v.tolist()}) for k, v in zip(vk, vecs)], 200)
    ck = list(comps)
    vecs = llm.embed([comps[k] for k in ck], name="index_competitors")
    upsert_v(conn, "Competitor", [(k, {"emb": v.tolist()}) for k, v in zip(ck, vecs)], 200)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
