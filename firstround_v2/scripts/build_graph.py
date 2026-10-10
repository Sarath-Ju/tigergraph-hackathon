"""Build the knowledge graph from any JSONL corpus, using TigerGraph GraphRAG.

    python scripts/build_graph.py                  the whole corpus
    python scripts/build_graph.py --sample 100     a small, varied sample first, to check time and cost
    python scripts/build_graph.py --no-records     plain GraphRAG, without our record layer

What it does, in order:
  1. creates the graph through the GraphRAG service
  2. works out the domain schema from the structured blocks (our record layer)
  3. lets GraphRAG set the graph up with that schema and install its retrievers
  4. loads the documents
  5. loads the typed records and connects each one to its document
  6. starts GraphRAG's build service, which chunks and embeds the documents

Every step can be run again safely, so if something stops halfway you just start the script again.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agr2 import config, tg  # noqa: E402
from agr2.records.blocks import find_blocks, vertex_id  # noqa: E402
from agr2.records.schema import build_records, infer_schema, iter_docs, query_guidance  # noqa: E402

OUT = config.RESULTS_DIR / "graph"


def stratified_sample(docs: list[dict], n: int, seed: int = 7) -> list[dict]:
    """Pick n documents spread evenly over the kinds of blocks (and documents without any), so a
    small test still touches every record type."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        b = find_blocks(d["text"])
        groups[b[0].kind if b else "<none>"].append(d)
    rnd = random.Random(seed)
    for g in groups.values():
        rnd.shuffle(g)
    out: list[dict] = []
    while len(out) < n and any(groups.values()):
        for g in sorted(groups, key=lambda k: -len(groups[k])):
            if groups[g] and len(out) < n:
                out.append(groups[g].pop())
    return out


def batched_upsert_vertices(vt: str, rows: dict[str, dict], batch: int = 1000):
    items = list(rows.items())
    for i in range(0, len(items), batch):
        tg.conn().upsertVertices(vt, items[i:i + batch])


def batched_upsert_edges(src: str, edge: str, tgt: str, pairs, batch: int = 2000):
    pairs = [(s, t, {}) for s, t in pairs]
    for i in range(0, len(pairs), batch):
        tg.conn().upsertEdges(src, edge, tgt, pairs[i:i + batch])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="only load a stratified sample of N documents")
    ap.add_argument("--with-gold", default="", help="also load the gold documents of these questions: "
                                                    "public:5 (first 5) or public:pub-001+pub-002")
    ap.add_argument("--no-records", action="store_true", help="skip the structured-record layer")
    ap.add_argument("--no-extract", action="store_true", help="load everything but do not trigger GraphRAG ECC")
    a = ap.parse_args()
    use_records = config.RECORDS_ENABLED and not a.no_records
    OUT.mkdir(parents=True, exist_ok=True)

    all_docs = list(iter_docs(config.CORPUS_PATH, config.CORPUS_FIELDS))
    docs = stratified_sample(all_docs, a.sample) if a.sample else all_docs
    if a.with_gold:
        want = set()
        for part in a.with_gold.split(","):
            split, sel = part.split(":")
            qs = [json.loads(l) for l in open(config.QUESTION_SPLITS[split], encoding="utf-8") if l.strip()]
            qs = qs[:int(sel)] if sel.isdigit() else [q for q in qs if q["qid"] in sel.split("+")]
            for q in qs:
                want.update(q.get("gold_doc_ids") or [])
        have = {d["id"] for d in docs}
        docs += [d for d in all_docs if d["id"] in want and d["id"] not in have]
    print(f"corpus: {len(all_docs)} documents; loading {len(docs)}", flush=True)

    tg.wait_healthy()
    # 1. graph
    if not tg.graph_exists():
        print("1. create graph", tg.create_graph().get("message"), flush=True)
    else:
        print(f"1. graph {config.TG_GRAPH} exists", flush=True)

    # 2. the schema is always worked out from the whole corpus, so a sample build gets the same schema
    schema_gsql = None
    schema = None
    if use_records:
        schema = infer_schema(all_docs, min_docs=config.RECORDS_MIN_DOCS,
                              min_coverage=config.RECORDS_MIN_FIELD_COVERAGE, reserved=tg.reserved_names())
        schema_gsql = schema.gsql()
        (OUT / "domain_schema.gsql").write_text(schema_gsql)
        print("2. inferred the domain schema, saved to", OUT / "domain_schema.gsql", flush=True)
        print("   " + schema.describe().replace("\n", "\n   "), flush=True)
    else:
        print("2. no domain schema (pure GraphRAG)", flush=True)

    # 3. let GraphRAG set the graph up, unless that has already happened
    vtypes = set(tg.conn().getVertexTypes())
    if "DocumentChunk" not in vtypes:
        st = tg.initialize_graph(schema_gsql)
        (OUT / "initialize_result.json").write_text(json.dumps(st, indent=1, default=str))
        print("3. initialised GraphRAG", flush=True)
    else:
        print("3. GraphRAG already initialised", flush=True)

    # 4. documents
    payload = [{"vid": vertex_id(d["id"]), "content": f"# {d['title']}\n\n{d['text']}"} for d in docs]
    t0 = time.time()
    print(f"4. loaded {tg.upsert_documents(payload)} documents in {time.time() - t0:.0f}s", flush=True)

    # 5. the typed records, their relations, and the link from each document to its records
    if schema is not None:
        rec = build_records(docs, schema)
        t0 = time.time()
        for vt, rows in rec.vertices.items():
            batched_upsert_vertices(vt, rows)
        for (src, edge, tgt), pairs in rec.edges.items():
            batched_upsert_edges(src, edge, tgt, pairs)
        for vt, pairs in rec.doc_links.items():
            batched_upsert_edges("Document", "CONTAINS_ENTITY", vt, pairs)
        (OUT / "records_counts.json").write_text(json.dumps(rec.counts(), indent=1))
        print(f"5. loaded records in {time.time() - t0:.0f}s: {json.dumps(rec.counts()['vertices'])}", flush=True)
        # hints for GraphRAG's query generators, with real example values of every field, built from all
        # documents so a sample build gets the same hints. GraphRAG reads them from its per-graph prompt folder.
        hints = query_guidance(schema, build_records(all_docs, schema) if len(docs) < len(all_docs) else rec)
        (OUT / "query_guidance.txt").write_text(hints)
        prompts = config.ROOT / "infra" / "configs" / "graph_configs" / config.TG_GRAPH / "prompts"
        prompts.mkdir(parents=True, exist_ok=True)
        (prompts / "query_guidance.txt").write_text(hints)
        print(f"   wrote query hints for GraphRAG ({len(hints) // 4} tokens) to {prompts}", flush=True)

    # 6. GraphRAG's build service chunks and embeds the new documents
    if not a.no_extract:
        print("6. GraphRAG ECC", tg.force_update(), flush=True)
        print("   follow progress: python scripts/graph_status.py --watch", flush=True)


if __name__ == "__main__":
    main()
