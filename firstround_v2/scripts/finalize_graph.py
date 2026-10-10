"""Run this once GraphRAG's build service has finished.

It does two things.

First, it connects every record to the chunk it came from. The structured block sits at the top of
its document, so that is always the document's first chunk. GraphRAG's hybrid search starts from
chunks that match the question and follows these connections, so a match on a chunk can now lead
to the typed record and its relations, such as DIRECTOR, GOLD or VENUE. Chunks only exist after the
build service has run, which is why this is a separate step.

Second, it keeps the record types clean. When LLM extraction is switched on, GraphRAG also writes
the entities it extracts into matching types. An actor's biography that mentions ten films would
add ten empty Film vertices, and counting questions would come out wrong. GraphRAG keeps its own
copy of every extracted entity anyway, so we remove the empty copies from our types. Our own
records always have a doc_id, and extracted copies never do.

    python scripts/finalize_graph.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agr2 import config, tg  # noqa: E402
from agr2.records.blocks import vertex_id  # noqa: E402
from agr2.records.schema import NAME_VT  # noqa: E402


def record_types() -> list[str]:
    p = config.RESULTS_DIR / "graph" / "records_counts.json"
    return [vt for vt in json.loads(p.read_text())["vertices"] if vt != NAME_VT] if p.exists() else []


def link_chunks(c) -> dict:
    existing = {v["v_id"] for v in c.getVertices("DocumentChunk", select="epoch_added")}
    linked = {}
    for vt in record_types():
        pairs = []
        for v in c.getVertices(vt, select="doc_id"):
            doc = vertex_id(v["attributes"]["doc_id"]) if v["attributes"].get("doc_id") else ""
            chunk = vertex_id(f"{doc}_chunk_0")
            if doc and chunk in existing:
                pairs.append((chunk, v["v_id"], {}))
        for i in range(0, len(pairs), 2000):
            c.upsertEdges("DocumentChunk", "CONTAINS_ENTITY", vt, pairs[i:i + 2000])
        linked[vt] = len(pairs)
    return linked


def main():
    c = tg.conn()
    print("linked records to their chunk:", json.dumps(link_chunks(c)))
    removed = {}
    for vt in record_types():
        removed[vt] = c.delVertices(vt, where='doc_id=""')
    print("removed extracted copies without doc_id:", json.dumps(removed))
    print(json.dumps(tg.vertex_counts(["Document", "DocumentChunk", "Entity", "Community"] + record_types())))


if __name__ == "__main__":
    main()
