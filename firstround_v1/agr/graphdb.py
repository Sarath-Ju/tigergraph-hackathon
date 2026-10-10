"""Domain-independent TigerGraph access shared by every pipeline and domain pack.

Two layers:
  * query helpers  - run an installed GSQL query with tracing; unpack printed vertex sets.
  * schema-driven  - introspect the live schema and filter / aggregate / traverse ANY vertex
                     type through TigerGraph's built-in REST endpoints, so the agent can answer
                     counting and ranking questions on a graph no one wrote queries for.
Nothing here knows about Olympics; domain packs add their own specialised queries on top.
"""
from __future__ import annotations

import json
import re
import threading
from collections import Counter

from . import config, llm, tracing
from .text import norm
from .tg import get_conn, run_query


# --------------------------------------------------------------------------- query helpers
def gsql(name: str, params: dict | None = None) -> list[dict]:
    with tracing.op("graph", name, params={k: v for k, v in (params or {}).items() if k != "qv"}) as o:
        res = run_query(name, params)
        o.detail["rows"] = sum(len(v) if isinstance(v, list) else 1 for r in res for v in r.values())
    return res


def vset(rows: list[dict], key: str) -> list[dict]:
    for r in rows:
        if key in r:
            # GSQL prints projected attributes as "<alias>.<attr>"; drop the alias
            return [{"id": v["v_id"], **{k.split(".", 1)[-1]: x for k, x in v["attributes"].items()}}
                    for v in r[key]]
    return []


def val(rows: list[dict], key: str):
    for r in rows:
        if key in r:
            return r[key]
    return None


def search_chunks(query: str, k: int = 6) -> list[dict]:
    rows = gsql("vs_chunks", {"qv": llm.embed_one(query), "k": k})
    dist = val(rows, "dist") or {}
    out = [{"chunk_id": c["id"], "doc_id": c["doc_id"], "doc_title": c["doc_title"], "text": c["text"],
            "distance": round(float(dist.get(c["id"], 0)), 4)} for c in vset(rows, "chunks")]
    return sorted(out, key=lambda c: c["distance"])


def document(doc_id: str, max_chars: int = 4000) -> dict:
    with tracing.op("graph", "get_document", doc_id=doc_id):
        v = get_conn().getVerticesById("Document", doc_id)
    if not v:
        return {"doc_id": doc_id, "error": "not found"}
    a = v[0]["attributes"]
    return {"doc_id": doc_id, "title": a["title"], "url": a["url"], "text": a["text"][:max_chars],
            "truncated": len(a["text"]) > max_chars}


# --------------------------------------------------------------------------- schema introspection
_HIDDEN_ATTRS = {"text", "emb", "embedding"}     # large or vector attributes are never shown to the LLM
_schema_lock = threading.Lock()
_schema_cache: dict | None = None


def schema() -> dict:
    """Vertex types (attributes, counts) and edge types of the live graph, cached per process."""
    global _schema_cache
    with _schema_lock:
        if _schema_cache is None:
            conn = get_conn()
            vts = {}
            for vt in conn.getVertexTypes():
                d = conn.getVertexType(vt)
                vts[vt] = {"id": d["PrimaryId"]["AttributeName"],
                           "attrs": {a["AttributeName"]: a["AttributeType"]["Name"] for a in d["Attributes"]
                                     if a["AttributeName"] not in _HIDDEN_ATTRS},
                           "count": conn.getVertexCount(vt)}
            ets, reverses = {}, set()
            for et in conn.getEdgeTypes():
                if et in reverses:          # skip the auto-generated reverse of an edge already listed
                    continue
                d = conn.getEdgeType(et)
                rev = (d.get("Config") or {}).get("REVERSE_EDGE")
                if rev:
                    reverses.add(rev)
                ets[et] = {"from": d.get("FromVertexTypeName"), "to": d.get("ToVertexTypeName"),
                           "attrs": [a["AttributeName"] for a in d.get("Attributes", [])], "reverse": rev}
            _schema_cache = {"vertices": vts, "edges": ets}
        return _schema_cache


def schema_text() -> str:
    """Compact schema description for an LLM prompt."""
    s = schema()
    lines = [f"  {vt}({', '.join(f'{a}:{t}' for a, t in v['attrs'].items())})  [{v['count']} vertices]"
             for vt, v in s["vertices"].items()]
    lines += [f"  {v['from']} -{et}{'(' + ','.join(v['attrs']) + ')' if v['attrs'] else ''}-> {v['to']}"
              for et, v in s["edges"].items()]
    return "\n".join(lines)


class SchemaError(ValueError):
    pass


def _check(vtype: str, attrs: list[str]):
    s = schema()["vertices"]
    if vtype not in s:
        raise SchemaError(f"Unknown vertex type '{vtype}'. Known: {', '.join(s)}")
    bad = [a for a in attrs if a and a not in s[vtype]["attrs"] and a != s[vtype]["id"]]
    if bad:
        raise SchemaError(f"{vtype} has no attribute(s) {bad}. Attributes: {', '.join(s[vtype]['attrs'])}")


_OPS = {"=": "=", "==": "=", "!=": "!=", ">": ">", ">=": ">=", "<": "<", "<=": "<="}


def _where(vtype: str, filters: list[dict]) -> str:
    """Build a REST ``where`` clause; string values are quoted, numbers passed as-is."""
    types = schema()["vertices"][vtype]["attrs"]
    parts = []
    for f in filters or []:
        attr, op, value = f["attr"], _OPS.get(str(f.get("op", "=")).strip(), None), f["value"]
        if op is None:
            raise SchemaError(f"Unsupported operator {f.get('op')!r}; use one of {sorted(set(_OPS))}")
        if types.get(attr, "STRING") in ("STRING", "DATETIME"):
            value = json.dumps(str(value), ensure_ascii=False)
        parts.append(f"{attr}{op}{value}")
    return ",".join(parts)


def find_vertices(vtype: str, filters: list[dict] | None = None, attrs: list[str] | None = None,
                  limit: int = 5000) -> list[dict]:
    """Server-side filtered scan of a vertex type via TigerGraph's built-in REST endpoint."""
    _check(vtype, [f["attr"] for f in filters or []] + list(attrs or []))
    where = _where(vtype, filters or [])
    with tracing.op("graph", "rest_vertices", vtype=vtype, where=where) as o:
        rows = get_conn().getVertices(vtype, where=where, select=",".join(attrs or []), limit=limit)
        o.detail["rows"] = len(rows)
    return [{"id": r["v_id"], **r["attributes"]} for r in rows]


def aggregate(vtype: str, filters: list[dict] | None, operation: str, metric: str | None = None,
              threshold: float | None = None, group_by: str | None = None, label: str | None = None) -> dict:
    """Filter + aggregate over any vertex type.

    operation: count | count_greater_than | count_at_least | count_less_than | sum | avg | min | max
               | argmax | argmin | group_count
    Filtering happens in TigerGraph; the (small) aggregation is computed here and every matching
    vertex id is returned so the answer is fully traceable.
    """
    s = schema()["vertices"].get(vtype)
    if s is None:
        _check(vtype, [])
    label = label or next((a for a in ("title", "name") if a in s["attrs"]), s["id"])
    need = [a for a in dict.fromkeys([metric, group_by, label]) if a and a in s["attrs"]]
    rows = find_vertices(vtype, filters, need)
    out = {"vertex_type": vtype, "filters": filters or [], "operation": operation, "n_matched": len(rows)}
    if metric:
        out["metric"] = metric
        known = [r for r in rows if isinstance(r.get(metric), (int, float)) and r[metric] >= 0]
        out["n_missing_metric"] = len(rows) - len(known)
    else:
        known = rows
    cmp = {"count_greater_than": lambda v: v > threshold, "count_at_least": lambda v: v >= threshold,
           "count_less_than": lambda v: v < threshold}
    if operation == "count":
        out["result"] = len(rows)
    elif operation in cmp:
        if threshold is None or not metric:
            raise SchemaError(f"{operation} needs both metric and threshold")
        hit = [r for r in known if cmp[operation](r[metric])]
        out.update(threshold=threshold, result=len(hit), matching=[r.get(label, r["id"]) for r in hit])
    elif operation in ("sum", "avg", "min", "max"):
        vals = [r[metric] for r in known]
        fn = {"sum": sum, "min": min, "max": max, "avg": lambda v: round(sum(v) / len(v), 3)}[operation]
        out["result"] = fn(vals) if vals else None
    elif operation in ("argmax", "argmin"):
        if not known:
            out["result"] = []
        else:
            best = (max if operation == "argmax" else min)(r[metric] for r in known)
            out.update(value=best, result=[r.get(label, r["id"]) for r in known if r[metric] == best])
    elif operation == "group_count":
        if not group_by:
            raise SchemaError("group_count needs group_by")
        out["result"] = Counter(str(r.get(group_by)) for r in rows).most_common(25)
    else:
        raise SchemaError(f"Unknown operation {operation!r}")
    if metric and len(rows) <= 200:
        out["rows"] = sorted(([r.get(label, r["id"]), r.get(metric)] for r in rows),
                             key=lambda x: -(x[1] if isinstance(x[1], (int, float)) else -1))
    out["ids"] = [r["id"] for r in rows]
    return out


def distinct_values(vtype: str, attr: str, contains: str = "", limit: int = 40) -> dict:
    """Distinct values of an attribute (with counts), optionally fuzzy-filtered, to learn exact spellings."""
    _check(vtype, [attr])
    rows = find_vertices(vtype, None, [attr])
    cnt = Counter(str(r.get(attr)) for r in rows)
    items = cnt.most_common()
    if contains:
        q = norm(contains)
        items = [(v, n) for v, n in items if q in norm(v)] or \
                sorted(items, key=lambda it: -_overlap(q, norm(it[0])))[:limit]
    return {"vertex_type": vtype, "attr": attr, "n_distinct": len(cnt), "values": items[:limit]}


def _overlap(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / (len(ta) or 1)


def neighbors(vtype: str, vid: str, edge_type: str = "", limit: int = 50) -> dict:
    """One hop from a vertex over any edge type, with the neighbours' attributes."""
    _check(vtype, [])
    conn = get_conn()
    with tracing.op("graph", "rest_edges", vtype=vtype, vid=vid, edge_type=edge_type) as o:
        edges = conn.getEdges(vtype, vid, edge_type or None)
        o.detail["rows"] = len(edges)
    by_type: dict[str, list] = {}
    for e in edges[:limit]:
        by_type.setdefault(e["to_type"], []).append(e)
    out = []
    for tt, es in by_type.items():
        attrs = {v["v_id"]: v["attributes"] for v in conn.getVerticesById(tt, [e["to_id"] for e in es])}
        for e in es:
            a = {k: x for k, x in attrs.get(e["to_id"], {}).items() if k not in _HIDDEN_ATTRS}
            out.append({"edge": e["e_type"], "edge_attrs": e.get("attributes", {}), "to_type": tt,
                        "to_id": e["to_id"], **{k: a[k] for k in list(a)[:12]}})
    return {"from": {"type": vtype, "id": vid}, "n_edges": len(edges), "neighbors": out}


# --------------------------------------------------------------------------- TigerGraph GraphRAG retrievers
_DOC_ID = re.compile(rf"\b({config.DOC_ID_PATTERN})(?:_chunk_\d+)?\b", re.I)


def tg_graphrag_search(query: str, method: str = "hybrid", top_k: int = 5) -> dict:
    """Retrieval-only call to TigerGraph's GraphRAG service (no answer generation, no question expansion).

    Needs the service running and a graph built by its pipeline (config.TG_GRAPHRAG_GRAPH).
    """
    import requests
    params = {"hybrid": {"indices": ["DocumentChunk", "Entity"], "top_k": top_k, "num_hops": 2, "num_seen_min": 2},
              "similarity": {"index": "DocumentChunk", "top_k": top_k, "withHyDE": False},
              "contextual": {"index": "DocumentChunk", "top_k": top_k, "lookahead": 2, "lookback": 2, "withHyDE": False},
              "community": {"community_level": 2, "top_k": top_k}}[method]
    params.update(expand=False, verbose=False)
    with tracing.op("graph", f"tg_graphrag_{method}", graph=config.TG_GRAPHRAG_GRAPH) as o:
        r = requests.post(f"{config.TG_GRAPHRAG_HOST}/{config.TG_GRAPHRAG_GRAPH}/graphrag/search",
                          json={"question": query, "method": method, "method_params": params},
                          auth=(config.TG_USERNAME, config.TG_PASSWORD), timeout=300)
        r.raise_for_status()
        res = r.json()
        o.detail["bytes"] = len(r.content)
    doc_ids = list(dict.fromkeys(m.group(1).upper() for m in _DOC_ID.finditer(json.dumps(res))))
    return {"method": method, "doc_ids": doc_ids, "result": res}
