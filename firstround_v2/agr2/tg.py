"""A small client for TigerGraph and the TigerGraph GraphRAG service.

Building the knowledge graph and answering questions both go through the GraphRAG service's own
endpoints, the same ones its UI uses. We only talk to the database directly to load data in bulk
and to look at what is in the graph.
"""
from __future__ import annotations

import time
from functools import lru_cache

import requests
from pyTigerGraph import TigerGraphConnection

from . import config

AUTH = (config.TG_USERNAME, config.TG_PASSWORD)
G = config.TG_GRAPH


@lru_cache(maxsize=1)
def conn() -> TigerGraphConnection:
    c = TigerGraphConnection(host=config.TG_HOST, restppPort=config.TG_PORT, gsPort=config.TG_PORT,
                             username=config.TG_USERNAME, password=config.TG_PASSWORD, graphname=G)
    return c


def _svc(method: str, path: str, *, timeout: int = 600, retries: int = 3, **kw) -> dict:
    url = f"{config.GRAPHRAG_HOST}{path}"
    for attempt in range(retries + 1):
        try:
            r = requests.request(method, url, auth=AUTH, timeout=timeout, **kw)
            if r.status_code >= 500 and attempt < retries:
                raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}")
            if r.status_code >= 400:
                raise RuntimeError(f"{method} {path} failed with {r.status_code}: {r.text[:1000]}")
            return r.json() if r.content else {}
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == retries:
                raise
            time.sleep(5 * (attempt + 1))
    return {}


# ------------------------------------------------------------------------------------------- service health

def wait_healthy(timeout_s: int = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if requests.get(f"{config.GRAPHRAG_HOST}/health", timeout=10).json().get("status") == "healthy":
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError("GraphRAG service not healthy; check `docker compose -f infra/docker-compose.yml logs graphrag`")


def reserved_names() -> set[str]:
    out = _svc("GET", "/ui/schema_reserved_names")
    names: set[str] = set()
    for v in out.values():
        if isinstance(v, list):
            names.update(str(x) for x in v)
    return names


# ------------------------------------------------------------------------------------------- graph lifecycle

def graph_exists() -> bool:
    return G in conn().gsql("SHOW GRAPH *")


def create_graph() -> dict:
    return _svc("POST", f"/ui/{G}/create_graph")


def initialize_graph(schema_gsql: str | None, poll_s: int = 10, timeout_s: int = 3600) -> dict:
    """Ask GraphRAG to set the graph up: its own types for documents and chunks, our domain schema
    if we have one, and its retriever queries. The service does this in the background, so we wait
    and report its progress."""
    body = {"schema_gsql": schema_gsql} if schema_gsql else {}
    _svc("POST", f"/ui/{G}/initialize_graph", json=body)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = _svc("GET", f"/ui/{G}/initialize_status")
        state = st.get("state") or st.get("status")
        if state in ("completed", "error"):
            if state == "error":
                raise RuntimeError(f"initialize_graph failed: {st.get('error') or st}")
            return st
        print(f"  initialize_graph: {state} {st.get('message', '')}", flush=True)
        time.sleep(poll_s)
    raise TimeoutError("initialize_graph did not finish")


def upsert_documents(docs: list[dict], batch: int = 200) -> int:
    """Load documents in the same shape GraphRAG's own loading job uses: a Document, its Content,
    and the edge between them. Documents already in the graph are left alone, so running this
    twice never makes GraphRAG process a document again."""
    c, n, now = conn(), 0, int(time.time())
    existing = {v["v_id"] for v in c.getVertices("Document", select="epoch_added")}
    docs = [d for d in docs if d["vid"] not in existing]  # never reset a document GraphRAG already processed
    for i in range(0, len(docs), batch):
        part = docs[i:i + batch]
        # zero timestamps tell GraphRAG's build service that these documents are new
        c.upsertVertices("Document", [(d["vid"], {"epoch_added": now, "epoch_processing": 0, "epoch_processed": 0})
                                      for d in part])
        # an empty content type means "use the chunker from the config"
        c.upsertVertices("Content", [(d["vid"], {"ctype": d.get("ctype", ""), "text": d["content"],
                                                 "epoch_added": now}) for d in part])
        c.upsertEdges("Document", "HAS_CONTENT", "Content", [(d["vid"], d["vid"], {}) for d in part])
        n += len(part)
    return n


def force_update() -> dict:
    """Start GraphRAG's build service (ECC) on the new documents. It chunks and embeds them, and also
    extracts entities and communities when that is switched on in the config."""
    return _svc("GET", f"/{G}/graphrag/forceupdate")


def build_status() -> dict:
    """How far the build service has got. We use rebuild_status on purpose: ECC's consistency_status
    address looks like a status check but actually starts a new build."""
    r = requests.get(f"{config.ECC_HOST}/{G}/graphrag/rebuild_status", auth=AUTH, timeout=30)
    r.raise_for_status()
    return r.json()


def vertex_counts(types: list[str] | None = None) -> dict[str, int]:
    c = conn()
    types = types or c.getVertexTypes()
    return {t: c.getVertexCount(t) for t in types}


# ------------------------------------------------------------------------------------------- answering

def search(question: str, method: str, params: dict, timeout: int = 300) -> dict:
    """Run one of the service's retrievers and return what it found, without writing an answer.

    We call the retrievers directly because the chat engine's classic mode first asks an LLM router
    whether to use the schema or the vector store. With a typed domain schema it almost always picks
    the schema, so the similarity and hybrid retrievers would never run.
    """
    out = _svc("POST", f"/{G}/graphrag/search", json={"question": question, "method": method,
                                                       "method_params": params}, timeout=timeout)
    return out[0] if isinstance(out, list) and out else (out or {})


def query(question: str, mode: str, rag_method: str, timeout: int = 900) -> dict:
    """Send a question to the service's chat engines.

    With mode "classic", rag_method picks the retriever: similarity, contextual, hybrid, community
    or auto. With mode "agentic", it picks the agent style: planned, reactive or auto.
    """
    body = {"query": question, "mode": mode, "rag_method": rag_method, "include_fields": ["all"]}
    return _svc("POST", f"/{G}/query", json=body, timeout=timeout)
