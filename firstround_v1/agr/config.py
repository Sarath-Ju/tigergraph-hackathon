"""Central configuration, read from environment variables or a local .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

TG_HOST = os.getenv("TG_HOST", "http://localhost")
TG_USERNAME = os.getenv("TG_USERNAME", "tigergraph")
TG_PASSWORD = os.getenv("TG_PASSWORD", "tigergraph")
TG_PORT = os.getenv("TG_PORT", "14240")
TG_GRAPH = os.getenv("TG_GRAPH", "OlympicsKG")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

CORPUS_PATH = Path(os.getenv("CORPUS_PATH", ROOT / "data" / "corpus" / "corpus.jsonl"))
PUBLIC_Q = Path(os.getenv("PUBLIC_Q", ROOT / "data" / "question" / "eval_public.jsonl"))
HIDDEN_Q = Path(os.getenv("HIDDEN_Q", ROOT / "data" / "question" / "eval_hidden.jsonl"))
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"

# Chunking for the RAG index
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Domain pack: which parser / schema / specialised tools to use (see agr/domains/).
DOMAIN = os.getenv("DOMAIN", "olympics")

# Corpus JSONL field names, so a differently-shaped dataset needs no code change.
CORPUS_FIELDS = {
    "id": os.getenv("CORPUS_ID_FIELD", "doc_id"),
    "title": os.getenv("CORPUS_TITLE_FIELD", "title"),
    "url": os.getenv("CORPUS_URL_FIELD", "url"),
    "text": os.getenv("CORPUS_TEXT_FIELD", "text"),
}

# Optional agent tool sets on top of a domain pack's own tools.
#   AGENT_GENERIC_TOOLS  schema-driven find / aggregate / distinct-values / neighbours tools
#   TG_GRAPHRAG_TOOLS    TigerGraph GraphRAG service retrievers (hybrid / community / ...) as a tool
_flag = lambda name, default: os.getenv(name, default).strip().lower() in ("1", "true", "yes")
AGENT_GENERIC_TOOLS = _flag("AGENT_GENERIC_TOOLS", "0")
TG_GRAPHRAG_TOOLS = _flag("TG_GRAPHRAG_TOOLS", "0")
TG_GRAPHRAG_HOST = os.getenv("TG_GRAPHRAG_HOST", "http://localhost:8000")
TG_GRAPHRAG_GRAPH = os.getenv("TG_GRAPHRAG_GRAPH", "corpus5")
# How corpus doc ids look inside free text (the GraphRAG service lowercases them); used to map citations back.
DOC_ID_PATTERN = os.getenv("DOC_ID_PATTERN", r"q\d+")
