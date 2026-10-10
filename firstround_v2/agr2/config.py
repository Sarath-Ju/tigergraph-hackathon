"""All the settings, read from the environment or from a local .env file.

Nothing here is tied to a topic. Pointing the project at a different corpus only means changing the
paths and, if needed, the field names below."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

_flag = lambda name, default: os.getenv(name, default).strip().lower() in ("1", "true", "yes")

# TigerGraph database and the TigerGraph GraphRAG service
TG_HOST = os.getenv("TG_HOST", "http://localhost")
TG_PORT = os.getenv("TG_PORT", "14240")
TG_USERNAME = os.getenv("TG_USERNAME", "tigergraph")
TG_PASSWORD = os.getenv("TG_PASSWORD", "tigergraph")
TG_GRAPH = os.getenv("TG_GRAPH", "CorpusKG")
GRAPHRAG_HOST = os.getenv("GRAPHRAG_HOST", "http://localhost:8000")
ECC_HOST = os.getenv("ECC_HOST", "http://localhost:8001")  # GraphRAG's build service

# Models (used by the service and by our evaluation judge)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
# retrieval settings, shared with the GraphRAG service config
TOP_K = int(os.getenv("TOP_K", "5"))
NUM_HOPS = int(os.getenv("NUM_HOPS", "2"))

# Dataset. Field names are configurable so a differently shaped JSONL corpus needs no code change.
CORPUS_PATH = Path(os.getenv("CORPUS_PATH", ROOT / "data" / "corpus" / "corpus.jsonl"))
QUESTION_SPLITS = {
    "public": Path(os.getenv("PUBLIC_Q", ROOT / "data" / "question" / "eval_public.jsonl")),
    "hidden": Path(os.getenv("HIDDEN_Q", ROOT / "data" / "question" / "eval_hidden.jsonl")),
    "general": Path(os.getenv("GENERAL_Q", ROOT / "data" / "question" / "eval_general.jsonl")),
}
CORPUS_FIELDS = {
    "id": os.getenv("CORPUS_ID_FIELD", "doc_id"),
    "title": os.getenv("CORPUS_TITLE_FIELD", "title"),
    "text": os.getenv("CORPUS_TEXT_FIELD", "text"),
    "url": os.getenv("CORPUS_URL_FIELD", "url"),
}

# The structured-record layer turns "key: value" blocks into typed records (see agr2/records/).
RECORDS_ENABLED = _flag("RECORDS_ENABLED", "1")
# a kind of block needs at least this many documents to get its own type
RECORDS_MIN_DOCS = int(os.getenv("RECORDS_MIN_DOCS", "5"))
# a field becomes an attribute when at least this share of the blocks of its kind have it
RECORDS_MIN_FIELD_COVERAGE = float(os.getenv("RECORDS_MIN_FIELD_COVERAGE", "0.3"))

RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"
