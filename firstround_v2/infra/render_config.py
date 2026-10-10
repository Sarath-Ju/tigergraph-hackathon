"""Write the GraphRAG service config from your .env settings.

The config needs your OpenAI key, so it is generated here instead of being kept in git. The template
next to this file holds everything else.

    python infra/render_config.py
"""
import os
import string
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")

DEFAULTS = {
    "TG_USERNAME": "tigergraph", "TG_PASSWORD": "tigergraph",
    "LLM_MODEL": "gpt-4.1-mini", "EMBED_MODEL": "text-embedding-3-small",
    "CHUNKER": "recursive", "CHUNK_SIZE": "2048", "CHUNK_OVERLAP": "256",
    "TOP_K": "5", "NUM_HOPS": "2", "INGEST_CONCURRENCY": "10",
    # LLM entity extraction and communities. Off by default, because chunks, embeddings and our record
    # layer cover this corpus for about $0.15, while extraction would add about $40 to $60.
    "EXTRACT_ENTITIES": "false",
}
values = {k: os.getenv(k, v) for k, v in DEFAULTS.items()}
values["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
values["EXTRACT_ENTITIES"] = "true" if values["EXTRACT_ENTITIES"].lower() in ("1", "true", "yes") else "false"

out = HERE / "configs" / "server_config.json"
out.write_text(string.Template((HERE / "server_config.template.json").read_text()).substitute(values))
out.chmod(0o600)
print(f"wrote {out}")

# Our planner instructions for TigerGraph's agent. The service reads per-graph prompt overrides from
# configs/graph_configs/<graph>/prompts, so we copy ours there. Without it the planner's structured
# output fails with OpenAI and the agent falls back to a single hybrid search on every question.
graph = os.getenv("TG_GRAPH", "CorpusKG")
prompts_out = HERE / "configs" / "graph_configs" / graph / "prompts"
prompts_out.mkdir(parents=True, exist_ok=True)
for src in (HERE / "prompts").glob("*.txt"):
    (prompts_out / src.name).write_text(src.read_text())
    print(f"wrote {prompts_out / src.name}")
