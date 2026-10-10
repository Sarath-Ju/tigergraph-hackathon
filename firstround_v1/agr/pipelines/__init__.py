"""Pipeline registry: name -> run(question_record) -> Result.

rag / graphrag / agentic are this project's pipelines; tg_classic / tg_agentic call
TigerGraph's own GraphRAG service (see tg_service.py) for a reference comparison.
"""
from . import agentic, graphrag, rag, tg_service

PIPELINES = {"rag": rag.run, "graphrag": graphrag.run, "agentic": agentic.run,
             "tg_classic": tg_service.run_classic, "tg_agentic": tg_service.run_agentic}
