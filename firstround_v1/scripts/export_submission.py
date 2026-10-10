"""Export the hidden-split outputs in submission form: one record per question and pipeline with the
answer, citations, token usage and the full agentic trace.

    python scripts/export_submission.py   ->  results/submission/hidden_<pipeline>.jsonl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agr import config  # noqa: E402

out_dir = config.RESULTS_DIR / "submission"
out_dir.mkdir(exist_ok=True)
for src in sorted((config.RESULTS_DIR / "hidden").glob("*.jsonl")):
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    with open(out_dir / f"hidden_{src.stem}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            t = r["trace"]
            f.write(json.dumps({
                "qid": r["qid"], "question": r["question"], "pipeline": r["pipeline"], "answer": r["answer"],
                "citations": r["citations"], "retrieved_doc_ids": r["retrieved_doc_ids"],
                "tokens": {k: t["totals"][k] for k in ("llm_input_tokens", "llm_output_tokens", "embed_tokens",
                                                       "context_tokens", "total_tokens")},
                "llm_calls": t["totals"]["llm_calls"], "latency_ms": t["totals"]["wall_ms"],
                "trace": {"events": t["events"], "ops": t["ops"]}, "error": r["error"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {out_dir / f'hidden_{src.stem}.jsonl'} ({len(rows)} questions)")
