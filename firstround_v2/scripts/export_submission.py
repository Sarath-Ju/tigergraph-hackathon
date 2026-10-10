"""Write the answers to the hidden questions in submission form.

Each line holds the answer, the cited documents, token usage, time taken and the agent's steps.
The files go to results/submission/hidden_<pipeline>.jsonl.

    python scripts/export_submission.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agr2 import config  # noqa: E402

out_dir = config.RESULTS_DIR / "submission"
out_dir.mkdir(parents=True, exist_ok=True)
for src in sorted((config.RESULTS_DIR / "hidden").glob("*.jsonl")):
    if src.stem == "judged":
        continue
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    with open(out_dir / f"hidden_{src.stem}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            t = r["trace"]
            f.write(json.dumps({
                "qid": r["qid"], "question": r["question"], "pipeline": r["pipeline"], "answer": r["answer"],
                "raw_answer": r["raw_answer"], "citations": r["citations"], "retrieved_doc_ids": r["retrieved_doc_ids"],
                "tokens": {k: t.get("totals", {}).get(k, 0) for k in ("llm_input_tokens", "llm_output_tokens",
                                                                       "total_tokens")},
                "latency_ms": t.get("totals", {}).get("wall_ms", 0), "engine": t.get("engine"),
                "steps": t.get("steps", []), "error": r["error"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {out_dir / f'hidden_{src.stem}.jsonl'} ({len(rows)} questions)")
