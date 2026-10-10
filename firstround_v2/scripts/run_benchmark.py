"""Run the questions of one set through the chosen approaches and save every answer.

    python scripts/run_benchmark.py --split public --pipelines rag,graphrag,agentic
    python scripts/run_benchmark.py --split general --pipelines agentic --limit 5
    python scripts/run_benchmark.py --split hidden --pipelines agentic

Answers go to results/<split>/<pipeline>.jsonl, one line per question. Questions that already have an
answer without an error are skipped, so if a run is interrupted you can simply start it again.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agr2 import config  # noqa: E402
from agr2.pipelines import PIPELINES, DocIds, run  # noqa: E402
from agr2.records.schema import iter_docs  # noqa: E402


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(config.QUESTION_SPLITS), default="public")
    ap.add_argument("--pipelines", default="rag,graphrag,agentic")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qids", default="", help="comma-separated qids to (re)run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    DocIds.load([d["id"] for d in iter_docs(config.CORPUS_PATH, config.CORPUS_FIELDS)])
    all_qs = load(config.QUESTION_SPLITS[a.split])
    qs = [q for q in all_qs if not a.qids or q["qid"] in set(a.qids.split(","))]
    qs = qs[:a.limit] if a.limit else qs
    order = {q["qid"]: i for i, q in enumerate(all_qs)}
    out_dir = config.RESULTS_DIR / a.split
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in a.pipelines.split(","):
        assert name in PIPELINES, f"unknown pipeline {name}; choose from {list(PIPELINES)}"
        path = out_dir / f"{name}.jsonl"
        done = {r["qid"]: r for r in load(path)} if path.exists() else {}
        todo = [q for q in qs if a.force or q["qid"] not in done or done[q["qid"]].get("error")]
        print(f"[{name}] {len(todo)} to run ({len(done)} already in {path.name})", flush=True)
        t0 = time.time()

        def save():
            with open(path, "w", encoding="utf-8") as f:
                for r in sorted(done.values(), key=lambda r: order.get(r["qid"], 1e9)):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(run, name, q): q for q in todo}
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result().to_dict()
                done[r["qid"]] = r
                tot = r["trace"].get("totals", {})
                flag = f"  ERROR {r['error'][:120]}" if r["error"] else ""
                print(f"  {i:>3}/{len(todo)} {r['qid']:<9} {tot.get('total_tokens', 0):>6} tok "
                      f"{tot.get('wall_ms', 0) / 1000:>5.1f}s  {r['answer'][:70]!r}{flag}", flush=True)
                if i % 10 == 0:
                    save()
        save()
        print(f"[{name}] wrote {path} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
