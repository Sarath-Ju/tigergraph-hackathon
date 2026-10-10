"""Run pipelines over a question split and write one JSONL record per (question, pipeline).

    python scripts/run_benchmark.py --split public --pipelines rag,graphrag,agentic
    python scripts/run_benchmark.py --split hidden --pipelines agentic
    python scripts/run_benchmark.py --split public --pipelines agentic --limit 10 --force

Output: results/<split>/<pipeline>.jsonl  (answer, citations, retrieved docs, full trace with tokens and timings).
Already-answered questions are skipped unless --force, so an interrupted run can simply be restarted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agr import config  # noqa: E402
from agr.graph_tools import EntityIndex  # noqa: E402
from agr.pipelines import PIPELINES  # noqa: E402


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["public", "hidden"], default="public")
    ap.add_argument("--pipelines", default="rag,graphrag,agentic")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qids", default="", help="comma-separated qids to (re)run")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    qs = load(config.PUBLIC_Q if a.split == "public" else config.HIDDEN_Q)
    if a.qids:
        wanted = set(a.qids.split(","))
        qs = [q for q in qs if q["qid"] in wanted]
    if a.limit:
        qs = qs[:a.limit]
    out_dir = config.RESULTS_DIR / a.split
    out_dir.mkdir(parents=True, exist_ok=True)
    EntityIndex.get()  # warm the entity catalogue once, before threads start

    for name in a.pipelines.split(","):
        path = out_dir / f"{name}.jsonl"
        done = {r["qid"]: r for r in load(path)} if path.exists() else {}
        todo = [q for q in qs if a.force or q["qid"] not in done or done[q["qid"]].get("error")]
        print(f"[{name}] {len(todo)} to run ({len(done)} already in {path.name})", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(PIPELINES[name], q): q for q in todo}
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result().to_dict()
                done[r["qid"]] = r
                flag = f" ERROR {r['error']}" if r["error"] else ""
                print(f"  {i:>3}/{len(todo)} {r['qid']:<9} {r['trace']['totals']['total_tokens']:>6} tok  "
                      f"{r['answer'][:60]!r}{flag}", flush=True)
        order = {q["qid"]: i for i, q in enumerate(load(config.PUBLIC_Q if a.split == "public" else config.HIDDEN_Q))}
        with open(path, "w", encoding="utf-8") as f:
            for r in sorted(done.values(), key=lambda r: order.get(r["qid"], 1e9)):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{name}] wrote {path} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
