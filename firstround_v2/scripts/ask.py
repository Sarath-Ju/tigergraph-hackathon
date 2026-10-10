"""Ask one question and see how each approach answers it.

For every pipeline you choose, it prints the answer, the cited documents, the tokens used, the time
taken and the steps the agent took.

    python scripts/ask.py "Who directed the film The English Patient?"
    python scripts/ask.py --pipelines rag,graphrag,agentic "How many films starring Johnny Depp were released after 1994?"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agr2 import config  # noqa: E402
from agr2.pipelines import DocIds, run  # noqa: E402
from agr2.records.schema import iter_docs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--pipelines", default="agentic")
    ap.add_argument("--raw", action="store_true", help="also print the service's full query_sources")
    a = ap.parse_args()
    DocIds.load([d["id"] for d in iter_docs(config.CORPUS_PATH, config.CORPUS_FIELDS)])
    for p in a.pipelines.split(","):
        r = run(p, {"qid": "ask", "question": a.question})
        t = r.trace.get("totals", {})
        print(f"\n=== {p} ===  {t.get('total_tokens', 0)} tokens  {t.get('wall_ms', 0) / 1000:.1f}s"
              + (f"  ERROR {r.error}" if r.error else ""))
        print("answer:   ", r.answer)
        print("citations:", ", ".join(r.citations) or "-")
        for i, s in enumerate(r.trace.get("steps", []), 1):
            print(f"  step {i}: {s['tool']}  {json.dumps(s['args'], ensure_ascii=False)[:150]}")
        if a.raw:
            print(json.dumps(r.trace.get("service", {}), ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
