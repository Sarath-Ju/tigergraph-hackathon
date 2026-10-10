"""Ask one question and print the answer with its trace.

    python -m agr.ask "Who won the gold medal in the event held at Richmond Olympic Oval on 14 February 2010?"
    python -m agr.ask --pipeline all "..."
"""
from __future__ import annotations

import argparse
import json

from .pipelines import PIPELINES


def show(res, verbose: bool):
    t = res.trace["totals"]
    print(f"\n=== {res.pipeline.upper()} ===")
    print(f"answer    : {res.answer}")
    print(f"citations : {', '.join(res.citations[:8])}{' ...' if len(res.citations) > 8 else ''}")
    print(f"tokens    : {t['total_tokens']} total (in {t['llm_input_tokens']}, out {t['llm_output_tokens']}, "
          f"embed {t['embed_tokens']}, context {t['context_tokens']}) | llm calls {t['llm_calls']} | {t['wall_ms']:.0f} ms")
    if res.error:
        print(f"error     : {res.error}")
    for ev in res.trace["events"]:
        kind = ev["kind"]
        if kind == "action":
            print(f"  [{ev['t_ms']:>7.0f}ms] {ev['agent']:<18} {ev['tool']}({json.dumps(ev['args'], ensure_ascii=False)})"
                  f"  -- {ev['reason']}")
        elif kind in ("strategy_change", "stop", "evaluation", "budget", "loop_guard"):
            rest = {k: v for k, v in ev.items() if k not in ("kind", "t_ms")}
            print(f"  [{ev['t_ms']:>7.0f}ms] >> {kind}: {json.dumps(rest, ensure_ascii=False)[:220]}")
    if verbose:
        for o in res.trace["ops"]:
            print(f"    op {o['step']:>2} {o['kind']:<6} {o['name']:<20} {o['agent']:<18} {o['duration_ms']:>7.0f}ms "
                  f"in={o['input_tokens']} out={o['output_tokens']} emb={o['embed_tokens']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--pipeline", default="agentic", choices=[*PIPELINES, "all"])
    ap.add_argument("-v", "--verbose", action="store_true", help="also print every low-level op")
    a = ap.parse_args()
    for name in (PIPELINES if a.pipeline == "all" else [a.pipeline]):
        show(PIPELINES[name]({"qid": "adhoc", "question": a.question}), a.verbose)


if __name__ == "__main__":
    main()
