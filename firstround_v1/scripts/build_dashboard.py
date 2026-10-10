"""Build the metrics dashboard (single self-contained HTML) from results/summary.json + traces.

    python scripts/build_dashboard.py

Writes results/dashboard.html (open in any browser) and results/dashboard_fragment.html
(the same page without the document wrapper, for hosting as an artifact).
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agr import config  # noqa: E402

PIPES = ["tg_classic", "tg_agentic", "rag", "graphrag", "agentic"]  # display order = validated colour order
TEMPLATE = Path(__file__).resolve().parent.parent / "dashboard" / "template.html"


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def main():
    s = json.loads((config.RESULTS_DIR / "summary.json").read_text())
    qs = s["questions"]
    types = sorted({q["qtype"] for q in qs})

    # per-question rows, one entry per pipeline
    rows = defaultdict(dict)
    for q in qs:
        r = rows[q["qid"]]
        r.update(qid=q["qid"], qtype=q["qtype"], question=q["question"], gold=q["gold"])
        r[q["pipeline"]] = {"answer": q["answer"], "score": q["judge_score"], "tokens": q["total_tokens"],
                            "llm": q["llm_calls"], "tools": q["tools"], "ms": q["wall_ms"],
                            "cite_recall": round(q["cite_recall"], 2)}

    # agentic traces (decision events only; low-level ops summarised)
    traces = {}
    for r in load(config.RESULTS_DIR / "public" / "agentic.jsonl"):
        traces[r["qid"]] = {
            "events": [{k: v for k, v in e.items() if k != "tb"} for e in r["trace"]["events"]],
            "ops": [{"k": o["kind"], "n": o["name"], "a": o["agent"], "ms": o["duration_ms"],
                     "tok": o["input_tokens"] + o["output_tokens"] + o["embed_tokens"]} for o in r["trace"]["ops"]],
            "citations": r["citations"],
        }

    # agentic behaviour by question type
    ag = [q for q in qs if q["pipeline"] == "agentic"]
    behaviour = {}
    for t in types:
        items = [q for q in ag if q["qtype"] == t]
        behaviour[t] = {
            "tool_calls": round(st.mean(q["tool_calls"] for q in items), 2),
            "llm_calls": round(st.mean(q["llm_calls"] for q in items), 2),
            "paths": Counter(" → ".join(q["tools"]) for q in items).most_common(3),
            "strategy_changes": sum(q["strategy_changes"] > 0 for q in items),
            "n": len(items),
        }
    tool_use = Counter(t for q in ag for t in q["tools"])
    agent_tokens = Counter()
    agent_ms = Counter()
    agent_calls = Counter()
    for tr in traces.values():
        for o in tr["ops"]:
            if o["k"] in ("llm", "tool"):
                agent_calls[o["a"] or "harness"] += 1
            agent_tokens[o["a"] or "harness"] += o["tok"]
            agent_ms[o["a"] or "harness"] += o["ms"]
    stops = Counter(q["stop_reason"] for q in ag)

    # hidden split: no answers, so efficiency only
    hidden = {}
    for p in PIPES:
        rs = load(config.RESULTS_DIR / "hidden" / f"{p}.jsonl")
        if rs:
            hidden[p] = {"n": len(rs), "avg_tokens": round(st.mean(r["trace"]["totals"]["total_tokens"] for r in rs)),
                         "avg_llm": round(st.mean(r["trace"]["totals"]["llm_calls"] for r in rs), 2),
                         "avg_ms": round(st.mean(r["trace"]["totals"]["wall_ms"] for r in rs)),
                         "errors": sum(1 for r in rs if r["error"])}

    data = {"pipes": [p for p in PIPES if p in s["overall"]], "model": s["model"], "embed_model": s["embed_model"], "types": types, "overall": s["overall"],
            "by_qtype": s["by_qtype"], "rows": list(rows.values()), "traces": traces, "behaviour": behaviour,
            "tool_use": tool_use.most_common(), "agent_tokens": dict(agent_tokens), "agent_calls": agent_calls.most_common(),
            "agent_ms": {k: round(v) for k, v in agent_ms.items()}, "stops": stops.most_common(), "hidden": hidden}

    tpl = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    frag = tpl.replace("/*__DATA__*/null", payload)
    head, body = frag.split("<!--BODY-->", 1)
    full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            f"{head}</head>\n<body>\n{body}</body>\n</html>\n")
    (config.RESULTS_DIR / "dashboard.html").write_text(full, encoding="utf-8")
    (config.RESULTS_DIR / "dashboard_fragment.html").write_text(head + body, encoding="utf-8")
    print(f"wrote {config.RESULTS_DIR / 'dashboard.html'} ({len(full) // 1024} KB)")


if __name__ == "__main__":
    main()
