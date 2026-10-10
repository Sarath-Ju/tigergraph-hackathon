"""Score benchmark results on the public split and build results/summary.json.

Per (question, pipeline):
  exact_match   normalised prediction == normalised gold answer
  judge_score   LLM judge: 1 correct, 0.5 partially correct / incomplete, 0 wrong (cached in results/public/judged.jsonl)
  complete      judge's completeness flag
  cite_recall   |citations ∩ gold_doc_ids| / |gold_doc_ids|
  cite_precision|citations ∩ gold_doc_ids| / |citations|
  retrieval_recall |retrieved docs ∩ gold_doc_ids| / |gold_doc_ids|

    python scripts/evaluate.py [--pipelines rag,graphrag,agentic]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agr import config, llm  # noqa: E402
from agr.parse import norm  # noqa: E402

JUDGE = """You grade answers to questions about a fixed document corpus against the gold answer.
Score 1 if the prediction states the same answer as the gold (ignore formatting, punctuation, separators between
team member names, accents, or an added article title prefix). Score 0.5 if it is partially right (e.g. one of
several required names, or the right event but wrong detail). Score 0 otherwise, including UNKNOWN.
Return JSON {"score": 0|0.5|1, "complete": bool, "reason": "<10 words"}."""


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def judge(q: dict, pred: str) -> dict:
    if not pred.strip() or pred.strip().upper() == "UNKNOWN":
        return {"score": 0, "complete": False, "reason": "no answer"}
    if any(norm(pred) == norm(g) for g in q["answer"]):
        return {"score": 1, "complete": True, "reason": "exact match"}
    out = llm.chat_json([{"role": "system", "content": JUDGE},
                         {"role": "user", "content": f"Question: {q['question']}\nGold: {' | '.join(q['answer'])}\n"
                                                     f"Prediction: {pred}"}], name="judge", max_tokens=60)
    s = out.get("score", 0)
    return {"score": s if s in (0, 0.5, 1) else 0, "complete": bool(out.get("complete")), "reason": out.get("reason", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipelines", default="rag,graphrag,agentic")
    a = ap.parse_args()
    gold = {q["qid"]: q for q in load(config.PUBLIC_Q)}
    rdir = config.RESULTS_DIR / "public"
    cache_path = rdir / "judged.jsonl"
    cache = {(r["pipeline"], r["qid"], r["answer"]): r["judge"] for r in load(cache_path)}

    rows = []
    for p in a.pipelines.split(","):
        for r in load(rdir / f"{p}.jsonl"):
            rows.append(r)
    todo = [r for r in rows if (r["pipeline"], r["qid"], r["answer"]) not in cache]
    with ThreadPoolExecutor(8) as ex:
        for r, j in zip(todo, ex.map(lambda r: judge(gold[r["qid"]], r["answer"]), todo)):
            cache[(r["pipeline"], r["qid"], r["answer"])] = j
    with open(cache_path, "w", encoding="utf-8") as f:
        for (p, qid, ans), j in cache.items():
            f.write(json.dumps({"pipeline": p, "qid": qid, "answer": ans, "judge": j}, ensure_ascii=False) + "\n")

    per_q = []
    for r in rows:
        q = gold[r["qid"]]
        g = set(q["gold_doc_ids"])
        cites, retrieved = set(r["citations"]), set(r["retrieved_doc_ids"])
        j = cache[(r["pipeline"], r["qid"], r["answer"])]
        t = r["trace"]["totals"]
        ev = r["trace"]["events"]
        ops = r["trace"]["ops"]
        per_q.append({
            "qid": r["qid"], "qtype": q["qtype"], "pipeline": r["pipeline"], "question": q["question"],
            "gold": q["answer"][0], "answer": r["answer"], "error": r["error"],
            "exact_match": int(any(norm(r["answer"]) == norm(x) for x in q["answer"])),
            "judge_score": j["score"], "complete": int(j["complete"]), "judge_reason": j["reason"],
            "cite_recall": len(cites & g) / len(g) if g else 0,
            "cite_precision": len(cites & g) / len(cites) if cites else 0,
            "retrieval_recall": len(retrieved & g) / len(g) if g else 0,
            "n_citations": len(cites), "n_retrieved_docs": len(retrieved),
            **t,
            "cached_input_tokens": sum(o["detail"].get("cached_input_tokens", 0) for o in ops),
            "tool_calls": sum(1 for e in ev if e["kind"] == "action"),
            "tools": [e["tool"] for e in ev if e["kind"] == "action"],
            "agents": sorted({o["agent"] for o in ops if o["agent"]}),
            "strategy_changes": sum(1 for e in ev if e["kind"] == "strategy_change"),
            "evaluator_rejections": sum(1 for e in ev if e["kind"] == "evaluation" and not e["accepted"]),
            "stop_reason": next((e["reason"] for e in ev if e["kind"] == "stop"), "single_pass"),
        })

    def agg(items: list[dict]) -> dict:
        m = lambda k: round(st.mean(x[k] for x in items), 4) if items else 0
        return {"n": len(items), "accuracy": m("judge_score"), "exact_match": m("exact_match"),
                "completeness": m("complete"), "cite_recall": m("cite_recall"), "cite_precision": m("cite_precision"),
                "retrieval_recall": m("retrieval_recall"), "avg_total_tokens": m("total_tokens"),
                "avg_input_tokens": m("llm_input_tokens"), "avg_output_tokens": m("llm_output_tokens"),
                "avg_context_tokens": m("context_tokens"), "avg_embed_tokens": m("embed_tokens"),
                "avg_llm_calls": m("llm_calls"), "avg_tool_calls": m("tool_calls"), "avg_latency_ms": m("wall_ms"),
                "strategy_change_rate": round(sum(x["strategy_changes"] > 0 for x in items) / len(items), 4) if items else 0,
                "total_tokens": sum(x["total_tokens"] for x in items),
                "tokens_per_correct": round(sum(x["total_tokens"] for x in items) / max(1e-9, sum(x["judge_score"] for x in items)), 1),
                "errors": sum(1 for x in items if x["error"])}

    by_p, by_pt = defaultdict(list), defaultdict(list)
    for x in per_q:
        by_p[x["pipeline"]].append(x)
        by_pt[(x["pipeline"], x["qtype"])].append(x)
    summary = {"model": config.LLM_MODEL, "embed_model": config.EMBED_MODEL,
               "overall": {p: agg(v) for p, v in by_p.items()},
               "by_qtype": {f"{p}|{t}": agg(v) for (p, t), v in sorted(by_pt.items())},
               "questions": per_q}
    (config.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    print(f"{'pipeline':<10} {'acc':>6} {'EM':>6} {'cite_R':>7} {'tok/q':>8} {'tok/correct':>12} {'llm':>5} {'ms':>7}")
    for p, s in summary["overall"].items():
        print(f"{p:<10} {s['accuracy']:>6.2f} {s['exact_match']:>6.2f} {s['cite_recall']:>7.2f} {s['avg_total_tokens']:>8.0f} "
              f"{s['tokens_per_correct']:>12.0f} {s['avg_llm_calls']:>5.1f} {s['avg_latency_ms']:>7.0f}")
    print("\naccuracy by question type")
    types = sorted({x["qtype"] for x in per_q})
    print(f"{'':<10}" + "".join(f"{t[:11]:>12}" for t in types))
    for p in by_p:
        print(f"{p:<10}" + "".join(f"{summary['by_qtype'].get(f'{p}|{t}', {}).get('accuracy', 0):>12.2f}" for t in types))


if __name__ == "__main__":
    main()
