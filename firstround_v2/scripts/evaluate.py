"""Score the saved answers and write results/summary.json.

Every set that comes with answers (public and general) is scored. For each question and approach:

  exact_match       the answer matches a gold answer once accents, case and punctuation are ignored
  judge_score       an LLM judge's grade: 1 for correct, 0.5 for partly correct, 0 for wrong
  cite_recall       how many of the gold documents the answer cited
  retrieval_recall  how many of the gold documents were retrieved at all
  tokens, tool calls and time, as reported by the GraphRAG service

Judge grades are cached, so scoring again costs nothing unless an answer changed.

    python scripts/evaluate.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agr2 import config  # noqa: E402
from agr2.records.blocks import norm  # noqa: E402

JUDGE = """You grade answers to questions about a fixed document corpus against the gold answer.
Score 1 if the prediction states the same answer as the gold (ignore formatting, punctuation, separators between
team member names, accents, extra explanation, or an added article title prefix/suffix). Score 0.5 if it is partially
right (e.g. one of several required names). Score 0 otherwise, including "I don't know" style answers.
Return JSON {"score": 0|0.5|1, "reason": "<10 words"}."""

_client = None


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def exact(pred: str, golds: list[str]) -> bool:
    p = norm(pred)
    return any(p == norm(g) or (len(norm(g)) >= 1 and f" {norm(g)} " in f" {p} " and len(p) <= 3 * len(norm(g)) + 12)
               for g in golds)


def judge(q: dict, pred: str) -> dict:
    global _client
    if not pred.strip():
        return {"score": 0, "reason": "no answer"}
    if exact(pred, q["answer"]):
        return {"score": 1, "reason": "exact match"}
    _client = _client or OpenAI()
    r = _client.chat.completions.create(
        model=config.LLM_MODEL, temperature=0, max_tokens=60, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": JUDGE},
                  {"role": "user", "content": f"Question: {q['question']}\nGold: {' | '.join(q['answer'])}\n"
                                              f"Prediction: {pred}"}])
    out = json.loads(r.choices[0].message.content or "{}")
    s = out.get("score", 0)
    return {"score": s if s in (0, 0.5, 1) else 0, "reason": out.get("reason", "")}


def agg(items: list[dict]) -> dict:
    m = lambda k: round(st.mean(x[k] for x in items), 4) if items else 0
    tot = sum(x["total_tokens"] for x in items)
    return {"n": len(items), "accuracy": m("judge_score"), "exact_match": m("exact_match"),
            "cite_recall": m("cite_recall"), "retrieval_recall": m("retrieval_recall"),
            "avg_total_tokens": m("total_tokens"), "avg_input_tokens": m("llm_input_tokens"),
            "avg_output_tokens": m("llm_output_tokens"), "avg_tool_calls": m("tool_calls"),
            "avg_latency_ms": m("wall_ms"), "total_tokens": tot,
            "tokens_per_correct": round(tot / max(1e-9, sum(x["judge_score"] for x in items)), 1),
            "errors": sum(1 for x in items if x["error"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="public,general")
    ap.add_argument("--pipelines", default="rag,graphrag,agentic,reactive")
    a = ap.parse_args()
    summary = {"model": config.LLM_MODEL, "embed_model": config.EMBED_MODEL, "splits": {}}

    for split in a.splits.split(","):
        gold = {q["qid"]: q for q in load(config.QUESTION_SPLITS[split]) if q.get("answer")}
        rdir = config.RESULTS_DIR / split
        rows = [r for p in a.pipelines.split(",") for r in load(rdir / f"{p}.jsonl") if r["qid"] in gold]
        if not rows:
            continue
        cache_path = rdir / "judged.jsonl"
        cache = {(r["pipeline"], r["qid"], r["answer"]): r["judge"] for r in load(cache_path)}
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
            g = set(q.get("gold_doc_ids") or [])
            cites, retrieved = set(r["citations"]), set(r["retrieved_doc_ids"])
            j = cache[(r["pipeline"], r["qid"], r["answer"])]
            t = r["trace"].get("totals", {})
            per_q.append({
                "qid": r["qid"], "qtype": q.get("qtype", ""), "pipeline": r["pipeline"], "question": q["question"],
                "gold": q["answer"][0], "answer": r["answer"], "error": r["error"],
                "exact_match": int(exact(r["answer"], q["answer"])),
                "judge_score": j["score"], "judge_reason": j["reason"],
                "cite_recall": len(cites & g) / len(g) if g else 0,
                "retrieval_recall": len(retrieved & g) / len(g) if g else 0,
                "total_tokens": t.get("total_tokens", 0), "llm_input_tokens": t.get("llm_input_tokens", 0),
                "llm_output_tokens": t.get("llm_output_tokens", 0), "tool_calls": t.get("tool_calls", 0),
                "wall_ms": t.get("wall_ms", 0), "tools": [s["tool"] for s in r["trace"].get("steps", [])],
            })
        by_p, by_pt = defaultdict(list), defaultdict(list)
        for x in per_q:
            by_p[x["pipeline"]].append(x)
            by_pt[(x["pipeline"], x["qtype"])].append(x)
        summary["splits"][split] = {"overall": {p: agg(v) for p, v in by_p.items()},
                                    "by_qtype": {f"{p}|{t}": agg(v) for (p, t), v in sorted(by_pt.items())},
                                    "questions": per_q}

        print(f"\n=== {split} ===")
        print(f"{'pipeline':<10} {'n':>4} {'acc':>6} {'EM':>6} {'cite_R':>7} {'tok/q':>8} {'tok/correct':>12} "
              f"{'tools':>6} {'sec':>6}")
        for p, s in summary["splits"][split]["overall"].items():
            print(f"{p:<10} {s['n']:>4} {s['accuracy']:>6.2f} {s['exact_match']:>6.2f} {s['cite_recall']:>7.2f} "
                  f"{s['avg_total_tokens']:>8.0f} {s['tokens_per_correct']:>12.0f} {s['avg_tool_calls']:>6.1f} "
                  f"{s['avg_latency_ms'] / 1000:>6.1f}")
        types = sorted({x["qtype"] for x in per_q})
        print(f"{'accuracy':<10}" + "".join(f"{t[:11]:>12}" for t in types))
        for p in by_p:
            print(f"{p:<10}" + "".join(f"{summary['splits'][split]['by_qtype'].get(f'{p}|{t}', {}).get('accuracy', 0):>12.2f}"
                                       for t in types))

    (config.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\nwrote {config.RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
