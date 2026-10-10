"""Create the general question set: questions about the documents that are not about the Olympics.

The hackathon's public and hidden questions are all about Olympic events, so on their own they
cannot show whether a system works on other topics. These questions cover the films and
officeholders in the corpus, in the same five styles as the public set:

  lookup       who directed a film, how long a film runs, who came before a politician in an office
  temporal     which film a director released just before another one
  multi_hop    who wrote the music for a director's film from a given year, or when a film's director was born
  aggregation  how many films with a given actor came out after a given year
  superlative  which film from a given year runs the longest

Every answer and every gold document is computed from the corpus itself, so the set can be checked.

    python scripts/make_general_eval.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agr2 import config  # noqa: E402
from agr2.records.blocks import find_blocks, parse_date, parse_number, split_names, split_suffix  # noqa: E402
from agr2.records.schema import iter_docs  # noqa: E402


def clean_names(v: str) -> list[str]:
    return [n for n in split_names(v or "") if 2 < len(n) < 40 and ":" not in n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    docs = list(iter_docs(config.CORPUS_PATH, config.CORPUS_FIELDS))

    films, holders, people = [], [], {}
    for d in docs:
        for b in find_blocks(d["text"]):
            kind = b.kind.lower()
            f = b.fields
            if kind.endswith("film") and b.seq == 0:
                iso, year = parse_date(f.get("released", ""))
                films.append({"doc": d["id"], "title": f.get("name") or d["title"], "doc_title": d["title"],
                              "year": year, "iso": iso,
                              "directors": clean_names(f.get("director", "")),
                              "stars": clean_names(f.get("starring", "")), "music": clean_names(f.get("music", "")),
                              "runtime": f.get("runtime", ""), "runtime_min": parse_number(f.get("runtime", ""))})
            elif kind.endswith("officeholder") and b.seq == 0:
                groups = defaultdict(dict)
                for k, v in f.items():
                    base, idx = split_suffix(k)
                    groups[idx or 0][base] = v
                holders.append({"doc": d["id"], "name": f.get("name") or d["title"], "groups": groups})
            if b.seq == 0 and f.get("birth_date"):
                _, by = parse_date(f["birth_date"])
                if by:
                    people[(f.get("name") or d["title"]).strip()] = {"doc": d["id"], "birth_year": by}

    out: dict[str, list[dict]] = defaultdict(list)

    def add(qtype, question, answers, gold_docs):
        answers = list(dict.fromkeys(str(x) for x in answers))
        out[qtype].append({"question": question, "qtype": qtype, "answer": answers,
                           "gold_doc_ids": sorted(set(gold_docs)), "answer_verified": True, "source": "generated"})

    titles_count = defaultdict(int)
    for fm in films:
        titles_count[fm["title"]] += 1
    films = [fm for fm in films if titles_count[fm["title"]] == 1]
    rnd.shuffle(films)

    # lookup
    for fm in [f for f in films if len(f["directors"]) == 1][:a.per_type // 2]:
        add("lookup", f"Who directed the film {fm['title']}?", fm["directors"], [fm["doc"]])
    for fm in [f for f in films if f["runtime_min"] and "minute" in f["runtime"].lower()][:a.per_type - a.per_type // 2]:
        add("lookup", f"According to the provided corpus, what is the runtime of the film {fm['title']}?",
            [fm["runtime"], f"{int(fm['runtime_min'])} minutes"], [fm["doc"]])

    # temporal: previous film by the same director
    by_dir = defaultdict(list)
    for fm in films:
        if len(fm["directors"]) == 1 and fm["iso"]:
            by_dir[fm["directors"][0]].append(fm)
    cands = []
    for dname, fs in by_dir.items():
        fs = sorted(fs, key=lambda x: x["iso"])
        if len(fs) >= 3 and len({x["iso"] for x in fs}) == len(fs):
            i = rnd.randrange(1, len(fs))
            cands.append((dname, fs[i - 1], fs[i]))
    rnd.shuffle(cands)
    for dname, prev, cur in cands[:a.per_type]:
        add("temporal", f"According to the provided corpus, which film directed by {dname} was released "
            f"immediately before {cur['title']}?", [prev["title"], prev["doc_title"]], [prev["doc"], cur["doc"]])

    # multi_hop
    by_dir_year = defaultdict(list)
    for fm in films:
        for dn in fm["directors"]:
            if fm["year"]:
                by_dir_year[(dn, fm["year"])].append(fm)
    mh = [(k, v[0]) for k, v in by_dir_year.items() if len(v) == 1 and len(v[0]["music"]) == 1
          and len(by_dir[k[0]]) >= 2]
    rnd.shuffle(mh)
    for (dn, yr), fm in mh[:a.per_type // 2]:
        add("multi_hop", f"According to the provided corpus, who composed the music for the {yr} film directed by "
            f"{dn}?", fm["music"], [fm["doc"]])
    born = [fm for fm in films if len(fm["directors"]) == 1 and fm["directors"][0] in people]
    rnd.shuffle(born)
    for fm in born[:a.per_type - a.per_type // 2]:
        p = people[fm["directors"][0]]
        add("multi_hop", f"In which year was the director of the film {fm['title']} born?", [p["birth_year"]],
            [fm["doc"], p["doc"]])

    # aggregation: films starring A released after Y
    by_star = defaultdict(list)
    for fm in films:
        for s in fm["stars"]:
            if fm["year"]:
                by_star[s].append(fm)
    stars = [s for s, fs in by_star.items() if len(fs) >= 4]
    rnd.shuffle(stars)
    for s in stars[:a.per_type]:
        years = sorted(f["year"] for f in by_star[s])
        y = years[len(years) // 2 - 1]
        hits = [f for f in by_star[s] if f["year"] > y]
        add("aggregation", f"According to the provided corpus, how many films starring {s} were released after {y}?",
            [len(hits)], [f["doc"] for f in by_star[s]])

    # superlative: longest runtime in a year
    by_year = defaultdict(list)
    for fm in films:
        if fm["year"] and fm["runtime_min"]:
            by_year[fm["year"]].append(fm)
    yrs = [y for y, fs in by_year.items() if len(fs) >= 8]
    rnd.shuffle(yrs)
    for y in yrs:
        fs = sorted(by_year[y], key=lambda x: -x["runtime_min"])
        if fs[0]["runtime_min"] == fs[1]["runtime_min"]:
            continue
        add("superlative", f"According to the provided corpus, which film released in {y} had the longest runtime?",
            [fs[0]["title"], fs[0]["doc_title"]], [f["doc"] for f in fs])
        if len(out["superlative"]) >= a.per_type:
            break

    # officeholders: predecessor in a specific office (pairs office<i> with predecessor<i>)
    rnd.shuffle(holders)
    for h in holders:
        opts = [(g["office"], clean_names(g.get("predecessor", ""))) for g in h["groups"].values()
                if g.get("office") and len(clean_names(g.get("predecessor", ""))) == 1
                and " and " not in g.get("predecessor", "") and "List of" not in g["office"]
                and not any(a.islower() and b.isupper() for a, b in zip(g["office"], g["office"][1:]))]
        if opts:
            office, pred = rnd.choice(opts)
            add("lookup", f"According to the provided corpus, who preceded {h['name']} as {office}?", pred, [h["doc"]])
        if sum(1 for q in out["lookup"] if "who preceded" in q["question"]) >= a.per_type // 2:
            break

    rows = [q for t in ("lookup", "temporal", "multi_hop", "aggregation", "superlative") for q in out[t]]
    path = config.QUESTION_SPLITS["general"]
    with open(path, "w", encoding="utf-8") as fh:
        for i, q in enumerate(rows, 1):
            fh.write(json.dumps({"qid": f"gen-{i:03d}", **q}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} questions to {path}")
    for q in rows:
        print(f"  [{q['qtype']:<11}] {q['question']}  => {q['answer'][0]}")


if __name__ == "__main__":
    main()
