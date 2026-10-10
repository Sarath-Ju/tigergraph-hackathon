"""Check that the chunks really have embeddings before we benchmark anything.

It takes a random sample of first chunks and searches with each chunk's own text. A chunk with an
embedding comes back in the top results for its own text, so a low hit rate means the build service
stopped part way and the vectors are missing. That happened once when a build was interrupted, and
every approach then scored far below what it should.

    python scripts/check_vectors.py            20 chunks, needs 80% to pass
    python scripts/check_vectors.py --n 50
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agr2 import tg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-rate", type=float, default=0.8)
    a = ap.parse_args()
    c = tg.conn()
    firsts = [v["v_id"] for v in c.getVertices("DocumentChunk", select="epoch_added") if v["v_id"].endswith("_chunk_0")]
    sample = random.Random(7).sample(firsts, min(a.n, len(firsts)))
    hits = 0
    for cid in sample:
        content = c.getVerticesById("Content", cid)
        text = content[0]["attributes"]["text"][:1500] if content else ""
        found = tg.search(text, "similarity", {"index": "DocumentChunk", "top_k": 3, "withHyDE": False})
        hits += cid in (found.get("final_retrieval") or {})
    rate = hits / max(1, len(sample))
    print(f"{hits} of {len(sample)} chunks found by their own text ({rate:.0%})")
    if rate < a.min_rate:
        print("too many chunks have no embedding; rebuild the graph before running the benchmark")
        sys.exit(1)


if __name__ == "__main__":
    main()
