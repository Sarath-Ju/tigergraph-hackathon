"""Show what is in the graph and how far GraphRAG's build service has got.

    python scripts/graph_status.py            check once
    python scripts/graph_status.py --watch    check every 30 seconds until the build is done
                                             (chunks exist and nothing changed for two checks)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agr2 import tg  # noqa: E402

CORE = ["Document", "DocumentChunk", "Entity", "Community"]


def snapshot() -> tuple[dict, object]:
    counts = tg.vertex_counts(CORE)
    try:
        st = tg.build_status()
    except Exception as e:  # the status is only for information, so a failed check is not fatal
        st = f"unavailable: {e}"
    return counts, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    idle_checks, last = 0, None
    while True:
        counts, st = snapshot()
        print(time.strftime("%H:%M:%S"), json.dumps(counts), str(st)[:300], flush=True)
        if not a.watch:
            break
        running = isinstance(st, dict) and (st.get("is_running") or st.get("status") in
                                            ("running", "already_running", "in_progress"))
        # the build service can look idle for a moment before it picks the documents up, so we only
        # stop once there are chunks and nothing has changed for two checks in a row
        idle_checks = idle_checks + 1 if (not running and counts["DocumentChunk"] > 0 and counts == last) else 0
        last = counts
        if idle_checks >= 2:
            break
        time.sleep(30)
    if not a.watch:
        print(json.dumps(tg.vertex_counts(), indent=1))


if __name__ == "__main__":
    main()
