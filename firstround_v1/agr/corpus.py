"""Load any JSONL corpus into ``Doc`` records.

Field names are configurable (``CORPUS_ID_FIELD`` etc. in .env), so a new dataset that
names its text field ``content`` instead of ``text`` needs no code change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class Doc:
    doc_id: str
    title: str
    url: str
    text: str
    approx_tokens: int
    meta: dict = field(default_factory=dict)   # domain packs may attach parsed structure here


def load_docs(path: Path | None = None) -> list[Doc]:
    f = config.CORPUS_FIELDS
    docs = []
    with open(path or config.CORPUS_PATH, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            text = d.get(f["text"], "")
            docs.append(Doc(doc_id=str(d[f["id"]]), title=d.get(f["title"], "") or str(d[f["id"]]),
                            url=d.get(f["url"], ""), text=text,
                            approx_tokens=int(d.get("approx_tokens") or len(text) // 4)))
    return docs
