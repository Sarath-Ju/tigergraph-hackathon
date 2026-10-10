"""Domain-independent text helpers."""
from __future__ import annotations

import re
import unicodedata


def norm(s: str) -> str:
    """Normalise a name for ids and matching: strip accents, casefold, squash punctuation."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold().replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def leading_int(s: str | None) -> int:
    if not s:
        return -1
    m = re.search(r"\d[\d,]*", s)
    return int(m.group(0).replace(",", "")) if m else -1
