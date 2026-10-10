"""Token-bounded, paragraph-aware chunker used for the similarity (RAG) index."""
from __future__ import annotations

from .llm import _enc
from . import config


def chunk_text(title: str, text: str, max_tokens: int = config.CHUNK_TOKENS,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split on blank lines, pack paragraphs up to ``max_tokens``; hard-split very long paragraphs.

    Each chunk is prefixed with the article title so it stays self-describing
    (important: most Olympic articles never repeat the year/sport in their body).
    """
    header = f"[{title}]\n"
    budget = max_tokens - len(_enc.encode(header))
    pieces: list[list[int]] = []
    for para in [p for p in text.split("\n\n") if p.strip()]:
        toks = _enc.encode(para + "\n\n", disallowed_special=())
        if len(toks) <= budget:
            pieces.append(toks)
        else:
            step = budget - overlap
            pieces.extend(toks[i:i + budget] for i in range(0, len(toks), step))
    chunks, cur = [], []
    for p in pieces:
        if cur and len(cur) + len(p) > budget:
            chunks.append(cur)
            cur = cur[-overlap:] if overlap else []
        cur = cur + p
    if cur:
        chunks.append(cur)
    return [header + _enc.decode(c).strip() for c in chunks]
