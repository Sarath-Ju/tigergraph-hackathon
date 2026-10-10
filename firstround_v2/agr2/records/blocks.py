"""Finding the structured blocks in a document and reading their values.

Many documents start with a small block of facts: a header in square brackets, then indented
"key: value" lines. In this corpus those are Wikipedia infoboxes, for example

    [Infobox film]
      director: Imtiaz Ali
      starring: Kareena KapoorShahid Kapoor
      released: 2007-10-26

This file only knows that shape. It does not know what a film or an Olympic event is. Deciding what
the blocks mean, and which fields are numbers, dates or names, happens in schema.py by looking at
the whole corpus.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

HEADER_RE = re.compile(r"^\[(?P<kind>[^\]\n]{1,80})\]\s*$")
FIELD_RE = re.compile(r"^\s+(?P<key>[A-Za-z][A-Za-z0-9_ ]{0,40}?)\s*:\s*(?P<val>.*\S)\s*$")
SUFFIX_RE = re.compile(r"^(?P<base>.*?[a-z_])(?P<idx>\d{1,2})$")  # "term_start2" is field "term_start", entry 2


@dataclass
class Block:
    kind: str                     # header text, e.g. "Infobox film"
    seq: int                      # position of the block in its document
    fields: dict[str, str] = field(default_factory=dict)


def find_blocks(text: str) -> list[Block]:
    """Return every block in the document, in the order they appear."""
    blocks: list[Block] = []
    cur: Block | None = None
    for line in text.split("\n"):
        h = HEADER_RE.match(line)
        if h:
            cur = Block(kind=h.group("kind").strip(), seq=len(blocks))
            blocks.append(cur)
            continue
        if cur is None:
            continue
        f = FIELD_RE.match(line)
        if f:
            cur.fields.setdefault(f.group("key").strip().lower().replace(" ", "_"), f.group("val").strip())
        elif line.strip():
            cur = None  # the first line that is not a field ends the block
    return [b for b in blocks if b.fields]


def split_suffix(key: str) -> tuple[str, int | None]:
    m = SUFFIX_RE.match(key)
    return (m.group("base"), int(m.group("idx"))) if m else (key, None)


# ---------------------------------------------------------------------------------------------- normalisation

def norm(s: str) -> str:
    """Make a string easy to compare: no accents, lower case, punctuation turned into spaces."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", " ", s.casefold())
    return re.sub(r"\s+", " ", s).strip()


def vertex_id(s: str) -> str:
    """Build a vertex id the same way TigerGraph GraphRAG does (lower case, underscores for spaces),
    so our ids and the service's ids always match."""
    return s.replace(" ", "_").lower().replace("/", "_").replace("(", "").replace(")", "")


# ---------------------------------------------------------------------------------------------- numbers

_MULT = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9,
         "trillion": 1e12, "lakh": 1e5, "crore": 1e7}
NUM_RE = re.compile(r"^[^\d\-]{0,4}(?P<num>-?\d[\d,]*(?:\.\d+)?)\s*(?P<mult>[A-Za-z]+)?")


def parse_number(s: str) -> float | None:
    """Read the number at the start of a value. Scale words are applied, so "24" gives 24,
    "$13 million" gives 13000000 and "142 minutes" gives 142. Returns None if there is no number."""
    m = NUM_RE.match(s.strip())
    # a colon right after the number means a time or a score (3:09.6, 6:4), which is not a quantity
    if not m or s.strip()[m.end("num"):m.end("num") + 1] == ":":
        return None
    try:
        x = float(m.group("num").replace(",", ""))
    except ValueError:
        return None
    mult = (m.group("mult") or "").lower()
    return x * _MULT.get(mult, 1.0)


# ---------------------------------------------------------------------------------------------- dates

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
ISO_RE = re.compile(r"\b(?P<y>1[5-9]\d\d|20\d\d)-(?P<m>\d\d)-(?P<d>\d\d)\b")
DMY_RE = re.compile(r"\b(?P<d>\d{1,2})\s+(?P<mon>[A-Za-z]{3,9})\.?,?\s+(?P<y>1[5-9]\d\d|20\d\d)\b")
MDY_RE = re.compile(r"\b(?P<mon>[A-Za-z]{3,9})\.?\s+(?P<d>\d{1,2}),?\s+(?P<y>1[5-9]\d\d|20\d\d)\b")
YEAR_RE = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
MONTH_RE = re.compile(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b", re.I)


def parse_date(s: str) -> tuple[str, int | None]:
    """Find the first date in a value. Returns the ISO date (or an empty string when only the year
    is known) and the year (or None when there is no year at all)."""
    m = ISO_RE.search(s)
    if m:
        return f"{m['y']}-{m['m']}-{m['d']}", int(m["y"])
    for rx in (DMY_RE, MDY_RE):
        m = rx.search(s)
        if m and m["mon"].lower() in MONTHS:
            return f"{m['y']}-{MONTHS[m['mon'].lower()]:02d}-{int(m['d']):02d}", int(m["y"])
    y = YEAR_RE.search(s)
    return "", int(y.group(1)) if y else None


def looks_like_date(s: str) -> bool:
    return bool(ISO_RE.search(s) or MONTH_RE.search(s) and (YEAR_RE.search(s) or re.search(r"\d", s)))


# ---------------------------------------------------------------------------------------------- names

# When a lower-case letter is directly followed by a capital, the source text usually glued two list
# items together ("Kareena KapoorShahid Kapoor"). Some names do that on purpose ("McDonald",
# "DiCaprio"), so we do not split after these short name particles.
_PARTICLES = {"mc", "mac", "di", "de", "da", "du", "le", "la", "van", "von", "o", "d", "al", "el", "st", "fitz"}
_GLUE_RE = re.compile(r"(?<=[a-zß-ÿ\.\)])(?=[A-ZÀ-Þ])")
_PAREN_RE = re.compile(r"\s*\([^)]*\)")


def split_names(value: str) -> list[str]:
    """Split a field that lists several names. Commas and semicolons come first, then names that
    were glued together. Notes in brackets such as "(acting)" are dropped."""
    value = _PAREN_RE.sub("", value)
    parts = [p.strip() for p in re.split(r"\s*(?:,|;|\s\|\s)\s*", value) if p.strip()]
    out: list[str] = []
    for p in parts:
        pieces, start = [], 0
        for m in _GLUE_RE.finditer(p):
            left = p[start:m.start()]
            last_word = left.split()[-1] if left.split() else ""
            if last_word.lower().rstrip(".") in _PARTICLES or len(last_word) <= 1:
                continue
            pieces.append(left)
            start = m.start()
        pieces.append(p[start:])
        out.extend(x.strip(" .:") for x in pieces if x.strip(" .:"))
    return list(dict.fromkeys(out))


def looks_like_names(value: str) -> bool:
    """True when the value looks like a list of names: short, capitalised items and no digits."""
    items = split_names(value)
    if not items or len(items) > 25 or re.search(r"\d", value):
        return False
    return all(len(i) <= 60 and len(i.split()) <= 6 and i[:1].isupper() for i in items)
