"""Deterministic parsing of the corpus into typed graph records.

Every Olympic event article starts with an ``[Infobox Olympic event]`` block of
``key: value`` lines, and its title follows
``<Sport> at the <YEAR> <Summer|Winter> Olympics – <Event>``.  We turn that into
structured Event / Games / Sport / Venue / Competitor / NOC records without any
LLM calls, which keeps graph construction cheap, fast and reproducible.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

TITLE_RE = re.compile(r"^(?P<sport>.+?) at the (?P<year>\d{4}) (?P<season>Summer|Winter) Olympics – (?P<event>.+)$")
INFOBOX_RE = re.compile(r"^\[Infobox (?P<kind>[^\]]+)\]")
FIELD_RE = re.compile(r"^\s+(?P<key>[A-Za-z_0-9]+):\s*(?P<val>.*)$")


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


def parse_infobox(text: str) -> tuple[str | None, dict[str, str]]:
    m = INFOBOX_RE.match(text)
    if not m:
        return None, {}
    fields: dict[str, str] = {}
    for line in text.split("\n")[1:]:
        if not line.strip():
            break
        fm = FIELD_RE.match(line)
        if fm:
            fields.setdefault(fm.group("key"), fm.group("val").strip())
    return m.group("kind"), fields


@dataclass
class Doc:
    doc_id: str
    title: str
    url: str
    text: str
    approx_tokens: int
    infobox_kind: str | None
    infobox: dict[str, str]


@dataclass
class EventRec:
    event_id: str          # = doc_id
    title: str
    sport: str
    event_name: str
    year: int
    season: str
    games_id: str          # e.g. "2012 Summer"
    gender: str
    competitors: int
    nations: int
    date: str
    venue: str
    win_value: str
    medals: list[tuple[str, str, str]] = field(default_factory=list)  # (medal, competitor, NOC)
    prev_year: int = -1
    next_year: int = -1


def load_docs(path: Path) -> list[Doc]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            kind, box = parse_infobox(d["text"])
            docs.append(Doc(d["doc_id"], d["title"], d.get("url", ""), d["text"], d.get("approx_tokens", 0), kind, box))
    return docs


def gender_of(event_name: str) -> str:
    n = event_name.lower()
    if n.startswith("women") or "women's" in n or "ladies" in n:
        return "women"
    if n.startswith("men") or "men's" in n:
        return "men"
    if "mixed" in n or "pairs" in n or "ice dance" in n:
        return "mixed"
    return "open"


def to_event(doc: Doc) -> EventRec | None:
    if doc.infobox_kind != "Olympic event":
        return None
    m = TITLE_RE.match(doc.title)
    if not m:
        return None
    b = doc.infobox
    year, season = int(m.group("year")), m.group("season")
    medals = []
    for medal in ("gold", "silver", "bronze"):
        for suffix in ("", "2", "3"):
            who = b.get(f"{medal}{suffix}")
            if who:
                medals.append((medal, who, b.get(f"{medal}NOC{suffix}", "")))
    return EventRec(
        event_id=doc.doc_id,
        title=doc.title,
        sport=m.group("sport"),
        event_name=m.group("event"),
        year=year,
        season=season,
        games_id=f"{year} {season}",
        gender=gender_of(m.group("event")),
        competitors=leading_int(b.get("competitors")),
        nations=leading_int(b.get("nations")),
        date=b.get("date") or b.get("dates") or "",
        venue=b.get("venue") or b.get("venues") or b.get("location") or "",
        win_value=b.get("win_value", ""),
        medals=medals,
        prev_year=leading_int(b.get("prev")),
        next_year=leading_int(b.get("next")),
    )
