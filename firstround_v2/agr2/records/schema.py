"""Working out a graph schema from the corpus, and turning blocks into records.

This is the heart of the structured-record layer. We look at every block in the corpus and let the
data decide the schema.

Each kind of block that appears in enough documents becomes its own vertex type, for example Film or
OlympicEvent. Its common fields become attributes, and we look at all the values of a field to decide
what else it deserves:

  - every field keeps its original text, for example director
  - a field that holds numbers also gets a numeric copy, for example competitors_num
  - a field that holds dates also gets a year and an ISO date, for example released_year
  - a field that lists names also becomes a relation, for example a DIRECTOR edge from the film to
    the director's Name

Each record is also connected to the Name of the thing it describes, through a NAMED edge. So when a
director or a predecessor is mentioned in one record, the graph can lead to that person's own record
if the corpus has one. Rare kinds of blocks share one generic Record type.

The output is ordinary GSQL that TigerGraph GraphRAG accepts as its domain schema, which means its
retrievers and its agent can see and query these types.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .blocks import (Block, find_blocks, looks_like_date, looks_like_names, parse_date, parse_number,
                     split_names, split_suffix, vertex_id)

NAME_VT, NAMED_EDGE, RECORD_VT = "Name", "NAMED", "Record"
NAME_KEYS = ("name", "title")  # field that names the record, else the document title
MAX_ATTR_FIELDS = 40


@dataclass
class FieldSpec:
    key: str
    coverage: float
    is_number: bool = False
    is_date: bool = False
    is_names: bool = False
    attr: str = ""                      # the attribute that keeps the original text
    base: str = ""                      # stem used for the extra attributes, such as <base>_num and <base>_year
    edge: str = ""                      # the relation name, when the field lists names


@dataclass
class VertexSpec:
    name: str
    kind: str
    n_docs: int
    fields: dict[str, FieldSpec] = field(default_factory=dict)

    def attributes(self) -> list[tuple[str, str]]:
        out = [("name", "STRING"), ("doc_id", "STRING"), ("doc_title", "STRING")]
        for f in self.fields.values():
            out.append((f.attr, "STRING"))
            if f.is_number:
                out.append((f.base + "_num", "DOUBLE"))
            if f.is_date:
                out += [(f.base + "_year", "INT"), (f.base + "_iso", "STRING")]
        return out


@dataclass
class Schema:
    vertices: dict[str, VertexSpec]                         # each vertex type and what it holds
    by_kind: dict[str, str]                                 # which vertex type each block kind went to
    edges: dict[str, list[str]] = field(default_factory=dict)  # each relation and the types it starts from

    # ------------------------------------------------------------------ GSQL
    def gsql(self) -> str:
        """Write the schema as GSQL, with a short comment above each type. GraphRAG reads those
        comments as descriptions, and its agent sees them when it plans graph queries."""
        lines = [f"// A named thing (person, place, organisation, work, ...) referenced by structured records.",
                 f"ADD VERTEX {NAME_VT}(PRIMARY_ID id STRING, name STRING);"]
        for v in self.vertices.values():
            attrs = ", ".join(f"{a} {t}" for a, t in v.attributes())
            num = [f.base + "_num" for f in v.fields.values() if f.is_number]
            yr = [f.base + "_year" for f in v.fields.values() if f.is_date]
            desc = (f"One '{v.kind}' record per document ({v.n_docs} documents). Raw field text in STRING attributes"
                    + (f"; numeric values in {', '.join(num[:8])}" if num else "")
                    + (f"; years in {', '.join(yr[:6])}" if yr else "") + ".")
            lines += [f"// {desc}", f"ADD VERTEX {v.name}(PRIMARY_ID id STRING, {attrs});"]
        lines += [f"// Generic record for rare structured block kinds; fields as 'key: value' lines.",
                  f"ADD VERTEX {RECORD_VT}(PRIMARY_ID id STRING, name STRING, doc_id STRING, doc_title STRING, "
                  f"kind STRING, fields STRING);"]
        all_vts = list(self.vertices) + [RECORD_VT]
        pairs = " | ".join(f"FROM {vt}, TO {NAME_VT}" for vt in all_vts)
        lines += [f"// The name of the thing a record describes.",
                  f"ADD DIRECTED EDGE {NAMED_EDGE}({pairs});"]
        for e, srcs in self.edges.items():
            pairs = " | ".join(f"FROM {vt}, TO {NAME_VT}" for vt in srcs)
            lines += [f"// Record field '{e.lower()}' naming another thing.", f"ADD DIRECTED EDGE {e}({pairs});"]
        return "\n".join(lines) + "\n"

    def describe(self) -> str:
        rows = [f"{v.name:<28} {v.n_docs:>5} docs  {len(v.fields):>3} fields  "
                f"{sum(f.is_number for f in v.fields.values())} num  {sum(f.is_date for f in v.fields.values())} date  "
                f"{sum(f.is_names for f in v.fields.values())} names" for v in self.vertices.values()]
        return "\n".join(rows)


# ---------------------------------------------------------------------- naming

def _camel(kind: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", kind)
    name = "".join(w[:1].upper() + w[1:] for w in words) or "Kind"
    return name if name[0].isalpha() else "K" + name


def _ident(key: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", key.lower()).strip("_") or "field"
    return s if s[0].isalpha() else "f_" + s


def _kind_label(header: str, common_prefix: str) -> str:
    return header[len(common_prefix):].strip() if common_prefix and header.startswith(common_prefix) else header


def _common_prefix_word(headers: Counter) -> str:
    """If almost every header starts with the same word (here "Infobox"), that word tells us nothing,
    so we leave it out of the type names."""
    first = Counter()
    for h, n in headers.items():
        w = h.split()
        if len(w) > 1:
            first[w[0]] += n
    if not first:
        return ""
    word, n = first.most_common(1)[0]
    return word + " " if n >= 0.8 * sum(headers.values()) else ""


# ---------------------------------------------------------------------- inference

def infer_schema(docs: Iterable[dict], *, min_docs: int, min_coverage: float, reserved: set[str]) -> Schema:
    """Look at every block in the corpus and decide the schema.

    docs are dicts with id, title and text. reserved holds names that GSQL or GraphRAG will not accept
    as type or attribute names; we rename around them."""
    per_kind_docs: dict[str, set] = defaultdict(set)
    values: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    blocks_per_kind: Counter = Counter()
    headers: Counter = Counter()
    for d in docs:
        for b in find_blocks(d["text"]):
            headers[b.kind] += 1
            per_kind_docs[b.kind].add(d["id"])
            blocks_per_kind[b.kind] += 1
            seen = set()
            for k, v in b.fields.items():
                base, _ = split_suffix(k)
                values[b.kind][base].append(v)
                seen.add(base)
    prefix = _common_prefix_word(headers)
    bad = {r.casefold() for r in reserved}
    used: set[str] = {NAME_VT.casefold(), NAMED_EDGE.casefold(), RECORD_VT.casefold()}

    def free(name: str) -> str:
        n, i = name, 2
        while n.casefold() in bad or n.casefold() in used:
            n, i = f"{name}{i}", i + 1
        used.add(n.casefold())
        return n

    vertices: dict[str, VertexSpec] = {}
    by_kind: dict[str, str] = {}
    edges: dict[str, list[str]] = defaultdict(list)
    edge_names: dict[str, str] = {}
    kinds = [k for k, d in sorted(per_kind_docs.items(), key=lambda kv: -len(kv[1])) if len(d) >= min_docs]
    # vertex types pick their names before any relation does, so a relation never takes a type's name
    vt_names = {k: free(_camel(_kind_label(k, prefix))) for k in kinds}
    for kind in kinds:
        vt = VertexSpec(name=vt_names[kind], kind=_kind_label(kind, prefix), n_docs=len(per_kind_docs[kind]))
        n_blocks = blocks_per_kind[kind]
        ranked = sorted(values[kind].items(), key=lambda kv: -len(kv[1]))
        for key, vals in ranked[:MAX_ATTR_FIELDS]:
            if key in NAME_KEYS:
                continue
            cov = min(1.0, len(vals) / n_blocks)
            if cov < min_coverage:
                continue
            f = FieldSpec(key=key, coverage=round(cov, 3))
            dates = sum(looks_like_date(v) for v in vals) / len(vals)
            nums = sum(parse_number(v) is not None for v in vals) / len(vals)
            names = sum(looks_like_names(v) for v in vals) / len(vals)
            f.is_date = dates >= 0.6
            f.is_number = not f.is_date and nums >= 0.8
            f.is_names = not (f.is_date or f.is_number) and names >= 0.6
            f.base = f.attr = _ident(key)
            if f.attr in {a for a, _ in vt.attributes()} or f.attr.casefold() in bad:
                f.attr = f.attr + "_text"
            if f.is_names:
                if key not in edge_names:
                    e = _ident(key).upper()
                    edge_names[key] = free(e if e.casefold() not in used and e.casefold() not in bad else "HAS_" + e)
                f.edge = edge_names[key]
                edges[f.edge].append(vt.name)
            vt.fields[key] = f
        vertices[vt.name] = vt
        by_kind[kind] = vt.name
    return Schema(vertices=vertices, by_kind=by_kind, edges=dict(edges))


# ---------------------------------------------------------------------- records

@dataclass
class Records:
    # records by vertex type and id, with their attributes
    vertices: dict[str, dict[str, dict]] = field(default_factory=lambda: defaultdict(dict))
    # relation pairs, grouped by (source type, relation, target type)
    edges: dict[tuple[str, str, str], set] = field(default_factory=lambda: defaultdict(set))
    # which document each record came from, by vertex type
    doc_links: dict[str, set] = field(default_factory=lambda: defaultdict(set))

    def counts(self) -> dict:
        return {"vertices": {vt: len(v) for vt, v in self.vertices.items()},
                "edges": dict(sum((Counter({e[1]: len(p)}) for e, p in self.edges.items()), Counter())),
                "doc_links": sum(len(v) for v in self.doc_links.values())}


def _name_vertex(rec: Records, name: str) -> str:
    vid = vertex_id(name.strip())[:200]
    if vid:
        rec.vertices[NAME_VT].setdefault(vid, {"name": name.strip()})
    return vid


def build_records(docs: Iterable[dict], schema: Schema) -> Records:
    rec = Records()
    for d in docs:
        doc_vid = vertex_id(d["id"])
        seen_kinds: Counter = Counter()
        for b in find_blocks(d["text"]):
            vt = schema.by_kind.get(b.kind, RECORD_VT)
            rid = doc_vid if seen_kinds[vt] == 0 else f"{doc_vid}_{b.seq}"
            seen_kinds[vt] += 1
            name = next((b.fields[k] for k in NAME_KEYS if b.fields.get(k)), d["title"])
            attrs = {"name": name, "doc_id": d["id"], "doc_title": d["title"]}
            if vt == RECORD_VT:
                attrs.update(kind=b.kind, fields="\n".join(f"{k}: {v}" for k, v in b.fields.items()))
            else:
                spec = schema.vertices[vt]
                grouped: dict[str, list[str]] = defaultdict(list)
                for k, v in b.fields.items():
                    grouped[split_suffix(k)[0]].append(v)
                for key, f in spec.fields.items():
                    vals = grouped.get(key)
                    if not vals:
                        continue
                    attrs[f.attr] = " | ".join(vals)
                    if f.is_number:
                        n = parse_number(vals[0])
                        if n is not None:
                            attrs[f.base + "_num"] = n
                    if f.is_date:
                        iso, year = parse_date(vals[0])
                        if year is not None:
                            attrs[f.base + "_year"] = year
                        if iso:
                            attrs[f.base + "_iso"] = iso
                    if f.is_names:
                        for v in vals:
                            for item in split_names(v):
                                tgt = _name_vertex(rec, item)
                                if tgt:
                                    rec.edges[(vt, f.edge, NAME_VT)].add((rid, tgt))
            rec.vertices[vt][rid] = attrs
            nid = _name_vertex(rec, name)
            if nid:
                rec.edges[(vt, NAMED_EDGE, NAME_VT)].add((rid, nid))
            rec.doc_links[vt].add((doc_vid, rid))
    return rec



def query_guidance(schema: Schema, rec: Records, examples: int = 1, max_len: int = 32) -> str:
    """Write hints for the LLM that turns questions into graph queries.

    The schema alone gives field names but not what the values look like, so the LLM guesses. On this
    corpus it searched games for "2018 winter olympics" when the field holds "2018 Winter", and found
    nothing. Here we show the most common values of every field, taken from the records themselves, and
    say which attributes to use for numbers and years. GraphRAG reads this file as "Domain Hints" for
    its query generators."""
    lines = [
        "These hints describe the typed record vertices built from the structured blocks in the documents.",
        "- Each record has name and doc_title, which hold the full title of the thing it describes. The title often "
        "contains words that the separate fields do not, so when the question names a specific thing, match it with "
        "toLower(r.name) CONTAINS a lowercase keyword, using one keyword per condition rather than the whole phrase.",
        "- Text fields hold short raw values like the examples below. Match them with CONTAINS on short keywords, never "
        "with a long phrase copied from the question.",
        "- For comparisons, counts and rankings on numbers, use the attributes ending in _num (DOUBLE). For years, use "
        "the attributes ending in _year (INT). A value of 0 in these attributes means the value is missing.",
        "- For \"how many\" and \"which had the most\" questions, return one row per matching record with its name, its doc_id and "
        "the attributes involved (for example RETURN r.name AS record_name, r.doc_id AS doc_id, r.x_num AS x_value ORDER BY x_value DESC), "
        "and do not use count(). The rows are counted or ranked afterwards, and they show the evidence.",
        "",
        "Record types, with the most common values of each field:",
    ]
    for v in schema.vertices.values():
        rows = list(rec.vertices.get(v.name, {}).values())
        if not rows:
            continue
        names = [r["name"][:max_len] for r in rows[:examples]]
        lines.append(f"{v.name} ({len(rows)} records), name examples: " + "; ".join(f'"{n}"' for n in names))
        for f in v.fields.values():
            common = Counter(r[f.attr] for r in rows if r.get(f.attr)).most_common(examples)
            if not common:
                continue
            vals = "; ".join(f'"{val[:max_len]}"' for val, _ in common)
            extra = []
            if f.is_number:
                extra.append(f"number in {f.base}_num")
            if f.is_date:
                extra.append(f"year in {f.base}_year")
            if f.is_names:
                extra.append(f"also a {f.edge} edge to Name")
            lines.append(f"  {f.attr}: {vals}" + (f" ({', '.join(extra)})" if extra else ""))
    return "\n".join(lines) + "\n"

def iter_docs(path, fields: dict) -> Iterator[dict]:
    import json
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                o = json.loads(line)
                yield {"id": str(o[fields["id"]]), "title": str(o.get(fields["title"], "")),
                       "text": str(o.get(fields["text"], "")), "url": str(o.get(fields["url"], ""))}
