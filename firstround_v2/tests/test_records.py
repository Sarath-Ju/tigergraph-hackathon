"""Unit tests for the structured-record layer. They need no database.

    python -m pytest tests -q
"""
from agr2.records.blocks import find_blocks, parse_date, parse_number, split_names, split_suffix
from agr2.records.schema import NAME_VT, NAMED_EDGE, build_records, infer_schema, query_guidance

FILM = """[Infobox film]
  name: Jab We Met
  director: Imtiaz Ali
  starring: Kareena KapoorShahid Kapoor
  released: 2007-10-26
  runtime: 142 minutes
  budget: 15 crore
[Infobox album]
  name: Jab We Met
  artist: Pritam

Jab We Met is a 2007 film."""


def test_find_blocks_stops_at_prose():
    b = find_blocks(FILM)
    assert [x.kind for x in b] == ["Infobox film", "Infobox album"]
    assert b[0].fields["director"] == "Imtiaz Ali" and "artist" not in b[0].fields


def test_numbers():
    assert parse_number("24") == 24
    assert parse_number("$13 million") == 13e6
    assert parse_number("142 minutes") == 142
    assert parse_number("15 crore") == 1.5e8
    assert parse_number("3:09.646") is None
    assert parse_number("Imtiaz Ali") is None


def test_dates():
    assert parse_date("2007-10-26") == ("2007-10-26", 2007)
    assert parse_date("21 September 2007 (India)") == ("2007-09-21", 2007)
    assert parse_date("March 2, 2017") == ("2017-03-02", 2017)
    assert parse_date("6 to 8 August") == ("", None)


def test_split_names():
    assert split_names("Kareena KapoorShahid Kapoor") == ["Kareena Kapoor", "Shahid Kapoor"]
    assert split_names("Jeremy Irons, Geneviève Bujold") == ["Jeremy Irons", "Geneviève Bujold"]
    assert split_names("Leonardo DiCaprio") == ["Leonardo DiCaprio"]
    assert split_names("Ronald McDonald") == ["Ronald McDonald"]
    assert split_names("Viktor Zubkov (acting), Dmitry Medvedev") == ["Viktor Zubkov", "Dmitry Medvedev"]


def test_suffix():
    assert split_suffix("term_start2") == ("term_start", 2)
    assert split_suffix("office") == ("office", None)
    print()

def test_schema_and_records():
    docs = [{"id": f"Q{i}", "title": f"Film {i}", "text": FILM.replace("Jab We Met", f"Film {i}")} for i in range(6)]
    s = infer_schema(docs, min_docs=5, min_coverage=0.3, reserved=set())
    assert set(s.vertices) == {"Film", "Album"}
    f = s.vertices["Film"].fields
    assert f["director"].is_names and f["runtime"].is_number and f["released"].is_date
    assert "ADD DIRECTED EDGE DIRECTOR(FROM Film, TO Name);" in s.gsql()
    r = build_records(docs, s)
    film = r.vertices["Film"]["q0"]
    assert film["runtime_num"] == 142 and film["released_year"] == 2007
    assert ("q0", "kareena_kapoor") in r.edges[("Film", "STARRING", NAME_VT)]
    assert ("q0", "film_0") in r.edges[("Film", NAMED_EDGE, NAME_VT)]
    assert ("q0", "q0") in r.doc_links["Film"]


def test_query_guidance():
    docs = [{"id": f"Q{i}", "title": f"Film {i}", "text": FILM.replace("Jab We Met", f"Film {i}")} for i in range(6)]
    s = infer_schema(docs, min_docs=5, min_coverage=0.3, reserved=set())
    g = query_guidance(s, build_records(docs, s))
    assert "Film (6 records)" in g
    assert 'runtime: "142 minutes" (number in runtime_num)' in g
    assert 'director: "Imtiaz Ali" (also a DIRECTOR edge to Name)' in g
    # GraphRAG strips single-brace placeholders from prompt files, so the hints must not contain any
    assert "{" not in g and "}" not in g
