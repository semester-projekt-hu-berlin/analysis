"""Comment index: a relink reads only the comments whose words can contain a new alias."""

import random
import re
from datetime import date, timedelta

import pytest

from analysis import pipeline
from analysis import textindex as ti
from analysis.firestore import from_value, to_value
from analysis.linker import Registry
from analysis.textutil import clean_text, today_berlin
from tests.test_budget import _limited, big
from tests.test_pipeline import comment
from tests.test_pipeline import fs as fs

TODAY = date.fromisoformat(today_berlin())


def pattern(aliases):
    alts = sorted({a.casefold() for a in aliases} | {a.lower() for a in aliases}, key=len, reverse=True)
    return alts, re.compile(r"(?<![\w])(" + "|".join(re.escape(a) for a in alts) + r")", re.IGNORECASE)


# --------------------------------------------------------------------------- keys and requirements


def test_keys_are_normalised_word_beginnings():
    assert ti.keys("Die Grünen-Wählerschaft, laut „Süddeutscher Zeitung“!") == {
        "die",
        "grunen",
        "wahlersc",
        "laut",
        "suddeuts",
        "zeitung",
    }
    assert ti.keys("STRAUSS und Strauß") == {"strauss", "und"}
    assert ti.keys("... !!! 😀") == set()


def test_requirement_of_an_alias():
    # a run followed by more of the alias is a complete word, the last run may be the beginning of one
    assert ti.requirement("die welt") == [("die", True), ("welt", False)]
    assert ti.requirement("t-online") == [("t", True), ("online", False)]
    assert ti.requirement("h&m") == [("h", True), ("m", False)]
    assert ti.requirement("bundesverfassungsgericht") == [("bundesve", True)]
    assert ti.requirement("c++") == [("c", True)]
    assert ti.requirement("&&") is None
    assert ti.can_match({"die", "weltweit"}, ti.requirement("die welt"))
    assert not ti.can_match({"diese", "welt"}, ti.requirement("die welt"))
    assert ti.can_match({"lidls"}, ti.requirement("lidl")) and not ti.can_match({"lid"}, ti.requirement("lidl"))


def test_index_never_misses_a_pattern_match():
    """Random texts and aliases over an alphabet with the awkward characters: wherever the relink
    pattern matches, the index selects the comment."""
    rnd = random.Random(7)
    letters = "aAäÄbBeEiIıİnNoOöÖsSßẞſtTuUüÜzZéÉçñ"
    other = "    --..,&_'’:/!?()0123456789№²"
    words = ["Merz", "merz", "MERZ", "Strauß", "STRAUSS", "Weiß", "t-online", "H&M", "Die", "Welt", "İstanbul"]

    def word():
        if rnd.random() < 0.3:
            return rnd.choice(words)
        return "".join(rnd.choice(letters) for _ in range(rnd.randint(1, 12)))

    def text(n):
        return "".join(word() + rnd.choice(other) * rnd.randint(1, 2) for _ in range(n))

    hits = 0
    for _ in range(4000):
        aliases = [text(rnd.randint(1, 3)).strip(" ") for _ in range(3)]
        aliases = [a for a in aliases if len(a) >= 2]
        if not aliases:
            continue
        alts, rx = pattern(aliases)
        m = ti.Matcher(alts)
        t = clean_text(text(rnd.randint(1, 12)) + " " + rnd.choice(aliases).swapcase() + rnd.choice(["", "s", " x"]))
        if rx.search(t):
            hits += 1
            assert not m.complete or m.select([("c", ti.keys(t))]) == ["c"], (aliases, t)
    assert hits > 1000


def test_norm_follows_the_case_insensitive_matching_of_re():
    """Characters that `re.IGNORECASE` treats as equal get the same normal form and are both
    word characters or both not (the index relies on it)."""
    _sre = pytest.importorskip("_sre")
    casefix = pytest.importorskip("re._casefix")
    for cp in range(0x110000):
        c = chr(cp)
        if 0xD800 <= cp <= 0xDFFF:
            continue
        low = _sre.unicode_tolower(cp)
        for other in {low, *casefix._EXTRA_CASES.get(low, ())}:
            d = chr(other)
            if d != c and not {c, d} & ti.UNSAFE:
                assert ti.norm(c) == ti.norm(d), (hex(cp), hex(other))
                assert (c.isalnum() or c == "_") == (d.isalnum() or d == "_"), (hex(cp), hex(other))
    assert ti.requirement("Ιωάννης") is None  # Greek iota: not looked up, the relink scans instead


def test_documents_roundtrip_and_hold_no_text():
    text = "Friedrich Merz redet über die Schuldenbremse."
    docs = ti.index_docs(
        [("c1", "2026-11-09", text), ("c2", "2026-11-09", "Lidl lohnt sich"), ("c3", "2026-11-10", "x")]
    )
    assert len(docs) == 2 and all(p.startswith("comment_index/2026-11-") for p in docs)
    doc = next(d for d in docs.values() if d["day"] == "2026-11-09")
    assert doc["n"] == 2 and isinstance(doc["blob"], bytes)
    entries = dict(ti.decode(doc["blob"]))
    assert entries["c2"] == {"lidl", "lohnt", "sich"}
    assert "schulden" in entries["c1"] and "Schuldenbremse" not in str(entries)  # prefixes, no order, no text
    # the same batch gives the same document (a repeated batch overwrites it)
    assert ti.index_docs([("c2", "2026-11-09", "Lidl lohnt sich"), ("c1", "2026-11-09", text)]).keys() <= docs.keys()
    # bytes survive the REST encoding
    assert ti.decode(from_value(to_value(doc["blob"]))) == ti.decode(doc["blob"])


# --------------------------------------------------------------------------- pipeline


def _corpus(fs, monkeypatch, future=8, past=4, noise=30):
    """Comments about Lidl before and after the start of the index, analysed without the entity."""
    for i in range(past):
        fs.docs[f"comments/p{i:02d}"] = comment(f"p{i:02d}", f"Bei Lidl ist es gut, sagt Merz {i}.", hours_ago=30 + i)
    for i in range(future):  # published after `indexSince` (today + 2 days)
        fs.docs[f"comments/f{i:02d}"] = comment(f"f{i:02d}", f"Lidl hat versagt, Nummer {i}.", hours_ago=-72 - i)
    for i in range(noise):
        fs.docs[f"comments/n{i:02d}"] = comment(f"n{i:02d}", f"Die SPD redet viel, Nummer {i}.", hours_ago=-72 - i)
    real = Registry.from_yaml.__func__
    monkeypatch.setattr(Registry, "from_yaml", classmethod(lambda cls, *a, **kw: real(cls, *a, generated=None)))
    pipeline.run(fs, limit=1000, batch=10, use_wikidata=False, budget=big(fs))
    monkeypatch.setattr(Registry, "from_yaml", classmethod(real))
    assert not any(p.endswith("_lidl") for p in fs.docs)


def test_run_writes_the_index_before_it_marks_the_comments(fs, monkeypatch):
    _corpus(fs, monkeypatch)
    since = (TODAY + timedelta(days=2)).isoformat()
    assert fs.docs["meta/analysis"]["indexSince"] == since
    idx = {p: d for p, d in fs.docs.items() if p.startswith("comment_index/")}
    indexed = {cid for d in idx.values() for cid, _ in ti.decode(d["blob"])}
    assert indexed == {p.split("/")[1] for p in fs.docs if p.startswith("comments/")}
    assert sum(d["n"] for d in idx.values()) == len(indexed)
    # order of the commits: the index document of a comment comes before its analyzed flag
    pos = {}
    for n, commit in enumerate(fs.commits):
        for w in commit:
            if w.path.startswith("comment_index/"):
                for cid, _ in ti.decode(w.data["blob"]):
                    pos.setdefault(("index", cid), n)
            elif w.path.startswith("comments/"):
                pos.setdefault(("flag", w.path.split("/")[1]), n)
    assert all(pos[("index", cid)] < pos[("flag", cid)] for cid in indexed)
    # the start of the index is written once
    pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    assert fs.docs["meta/analysis"]["indexSince"] == since


def test_relink_reads_only_candidates_of_indexed_days(fs, monkeypatch):
    _corpus(fs, monkeypatch)
    mark = fs.ops_mark()
    out = pipeline.relink(fs, days=14, entity_ids={"lidl"}, budget=big(fs))
    assert out["done"] is True and out["newMentions"] == 12
    assert fs.docs["entities/lidl"]["mentions"] == 12
    reads = fs.ops_summary(mark)["reads"]
    past = sum(
        1 for p, d in fs.docs.items() if p.startswith("comments/") and d["day"] < fs.docs["meta/analysis"]["indexSince"]
    )
    # days before the index: every comment is read; after: the 8 that contain the alias, not the 30 others
    assert reads["comments"] == past + 8
    assert reads["comment_index"] == sum(
        1
        for p in fs.docs
        if p.startswith("comment_index/f") is False
        and p[:14] == "comment_index/"
        and fs.docs[p]["day"] >= fs.docs["meta/analysis"]["indexSince"]
    )
    assert out["scanned"] == sum(1 for p in fs.docs if p.startswith("comments/"))
    assert "relink" not in fs.docs["meta/analysis"].get("jobs", {})


def test_relink_with_index_equals_relink_by_scan(fs, monkeypatch):
    import copy

    _corpus(fs, monkeypatch)
    base = copy.deepcopy(fs.docs)
    pipeline.relink(fs, days=14, entity_ids={"lidl"}, budget=big(fs))
    with_index = {p: d for p, d in fs.docs.items() if p.startswith(("mentions/", "entities/lidl"))}
    fs2 = type(fs)(base)
    monkeypatch.setenv("ANALYSIS_TEXT_INDEX", "0")
    out = pipeline.relink(fs2, days=14, entity_ids={"lidl"}, budget=big(fs2))
    assert out["done"] is True and out["newMentions"] == 12
    by_scan = {p: d for p, d in fs2.docs.items() if p.startswith(("mentions/", "entities/lidl"))}

    def strip(docs):
        return {p: {k: v for k, v in d.items() if k != "updatedAt"} for p, d in docs.items()}

    assert strip(with_index) == strip(by_scan)


def test_relink_resumes_inside_the_index(fs, monkeypatch):
    _corpus(fs, monkeypatch)
    # small steps: one index document at a time, and a budget that ends in the index phase
    out = pipeline.relink(fs, days=14, entity_ids={"lidl"}, page=4, budget=_limited(fs, run_writes=40, reserve=0))
    job = fs.docs["meta/analysis"]["jobs"]["relink"]
    assert out["done"] is False and job["scanDone"] is True and job["icursor"]
    partial = sum(1 for p in fs.docs if p.endswith("_lidl"))
    assert 4 <= partial < 12 and fs.docs["entities/lidl"]["mentions"] == partial
    out = pipeline.relink(fs, days=14, page=4, budget=big(fs))
    assert out["done"] is True
    assert sum(1 for p in fs.docs if p.endswith("_lidl")) == 12 == fs.docs["entities/lidl"]["mentions"]


def test_job_started_before_the_index_keeps_scanning_its_days(fs, monkeypatch):
    _corpus(fs, monkeypatch)
    # a job as the previous version stored it: a scan cursor, nothing about the index
    first = sorted((d["day"], p.split("/")[1]) for p, d in fs.docs.items() if p.startswith("comments/"))[1]
    pipeline.queue_job(fs, "relink", {"cutoff": "2000-01-01", "entityIds": ["lidl"]})
    fs.docs["meta/analysis"]["jobs"]["relink"]["cursor"] = list(first)
    out = pipeline.relink(fs, budget=big(fs))
    assert out["done"] is True
    # everything after the cursor is found: by the scan before `indexSince`, through the index after it
    want = {
        p.split("/")[1]
        for p, d in fs.docs.items()
        if p.startswith("comments/") and "Lidl" in d["text"] and (d["day"], p.split("/")[1]) > first
    }
    assert {p.split("/")[1].split("_")[0] for p in fs.docs if p.endswith("_lidl")} == want


def test_index_reset_makes_a_running_job_scan_the_days_again(fs, monkeypatch):
    _corpus(fs, monkeypatch)
    pipeline.relink(fs, days=14, entity_ids={"lidl"}, page=4, budget=_limited(fs, run_writes=40, reserve=0))
    assert fs.docs["meta/analysis"]["jobs"]["relink"]["scanDone"] is True
    done_before = sum(1 for p in fs.docs if p.endswith("_lidl"))
    # the index is switched off: a run removes `indexSince`, the job falls back to the scan
    monkeypatch.setenv("ANALYSIS_TEXT_INDEX", "0")
    fs.docs["comments/z1"] = comment("z1", "Noch ein Kommentar über Merz.")
    out = pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    assert "indexSince" not in fs.docs["meta/analysis"]
    assert out["jobs"]["relink"]["done"] is True
    assert sum(1 for p in fs.docs if p.endswith("_lidl")) == 12 > done_before


def test_alias_without_a_word_run_falls_back_to_the_scan(fs, monkeypatch):
    _corpus(fs, monkeypatch)
    monkeypatch.setattr(ti.Matcher, "complete", False, raising=False)
    monkeypatch.setattr(ti.Matcher, "__init__", lambda self, aliases: setattr(self, "reqs", []))
    mark = fs.ops_mark()
    out = pipeline.relink(fs, days=14, entity_ids={"lidl"}, budget=big(fs))
    assert out["done"] is True and out["newMentions"] == 12
    reads = fs.ops_summary(mark)["reads"]
    assert reads["comments"] == sum(1 for p in fs.docs if p.startswith("comments/")) and "comment_index" not in reads


def test_old_index_documents_are_deleted(fs):
    old = (TODAY - timedelta(days=pipeline.INDEX_KEEP_DAYS + 1)).isoformat()
    keep = (TODAY - timedelta(days=pipeline.INDEX_KEEP_DAYS - 1)).isoformat()
    fs.docs[f"comment_index/{old}_a"] = {"day": old, "n": 1, "blob": ti.encode([("x", {"a"})])}
    fs.docs[f"comment_index/{keep}_a"] = {"day": keep, "n": 1, "blob": ti.encode([("y", {"a"})])}
    pipeline.run(fs, limit=100, use_wikidata=False)
    assert f"comment_index/{old}_a" not in fs.docs and f"comment_index/{keep}_a" in fs.docs


def test_failing_index_write_does_not_stop_the_run(fs, monkeypatch):
    pipeline.run(fs, limit=100, use_wikidata=False)
    assert fs.docs["meta/analysis"]["indexSince"]
    real = fs.commit

    def commit(writes):
        if any(w.path.startswith("comment_index/") for w in writes):
            raise RuntimeError("Firestore 400: INVALID_ARGUMENT")
        return real(writes)

    monkeypatch.setattr(fs, "commit", commit)
    fs.docs["comments/z1"] = comment("z1", "Merz ist gut.")
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    assert out["comments"] == 1 and fs.docs["comments/z1"]["analyzed"] is True
    assert "mentions/z1_friedrich-merz" in fs.docs
    # the index has a gap now: its start is removed, relinks scan until a later run sets it again
    assert "indexSince" not in fs.docs["meta/analysis"]
