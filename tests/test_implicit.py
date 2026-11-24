"""Implicit mentions (analysis/implicit.py): cues, rules A/B/C, the pipeline with fake models."""

from datetime import UTC, datetime, timedelta

import pytest

from analysis import implicit as imp
from analysis import pipeline
from analysis.analyze import ImplicitInput, analyze_comments
from analysis.linker import Registry
from analysis.textutil import day_key
from tests.fakes import FakeFirestore, FakeNer, FakeSentiment

REG = Registry.from_yaml()
NOW = datetime.now(UTC)
DAY = day_key(NOW - timedelta(hours=1))
ALL_RULES = frozenset({("A", "*"), ("B", "*"), ("C", "*")})


def kinds(text, top_level=True):
    return [(c.kind, text[c.start : c.end]) for c in imp.cues(text, top_level)]


def test_cues():
    assert kinds("Endlich tritt er zurück") == [("pron-m", "er")]
    assert kinds("Der muss weg.") == [("dem-m", "Der")]
    assert kinds("Der neue Kanzler ist schwach")[0][0] == "noun"
    assert kinds("Das kann sein.") == [("short", "")]  # the verb "sein" is no pronoun
    assert kinds("Das kann sein.", top_level=False) == []
    assert kinds("Sein Rücktritt ist überfällig") == [("pron-m", "Sein")]
    assert kinds("Die App ist Müll")[0] == ("noun", "Die App")
    assert kinds("Ein langer Kommentar ohne jeden Bezug auf irgendwen, der hier mehr als zehn Wörter hat.") == []


def test_gender_and_compatibility():
    merz, weidel, spd, whatsapp = (REG.entities[i] for i in ("friedrich-merz", "alice-weidel", "spd", "whatsapp"))
    assert imp.gender(merz) == "m" and imp.gender(weidel) == "f"
    pron_m = imp.Cue("pron-m", 0, 2)
    assert imp.compatible(pron_m, merz) == "pron-m" and imp.compatible(pron_m, weidel) is None
    assert imp.compatible(imp.Cue("noun", 0, 0, imp._stem("Kanzler")), merz) == "role"
    assert imp.compatible(imp.Cue("noun", 0, 0, imp._stem("Partei")), spd) == "role"
    assert imp.compatible(imp.Cue("noun", 0, 0, imp._stem("App")), whatsapp) in ("role", "kind")
    assert imp.compatible(imp.Cue("noun", 0, 0, imp._stem("App")), merz) is None


def test_titles_hold_for_their_term_only():
    for t in imp.TITLES:
        assert t.entity_id in REG.entities, t
    assert [e for e, *_ in imp.title_mentions("Der Kanzler hat recht", "DE", "2026-11-20")] == ["friedrich-merz"]
    assert [e for e, *_ in imp.title_mentions("Der Kanzler hat recht", "AT", "2026-11-20")] == ["christian-stocker"]
    assert imp.title_mentions("Der Kanzler hat recht", "CH", "2026-11-20") == []
    assert imp.title_mentions("Der Kanzler hat recht", "DE", "2025-01-01") == []  # before Merz took office
    assert imp.title_mentions("Der nächste Kanzler wird es schwer haben", "DE", "2026-11-20") == []
    assert imp.title_mentions("Der Kanzler Scholz war besser", "DE", "2026-11-20") == []  # a name follows
    got = imp.title_mentions("Der Vizekanzler schweigt", "DE", "2026-11-20")
    assert [e for e, *_ in got] == ["lars-klingbeil"]


def ctx(**kw):
    return imp.CommentContext(country="DE", day="2026-11-20", **kw)


def test_rule_a_headline_needs_exactly_one_compatible_target():
    c = imp.find("Endlich tritt er zurück", set(), ctx(title="x", title_entities=["friedrich-merz"]), REG)
    assert [(x.rule, x.via, x.cue, x.entity_id) for x in c] == [("A", "headline", "pron-m", "friedrich-merz")]
    # two men in the headline: "er" is ambiguous
    two = ctx(title="x", title_entities=["friedrich-merz", "markus-soeder"])
    assert imp.find("Endlich tritt er zurück", set(), two, REG) == []
    # a man and a woman: "sie" can only be her
    mixed = ctx(title="x", title_entities=["friedrich-merz", "alice-weidel"])
    assert [x.entity_id for x in imp.find("Sie hat keine Ahnung", set(), mixed, REG)] == ["alice-weidel"]
    # a comment that names an entity gets no implicit one
    assert imp.find("Endlich tritt er zurück", {"spd"}, ctx(title="x", title_entities=["friedrich-merz"]), REG) == []
    # no headline (social pseudo-article): nothing
    assert imp.find("Endlich tritt er zurück", set(), ctx(title_entities=["friedrich-merz"]), REG) == []


def test_rule_a_article_and_rule_b_parent():
    counts = {"friedrich-merz": 4, "spd": 1}
    from collections import Counter

    got = imp.find("Er hat keine Ahnung", set(), ctx(title="x", article_counts=Counter(counts)), REG)
    assert [(x.via, x.entity_id) for x in got] == [("article", "friedrich-merz")]
    reply = ctx(title="x", title_entities=["alice-weidel"], is_reply=True, parent_entities=["friedrich-merz"])
    got = imp.find("Er hat keine Ahnung", set(), reply, REG)
    assert [(x.rule, x.via, x.entity_id) for x in got] == [("B", "parent", "friedrich-merz")]
    # a short reply is a reaction to the parent's opinion, not to an entity
    assert imp.find("Genau so ist es", set(), ctx(title="x", is_reply=True, parent_entities=["spd"]), REG) == []


def test_premise_and_window():
    w = imp.window("Endlich tritt er zurück", 14, 16)
    assert w == "Endlich tritt [er] zurück"
    assert imp.premise(w, "Merz kündigt Rücktritt an") == "Artikel: Merz kündigt Rücktritt an | Kommentar: " + w
    assert imp.premise(w, "") == w
    assert "Antwort auf: Merz" in imp.premise(w, "T", "Merz hat recht")


def comment(text, article="a1", parent=None, source="spiegel", hours_ago=1):
    ts = NOW - timedelta(hours=hours_ago)
    return {
        "articleId": article,
        "sourceId": source,
        "text": text,
        "publishedAt": ts,
        "crawledAt": ts,
        "day": day_key(ts),
        "parentId": parent,
        "analyzed": False,
        "analyzedAt": None,
        "analysisVersion": None,
    }


def test_analyze_comments_writes_implicit_mentions(monkeypatch):
    monkeypatch.setattr(imp, "ACCEPTED", ALL_RULES)
    inp = ImplicitInput(titles={"a1": "Merz kündigt Rücktritt an"})
    cs = [{"id": "c1", **comment("Endlich tritt er zurück, super.")}, {"id": "c2", **comment("Wetter ist gut")}]
    out = analyze_comments(cs, REG, FakeNer(), FakeSentiment(), source_country={"spiegel": "DE"}, implicit=inp)
    (m,) = out["c1"]
    assert (m["entityId"], m["implicit"], m["via"], m["surface"], m["label"]) == (
        "friedrich-merz",
        True,
        "headline",
        "er",
        "pos",
    )
    assert inp.title_entities == {"a1": ["friedrich-merz"]}
    # no ACCEPTED rule: candidates are found but not written
    monkeypatch.setattr(imp, "ACCEPTED", frozenset())
    out = analyze_comments(cs, REG, FakeNer(), FakeSentiment(), source_country={"spiegel": "DE"}, implicit=inp)
    assert out["c1"] == []


@pytest.fixture
def fs(monkeypatch):
    monkeypatch.setattr("analysis.ner.load_ner", lambda name: FakeNer())
    monkeypatch.setattr("analysis.sentiment.load_sentiment", lambda key, **kw: FakeSentiment())
    monkeypatch.setattr(imp, "ACCEPTED", ALL_RULES)
    reg = Registry.from_yaml()
    docs = {f"entities/{e.id}": {**e.metadata(), **pipeline.ZERO_STATS} for e in reg.entities.values()}
    docs["sources/spiegel"] = {"name": "DER SPIEGEL", "country": "DE", "sourceType": "news"}
    docs["sources/bluesky"] = {"name": "Bluesky", "country": "DE", "sourceType": "social"}
    docs["articles/a1"] = {"title": "Merz kündigt Rücktritt an", "sourceId": "spiegel", "sourceType": "news"}
    docs["articles/s1"] = {"title": "Bluesky · 2026-11-20 · Merz", "sourceId": "bluesky", "sourceType": "social"}
    docs["articles/a2"] = {"title": "Wetter", "sourceId": "spiegel"}
    docs["comments/old"] = {**comment("Merz hat versagt.", article="a2", hours_ago=30), "analyzed": True}
    docs["comments/c1"] = comment("Endlich tritt er zurück, super.")
    docs["comments/c2"] = comment("Die SPD ist gut.")
    docs["comments/c3"] = comment("Er hat versagt.", article="a2", parent="old")  # parent from an earlier run
    docs["comments/c4"] = comment("Endlich tritt er zurück.", article="s1", source="bluesky")  # social: nothing
    return FakeFirestore(docs)


def test_pipeline_run_implicit(fs):
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    m = fs.docs["mentions/c1_friedrich-merz"]
    assert m["implicit"] is True and m["via"] == "headline" and m["surface"] == "er"
    r = fs.docs["mentions/c3_friedrich-merz"]
    assert r["via"] == "parent" and r["label"] == "neg"
    assert "mentions/c4_friedrich-merz" not in fs.docs
    assert "implicit" not in fs.docs["mentions/c2_spd"]
    assert out["implicitMentions"] == {"A/pron-m": 1, "B/pron-m": 1}
    # implicit mentions are counted in the main aggregates
    assert fs.docs["entities/friedrich-merz"]["mentions"] == 2
    # privacy: a mention keeps the referring span only
    assert set(m) <= {*pipeline.MENTION_FIELDS, "surface", "confidence", "implicit", "via"}


def test_pipeline_run_without_implicit_reads_no_parents(fs, monkeypatch):
    monkeypatch.setattr(imp, "ACCEPTED", frozenset())
    pipeline.run(fs, limit=100, use_wikidata=False)
    assert not any(m.get("implicit") for p, m in fs.docs.items() if p.startswith("mentions/"))
