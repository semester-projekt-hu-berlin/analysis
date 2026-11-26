"""Topic words ("Worüber gesprochen wird"): extraction, day docs, privacy thresholds, lift, backfill."""

import math
import re
from datetime import UTC, date, datetime, timedelta

import pytest

from analysis import aggregate as agg
from analysis import pipeline
from analysis import topics as tp
from analysis.analyze import analyze_comments
from analysis.linker import Registry
from analysis.textutil import day_key
from tests.fakes import FakeNer, FakeSentiment
from tests.test_budget import big
from tests.test_pipeline import comment
from tests.test_pipeline import fs as fs

REG = Registry.from_yaml()
NOW = datetime.now(UTC)


def terms_of(text, entity="friedrich-merz", ner=None):
    c = {"id": "c", "text": text, "sourceId": "spiegel", "articleId": "a", "publishedAt": NOW}
    out = analyze_comments([c], REG, FakeNer(ner or {}), FakeSentiment(), source_country={"spiegel": "DE"})["c"]
    by = {m["entityId"]: m for m in out}
    assert entity in by, out
    return by[entity]["terms"]


# ---------------------------------------------------------------------------- extraction


def test_noun_in_the_mention_sentence():
    assert terms_of("Die Rente wird von Merz kaputt gemacht.") == ["Rente"]


def test_sentence_initial_words_are_not_terms():
    assert terms_of("Merz redet. Wehrpflicht ist sein Ding, die Rente auch.") == ["Rente"]
    assert terms_of("Merz sagt: Steuern runter, aber die Rente bleibt.") == ["Rente"]  # after a colon


def test_entities_aliases_and_person_names_are_not_terms():
    # linked entities (SPD, Grüne) and an NER person span (a private person) are left out
    got = terms_of(
        "Merz streitet mit der SPD, den Grünen und Mösenbacher über die Schuldenbremse.", ner={"Mösenbacher": "PER"}
    )
    assert got == ["Schuldenbremse"]
    # names the NER missed: after a form of address or a known first name
    assert terms_of("Merz traf Klaus Meier und Frau Hinterhuber wegen der Rente.") == ["Rente"]
    is_name = tp.name_check(REG)
    assert is_name("Scholz") and is_name("Trumps")  # an alias, also inflected
    assert is_name("Klaus")  # first name of a known person: private persons' first names too
    assert not is_name("Rente") and not is_name("Wehrpflicht")


def test_digits_urls_handles_and_all_caps_are_dropped():
    got = terms_of("Merz und die CO2-Steuer, siehe https://x.de/Rentenreform @Wehrpflicht. Er sagt NEIN zur Rente.")
    assert got == ["Rente"]


def test_hyphenated_compounds_and_hashtags():
    assert terms_of("Merz will das Verbrenner-Aus kippen.") == ["Verbrenner-Aus"]
    assert terms_of("Merz und #Bürgergeld") == ["Bürgergeld"]  # the hashtag as a word


def test_window_is_the_sentence_plus_minus_one():
    got = terms_of("Merz redet. Das ist gut so. Noch ein Satz zur Wehrpflicht. Hier steht die Rente.")
    assert got == []
    assert terms_of("Das Bürgergeld ist teuer. Merz redet. Die Wehrpflicht kommt.") == ["Bürgergeld", "Wehrpflicht"]


def test_stopwords_inflection_and_limit():
    assert terms_of("Merz hat keine Ahnung von der Rente, die Renten sinken, sagt Herr Müller.") == ["Rente"]
    text = (
        "Merz spricht über Rente, Steuern, Migration, Wehrpflicht, Bürgergeld, Schuldenbremse, Mindestlohn"
        " und Heizungsgesetz."
    )
    got = terms_of(text)
    assert len(got) == tp.MAX_TERMS_PER_MENTION and got == sorted(got)
    assert "Rente" in got and "Heizungsgesetz" not in got  # the nearest words win


def test_terms_are_single_words():
    got = terms_of("Merz und die Rente. Die Wehrpflicht-Debatte und der Mindestlohn sind wichtig, sagt Merz heute.")
    assert got and all(re.fullmatch(r"[^\W\d_]+(?:-[^\W\d_]+)*", t) for t in got)


# ---------------------------------------------------------------------------- day docs


def _m(cid, terms, src="spiegel", score=-0.5, day="2026-11-20", eid="friedrich-merz"):
    return {
        "commentId": cid,
        "entityId": eid,
        "sourceId": src,
        "day": day,
        "label": "neg" if score < 0 else "pos",
        "score": score,
        "publishedAt": NOW,
        "coEntityIds": [],
        "terms": terms,
    }


def test_day_doc_terms_shape_and_caps():
    ms = [_m(f"c{i}", ["Rente"], src=f"s{i}") for i in range(5)] + [_m("x", ["Wehrpflicht"], score=0.5)]
    ms += [_m(f"y{i}", [f"Wort{chr(97 + i % 26)}{chr(97 + i // 26)}"]) for i in range(60)]
    doc = agg.build_day_docs(ms)[("friedrich-merz", "2026-11-20")]
    assert doc["terms"]["Rente"] == [5, -2.5, "s0", "s1", "s2"]  # count, scoreSum, up to 3 sources
    assert doc["terms"]["Wehrpflicht"] == [1, 0.5, "spiegel"]
    assert len(doc["terms"]) == tp.TERMS_PER_DAY
    # without terms, a day doc looks as before (no empty field, unchanged docs are not rewritten)
    plain = agg.build_day_docs([{**m, "terms": []} for m in ms[:2]])[("friedrich-merz", "2026-11-20")]
    assert "terms" not in plain
    assert agg.same_day_doc(dict(plain), plain)
    assert not agg.same_day_doc(plain, doc)


# ---------------------------------------------------------------------------- topics7d

TODAY = "2026-11-26"


def _entity(days: dict[str, dict], mentions_per_day: int = 20) -> dict:
    """An entity doc with `mentions_per_day` mentions on each day of `days` ({day: terms map})."""
    return {
        "allowlisted": True,
        "recentDays": {d: {"mentions": mentions_per_day, "scoreSum": 0.0} for d in days},
        "recentTerms": days,
    }


def _day(k: int) -> str:
    return (date.fromisoformat(TODAY) - timedelta(days=k)).isoformat()


def _world(merz_terms: dict, other_terms: dict | None = None) -> dict:
    other = {_day(k): other_terms or {"Wetter": [10, 0.0, "a", "b"]} for k in range(7)}
    return {"friedrich-merz": _entity({_day(0): merz_terms}), "spd": _entity(other, 200)}


def _topics(world, eid="friedrich-merz"):
    g, total = tp.global_terms(world, TODAY)
    return tp.topics(world[eid], g, total, TODAY)


def test_privacy_thresholds():
    # 5 mentions from 2 sources: published; 4 mentions, or one source only: not
    got = _topics(_world({"Rente": [5, -4.0, "spiegel", "zeit"]}))
    assert [t["term"] for t in got] == ["Rente"]
    assert _topics(_world({"Rente": [4, -4.0, "spiegel", "zeit"]})) == []
    assert _topics(_world({"Rente": [9, -4.0, "spiegel"]})) == []
    # sources are counted over the days: one source each day, two different ones in the week
    world = _world({"Rente": [3, -1.0, "spiegel"]})
    world["friedrich-merz"]["recentTerms"][_day(1)] = {"Rente": [3, -1.0, "zeit"]}
    world["friedrich-merz"]["recentDays"][_day(1)] = {"mentions": 20, "scoreSum": 0.0}
    assert [t["term"] for t in _topics(world)] == ["Rente"]


def test_lift_and_ranking():
    # "Wetter" is as frequent for Merz as for everybody: lift ~1, not a topic
    world = _world(
        {"Rente": [12, -9.6, "a", "b"], "Wetter": [5, 0.0, "a", "b"], "Wehrpflicht": [6, -1.2, "a", "c"]},
        other_terms={"Wetter": [10, 0.0, "a", "b"], "Rente": [2, 0.0, "a", "b"]},
    )
    world["friedrich-merz"]["recentDays"][_day(0)]["mentions"] = 100
    got = _topics(world)
    assert [t["term"] for t in got] == ["Rente", "Wehrpflicht"]
    r = got[0]
    assert set(r) == {"term", "mentions", "share", "avgSentiment", "lift", "prev"}
    assert r["mentions"] == 12 and r["share"] == 0.12 and r["avgSentiment"] == -0.8 and r["prev"] == 0
    total = 100 + 7 * 200
    assert r["lift"] == round((12 / 100) / ((12 + 14) / total), 1)
    assert got[1]["lift"] == 15.0  # (6 / 100) / (6 / 1500)
    world["spd"]["recentTerms"] = {_day(k): {"Wehrpflicht": [100, 0.0, "a", "b"]} for k in range(7)}
    assert "Wehrpflicht" not in [t["term"] for t in _topics(world)]  # everybody talks about it


def test_prev_window_inflection_and_top10():
    terms = {f"Thema{chr(97 + i)}": [5 + i, -1.0, "a", "b"] for i in range(14)}
    world = _world({**terms, "Rente": [3, -1.5, "a"], "Renten": [3, -1.5, "b"]})
    world["friedrich-merz"]["recentDays"][_day(0)]["mentions"] = 100
    world["friedrich-merz"]["recentTerms"][_day(9)] = {"Rente": [4, 0.0, "a"]}
    world["friedrich-merz"]["recentDays"][_day(9)] = {"mentions": 4, "scoreSum": 0.0}
    got = _topics(world)
    assert len(got) == tp.TOPICS_TOP
    assert [t["mentions"] for t in got] == list(range(18, 8, -1))  # equal lift: the most mentioned
    assert got == sorted(got, key=lambda t: -t["mentions"] * math.log(t["lift"]))
    rente = tp.topics(
        world["friedrich-merz"], *tp.global_terms({"friedrich-merz": world["friedrich-merz"]}, TODAY), TODAY
    )
    assert rente == []  # alone in the world: lift 1
    world["friedrich-merz"]["recentTerms"][_day(0)] = {"Rente": [3, -1.5, "a"], "Renten": [3, -1.5, "b"]}
    got = _topics(world)
    assert got[0]["term"] == "Rente" and got[0]["mentions"] == 6 and got[0]["prev"] == 4


def test_too_few_mentions_give_no_topics():
    world = _world({"Rente": [5, -4.0, "a", "b"]})
    world["friedrich-merz"]["recentDays"] = {_day(0): {"mentions": 4, "scoreSum": 0.0}}
    assert _topics(world) == []


# ---------------------------------------------------------------------------- pipeline


def _rente_comments(fs, n=6, sources=("spiegel", "derstandard")):
    for i in range(n):
        cid = f"r{i}"
        fs.docs[f"comments/{cid}"] = comment(cid, f"Merz und die Rente, Teil {i}.", source=sources[i % len(sources)])
    for i in range(20):  # background: other entities, other words
        cid = f"b{i}"
        fs.docs[f"comments/{cid}"] = comment(cid, f"Die SPD und das Wetter, Teil {i}.")


def _published(fs, eid="friedrich-merz"):
    return fs.docs[f"entities/{eid}"].get("topics7d") or []


def test_run_publishes_topics_and_stores_no_text(fs):
    _rente_comments(fs)
    pipeline.run(fs, limit=1000, use_wikidata=False)
    top = _published(fs)
    assert top and top[0]["term"] == "Rente" and top[0]["mentions"] == 6 and top[0]["prev"] == 0
    assert top[0]["share"] == pytest.approx(6 / 8)  # Merz has 8 mentions (c1, c2 of the fixture)
    day = day_key(NOW - timedelta(hours=1))
    assert fs.docs[f"entities/friedrich-merz/days/{day}"]["terms"]["Rente"][0] == 6
    assert "Rente" in fs.docs["entities/friedrich-merz"]["recentTerms"][day]
    assert _published(fs, "spd") == [] or all(t["term"] != "Rente" for t in _published(fs, "spd"))
    for p, d in fs.docs.items():
        if p.startswith("mentions/"):
            assert "text" not in d and all(" " not in t for t in d["terms"])


def test_one_source_is_not_enough(fs):
    _rente_comments(fs, n=8, sources=("spiegel",))
    pipeline.run(fs, limit=1000, use_wikidata=False)
    assert all(t["term"] != "Rente" for t in _published(fs))


def test_topics_are_idempotent_and_rebuilt_by_reaggregate(fs):
    _rente_comments(fs)
    pipeline.run(fs, limit=1000, use_wikidata=False)
    first = {p: d for p, d in fs.docs.items() if p.startswith("entities/friedrich-merz")}
    for p in list(fs.docs):
        if p.startswith("comments/"):
            fs.docs[p].update(analyzed=False, analysisVersion=None)
    mark = fs.ops_mark()
    pipeline.run(fs, limit=1000, use_wikidata=False)
    # the re-run writes each mention once (as before), no extra writes for the topic words
    assert fs.ops_summary(mark)["writes"]["mentions"] == sum(1 for p in fs.docs if p.startswith("mentions/"))
    pipeline.reaggregate(fs)
    for p, d in first.items():
        if "/days/" in p:
            assert fs.docs[p]["terms"] == d["terms"]
    assert fs.docs["entities/friedrich-merz"]["topics7d"] == first["entities/friedrich-merz"]["topics7d"]


def _as_before_the_feature(fs):
    """Mentions, day docs and entity docs as the analysis left them before topic words existed."""
    for p, d in fs.docs.items():
        if p.startswith("mentions/"):
            d.pop("terms", None)
        if "/days/" in p:
            d.pop("terms", None)
        if p.startswith("entities/") and "/" not in p[len("entities/") :]:
            d.pop("recentTerms", None)
            d.pop("topics7d", None)
    fs.docs["meta/analysis"].pop("topics", None)


def test_first_run_queues_the_backfill_and_run_continues_it(fs):
    _rente_comments(fs)
    pipeline.run(fs, limit=1000, use_wikidata=False)
    want = _published(fs)
    _as_before_the_feature(fs)
    fs.docs["comments/new"] = comment("new", "Nichts Relevantes hier.")
    out = pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    res = out["jobs"]["topics"]
    assert res["done"] and res["entities"] >= 2
    assert "topics" not in fs.docs["meta/analysis"].get("jobs", {})
    assert fs.docs["meta/analysis"]["topics"]["backfilledAt"]
    assert _published(fs) == want
    assert fs.docs["mentions/r0_friedrich-merz"]["terms"] == ["Rente"]
    # a later run does not queue it again
    out = pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    assert "topics" not in (out.get("jobs") or {})


def test_backfill_is_budget_capped_and_resumable(fs):
    _rente_comments(fs)
    # comments naming Merz and the SPD: Merz's mention gets its terms in the SPD's step
    for i in range(3):
        fs.docs[f"comments/m{i}"] = comment(f"m{i}", f"Merz und die SPD streiten über die Rente, Teil {i}.")
    pipeline.run(fs, limit=1000, use_wikidata=False)
    want = _published(fs)
    _as_before_the_feature(fs)
    pipeline.queue_job(fs, "topics", {"days": 7})
    budget = big(fs)
    calls = []
    budget.allows = lambda writes=0, reads=0: calls.append(writes) or len(calls) == 1  # one entity only
    first = pipeline.topics_backfill(fs, budget=budget)
    assert not first["done"] and "budget" in first["stopped"] and first["next"] == 1
    job = fs.docs["meta/analysis"]["jobs"]["topics"]
    assert job["order"][:2] == ["spd", "friedrich-merz"] and job["next"] == 1  # most mentions first
    assert fs.docs["mentions/m0_friedrich-merz"]["terms"] == ["Rente"]  # written with the SPD's step
    assert "terms" not in fs.docs["mentions/r0_friedrich-merz"]
    mark = fs.ops_mark()
    second = pipeline.topics_backfill(fs, budget=big(fs))
    assert second["done"] and "topics" not in fs.docs["meta/analysis"]["jobs"]
    # only Merz's comments without terms yet are read (c1, c2, r0-r5), not the shared m0-m2
    assert fs.ops_summary(mark)["reads"]["comments"] == 8
    assert _published(fs) == want


def test_backfill_skips_a_stale_co_mention(fs, monkeypatch):
    fs.docs["comments/m0"] = comment("m0", "Merz und die SPD streiten über die Rente.")
    for i in range(6):
        fs.docs[f"comments/s{i}"] = comment(f"s{i}", f"Die SPD und Merz, Teil {i}.")
        fs.docs[f"comments/t{i}"] = comment(f"t{i}", f"Die SPD, Teil {i}.")
    pipeline.run(fs, limit=1000, use_wikidata=False)
    _as_before_the_feature(fs)
    del fs.docs["mentions/m0_friedrich-merz"]  # coEntityIds of m0_spd still names it
    pipeline.queue_job(fs, "topics", {"days": 7})
    calls = []
    real = pipeline._commit_terms
    monkeypatch.setattr(pipeline, "_commit_terms", lambda f, groups: calls.append(len(groups)) or real(f, groups))
    out = pipeline.topics_backfill(fs, budget=big(fs))
    assert out["done"] and fs.docs["meta/analysis"].get("jobs", {}).get("topics") is None
    assert out["entities"] == 2 and out["written"] == 1 and calls[0] == 2  # SPD first: its own + the stale one
    assert fs.docs["mentions/m0_spd"]["terms"] == ["Rente"]
    assert "mentions/m0_friedrich-merz" not in fs.docs  # not created by the masked write
