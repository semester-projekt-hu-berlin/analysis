"""End-to-end pipeline on an in-memory Firestore: idempotency and privacy."""

from datetime import UTC, datetime, timedelta

import pytest

from analysis import pipeline
from analysis.linker import Registry
from analysis.textutil import day_key
from tests.fakes import FakeFirestore, FakeNer, FakeSentiment

NOW = datetime.now(UTC)


def comment(cid, text, hours_ago=1, source="spiegel", article="a1"):
    ts = NOW - timedelta(hours=hours_ago)
    return {
        "articleId": article,
        "sourceId": source,
        "text": text,
        "publishedAt": ts,
        "crawledAt": ts,
        "day": day_key(ts),
        "parentId": None,
        "upvotes": 3,
        "analyzed": False,
        "analyzedAt": None,
        "analysisVersion": None,
    }


@pytest.fixture
def fs(monkeypatch):
    monkeypatch.setattr("analysis.ner.load_ner", lambda name: FakeNer({"Klaus Müller": "PER"}))
    monkeypatch.setattr("analysis.sentiment.SentimentModel", lambda key: FakeSentiment())
    reg = Registry.from_yaml()
    docs = {f"entities/{e.id}": {**e.metadata(), **pipeline.ZERO_STATS} for e in reg.entities.values()}
    docs["sources/spiegel"] = {"name": "DER SPIEGEL", "country": "DE"}
    docs["sources/derstandard"] = {"name": "DER STANDARD", "country": "AT"}
    docs["articles/a1"] = {"title": "Koalitionsstreit in Berlin", "sourceId": "spiegel"}
    docs["comments/c1"] = comment("c1", "Merz macht das gut. Die SPD hat versagt.")
    docs["comments/c2"] = comment("c2", "Kanzler Merz ist eine Katastrophe, Klaus Müller sagt das auch.")
    docs["comments/c3"] = comment("c3", "Die Grünen sind super.", source="derstandard")
    docs["comments/c4"] = comment("c4", "Nichts Relevantes hier.")
    return FakeFirestore(docs)


def snapshot(fs):
    keep = ("entities/", "mentions/", "meta/trending")
    out = {}
    for p, d in fs.docs.items():
        if p.startswith(keep):
            d = {k: v for k, v in d.items() if k not in ("updatedAt",)}
            out[p] = d
    return out


def test_run_aggregates_and_marks_comments(fs):
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    assert out["comments"] == 4
    assert all(fs.docs[f"comments/c{i}"]["analyzed"] is True for i in range(1, 5))
    assert fs.docs["comments/c1"]["analysisVersion"].startswith("fake-ner@1+fake-sent@1")
    merz = fs.docs["entities/friedrich-merz"]
    assert merz["mentions"] == 2 and merz["pos"] == 1 and merz["neg"] == 1
    assert merz["mentions7d"] == 2
    day = fs.docs[f"entities/friedrich-merz/days/{day_key(NOW - timedelta(hours=1))}"]
    assert day["mentions"] == 2 and day["coMentions"] == {"spd": 1}
    # target-dependent window: "Merz macht das gut" vs "Die SPD hat versagt"
    assert fs.docs["mentions/c1_friedrich-merz"]["label"] == "pos"
    assert fs.docs["mentions/c1_spd"]["label"] == "neg"
    # country disambiguation
    assert "mentions/c3_gruene-at" in fs.docs and "mentions/c3_gruene" not in fs.docs
    trending = fs.docs["meta/trending"]
    assert trending["top"][0]["entityId"] == "friedrich-merz"
    assert fs.docs["meta/stats"]["mentions"] == 4


def test_privacy_mentions_store_only_surface(fs):
    pipeline.run(fs, limit=100, use_wikidata=False)
    for p, d in fs.docs.items():
        if p.startswith("mentions/"):
            assert set(d) == {
                "commentId",
                "articleId",
                "sourceId",
                "entityId",
                "day",
                "publishedAt",
                "surface",
                "label",
                "score",
                "confidence",
            }
            assert len(d["surface"]) <= 40
    # the private person never becomes an entity
    assert not any("mueller" in p or "muller" in p for p in fs.docs)
    for p, d in fs.docs.items():
        if p.startswith(("entities/", "meta/")):
            assert "Klaus" not in str(d)
            assert "Katastrophe" not in str(d)


def test_rerun_is_idempotent(fs):
    pipeline.run(fs, limit=100, use_wikidata=False)
    first = snapshot(fs)
    # re-run on the same comments (e.g. after a crash before they were marked analyzed)
    for i in range(1, 5):
        fs.docs[f"comments/c{i}"]["analyzed"] = False
        fs.docs[f"comments/c{i}"]["analysisVersion"] = None
    pipeline.run(fs, limit=100, use_wikidata=False)
    assert snapshot(fs) == first
    # and reaggregate from scratch gives the same counters
    pipeline.reaggregate(fs)
    second = snapshot(fs)
    for p in first:
        if p.startswith("entities/") and "/days/" not in p:
            for k in ("mentions", "pos", "neu", "neg", "mentions7d", "avgSentiment7d"):
                assert second[p][k] == pytest.approx(first[p][k]), (p, k)
        elif "/days/" in p:
            assert second[p] == first[p]


def test_reanalysis_removes_stale_mentions(fs):
    pipeline.run(fs, limit=100, use_wikidata=False)
    assert fs.docs["entities/spd"]["mentions"] == 1
    fs.docs["comments/c1"].update({"analyzed": False, "text": "Merz macht das gut."})
    pipeline.run(fs, limit=100, use_wikidata=False)
    assert "mentions/c1_spd" not in fs.docs
    assert fs.docs["entities/spd"]["mentions"] == 0
    assert fs.docs["entities/friedrich-merz"]["mentions"] == 2
    assert not any(p.startswith("entities/spd/days/") for p in fs.docs)


def test_new_comments_accumulate(fs):
    pipeline.run(fs, limit=100, use_wikidata=False)
    fs.docs["comments/c5"] = comment("c5", "Merz ist stark.")
    pipeline.run(fs, limit=100, use_wikidata=False)
    assert fs.docs["entities/friedrich-merz"]["mentions"] == 3
    assert fs.docs["entities/friedrich-merz"]["pos"] == 2


def test_crash_before_aggregation_is_recovered(fs, monkeypatch):
    calls = {"n": 0}
    real = pipeline.aggregate_days

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(pipeline, "aggregate_days", boom)
    with pytest.raises(RuntimeError):
        pipeline.run(fs, limit=100, use_wikidata=False)
    assert fs.docs["comments/c1"]["analyzed"] is True  # mentions + flags were written
    assert fs.docs["entities/friedrich-merz"]["mentions"] == 0  # but not aggregated
    assert fs.docs["meta/analysis"]["pendingDays"]
    monkeypatch.setattr(pipeline, "aggregate_days", real)
    out = pipeline.run(fs, limit=100, use_wikidata=False)  # no new comments
    assert out["comments"] == 0 and out["recoveredDays"]
    assert fs.docs["entities/friedrich-merz"]["mentions"] == 2
    assert fs.docs["meta/analysis"]["pendingDays"] == []
