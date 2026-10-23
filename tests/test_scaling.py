"""Per-run Firestore reads must not grow with the number of known entities."""

from analysis import pipeline
from analysis.entity_requests import process_entity_requests
from analysis.registry_store import ACTIVE_DOC, REGISTRY_DOC
from tests.test_pipeline import StubClient, comment, wd_item
from tests.test_pipeline import fs as fs


def _second_run_reads(fs, n_extra):
    for i in range(n_extra):  # never-mentioned entities
        fs.docs[f"entities/dummy-{i:05d}"] = {"name": f"Dummy {i}", "type": "ORG", "allowlisted": True, "mentions": 0}
    pipeline.run(fs, limit=100, use_wikidata=False)  # first run: bootstraps meta/active and meta/registry
    for i in range(5, 9):
        fs.docs[f"comments/n{i}"] = comment(f"n{i}", f"Merz ist gut, Nummer {i}.")
    r0 = fs.reads
    pipeline.run(fs, limit=100, use_wikidata=False)
    return fs.reads - r0


def test_reads_per_run_do_not_grow_with_entities(fs, monkeypatch):
    import copy

    base = copy.deepcopy(fs.docs)
    small = _second_run_reads(fs, 0)
    fs2 = type(fs)(base)
    big = _second_run_reads(fs2, 5000)
    assert big == small
    assert small < 150  # comments, a few entity docs, meta docs; not ~2,000 entity docs
    assert set(fs2.docs[ACTIVE_DOC]["ids"]) >= {"friedrich-merz", "spd"}
    assert REGISTRY_DOC in fs2.docs


def test_entities_leave_the_active_set(fs):
    pipeline.run(fs, limit=100, use_wikidata=False)
    assert "friedrich-merz" in fs.docs[ACTIVE_DOC]["ids"]
    # all mentions gone (e.g. pruned): after aggregation the entity needs no upkeep any more
    for p in [p for p in fs.docs if p.startswith("mentions/") and p.endswith("_friedrich-merz")]:
        del fs.docs[p]
    days = {p.split("/")[-1] for p in fs.docs if p.startswith("entities/friedrich-merz/days/")}
    docs, _ = pipeline.load_active_docs(fs, pipeline.ENTITY_READ_FIELDS)
    pipeline.aggregate_pairs(fs, {("friedrich-merz", d) for d in days}, docs)
    assert fs.docs["entities/friedrich-merz"]["mentions"] == 0
    assert "friedrich-merz" not in fs.docs[ACTIVE_DOC]["ids"]
    assert "spd" in fs.docs[ACTIVE_DOC]["ids"]


def test_discovered_entities_are_registered(fs, monkeypatch):
    from tests.fakes import FakeNer

    monkeypatch.setattr("analysis.ner.load_ner", lambda name: FakeNer({"Hanno Testmeier": "PER"}))
    fs.docs["entity_requests/Q1"] = {"qid": "Q1", "status": "accepted"}
    process_entity_requests(fs, client=StubClient({"Q1": wd_item("Hanno Testmeier", ["Q5"], sitelinks=40)}))
    assert fs.docs[REGISTRY_DOC]["entities"]["hanno-testmeier"]["wikidataId"] == "Q1"
    fs.docs["comments/k1"] = comment("k1", "Hanno Testmeier ist super.")
    pipeline.run(fs, limit=100, use_wikidata=False)
    assert fs.docs["mentions/k1_hanno-testmeier"]["label"] == "pos"


def test_sources_without_country_keep_baselines(fs):
    fs.docs["sources/feddit"] = {"name": "feddit.org"}  # no country field
    fs.docs["comments/f1"] = comment("f1", "Merz ist gut.", source="feddit")
    pipeline.run(fs, limit=100, use_wikidata=False)
    by_source = fs.docs["meta/stats"]["baselineBySource7d"]
    assert "feddit" in by_source and "spiegel" in by_source
    assert fs.docs["mentions/f1_friedrich-merz"]["sourceId"] == "feddit"
