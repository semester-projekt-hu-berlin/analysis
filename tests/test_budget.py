"""Daily Firestore budget (meta/budget): day rollover, stop condition, reserved aggregation, resuming."""

from analysis import pipeline
from analysis.budget import BUDGET_DOC, Budget
from analysis.linker import Registry
from analysis.textutil import today_berlin
from tests.test_pipeline import comment
from tests.test_pipeline import fs as fs

TODAY = today_berlin()


def budget_doc(day, writes=None, reads=None):
    return {
        "day": day,
        "writes": {"crawler": 0, "analysis": 0, "api": 0, **(writes or {})},
        "reads": {"crawler": 0, "analysis": 0, "api": 0, **(reads or {})},
    }


def big(fs, **kw):
    kw = {"write_limit": 10**6, "read_limit": 10**7, "share_writes": 10**6, "share_reads": 10**7, **kw}
    return Budget.load(fs, **kw)


# ---------------------------------------------------------------------------- accounting


def test_other_day_counts_as_zero_and_rollover_resets(fs):
    fs.docs[BUDGET_DOC] = budget_doc("2000-01-01", writes={"crawler": 99999, "analysis": 99999})
    b = Budget.load(fs, write_limit=60000, read_limit=200000, share_writes=30000, share_reads=120000)
    assert b.day_writes == {"crawler": 0, "analysis": 0, "api": 0}
    assert b.exhausted() is None
    fs.commit([pipeline.Write("x/1", {"a": 1}), pipeline.Write("x/2", {"a": 1})])
    b.record()
    d = fs.docs[BUDGET_DOC]
    assert d["day"] == TODAY
    assert d["writes"] == {"crawler": 0, "analysis": 3, "api": 0}  # 2 + the budget write itself
    assert d["reads"]["analysis"] == 1 and d["reads"]["crawler"] == 0


def test_same_day_increments_only_own_counters(fs):
    fs.docs[BUDGET_DOC] = budget_doc(TODAY, writes={"crawler": 500, "analysis": 10}, reads={"api": 7})
    b = big(fs)
    fs.commit([pipeline.Write("x/1", {"a": 1})])
    b.record()
    d = fs.docs[BUDGET_DOC]
    assert d["writes"] == {"crawler": 500, "analysis": 12, "api": 0}
    assert d["reads"] == {"crawler": 0, "analysis": 1, "api": 7}


def test_concurrent_rollover_is_not_overwritten(fs):
    fs.docs[BUDGET_DOC] = budget_doc("2000-01-01", writes={"crawler": 5})
    b = big(fs)
    # the crawler rolls the day over between our read and our write
    fs.commit([pipeline.Write(BUDGET_DOC, budget_doc(TODAY, writes={"crawler": 40}))])
    b.record()
    d = fs.docs[BUDGET_DOC]
    assert d["day"] == TODAY and d["writes"]["crawler"] == 40 and d["writes"]["analysis"] >= 2


def test_missing_doc_is_created(fs):
    b = big(fs)
    b.record()
    assert fs.docs[BUDGET_DOC]["day"] == TODAY and fs.docs[BUDGET_DOC]["writes"]["analysis"] == 1


# ---------------------------------------------------------------------------- run


def test_run_skips_when_share_is_used_up(fs):
    fs.docs[BUDGET_DOC] = budget_doc(TODAY, writes={"analysis": 30000})
    before = dict(fs.docs[BUDGET_DOC])
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    assert out["skipped"] and out["comments"] == 0
    assert fs.docs["comments/c1"]["analyzed"] is False
    assert fs.docs[BUDGET_DOC] == before  # nothing recorded for a skipped run (1 read only)


def test_run_skips_when_combined_limit_is_used_up(fs):
    fs.docs[BUDGET_DOC] = budget_doc(TODAY, writes={"crawler": 59000}, reads={"api": 10})
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    assert out["skipped"]


def _many(fs, n):
    for i in range(n):
        fs.docs[f"comments/m{i:03d}"] = comment(f"m{i:03d}", f"Merz ist gut, Nummer {i}.", hours_ago=2 + i / 100)


def _limited(fs, run_writes, reserve):
    """A budget with `run_writes` writes left today (on top of the fixture's setup) and a reserve."""
    b = big(fs, reserve_writes=reserve, reserve_reads=0)
    b.share_writes = b.day_writes["analysis"] + run_writes
    return b


def test_run_stops_at_budget_and_aggregates_what_was_processed(fs):
    _many(fs, 30)
    b = _limited(fs, run_writes=200, reserve=150)
    out = pipeline.run(fs, limit=1000, batch=5, use_wikidata=False, budget=b)
    assert out["stopped"] and not out["backlogDone"]
    analyzed = [p for p, d in fs.docs.items() if p.startswith("comments/") and d.get("analyzed")]
    assert 0 < len(analyzed) < 34
    # the reserve was used for the aggregation: counters match the mentions written so far
    merz = fs.docs["entities/friedrich-merz"]
    n = sum(1 for p, d in fs.docs.items() if p.startswith("mentions/") and d["entityId"] == "friedrich-merz")
    assert merz["mentions"] == n > 0
    assert fs.docs["meta/analysis"]["pendingPairs"] == []
    assert b.run_writes <= 200
    # next day: the rest is processed
    out = pipeline.run(fs, limit=1000, batch=5, use_wikidata=False, budget=big(fs))
    assert all(d.get("analyzed") for p, d in fs.docs.items() if p.startswith("comments/"))
    n = sum(1 for p, d in fs.docs.items() if p.startswith("mentions/") and d["entityId"] == "friedrich-merz")
    assert fs.docs["entities/friedrich-merz"]["mentions"] == n == 2 + 30


def test_fetch_is_capped_by_the_remaining_budget(fs, monkeypatch):
    _many(fs, 30)
    b = _limited(fs, run_writes=100, reserve=50)  # (100 - 50) / 1.5 writes per comment -> 33 at most
    seen = {}
    real = pipeline.fetch_unanalyzed

    def spy(fs_, limit, **kw):
        seen["limit"] = limit
        return real(fs_, limit, **kw)

    monkeypatch.setattr(pipeline, "fetch_unanalyzed", spy)
    pipeline.run(fs, limit=1000, batch=5, use_wikidata=False, budget=b)
    assert seen["limit"] < 40


# ---------------------------------------------------------------------------- one-off jobs


def _relink_setup(fs, monkeypatch):
    for i in range(12):
        fs.docs[f"comments/r{i:02d}"] = comment(f"r{i:02d}", f"Bei Lidl ist es gut, sagt Merz {i}.", hours_ago=3 + i)
    real = Registry.from_yaml.__func__
    monkeypatch.setattr(Registry, "from_yaml", classmethod(lambda cls, *a, **kw: real(cls, *a, generated=None)))
    pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    monkeypatch.setattr(Registry, "from_yaml", classmethod(real))
    assert not any(p.endswith("_lidl") for p in fs.docs)


def test_relink_resumes_across_days(fs, monkeypatch):
    _relink_setup(fs, monkeypatch)
    b = _limited(fs, run_writes=12, reserve=0)
    out = pipeline.relink(fs, days=14, entity_ids={"lidl"}, page=4, budget=b)
    assert out["done"] is False and out["stopped"]
    job = fs.docs["meta/analysis"]["jobs"]["relink"]
    assert job["cursor"] and job["entityIds"] == ["lidl"]
    partial = sum(1 for p in fs.docs if p.endswith("_lidl"))
    assert 0 < partial < 12
    # what was linked is aggregated already
    assert fs.docs["entities/lidl"]["mentions"] == partial
    # next day: continues from the cursor (entity set and cutoff come from the job)
    out = pipeline.relink(fs, days=14, page=4, budget=big(fs))
    assert out["done"] is True
    assert "relink" not in fs.docs["meta/analysis"].get("jobs", {})
    assert sum(1 for p in fs.docs if p.endswith("_lidl")) == 12
    assert fs.docs["entities/lidl"]["mentions"] == 12
    assert "lidl" in fs.docs["mentions/r00_friedrich-merz"]["coEntityIds"]


def test_run_continues_unfinished_jobs_after_its_backlog(fs, monkeypatch):
    _relink_setup(fs, monkeypatch)
    pipeline.relink(fs, days=14, entity_ids={"lidl"}, page=4, budget=_limited(fs, run_writes=12, reserve=0))
    assert fs.docs["meta/analysis"]["jobs"]["relink"]
    out = pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    assert out["jobs"]["relink"]["done"] is True
    assert fs.docs["entities/lidl"]["mentions"] == 12


def test_reaggregate_resumes_and_matches_a_full_rebuild(fs):
    _many(fs, 10)
    pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    expected = {p: {k: v for k, v in d.items() if k != "updatedAt"} for p, d in fs.docs.items() if "/days/" in p}
    for p in [p for p in fs.docs if "/days/" in p]:
        del fs.docs[p]  # simulate lost day docs
    out = pipeline.reaggregate(fs, budget=_limited(fs, run_writes=6, reserve=0))
    assert out["done"] is False and fs.docs["meta/analysis"]["jobs"]["reaggregate"]["cursor"]
    out = pipeline.reaggregate(fs, budget=big(fs))
    assert out["done"] is True and "reaggregate" not in fs.docs["meta/analysis"].get("jobs", {})
    got = {p: {k: v for k, v in d.items() if k != "updatedAt"} for p, d in fs.docs.items() if "/days/" in p}
    assert got == expected


def test_rescore_resumes(fs, monkeypatch):
    from tests.fakes import FakeSentiment

    _many(fs, 10)
    pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    monkeypatch.setattr("analysis.sentiment.load_sentiment", lambda key, **kw: FakeSentiment(invert=True))
    out = pipeline.rescore(fs, sentiment_key="fake", batch=3, budget=_limited(fs, run_writes=12, reserve=0))
    assert out["done"] is False and fs.docs["meta/analysis"]["jobs"]["rescore"]["cursor"]
    out = pipeline.rescore(fs, sentiment_key="fake", batch=3, budget=big(fs))
    assert out["done"] is True and "rescore" not in fs.docs["meta/analysis"].get("jobs", {})
    merz = [d for p, d in fs.docs.items() if p.startswith("mentions/m") and d["entityId"] == "friedrich-merz"]
    assert merz and all(d["label"] == "neg" for d in merz)
    assert fs.docs["entities/friedrich-merz"]["neg"] == 10 + 1
