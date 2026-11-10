"""Rescore after a model change: day by day (newest first), within the budget, queued by `run`."""

from datetime import UTC, datetime, timedelta

from analysis import pipeline
from analysis.budget import Budget
from analysis.textutil import BERLIN, day_key
from tests.fakes import FakeSentiment
from tests.test_budget import _limited, big
from tests.test_pipeline import NOW, comment
from tests.test_pipeline import fs as fs


class OtherSentiment(FakeSentiment):
    version = "fake-sent@2"

    def __init__(self):
        super().__init__(invert=True)


def _three_days(fs, sizes=(4, 4, 4)):
    """Comments about Merz on three days (today, 2 and 5 days ago); returns the days, newest first."""
    for k, days_ago in enumerate((0, 2, 5)):
        for i in range(sizes[k]):
            cid = f"d{k}x{i}"
            fs.docs[f"comments/{cid}"] = comment(cid, f"Merz ist gut, Nummer {i}.", hours_ago=1 + 24 * days_ago)
    return [day_key(NOW - timedelta(hours=1 + 24 * d)) for d in (0, 2, 5)]


def _labels(fs, day):
    """Labels of the test's own mentions (comments d<k>x<i>, all "Merz ist gut") of one day."""
    return [d["label"] for p, d in sorted(fs.docs.items()) if p.startswith("mentions/d") and d["day"] == day]


def _mentions(fs, day):
    return sum(1 for p, d in fs.docs.items() if p.startswith("mentions/") and d["day"] == day)


def _day_docs(fs):
    return {p: {k: v for k, v in d.items() if k != "updatedAt"} for p, d in fs.docs.items() if "/days/" in p}


def test_rescore_converts_whole_days_newest_first(fs, monkeypatch):
    newest, middle, oldest = _three_days(fs, sizes=(4, 40, 4))
    pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    versions = {p: d["analysisVersion"] for p, d in fs.docs.items() if p.startswith("comments/")}
    n_newest = _mentions(fs, newest)
    monkeypatch.setattr("analysis.sentiment.load_sentiment", lambda key, **kw: FakeSentiment(invert=True))
    mark = fs.ops_mark()
    # the budget covers the newest day, but not all of the next one (40 mentions): that day is not started
    out = pipeline.rescore(fs, sentiment_key="fake", batch=10, budget=_limited(fs, run_writes=120, reserve=0))
    assert out["done"] is False and out["days"] == 1 and out["nextDay"] == middle
    assert "waits for the next run" in out["stopped"]
    assert _labels(fs, newest) == ["neg"] * 4  # "gut" is negative for the new model
    assert fs.docs["mentions/c1_spd"]["label"] == "pos"  # the other mentions of that day too
    assert _labels(fs, middle) == ["pos"] * 40 and _labels(fs, oldest) == ["pos"] * 4  # untouched
    job = fs.docs["meta/analysis"]["jobs"]["rescore"]
    assert job["day"] == middle and job["cursor"] == "" and job["pairs"] == []
    # only the processed day was read, and no comment was written
    ops = fs.ops_summary(mark)
    assert ops["writes"].get("comments", 0) == 0
    assert {p: d["analysisVersion"] for p, d in fs.docs.items() if p.startswith("comments/")} == versions
    assert n_newest <= ops["reads"]["mentions"] <= 3 * n_newest + 10
    # the finished day is aggregated: its day docs match its mentions (a full rebuild changes nothing)
    before = _day_docs(fs)
    pipeline.reaggregate(fs, budget=big(fs))
    assert _day_docs(fs) == before

    out = pipeline.rescore(fs, sentiment_key="fake", batch=50, budget=big(fs))
    assert out["done"] is True and out["nextDay"] is None and out["days"] == 3
    assert _labels(fs, middle) == ["neg"] * 40 and _labels(fs, oldest) == ["neg"] * 4
    state = fs.docs["meta/analysis"]
    assert "rescore" not in state.get("jobs", {})
    assert state["sentiment"]["rescored"]["version"] == "fake-sent@1"
    before = _day_docs(fs)
    pipeline.reaggregate(fs, budget=big(fs))
    assert _day_docs(fs) == before


def test_rescore_resumes_inside_a_day(fs, monkeypatch):
    newest, *_ = _three_days(fs, sizes=(9, 9, 9))
    pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    monkeypatch.setattr("analysis.sentiment.load_sentiment", lambda key, **kw: FakeSentiment(invert=True))
    out = pipeline.rescore(fs, sentiment_key="fake", batch=3, budget=_limited(fs, run_writes=14, reserve=0))
    job = fs.docs["meta/analysis"]["jobs"]["rescore"]
    assert out["done"] is False and job["day"] == newest and job["cursor"] and 0 < out["mentions"] < 9
    # what was rewritten is aggregated, although the day is not finished
    before = _day_docs(fs)
    pipeline.reaggregate(fs, budget=big(fs))
    assert _day_docs(fs) == before
    out = pipeline.rescore(fs, sentiment_key="fake", batch=3, budget=big(fs))
    assert out["done"] is True
    assert all(d["label"] != "pos" for p, d in fs.docs.items() if p.startswith("mentions/d"))


def test_a_new_model_queues_the_rescore_and_the_run_continues_it(fs, monkeypatch):
    days = _three_days(fs)
    pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    state = fs.docs["meta/analysis"]
    assert state["sentiment"]["version"] == "fake-sent@1" and "rescore" not in state.get("jobs", {})
    # same model again: nothing is queued
    fs.docs["comments/n1"] = comment("n1", "Merz ist gut.")
    pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    assert "rescore" not in fs.docs["meta/analysis"].get("jobs", {})
    # a new model: the run records it, queues the rescore and continues it once its backlog is done
    monkeypatch.setattr("analysis.sentiment.load_sentiment", lambda key, **kw: OtherSentiment())
    fs.docs["comments/n2"] = comment("n2", "Merz ist gut.")
    out = pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    assert out["jobs"]["rescore"]["done"] is True and out["jobs"]["rescore"]["version"] == "fake-sent@2"
    state = fs.docs["meta/analysis"]
    assert state["sentiment"]["version"] == "fake-sent@2" and state["sentiment"]["rescored"]["version"] == "fake-sent@2"
    assert "rescore" not in state.get("jobs", {})
    for day in days:
        assert _labels(fs, day) == ["neg"] * 4  # every "gut" is negative for the new model
    merz = fs.docs["entities/friedrich-merz"]
    mine = [d for p, d in fs.docs.items() if p.startswith("mentions/") and d["entityId"] == "friedrich-merz"]
    assert merz["mentions"] == len(mine) and merz["neg"] == sum(d["label"] == "neg" for d in mine) >= 12


def test_first_run_with_the_record_treats_existing_mentions_as_the_old_model(fs, monkeypatch):
    pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    del fs.docs["meta/analysis"]["sentiment"]  # a database from before the record existed
    queued = []
    monkeypatch.setattr(pipeline, "continue_jobs", lambda *a, **kw: queued.append(1) or {})
    fs.docs["comments/n1"] = comment("n1", "Merz ist gut.")
    pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    job = fs.docs["meta/analysis"]["jobs"]["rescore"]
    assert job["replaces"] == pipeline.LEGACY_SENTIMENT and job["from"] == day_key(NOW) and queued


def test_a_fresh_database_queues_nothing(fs):
    assert "meta/analysis" not in fs.docs
    pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    assert "rescore" not in fs.docs["meta/analysis"].get("jobs", {})


def test_a_long_job_leaves_the_budget_of_the_later_runs(fs):
    b = Budget(fs, None, write_limit=60000, read_limit=200000, share_writes=30000, share_reads=120000)
    fs.writes += 8000  # what the run itself has used
    fs.reads += 21000
    morning = datetime(2026, 11, 11, 6, 30, tzinfo=BERLIN).astimezone(UTC)
    evening = datetime(2026, 11, 11, 18, 40, tzinfo=BERLIN).astimezone(UTC)
    assert pipeline._keep_for_later_runs(b, morning) == (10400, 27300)  # one more run today
    assert pipeline._keep_for_later_runs(b, evening) == (0, 0)
    late_morning = datetime(2026, 11, 11, 12, 30, tzinfo=BERLIN).astimezone(UTC)
    assert pipeline._keep_for_later_runs(b, late_morning)[0] == 10400  # a delayed run: the evening run still comes
