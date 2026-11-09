"""Soft time budget (analysis/timebudget.py): a run stops cleanly before the CI job timeout."""

import logging

import pytest

from analysis import pipeline, timebudget
from analysis.timebudget import TimeBudget
from tests.test_budget import _relink_setup, big
from tests.test_pipeline import comment
from tests.test_pipeline import fs as fs


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    c = Clock()
    yield c
    timebudget.set_current(None)


def slow_analysis(monkeypatch, clock, seconds):
    """Every analysed batch takes `seconds` on the budget's clock."""
    real = pipeline.analyze_comments

    def wrapped(*a, **kw):
        clock.t += seconds
        return real(*a, **kw)

    monkeypatch.setattr(pipeline, "analyze_comments", wrapped)


def test_time_budget_from_env(monkeypatch, clock):
    monkeypatch.setenv("ANALYSIS_TIME_BUDGET_MIN", "10")
    monkeypatch.setenv("ANALYSIS_JOB_START", "400")  # the CI job started 10 minutes before the clock
    tb = TimeBudget(clock=clock)
    assert tb.deadline == 400 + 600
    assert tb.reached() is True
    monkeypatch.setenv("ANALYSIS_TIME_BUDGET_MIN", "0")  # 0 = no limit
    assert TimeBudget(clock=clock).reached() is False
    monkeypatch.delenv("ANALYSIS_TIME_BUDGET_MIN")
    monkeypatch.delenv("ANALYSIS_JOB_START")
    tb = TimeBudget(clock=clock)
    assert tb.minutes == timebudget.DEFAULT_MINUTES and tb.start == clock.t


def test_reached_uses_the_reserve_and_logs_once(clock, caplog):
    tb = TimeBudget(minutes=10, start=clock.t, clock=clock)
    clock.t += 500
    assert tb.reached(reserve=60) is False
    with caplog.at_level(logging.WARNING, logger="analysis.timebudget"):
        assert tb.reached("relink", reserve=120) is True  # the next step would not end in time
        assert tb.reached("relink", reserve=120) is True
    assert sum("time budget reached" in r.getMessage() for r in caplog.records) == 1
    assert tb.summary() == {"minutes": 10, "usedMinutes": 8.3, "reached": "relink"}


def test_run_stops_cleanly_when_the_time_is_up(fs, monkeypatch, clock, caplog):
    for i in range(6):
        fs.docs[f"comments/t{i}"] = comment(f"t{i}", f"Merz sagt dies und das {i}.", hours_ago=2 + i)
    timebudget.set_current(TimeBudget(minutes=5, start=clock.t, clock=clock))
    slow_analysis(monkeypatch, clock, 100)  # 5 minutes = 3 batches
    with caplog.at_level(logging.WARNING):
        out = pipeline.run(fs, limit=100, batch=2, use_wikidata=False, budget=big(fs))
    assert out["stopped"] == "time budget reached after 6 comments"
    assert out["comments"] == 6 and out["backlogDone"] is False and "jobs" not in out
    assert out["timeBudget"]["reached"] == "the analysis run"
    assert any("time budget reached" in r.getMessage() for r in caplog.records)
    analyzed = [p for p, d in fs.docs.items() if p.startswith("comments/") and d.get("analyzed")]
    assert len(analyzed) == 6  # the other 4 wait for the next run
    # what was processed is aggregated, nothing is left pending, and the usage is recorded
    merz = sum(1 for p in fs.docs if p.startswith("mentions/") and p.endswith("_friedrich-merz"))
    assert merz and fs.docs["entities/friedrich-merz"]["mentions"] == merz
    assert fs.docs["meta/analysis"]["pendingPairs"] == []
    assert fs.docs["meta/budget"]["writes"]["analysis"] > 0
    # the next run (fresh time budget) analyses the rest
    timebudget.set_current(None)
    out = pipeline.run(fs, limit=100, batch=2, use_wikidata=False, budget=big(fs))
    assert out["stopped"] is None and out["comments"] == 4


def test_relink_saves_its_cursor_when_the_time_is_up(fs, monkeypatch, clock):
    _relink_setup(fs, monkeypatch)
    # only the Lidl comments: the fixture's other comments (1 hour old) sort first on the same day
    # after ~11:00 Berlin time, and a page without candidates uses no time
    for path in [p for p in fs.docs if p.startswith("comments/c")]:
        del fs.docs[path]
    timebudget.set_current(TimeBudget(minutes=1, start=clock.t, clock=clock))
    slow_analysis(monkeypatch, clock, 100)  # the first page uses up the minute
    out = pipeline.relink(fs, days=14, entity_ids={"lidl"}, page=4, budget=big(fs))
    assert out["done"] is False and out["stopped"].startswith("time budget reached after 4 comments")
    job = fs.docs["meta/analysis"]["jobs"]["relink"]
    assert job["cursor"] and job["scanned"] == 4 and job["pairs"] == []
    partial = sum(1 for p in fs.docs if p.endswith("_lidl"))
    assert partial == 4 and fs.docs["entities/lidl"]["mentions"] == partial  # aggregated before stopping
    # a run whose time is up does not start the job at all
    out = pipeline.continue_jobs(fs, big(fs))
    assert out == {"relink": "waiting for time"}
    # the next run continues from the cursor
    timebudget.set_current(None)
    out = pipeline.relink(fs, days=14, page=4, budget=big(fs))
    assert out["done"] is True and fs.docs["entities/lidl"]["mentions"] == 12


def test_prune_and_reaggregate_respect_the_time(fs, clock):
    pipeline.run(fs, limit=100, use_wikidata=False, budget=big(fs))
    timebudget.set_current(TimeBudget(minutes=1, start=clock.t - 120, clock=clock))
    assert pipeline.prune(fs, {"friedrich-merz"}, budget=big(fs))["stopped"].startswith("time budget")
    assert "mentions/c1_friedrich-merz" in fs.docs
    out = pipeline.reaggregate_since(fs, "2000-01-01", budget=big(fs))
    assert out["done"] is False and out["stopped"].startswith("time budget reached")
    assert fs.docs["meta/analysis"]["jobs"]["reaggregate_since"]["since"] == "2000-01-01"
