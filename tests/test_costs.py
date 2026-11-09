"""Firestore cost of a run: where the reads and writes go, and what a run must not pay twice."""

import copy
from datetime import timedelta

from analysis import pipeline
from analysis.firestore import op_kind
from analysis.textutil import day_key
from tests.test_budget import _relink_setup, big
from tests.test_pipeline import NOW, comment, snapshot
from tests.test_pipeline import fs as fs

DAY = day_key(NOW - timedelta(hours=1))


def more_comments(fs, n, start=0):
    for i in range(start, start + n):
        fs.docs[f"comments/m{i:03d}"] = comment(f"m{i:03d}", f"Merz ist gut, die SPD hat versagt, Nummer {i}.")


def test_op_kind():
    assert op_kind("comments/abc") == "comments"
    assert op_kind("entities/spd/days/2026-11-09") == "entities/days"
    assert op_kind("meta/registry_2") == "meta/registry"
    assert op_kind("meta/analysis") == "meta/analysis"


def test_run_reports_operations_per_collection(fs):
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    ops = out["firestoreOps"]
    assert ops["reads"]["comments"] == 4 and ops["writes"]["comments"] == 4
    assert ops["writes"]["mentions"] == out["mentions"] == 4
    assert ops["writes"]["meta/budget"] == 1
    assert sum(ops["reads"].values()) == out["budget"]["runReads"]
    assert sum(ops["writes"].values()) == out["budget"]["runWrites"]


def test_one_aggregation_per_run_reads_each_mention_once(fs):
    more_comments(fs, 12)
    base = copy.deepcopy(fs.docs)
    out = pipeline.run(fs, limit=100, batch=4, use_wikidata=False)
    assert out["comments"] == 16
    mentions = sum(1 for p in fs.docs if p.startswith("mentions/"))
    once = out["firestoreOps"]
    assert once["reads"]["mentions"] == mentions  # every mention read exactly once
    assert once["writes"]["meta/trending"] == 1
    # with intermediate rounds the same pairs are read and written again; the result is the same
    fs2 = type(fs)(base)
    out2 = pipeline.run(fs2, limit=100, batch=4, agg_every=4, use_wikidata=False)
    progressive = out2["firestoreOps"]
    assert progressive["reads"]["mentions"] > 2 * mentions
    assert progressive["writes"]["entities"] > once["writes"]["entities"]
    assert progressive["writes"]["entities/days"] > once["writes"]["entities/days"]
    assert snapshot(fs2) == snapshot(fs)


def test_pending_pairs_are_bounded(fs, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_PENDING_PAIRS", 2)
    more_comments(fs, 8)
    out = pipeline.run(fs, limit=100, batch=4, use_wikidata=False)
    assert out["firestoreOps"]["writes"]["meta/trending"] > 1  # aggregated before the end as well
    assert fs.docs["meta/analysis"]["pendingPairs"] == []


def test_community_docs_are_only_read_where_one_can_exist(fs):
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    assert "entities/community_text" not in out["firestoreOps"]["reads"]
    # a community text: its (entity, day) doc is read and written, the others still are not
    fs.docs["comments/u1"] = {
        **comment("u1", "Merz ist super gut.", source="community", article=f"community-{DAY}"),
        "sourceType": "community",
    }
    fs.docs["comments/n1"] = comment("n1", "Die SPD hat versagt.")
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    assert out["firestoreOps"]["reads"]["entities/community_text"] == 1
    path = f"entities/friedrich-merz/community_text/{DAY}"
    assert fs.docs[path]["mentions"] == 1
    # the community text is analysed again and no longer mentions the entity: the doc is found
    # through the entity's communityRecentDays and deleted
    fs.docs["comments/u1"].update(text="Nichts Relevantes mehr.", analyzed=False)
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    assert out["firestoreOps"]["reads"]["entities/community_text"] == 1
    assert path not in fs.docs
    assert fs.docs["entities/friedrich-merz"]["communityText7d"]["mentions"] == 0


def test_stats_mentions_follow_the_day_docs_without_counting(fs):
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    stats = fs.docs["meta/stats"]
    assert stats["mentions"] == 4 and stats["mentionsCountedAt"]
    assert out["firestoreOps"]["reads"]["count:mentions"] == 2  # first run: counted once (all, community)
    more_comments(fs, 3)
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    assert "count:mentions" not in out["firestoreOps"]["reads"]
    assert fs.docs["meta/stats"]["mentions"] == 10 == sum(1 for p in fs.docs if p.startswith("mentions/"))
    # a comment analysed again loses a mention: the total follows
    fs.docs["comments/m000"].update(text="Nur noch Merz ist gut.", analyzed=False)
    pipeline.run(fs, limit=100, use_wikidata=False)
    assert fs.docs["meta/stats"]["mentions"] == 9 == sum(1 for p in fs.docs if p.startswith("mentions/"))


def test_stats_mentions_are_recounted_weekly_and_after_a_rebuild(fs):
    pipeline.run(fs, limit=100, use_wikidata=False)
    fs.docs["meta/stats"]["mentions"] = 999  # drift
    pipeline.run(fs, limit=100, use_wikidata=False)  # nothing new: no refresh at all
    assert fs.docs["meta/stats"]["mentions"] == 999
    fs.docs["meta/stats"]["mentionsCountedAt"] -= timedelta(days=pipeline.MENTIONS_RECOUNT_DAYS, minutes=1)
    more_comments(fs, 1)
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    assert out["firestoreOps"]["reads"]["count:mentions"] == 2
    assert fs.docs["meta/stats"]["mentions"] == 6
    fs.docs["meta/stats"]["mentions"] = 999
    pipeline.reaggregate(fs)
    assert fs.docs["meta/stats"]["mentions"] == 6


def test_run_without_news_costs_a_few_reads(fs, monkeypatch):
    pipeline.run(fs, limit=100, use_wikidata=False)
    r0, w0 = fs.reads, fs.writes
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    assert out["comments"] == 0 and out["refresh"] == "not needed"
    assert fs.reads - r0 <= 6 and fs.writes - w0 == 1  # budget, requests, comments, state, stats | budget
    # on the next day the windows have to move: the run refreshes once
    tomorrow = day_key(NOW + timedelta(days=1))
    monkeypatch.setattr(pipeline, "today_berlin", lambda now=None: day_key(now) if now else tomorrow)
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    assert "refresh" not in out and out["firestoreOps"]["reads"]["entities"] >= 1


def test_interrupted_run_is_still_recovered_by_a_run_without_comments(fs, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("killed")

    monkeypatch.setattr(pipeline, "aggregate_pairs", boom)
    try:
        pipeline.run(fs, limit=100, use_wikidata=False)
    except RuntimeError:
        pass
    monkeypatch.undo()
    assert fs.docs["meta/analysis"]["pendingPairs"]
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    assert out["recoveredPairs"] > 0 and fs.docs["entities/friedrich-merz"]["mentions"] == 2
    assert fs.docs["meta/stats"]["mentions"] == 4


def test_article_context_is_read_once_per_run(fs):
    more_comments(fs, 8)  # all on article a1
    out = pipeline.run(fs, limit=100, batch=3, use_wikidata=False)
    assert out["firestoreOps"]["reads"]["articles"] == 1


def test_continued_jobs_reuse_the_entity_docs_of_the_run(fs, monkeypatch):
    _relink_setup(fs, monkeypatch)
    pipeline.queue_job(fs, "relink", {"cutoff": "2000-01-01", "entityIds": ["lidl"]})
    active = len(fs.docs["meta/active"]["ids"])
    fs.docs["comments/n1"] = comment("n1", "Merz ist gut.")
    out = pipeline.run(fs, limit=1000, use_wikidata=False, budget=big(fs))
    assert out["jobs"]["relink"]["done"] is True and fs.docs["entities/lidl"]["mentions"] == 12
    # the active entity docs are read once (by the run), not again by the relink it continues;
    # lidl itself is new to the run and read once
    assert out["firestoreOps"]["reads"]["entities"] <= active + 1
    # meta/active: loaded once; the other two reads are the unchanged-checks of the two refreshes
    assert out["firestoreOps"]["reads"]["meta/active"] == 3
