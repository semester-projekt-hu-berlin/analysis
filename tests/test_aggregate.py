from datetime import UTC, datetime

import pytest

from analysis import aggregate as agg


def m(cid, eid, day, label, score, src="spiegel"):
    return {
        "commentId": cid,
        "entityId": eid,
        "day": day,
        "label": label,
        "score": score,
        "sourceId": src,
        "publishedAt": datetime(2026, 10, int(day[-2:]), 12, tzinfo=UTC),
    }


def test_build_day_docs_counts_bysource_and_comentions():
    ms = [
        m("c1", "spd", "2026-10-20", "neg", -0.5),
        m("c1", "cdu", "2026-10-20", "pos", 0.5),
        m("c2", "spd", "2026-10-20", "pos", 0.9, src="welt"),
        m("c3", "spd", "2026-10-21", "neu", 0.0),
    ]
    docs = agg.build_day_docs(ms)
    d = docs[("spd", "2026-10-20")]
    assert d["mentions"] == 2 and d["pos"] == 1 and d["neg"] == 1 and d["neu"] == 0
    assert d["scoreSum"] == pytest.approx(0.4)
    assert d["avgSentiment"] == pytest.approx(0.2)
    assert d["bySource"]["spiegel"] == {"mentions": 1, "pos": 0, "neu": 0, "neg": 1, "scoreSum": -0.5}
    assert d["bySource"]["welt"]["mentions"] == 1
    assert d["coMentions"] == {"cdu": 1}
    assert docs[("cdu", "2026-10-20")]["coMentions"] == {"spd": 1}
    assert docs[("spd", "2026-10-21")]["coMentions"] == {}
    assert d["lastMentionAt"] == datetime(2026, 10, 20, 12, tzinfo=UTC)


def test_build_day_docs_score_sq_sum():
    ms = [
        m("c1", "spd", "2026-10-20", "neg", -0.5),
        m("c2", "spd", "2026-10-20", "pos", 0.9, src="welt"),
        m("c3", "spd", "2026-10-20", "neu", 0.1),
    ]
    d = agg.build_day_docs(ms)[("spd", "2026-10-20")]
    assert d["scoreSqSum"] == pytest.approx(0.25 + 0.81 + 0.01)
    # order-independent and recomputed from scratch, so re-aggregation is idempotent
    assert agg.build_day_docs(list(reversed(ms)))[("spd", "2026-10-20")]["scoreSqSum"] == d["scoreSqSum"]
    # only on the day doc itself, not per source
    assert "scoreSqSum" not in d["bySource"]["spiegel"]
    # removing a mention (prune / relink) lowers it by exactly that mention's square
    d2 = agg.build_day_docs(ms[1:])[("spd", "2026-10-20")]
    assert d["scoreSqSum"] - d2["scoreSqSum"] == pytest.approx(0.25)


def test_same_day_doc_detects_missing_score_sq_sum():
    ms = [m("c1", "spd", "2026-10-20", "neg", -0.5), m("c2", "spd", "2026-10-20", "pos", 0.9)]
    new = agg.build_day_docs(ms)[("spd", "2026-10-20")]
    old = {k: v for k, v in new.items() if k != "scoreSqSum"}  # written before the field existed
    assert agg.same_day_doc(new, dict(new))
    assert not agg.same_day_doc(old, new)  # a re-aggregation fills it in
    assert not agg.same_day_doc({**new, "scoreSqSum": 9.0}, new)


def test_comentions_top20():
    ms = []
    for i in range(30):
        ms.append(m(f"c{i}", "spd", "2026-10-20", "neu", 0.0))
        ms.append(m(f"c{i}", f"e{i:02d}", "2026-10-20", "neu", 0.0))
    ms.append(m("c0x", "spd", "2026-10-20", "neu", 0.0))
    ms.append(m("c0x", "e00", "2026-10-20", "neu", 0.0))
    co = agg.build_day_docs(ms)[("spd", "2026-10-20")]["coMentions"]
    assert len(co) == 20
    assert co["e00"] == 2


def test_window_stats():
    recent = {
        "2026-10-22": {"mentions": 2, "scoreSum": 1.0},
        "2026-10-16": {"mentions": 2, "scoreSum": -1.0},  # still in 7d window (today-6)
        "2026-10-15": {"mentions": 4, "scoreSum": 2.0},  # previous window
        "2026-10-09": {"mentions": 1, "scoreSum": 1.0},  # previous window (today-13)
        "2026-10-08": {"mentions": 100, "scoreSum": 0.0},  # outside both
    }
    w = agg.window_stats(recent, "2026-10-22")
    assert w == {"mentions7d": 4, "avgSentiment7d": 0.0, "mentionsPrev7d": 5, "avgSentimentPrev7d": 0.6}
    assert agg.window_stats({}, "2026-10-22")["avgSentiment7d"] is None


def test_apply_day_changes_is_delta_based():
    entity = {"mentions": 10, "pos": 5, "neu": 3, "neg": 2, "scoreSum": 3.0, "recentDays": {}}
    old = {"mentions": 2, "pos": 1, "neu": 1, "neg": 0, "scoreSum": 0.5}
    new = {"mentions": 3, "pos": 1, "neu": 1, "neg": 1, "scoreSum": -0.2}
    out = agg.apply_day_changes(entity, [("2026-10-22", old, new)], "2026-10-22")
    assert out["mentions"] == 11 and out["neg"] == 3 and out["pos"] == 5
    assert out["scoreSum"] == pytest.approx(2.3)
    assert out["avgSentiment"] == pytest.approx(2.3 / 11, abs=1e-4)
    assert out["recentDays"]["2026-10-22"]["mentions"] == 3
    assert out["mentions7d"] == 3
    # applying the same (new -> new) change again is a no-op
    out2 = agg.apply_day_changes(out, [("2026-10-22", new, new)], "2026-10-22")
    assert out2["mentions"] == 11 and out2["scoreSum"] == pytest.approx(2.3)


def test_day_removed():
    entity = {
        "mentions": 2,
        "pos": 2,
        "neu": 0,
        "neg": 0,
        "scoreSum": 1.0,
        "recentDays": {"2026-10-22": {"mentions": 2}},
    }
    old = {"mentions": 2, "pos": 2, "neu": 0, "neg": 0, "scoreSum": 1.0}
    out = agg.apply_day_changes(entity, [("2026-10-22", old, None)], "2026-10-22")
    assert out["mentions"] == 0 and out["avgSentiment"] is None and out["recentDays"] == {}


def test_totals_from_days_matches_incremental():
    days = {
        "2026-10-20": {"mentions": 3, "pos": 1, "neu": 1, "neg": 1, "scoreSum": 0.3},
        "2026-10-22": {"mentions": 1, "pos": 1, "neu": 0, "neg": 0, "scoreSum": 0.9},
    }
    full = agg.totals_from_days(days, "2026-10-22")
    inc = agg.apply_day_changes({}, [("2026-10-20", None, days["2026-10-20"])], "2026-10-22")
    inc = agg.apply_day_changes(inc, [("2026-10-22", None, days["2026-10-22"])], "2026-10-22")
    for k in ("mentions", "pos", "neu", "neg", "scoreSum", "avgSentiment", "mentions7d", "avgSentiment7d"):
        assert full[k] == pytest.approx(inc[k])


def test_trending_top_risers_fallers():
    ents = {
        "a": {
            "allowlisted": True,
            "name": "A",
            "type": "PER",
            "mentions7d": 50,
            "avgSentiment7d": 0.4,
            "mentionsPrev7d": 40,
            "avgSentimentPrev7d": 0.1,
        },
        "b": {
            "allowlisted": True,
            "name": "B",
            "type": "ORG",
            "mentions7d": 30,
            "avgSentiment7d": -0.5,
            "mentionsPrev7d": 25,
            "avgSentimentPrev7d": 0.0,
        },
        "c": {
            "allowlisted": True,
            "name": "C",
            "type": "PER",
            "mentions7d": 5,
            "avgSentiment7d": 0.9,
            "mentionsPrev7d": 5,
            "avgSentimentPrev7d": -0.9,
        },
        "d": {"allowlisted": False, "name": "D", "type": "PER", "mentions7d": 500, "avgSentiment7d": 0.0},
        "e": {"allowlisted": True, "name": "E", "type": "PER", "mentions7d": 0},
        "f": {
            "allowlisted": True,
            "name": "F",
            "type": "PARTY",
            "mentions7d": 25,
            "avgSentiment7d": 0.2,
            "mentionsPrev7d": 0,
            "avgSentimentPrev7d": None,
        },
    }
    t = agg.trending(ents)
    assert [x["entityId"] for x in t["top"]] == ["a", "b", "f"]  # c: < 10 mentions
    assert {k: v for k, v in t["top"][0].items() if k not in ("spark", "relative7d", "imageUrl", "imageKind")} == {
        "entityId": "a",
        "name": "A",
        "type": "PER",
        "mentions7d": 50,
        "avgSentiment7d": 0.4,
        "delta": 0.3,
    }
    assert t["top"][2]["delta"] is None
    assert [x["entityId"] for x in t["risers"]] == ["a"]  # c below min mentions
    assert [x["entityId"] for x in t["fallers"]] == ["b"]


def test_sparkline():
    recent = {"2026-10-22": {"mentions": 2, "scoreSum": 1.0}, "2026-09-23": {"mentions": 1, "scoreSum": -1.0}}
    sp = agg.sparkline(recent, "2026-10-22")
    assert len(sp) == 30 and sp[-1] == 0.5 and sp[0] == -1.0 and sp[1] is None


def test_comentions_from_coentityids():
    a = {**m("c1", "spd", "2026-10-20", "neg", -0.5), "coEntityIds": ["cdu"]}
    b = {**m("c2", "spd", "2026-10-20", "neg", -0.5), "coEntityIds": []}
    assert agg.build_day_docs([a, b])[("spd", "2026-10-20")]["coMentions"] == {"cdu": 1}


def test_baselines_and_relative():
    ents = {
        "a": {
            "allowlisted": True,
            "mentions7d": 10,
            "avgSentiment7d": 0.0,
            "recentDays": {
                "2026-10-22": {"mentions": 10, "scoreSum": 0.0, "bySource": {"x": {"mentions": 10, "scoreSum": 0.0}}}
            },
        },
        "b": {
            "allowlisted": True,
            "mentions7d": 30,
            "avgSentiment7d": -0.4,
            "recentDays": {
                "2026-10-21": {
                    "mentions": 30,
                    "scoreSum": -12.0,
                    "bySource": {"y": {"mentions": 30, "scoreSum": -12.0}},
                }
            },
        },
        "c": {
            "allowlisted": True,
            "mentions7d": 3,
            "avgSentiment7d": 0.9,
            "recentDays": {"2026-10-01": {"mentions": 3, "scoreSum": 2.7}},
        },  # outside 7 days
    }
    base, by_src = agg.baselines(ents, "2026-10-22")
    assert base == -0.3 and by_src == {"x": 0.0, "y": -0.4}
    assert agg.relative(ents["a"], base) == 0.3
    assert agg.relative(ents["c"], base) is None  # < 10 mentions


def test_peer_ranks_percentiles():
    ents = {
        "a": {"allowlisted": True, "type": "PER", "mentions7d": 12, "avgSentiment7d": 0.2, "name": "A"},
        "b": {"allowlisted": True, "type": "PER", "mentions7d": 30, "avgSentiment7d": -0.4, "name": "B"},
        "c": {"allowlisted": True, "type": "PER", "mentions7d": 15, "avgSentiment7d": 0.0, "name": "C"},
        "d": {"allowlisted": True, "type": "ORG", "mentions7d": 50, "avgSentiment7d": -0.1, "name": "D"},
        "e": {"allowlisted": True, "type": "PER", "mentions7d": 3, "avgSentiment7d": 0.9, "name": "E"},  # unreliable
    }
    f, strip = agg.peer_ranks(ents)
    assert f["a"] == {
        "rankAll": 1,
        "peersAll": 4,
        "percentileAll": 100.0,
        "rankInType": 1,
        "peersInType": 3,
        "percentileInType": 100.0,
    }
    assert f["b"]["rankInType"] == 3 and f["b"]["percentileInType"] == 0.0
    assert f["c"]["percentileInType"] == 50.0
    assert f["d"]["peersInType"] == 1 and f["d"]["percentileInType"] == 100.0
    assert "e" not in f
    assert [x["id"] for x in strip["PER"]] == ["a", "c", "b"]
