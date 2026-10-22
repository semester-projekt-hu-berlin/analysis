"""Pure aggregation math (no I/O), unit-tested in tests/test_aggregate.py.

Idempotency strategy: mentions have deterministic ids (commentId_entityId), so
re-analysing a comment overwrites instead of duplicating. Day documents are always
*recomputed* from all mentions of that day (never incremented). Entity totals are
updated by the difference between the old and the recomputed day document, committed
atomically together with the day documents, so re-running yields a zero delta.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

LABELS = ("pos", "neu", "neg")
CO_MENTIONS_TOP = 20
TRENDING_TOP = 50
TOP_MIN_MENTIONS = 10  # an entity needs this many mentions in 7 days to appear in `top`
RELATIVE_MIN_MENTIONS = 10
MOVERS_TOP = 10
MOVERS_MIN_MENTIONS = 20
RECENT_DAYS = 30  # recentDays window; also feeds the 30-day sparkline in meta/trending
SPARK_DAYS = 30


def r4(x: float) -> float:
    return round(float(x), 4)


def empty_counts() -> dict:
    return {"mentions": 0, "pos": 0, "neu": 0, "neg": 0, "scoreSum": 0.0}


def build_day_docs(mentions: list[dict]) -> dict[tuple[str, str], dict]:
    """Recompute day documents from mention records.

    Each mention needs: commentId, entityId, sourceId, day, label, score, publishedAt.
    Returns {(entityId, day): dayDoc}.
    """
    by_key: dict[tuple[str, str], dict] = {}
    comments: dict[tuple[str, str], set[str]] = defaultdict(set)  # (commentId, day) -> entity ids
    mention_index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in mentions:
        key = (m["entityId"], m["day"])
        doc = by_key.get(key)
        if doc is None:
            doc = by_key[key] = {
                **empty_counts(),
                "day": m["day"],
                "bySource": {},
                "coMentions": {},
                "lastMentionAt": None,
            }
        lab = m["label"] if m["label"] in LABELS else "neu"
        doc["mentions"] += 1
        doc[lab] += 1
        doc["scoreSum"] += float(m["score"])
        src = doc["bySource"].setdefault(m.get("sourceId") or "unknown", empty_counts())
        src["mentions"] += 1
        src[lab] += 1
        src["scoreSum"] += float(m["score"])
        ts = m.get("publishedAt")
        if isinstance(ts, datetime) and (doc["lastMentionAt"] is None or ts > doc["lastMentionAt"]):
            doc["lastMentionAt"] = ts
        comments[(m["commentId"], m["day"])].add(m["entityId"])
        mention_index[(m["commentId"], m["day"])].append(m)
    co: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for m in mentions:
        # Preferred: co-mentioned entities stored on the mention itself (so one (entity, day)
        # can be recomputed from its own mentions only).
        if "coEntityIds" in m:
            for other in m.get("coEntityIds") or []:
                co[(m["entityId"], m["day"])][other] += 1
    for (cid, day), ents in comments.items():
        if any("coEntityIds" in m for m in mention_index.get((cid, day), [])):
            continue
        if len(ents) < 2:
            continue
        for a in ents:
            for b in ents:
                if a != b:
                    co[(a, day)][b] += 1
    for key, doc in by_key.items():
        top = sorted(co.get(key, Counter()).items(), key=lambda kv: (-kv[1], kv[0]))[:CO_MENTIONS_TOP]
        doc["coMentions"] = dict(top)
        doc["scoreSum"] = r4(doc["scoreSum"])
        doc["avgSentiment"] = r4(doc["scoreSum"] / doc["mentions"]) if doc["mentions"] else None
        for s in doc["bySource"].values():
            s["scoreSum"] = r4(s["scoreSum"])
    return by_key


def day_counts(doc: dict | None) -> dict:
    if not doc:
        return empty_counts()
    return {k: doc.get(k, 0) or 0 for k in ("mentions", "pos", "neu", "neg", "scoreSum")}


def same_day_doc(a: dict | None, b: dict | None) -> bool:
    if not a or not b:
        return a == b
    keys = ("mentions", "pos", "neu", "neg", "scoreSum", "bySource", "coMentions")
    return all(_norm(a.get(k)) == _norm(b.get(k)) for k in keys)


def _norm(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, int):
        return round(float(v), 4)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def apply_day_changes(entity: dict, changes: list[tuple[str, dict | None, dict | None]], today: str) -> dict:
    """Update all-time totals and the recentDays map of an entity.

    `changes`: [(day, oldDayDoc, newDayDoc)]. Returns the updated count fields.
    """
    totals = {k: entity.get(k, 0) or 0 for k in ("mentions", "pos", "neu", "neg")}
    score_sum = entity.get("scoreSum")
    if score_sum is None:
        score_sum = (entity.get("avgSentiment") or 0.0) * totals["mentions"]
    recent = {d: dict(v) for d, v in (entity.get("recentDays") or {}).items()}
    last = entity.get("lastMentionAt")
    for day, old, new in changes:
        o, n = day_counts(old), day_counts(new)
        for k in totals:
            totals[k] += n[k] - o[k]
        score_sum += n["scoreSum"] - o["scoreSum"]
        if n["mentions"]:
            recent[day] = {k: n[k] for k in ("mentions", "pos", "neu", "neg", "scoreSum")}
            by_src = (new or {}).get("bySource") or {}
            if by_src:
                recent[day]["bySource"] = {
                    src: {"mentions": v.get("mentions", 0), "scoreSum": r4(v.get("scoreSum", 0.0))}
                    for src, v in by_src.items()
                }
        else:
            recent.pop(day, None)
        ts = (new or {}).get("lastMentionAt")
        if isinstance(ts, datetime) and (not isinstance(last, datetime) or ts > last):
            last = ts
    for k in totals:
        totals[k] = max(0, int(totals[k]))
    if totals["mentions"] == 0:
        score_sum = 0.0
    out = {
        **totals,
        "scoreSum": r4(score_sum),
        "avgSentiment": r4(score_sum / totals["mentions"]) if totals["mentions"] else None,
        "lastMentionAt": last,
        "recentDays": prune_recent(recent, today),
    }
    out.update(window_stats(out["recentDays"], today))
    return out


def totals_from_days(day_docs: dict[str, dict], today: str) -> dict:
    """Full recomputation of entity counts from all its day docs (used by reaggregate)."""
    fake = {"mentions": 0, "pos": 0, "neu": 0, "neg": 0, "scoreSum": 0.0, "recentDays": {}, "lastMentionAt": None}
    return apply_day_changes(fake, [(d, None, doc) for d, doc in sorted(day_docs.items())], today)


def prune_recent(recent: dict[str, dict], today: str) -> dict[str, dict]:
    cutoff = (date.fromisoformat(today) - timedelta(days=RECENT_DAYS - 1)).isoformat()
    return {d: v for d, v in sorted(recent.items()) if cutoff <= d <= today}


def window_stats(recent: dict[str, dict], today: str) -> dict:
    t = date.fromisoformat(today)
    cur_from = (t - timedelta(days=6)).isoformat()
    prev_from = (t - timedelta(days=13)).isoformat()
    prev_to = (t - timedelta(days=7)).isoformat()
    m7 = s7 = mp = sp = 0.0
    for d, v in recent.items():
        if cur_from <= d <= today:
            m7 += v.get("mentions", 0)
            s7 += v.get("scoreSum", 0.0)
        elif prev_from <= d <= prev_to:
            mp += v.get("mentions", 0)
            sp += v.get("scoreSum", 0.0)
    return {
        "mentions7d": int(m7),
        "avgSentiment7d": r4(s7 / m7) if m7 else None,
        "mentionsPrev7d": int(mp),
        "avgSentimentPrev7d": r4(sp / mp) if mp else None,
    }


def sparkline(recent: dict[str, dict], today: str, days: int = SPARK_DAYS) -> list[float | None]:
    """Daily avgSentiment for the last `days` days (oldest first, ending today), None without mentions."""
    t = date.fromisoformat(today)
    out: list[float | None] = []
    for k in range(days - 1, -1, -1):
        v = recent.get((t - timedelta(days=k)).isoformat())
        out.append(r4(v["scoreSum"] / v["mentions"]) if v and v.get("mentions") else None)
    return out


def trending(entities: dict[str, dict], today: str | None = None) -> dict:
    """Build meta/trending from entity docs (id -> doc with name, type, window stats)."""
    items = []
    for eid, e in entities.items():
        if not e.get("allowlisted") or not e.get("mentions7d"):
            continue
        a7, ap = e.get("avgSentiment7d"), e.get("avgSentimentPrev7d")
        delta = r4(a7 - ap) if a7 is not None and ap is not None and e.get("mentionsPrev7d") else None
        items.append(
            {
                "entityId": eid,
                "name": e.get("name", eid),
                "type": e.get("type", "MISC"),
                "mentions7d": int(e["mentions7d"]),
                "avgSentiment7d": a7,
                "delta": delta,
                "relative7d": e.get("relative7d"),
                "spark": sparkline(e.get("recentDays") or {}, today) if today else [None] * SPARK_DAYS,
                "_prev": int(e.get("mentionsPrev7d") or 0),
            }
        )
    top = sorted(
        [x for x in items if x["mentions7d"] >= TOP_MIN_MENTIONS], key=lambda x: (-x["mentions7d"], x["entityId"])
    )[:TRENDING_TOP]
    movers = [
        x
        for x in items
        if x["delta"] is not None and x["mentions7d"] >= MOVERS_MIN_MENTIONS and x["_prev"] >= MOVERS_MIN_MENTIONS
    ]
    risers = sorted([x for x in movers if x["delta"] > 0], key=lambda x: (-x["delta"], x["entityId"]))[:MOVERS_TOP]
    fallers = sorted([x for x in movers if x["delta"] < 0], key=lambda x: (x["delta"], x["entityId"]))[:MOVERS_TOP]

    def clean(xs):
        return [{k: v for k, v in x.items() if not k.startswith("_")} for x in xs]

    return {"top": clean(top), "risers": clean(risers), "fallers": clean(fallers)}


def baselines(entities: dict[str, dict], today: str) -> tuple[float | None, dict[str, float]]:
    """Mention-weighted mean sentiment over all allowlisted entities in the last 7 days, overall
    and per source (from recentDays)."""
    t = date.fromisoformat(today)
    cur_from = (t - timedelta(days=6)).isoformat()
    m = s = 0.0
    src: dict[str, list[float]] = {}
    for e in entities.values():
        if not e.get("allowlisted"):
            continue
        for d, v in (e.get("recentDays") or {}).items():
            if not (cur_from <= d <= today):
                continue
            m += v.get("mentions", 0)
            s += v.get("scoreSum", 0.0)
            for sid, sv in (v.get("bySource") or {}).items():
                acc = src.setdefault(sid, [0.0, 0.0])
                acc[0] += sv.get("mentions", 0)
                acc[1] += sv.get("scoreSum", 0.0)
    return (r4(s / m) if m else None), {k: r4(v[1] / v[0]) for k, v in sorted(src.items()) if v[0]}


def relative(entity: dict, baseline: float | None) -> float | None:
    if baseline is None or entity.get("avgSentiment7d") is None:
        return None
    if (entity.get("mentions7d") or 0) < RELATIVE_MIN_MENTIONS:
        return None
    return r4(entity["avgSentiment7d"] - baseline)
