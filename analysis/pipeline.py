"""Firestore orchestration: run (analyze new comments), reaggregate, seed."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import yaml

from . import aggregate as agg
from .analyze import Stats, analyze_comments
from .firestore import Firestore, Write
from .linker import DATA_DIR, Entity, Registry
from .textutil import today_berlin

log = logging.getLogger(__name__)

COMMENT_FIELDS = ["articleId", "sourceId", "text", "publishedAt", "crawledAt", "day", "analysisVersion"]
MENTION_FIELDS = ["commentId", "entityId", "sourceId", "day", "label", "score", "publishedAt"]
ENTITY_STAT_FIELDS = [
    "mentions",
    "pos",
    "neu",
    "neg",
    "scoreSum",
    "avgSentiment",
    "mentions7d",
    "avgSentiment7d",
    "mentionsPrev7d",
    "avgSentimentPrev7d",
    "lastMentionAt",
    "recentDays",
    "updatedAt",
]
ENTITY_READ_FIELDS = [
    "name",
    "type",
    "description",
    "wikidataId",
    "imageUrl",
    "aliases",
    "allowlisted",
    *ENTITY_STAT_FIELDS,
]
ZERO_STATS = {
    "mentions": 0,
    "pos": 0,
    "neu": 0,
    "neg": 0,
    "scoreSum": 0.0,
    "avgSentiment": None,
    "mentions7d": 0,
    "avgSentiment7d": None,
    "mentionsPrev7d": 0,
    "avgSentimentPrev7d": None,
    "lastMentionAt": None,
    "recentDays": {},
}


def now() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- helpers


def load_entity_docs(fs: Firestore) -> dict[str, dict]:
    return {d.id: d.data for d in fs.query("entities", select=ENTITY_READ_FIELDS)}


def build_registry(entity_docs: dict[str, dict]) -> Registry:
    """Seed registry + allowlisted entities discovered earlier (stored in Firestore)."""
    reg = Registry.from_yaml()
    for eid, d in entity_docs.items():
        if eid in reg.entities or not d.get("allowlisted") or not d.get("wikidataId"):
            continue
        if d["wikidataId"] in reg.by_qid or d.get("type") not in ("PER", "ORG", "PARTY"):
            continue
        reg.add(
            Entity(
                id=eid,
                name=d.get("name", eid),
                type=d["type"],
                description=d.get("description", ""),
                wikidataId=d["wikidataId"],
                aliases=list(d.get("aliases") or [d.get("name", eid)]),
                imageUrl=d.get("imageUrl"),
                source="wikidata",
            )
        )
    return reg


def source_countries(fs: Firestore) -> dict[str, str]:
    try:
        return {d.id: d.data.get("country") or "" for d in fs.query("sources", select=["country"])}
    except RuntimeError as e:
        log.warning("could not read sources: %s", e)
        return {}


def article_context(fs: Firestore, article_ids: set[str]) -> dict[str, str]:
    if not article_ids:
        return {}
    docs = fs.batch_get([f"articles/{a}" for a in article_ids], mask=["title", "topics"])
    out = {}
    for path, d in docs.items():
        if d is None:
            continue
        topics = d.data.get("topics") or []
        out[path.split("/", 1)[1]] = " ".join([d.data.get("title") or "", *[str(t) for t in topics]])
    return out


def new_entity_write(e: Entity) -> Write:
    return Write(f"entities/{e.id}", {**e.metadata(), **ZERO_STATS, "updatedAt": now()})


# --------------------------------------------------------------------------- seed


def seed(fs: Firestore, with_images: bool = True) -> dict:
    reg = Registry.from_yaml()
    existing = {d.id: d.data for d in fs.query("entities", select=["imageUrl", "mentions"])}
    if with_images:
        _fill_images(reg, existing)
    writes = []
    for e in reg.entities.values():
        meta = e.metadata()
        if e.id in existing:
            if not e.imageUrl and existing[e.id].get("imageUrl"):
                meta["imageUrl"] = existing[e.id]["imageUrl"]
            writes.append(Write(f"entities/{e.id}", {**meta, "updatedAt": now()}, mask=[*meta.keys(), "updatedAt"]))
        else:
            writes.append(new_entity_write(e))
    events = yaml.safe_load((DATA_DIR / "events.yaml").read_text(encoding="utf-8"))
    unknown = [eid for ev in events for eid in ev.get("entityIds", []) if eid not in reg.entities]
    if unknown:
        raise ValueError(f"events reference unknown entities: {unknown}")
    for ev in events:
        data = {k: ev.get(k) for k in ("day", "title", "description", "entityIds", "url")}
        data["day"] = str(data["day"])
        data["entityIds"] = data["entityIds"] or []
        writes.append(Write(f"events/{ev['id']}", data))
    fs.commit(writes)
    new = len(reg.entities.keys() - existing.keys())
    log.info("seeded %d entities (%d new) and %d events", len(reg.entities), new, len(events))
    return {"entities": len(reg.entities), "newEntities": new, "events": len(events)}


def _fill_images(reg: Registry, existing: dict[str, dict]) -> None:
    from .wikidata import WikidataClient, image_url

    qids = [e.wikidataId for e in reg.entities.values() if e.wikidataId]
    try:
        items = WikidataClient(min_interval=1.0).get_entities(qids)
    except RuntimeError as e:
        log.warning("could not fetch images: %s", e)
        return
    for e in reg.entities.values():
        it = items.get(e.wikidataId or "")
        if it:
            e.imageUrl = image_url(it)


# --------------------------------------------------------------------------- run


def fetch_unanalyzed(fs: Firestore, limit: int) -> list[dict]:
    docs = fs.query(
        "comments",
        where=[("analyzed", "EQUAL", False)],
        order_by=[("crawledAt", "ASCENDING")],
        limit=limit,
        select=COMMENT_FIELDS,
    )
    return [{"id": d.id, **d.data} for d in docs]


def run(
    fs: Firestore,
    limit: int = 3000,
    batch: int = 250,
    ner_backend: str = "flair",
    sentiment_key: str = "guhr",
    use_wikidata: bool = True,
    max_wikidata_lookups: int = 150,
) -> dict:
    from .ner import load_ner
    from .sentiment import SentimentModel

    t0 = time.time()
    comments = fetch_unanalyzed(fs, limit)
    log.info("%d unanalyzed comments", len(comments))
    entity_docs = load_entity_docs(fs)
    if not comments:
        refresh_rolling(fs, entity_docs, set())
        return {"comments": 0}
    registry = build_registry(entity_docs)
    ner = load_ner(ner_backend)
    sentiment = SentimentModel(sentiment_key)
    version = f"{ner.version}+{sentiment.version}+link@1"
    wd = None
    if use_wikidata:
        from .wikidata import WikidataLinker

        wd = WikidataLinker(max_lookups=max_wikidata_lookups)
    countries = source_countries(fs)
    stats = Stats()
    affected_days: set[str] = set()
    stale_entities: dict[str, set[str]] = {}  # day -> entity ids that lost a mention
    known_entities = set(entity_docs)

    for i in range(0, len(comments), batch):
        chunk = comments[i : i + batch]
        ctx = article_context(fs, {c.get("articleId") for c in chunk if c.get("articleId")})
        result = analyze_comments(
            chunk, registry, ner, sentiment, source_country=countries, article_context=ctx, wikidata=wd, stats=stats
        )
        writes: list[Write] = []
        # New entities discovered through Wikidata must exist before mentions reference them.
        for eid, e in registry.entities.items():
            if eid not in known_entities:
                writes.append(new_entity_write(e))
                known_entities.add(eid)
                entity_docs[eid] = {**e.metadata(), **ZERO_STATS}
        # Re-analysis: drop mentions of entities that no longer match.
        reanalysed = [c for c in chunk if c.get("analysisVersion")]
        for c in reanalysed:
            new_ids = {m["entityId"] for m in result[c["id"]]}
            for d in fs.query("mentions", where=[("commentId", "EQUAL", c["id"])], select=["entityId", "day"]):
                if d.data.get("entityId") not in new_ids:
                    writes.append(Write(f"mentions/{d.id}", None))
                    stale_entities.setdefault(d.data.get("day", ""), set()).add(d.data.get("entityId", ""))
                    affected_days.add(d.data.get("day", ""))
        ts = now()
        for c in chunk:
            for m in result[c["id"]]:
                writes.append(Write(f"mentions/{m['commentId']}_{m['entityId']}", m))
                affected_days.add(m["day"])
            writes.append(
                Write(
                    f"comments/{c['id']}",
                    {"analyzed": True, "analyzedAt": ts, "analysisVersion": version},
                    mask=["analyzed", "analyzedAt", "analysisVersion"],
                    exists=True,
                )
            )
        fs.commit(writes)
        log.info(
            "batch %d-%d: %d mentions so far (ner %.0fs, sentiment %.0fs)",
            i,
            i + len(chunk),
            stats.mentions,
            stats.ner_seconds,
            stats.sentiment_seconds,
        )
    if wd is not None:
        wd.save()
    affected_days.discard("")
    agg_info = aggregate_days(fs, affected_days, entity_docs, stale_entities)
    summary = {
        "comments": stats.comments,
        "sentences": stats.sentences,
        "mentions": stats.mentions,
        "nerSeconds": round(stats.ner_seconds, 1),
        "sentimentSeconds": round(stats.sentiment_seconds, 1),
        "linkSeconds": round(stats.link_seconds, 1),
        "wikidataLookups": stats.wikidata_lookups,
        "discovered": stats.discovered,
        "days": sorted(affected_days),
        "version": version,
        "seconds": round(time.time() - t0, 1),
        **agg_info,
    }
    log.info("run summary: %s", summary)
    return summary


# --------------------------------------------------------------------------- aggregation


def aggregate_days(
    fs: Firestore,
    days: set[str],
    entity_docs: dict[str, dict],
    stale_entities: dict[str, set[str]] | None = None,
) -> dict:
    """Recompute day docs for `days` from mentions, then update entity totals + trending."""
    today = today_berlin()
    stale_entities = stale_entities or {}
    changes: dict[str, list[tuple[str, dict | None, dict | None]]] = {}
    new_docs: dict[tuple[str, str], dict] = {}
    for day in sorted(days):
        mentions = [d.data for d in fs.query("mentions", where=[("day", "EQUAL", day)], select=MENTION_FIELDS)]
        mentions = [
            m
            for m in mentions
            if m.get("entityId") in entity_docs and entity_docs[m["entityId"]].get("allowlisted", True)
        ]
        docs = agg.build_day_docs(mentions)
        ents = {e for e, _ in docs} | stale_entities.get(day, set())
        old = fs.batch_get([f"entities/{e}/days/{day}" for e in ents])
        for e in ents:
            o = old.get(f"entities/{e}/days/{day}")
            od = o.data if o else None
            nd = docs.get((e, day))
            if od is None and nd is None:
                continue
            if agg.same_day_doc(od, nd):
                continue
            changes.setdefault(e, []).append((day, od, nd))
            if nd is not None:
                new_docs[(e, day)] = nd
    _commit_entity_changes(fs, changes, new_docs, entity_docs, today)
    refresh_rolling(fs, entity_docs, set(changes))
    return {"changedEntities": len(changes), "changedDays": sum(len(v) for v in changes.values())}


def _commit_entity_changes(fs, changes, new_docs, entity_docs, today) -> None:
    """Commit day docs and the entity totals of each entity atomically (same commit)."""
    pending: list[Write] = []
    ts = now()
    for eid, ch in sorted(changes.items()):
        group: list[Write] = []
        for day, _old, new in ch:
            path = f"entities/{eid}/days/{day}"
            group.append(Write(path, None) if new is None else Write(path, new_docs[(eid, day)]))
        upd = agg.apply_day_changes(entity_docs.get(eid, {}), ch, today)
        upd["updatedAt"] = ts
        entity_docs.setdefault(eid, {}).update(upd)
        group.append(Write(f"entities/{eid}", upd, mask=ENTITY_STAT_FIELDS))
        if len(group) > 500:  # one entity with > 499 changed days: only possible on reaggregate
            for i in range(0, len(group) - 1, 499):
                fs.commit_atomic(group[i : i + 499])
            continue
        if len(pending) + len(group) > 500:
            fs.commit_atomic(pending)
            pending = []
        pending.extend(group)
    if pending:
        fs.commit_atomic(pending)


def refresh_rolling(fs: Firestore, entity_docs: dict[str, dict], already: set[str]) -> None:
    """Slide 7-day windows for all entities (windows move even without new mentions), then trending + stats."""
    today = today_berlin()
    writes = []
    ts = now()
    for eid, d in entity_docs.items():
        if eid in already or (not d.get("recentDays") and not d.get("mentions7d") and not d.get("mentionsPrev7d")):
            continue
        recent = agg.prune_recent(d.get("recentDays") or {}, today)
        win = agg.window_stats(recent, today)
        cur = {k: d.get(k) for k in win}
        if cur != win or recent != (d.get("recentDays") or {}):
            upd = {**win, "recentDays": recent, "updatedAt": ts}
            d.update(upd)
            writes.append(Write(f"entities/{eid}", upd, mask=list(upd)))
    fs.commit(writes)
    trend = agg.trending(entity_docs)
    trend["updatedAt"] = ts
    total_mentions = fs.count("mentions")
    tracked = sum(1 for d in entity_docs.values() if d.get("allowlisted") and (d.get("mentions") or 0) > 0)
    fs.commit(
        [
            Write("meta/trending", trend),
            Write(
                "meta/stats",
                {"mentions": total_mentions, "entities": tracked, "lastAnalysisAt": ts},
                mask=["mentions", "entities", "lastAnalysisAt"],
            ),
        ]
    )


def reaggregate(fs: Firestore) -> dict:
    """Rebuild every day doc and all entity counters from scratch from `mentions`."""
    today = today_berlin()
    entity_docs = load_entity_docs(fs)
    mentions = [d.data for d in fs.query("mentions", select=MENTION_FIELDS)]
    mentions = [
        m for m in mentions if m.get("entityId") in entity_docs and entity_docs[m["entityId"]].get("allowlisted")
    ]
    docs = agg.build_day_docs(mentions)
    existing_days = [
        d.path
        for d in fs.run_query(
            {
                "from": [{"collectionId": "days", "allDescendants": True}],
                "select": {"fields": [{"fieldPath": "__name__"}]},
            }
        )
    ]
    by_entity: dict[str, dict[str, dict]] = {}
    for (e, day), doc in docs.items():
        by_entity.setdefault(e, {})[day] = doc
    writes: list[Write] = []
    wanted = {f"entities/{e}/days/{d}" for e, d in docs}
    for path in existing_days:
        if path not in wanted:
            writes.append(Write(path, None))
    for (e, day), doc in docs.items():
        writes.append(Write(f"entities/{e}/days/{day}", doc))
    ts = now()
    for eid, d in entity_docs.items():
        upd = agg.totals_from_days(by_entity.get(eid, {}), today) if eid in by_entity else {**ZERO_STATS}
        upd["updatedAt"] = ts
        d.update(upd)
        writes.append(Write(f"entities/{eid}", upd, mask=ENTITY_STAT_FIELDS))
    fs.commit(writes)
    refresh_rolling(fs, entity_docs, set(entity_docs))
    return {
        "mentions": len(mentions),
        "dayDocs": len(docs),
        "entitiesWithMentions": len(by_entity),
        "deletedDayDocs": len(existing_days) - len(wanted & set(existing_days)),
    }
