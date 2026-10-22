"""Firestore orchestration: run (analyze new comments), reaggregate, seed."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from urllib.parse import unquote

import yaml

from . import aggregate as agg
from .analyze import Stats, analyze_comments
from .firestore import Firestore, Write
from .linker import DATA_DIR, Entity, Registry
from .textutil import today_berlin

log = logging.getLogger(__name__)

COMMENT_FIELDS = ["articleId", "sourceId", "text", "publishedAt", "crawledAt", "day", "analysisVersion", "analyzedAt"]
MENTION_FIELDS = ["commentId", "entityId", "sourceId", "day", "label", "score", "publishedAt", "coEntityIds"]
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
STATE_DOC = "meta/analysis"  # internal: (entity, day) pairs still to aggregate after an interrupted run
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
    """Write curated entities and events; only documents whose content changed are written."""
    reg = Registry.from_yaml()
    meta_fields = list(next(iter(reg.entities.values())).metadata().keys())
    existing = {d.id: d.data for d in fs.query("entities", select=meta_fields)}
    if with_images:
        _fill_images(reg)
    writes = []
    changed_images = 0
    for e in reg.entities.values():
        meta = e.metadata()
        cur = existing.get(e.id)
        if cur is None:
            writes.append(new_entity_write(e))
            continue
        if not e.imageUrl and cur.get("imageUrl") and not with_images:
            meta["imageUrl"] = cur["imageUrl"]
        diff = [k for k in meta if cur.get(k) != meta[k]]
        if diff:
            changed_images += "imageUrl" in diff
            writes.append(Write(f"entities/{e.id}", {**meta, "updatedAt": now()}, mask=[*diff, "updatedAt"]))
    events = yaml.safe_load((DATA_DIR / "events.yaml").read_text(encoding="utf-8"))
    unknown = [eid for ev in events for eid in ev.get("entityIds", []) if eid not in reg.entities]
    if unknown:
        raise ValueError(f"events reference unknown entities: {unknown}")
    old_events = {d.id: d.data for d in fs.query("events")}
    for ev in events:
        data = {k: ev.get(k) for k in ("day", "title", "description", "entityIds", "url")}
        data["day"] = str(data["day"])
        data["entityIds"] = data["entityIds"] or []
        if old_events.get(ev["id"]) != data:
            writes.append(Write(f"events/{ev['id']}", data))
    fs.commit(writes)
    new = len(reg.entities.keys() - existing.keys())
    out = {
        "entities": len(reg.entities),
        "newEntities": new,
        "updatedEntities": sum(1 for w in writes if w.path.startswith("entities/")) - new,
        "changedImages": changed_images,
        "withImage": sum(1 for e in reg.entities.values() if e.imageUrl),
        "events": len(events),
    }
    log.info("seed: %s", out)
    return out


def _fill_images(reg: Registry) -> None:
    """Thumbnail (250 px) per entity from Wikidata in one SPARQL query: logo (P154) first for
    organisations and parties, photo (P18) first for persons."""
    import requests

    from .wikidata import USER_AGENT, commons_thumb

    qids = sorted({e.wikidataId for e in reg.entities.values() if e.wikidataId})
    files: dict[str, dict[str, str]] = {}
    for i in range(0, len(qids), 150):
        values = " ".join(f"wd:{q}" for q in qids[i : i + 150])
        query = (
            "SELECT ?item ?img ?logo WHERE { VALUES ?item { " + values + " } "
            "OPTIONAL { ?item wdt:P18 ?img } OPTIONAL { ?item wdt:P154 ?logo } }"
        )
        try:
            r = requests.post(
                "https://query.wikidata.org/sparql",
                data={"query": query},
                headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT},
                timeout=120,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            log.warning("could not fetch images: %s", e)
            return
        for b in r.json()["results"]["bindings"]:
            q = b["item"]["value"].rsplit("/", 1)[1]
            d = files.setdefault(q, {})
            for key in ("img", "logo"):
                if key in b and key not in d:
                    d[key] = unquote(b[key]["value"].rsplit("/", 1)[1])
    for e in reg.entities.values():
        f = files.get(e.wikidataId or "", {})
        order = ("img", "logo") if e.type == "PER" else ("logo", "img")
        e.imageUrl = next((u for k in order if k in f and (u := commons_thumb(f[k]))), None)


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
    limit: int = 20000,
    batch: int = 250,
    ner_backend: str = "flair",
    sentiment_key: str = "ens",
    use_wikidata: bool = True,
    max_wikidata_lookups: int = 150,
    agg_every: int = 2000,
) -> dict:
    from .ner import load_ner
    from .sentiment import SentimentModel

    t0 = time.time()
    comments = fetch_unanalyzed(fs, limit)
    log.info("%d unanalyzed comments", len(comments))
    entity_docs = load_entity_docs(fs)
    # (entity, day) pairs whose mentions were written by a run that stopped before aggregating.
    state = fs.get(STATE_DOC)
    pending = {tuple(p.split("|", 1)) for p in ((state.data.get("pendingPairs") or []) if state else [])}
    if not comments:
        info = aggregate_pairs(fs, pending, entity_docs)
        if pending:
            fs.commit([Write(STATE_DOC, {"pendingPairs": []}, mask=["pendingPairs"])])
        return {"comments": 0, "recoveredPairs": len(pending), **info}
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
    to_aggregate: set[tuple[str, str]] = set(pending)
    all_days: set[str] = set()
    changed_entities = 0
    last_agg = 0
    known_entities = set(entity_docs)

    for i in range(0, len(comments), batch):
        chunk = comments[i : i + batch]
        ctx = article_context(fs, {c.get("articleId") for c in chunk if c.get("articleId")})
        result = analyze_comments(
            chunk, registry, ner, sentiment, source_country=countries, article_context=ctx, wikidata=wd, stats=stats
        )
        pre: list[Write] = []
        # New entities discovered through Wikidata must exist before mentions reference them.
        for eid, e in registry.entities.items():
            if eid not in known_entities:
                pre.append(new_entity_write(e))
                known_entities.add(eid)
                entity_docs[eid] = {**e.metadata(), **ZERO_STATS}
        groups: list[list[Write]] = []  # one atomic group per comment: its mentions + its flags
        batch_pairs: set[tuple[str, str]] = set()
        ts = now()
        for c in chunk:
            g: list[Write] = []
            new_ids = {m["entityId"] for m in result[c["id"]]}
            # Only a comment analysed before can have mentions to remove (saves a query per new comment).
            if c.get("analysisVersion") or c.get("analyzedAt"):
                for d in fs.query("mentions", where=[("commentId", "EQUAL", c["id"])], select=["entityId", "day"]):
                    if d.data.get("entityId") not in new_ids:
                        g.append(Write(f"mentions/{d.id}", None))
                        batch_pairs.add((d.data.get("entityId", ""), d.data.get("day", "")))
            for m in result[c["id"]]:
                g.append(Write(f"mentions/{m['commentId']}_{m['entityId']}", m))
                batch_pairs.add((m["entityId"], m["day"]))
            g.append(
                Write(
                    f"comments/{c['id']}",
                    {"analyzed": True, "analyzedAt": ts, "analysisVersion": version},
                    mask=["analyzed", "analyzedAt", "analysisVersion"],
                    exists=True,
                )
            )
            groups.append(g)
        batch_pairs = {p for p in batch_pairs if p[0] and p[1]}
        new_pending = batch_pairs - to_aggregate
        to_aggregate |= batch_pairs
        # Remember what still has to be aggregated *before* the mentions land.
        if new_pending:
            pre.append(
                Write(STATE_DOC, {"pendingPairs": sorted(f"{e}|{d}" for e, d in to_aggregate)}, mask=["pendingPairs"])
            )
        fs.commit(pre)
        fs.commit_groups(groups)
        log.info(
            "batch %d-%d: %d mentions so far (ner %.0fs, sentiment %.0fs, %d reads, %d writes)",
            i,
            i + len(chunk),
            stats.mentions,
            stats.ner_seconds,
            stats.sentiment_seconds,
            fs.reads,
            fs.writes,
        )
        # Aggregate every ~agg_every comments and at the end, so the dashboard fills progressively.
        done = i + len(chunk)
        if done - last_agg < agg_every and done < len(comments):
            continue
        last_agg = done
        info = aggregate_pairs(fs, to_aggregate, entity_docs)
        changed_entities += info["changedEntities"]
        all_days |= {d for _, d in to_aggregate}
        to_aggregate = set()
        fs.commit([Write(STATE_DOC, {"pendingPairs": []}, mask=["pendingPairs"])])
        if wd is not None:
            wd.save()
    summary = {
        "comments": stats.comments,
        "sentences": stats.sentences,
        "nerSentences": stats.ner_sentences,
        "mentions": stats.mentions,
        "nerSeconds": round(stats.ner_seconds, 1),
        "sentimentSeconds": round(stats.sentiment_seconds, 1),
        "linkSeconds": round(stats.link_seconds, 1),
        "wikidataLookups": stats.wikidata_lookups,
        "discovered": stats.discovered,
        "days": sorted(all_days),
        "version": version,
        "seconds": round(time.time() - t0, 1),
        "changedEntities": changed_entities,
    }
    log.info("run summary: %s", summary)
    return summary


# --------------------------------------------------------------------------- aggregation


def aggregate_pairs(fs: Firestore, pairs: set[tuple[str, str]], entity_docs: dict[str, dict]) -> dict:
    """Recompute the day docs of the affected (entity, day) pairs from their mentions.

    Reads only the mentions of those pairs (query entityId == X AND day == D) and the old day
    docs; entity totals change by (new - old) in the same atomic commit as the day docs.
    Then slide rolling windows, rewrite trending and stats.
    """
    today = today_berlin()
    pairs = {(e, d) for e, d in pairs if e in entity_docs and entity_docs[e].get("allowlisted", True)}
    old = fs.batch_get([f"entities/{e}/days/{d}" for e, d in sorted(pairs)]) if pairs else {}
    changes: dict[str, list[tuple[str, dict | None, dict | None]]] = {}
    new_docs: dict[tuple[str, str], dict] = {}
    for e, day in sorted(pairs):
        mentions = [
            x.data
            for x in fs.query(
                "mentions", where=[("entityId", "EQUAL", e), ("day", "EQUAL", day)], select=MENTION_FIELDS
            )
        ]
        nd = agg.build_day_docs(mentions).get((e, day))
        o = old.get(f"entities/{e}/days/{day}")
        od = o.data if o else None
        if (od is None and nd is None) or agg.same_day_doc(od, nd):
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
        if len(pending) + len(group) > 100:
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
    trend = agg.trending(entity_docs, today)
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


def reaggregate_since(fs: Firestore, since: str) -> dict:
    """Recompute only the (entity, day) pairs with mentions on or after `since` (cheap repair)."""
    entity_docs = load_entity_docs(fs)
    pairs = {
        (d.data.get("entityId", ""), d.data.get("day", ""))
        for d in fs.query("mentions", where=[("day", "GREATER_THAN_OR_EQUAL", since)], select=["entityId", "day"])
    }
    return {"pairs": len(pairs), **aggregate_pairs(fs, pairs, entity_docs)}


def reaggregate(fs: Firestore) -> dict:
    """Rebuild every day doc and all entity counters from scratch from `mentions` (full scan)."""
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
        if eid not in by_entity and not d.get("mentions") and not d.get("recentDays"):
            continue  # already zero: save a write
        upd["updatedAt"] = ts
        d.update(upd)
        writes.append(Write(f"entities/{eid}", upd, mask=ENTITY_STAT_FIELDS))
    # Small commits: a full rebuild does not need cross-document atomicity.
    for i in range(0, len(writes), 100):
        fs.commit_atomic(writes[i : i + 100])
    refresh_rolling(fs, entity_docs, set(entity_docs))
    return {
        "mentions": len(mentions),
        "dayDocs": len(docs),
        "entitiesWithMentions": len(by_entity),
        "deletedDayDocs": len(existing_days) - len(wanted & set(existing_days)),
    }
