"""Firestore orchestration: run (analyze new comments), reaggregate, seed."""

from __future__ import annotations

import logging
import re
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

COMMENT_FIELDS = [
    "articleId",
    "sourceId",
    "sourceType",
    "text",
    "publishedAt",
    "crawledAt",
    "day",
    "analysisVersion",
    "analyzedAt",
]
MENTION_FIELDS = [
    "commentId",
    "articleId",
    "entityId",
    "sourceId",
    "sourceType",
    "day",
    "label",
    "score",
    "publishedAt",
    "coEntityIds",
]
PEER_FIELDS = ["rankInType", "peersInType", "percentileInType", "rankAll", "peersAll", "percentileAll"]
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
    "imageKind",
    "aliases",
    "allowlisted",
    *ENTITY_STAT_FIELDS,
    "relative7d",
    "communityRecentDays",
    "communityText7d",
    *PEER_FIELDS,
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
                imageKind=d.get("imageKind"),
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
        if not with_images:  # keep stored images
            meta["imageUrl"] = cur.get("imageUrl")
            meta["imageKind"] = cur.get("imageKind")
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


IMAGE_PROPS = {"P18": "img", "P154": "logo", "P8972": "icon", "P158": "seal"}


def _fill_images(reg: Registry) -> None:
    """Thumbnail per entity from Wikidata (one SPARQL query per 150 items).

    Persons: portrait (P18). Organisations, parties, clubs, media: logo only (P154, then P8972
    small logo/icon, then P158 seal); never P18, which is often a photo of a building. No logo
    -> imageUrl None (the frontend shows initials).
    """
    import requests

    from .wikidata import USER_AGENT, commons_thumb

    qids = sorted({e.wikidataId for e in reg.entities.values() if e.wikidataId})
    files: dict[str, dict[str, str]] = {}
    for i in range(0, len(qids), 150):
        values = " ".join(f"wd:{q}" for q in qids[i : i + 150])
        optional = " ".join(f"OPTIONAL {{ ?item wdt:{p} ?{v} }}" for p, v in IMAGE_PROPS.items())
        variables = " ".join("?" + v for v in IMAGE_PROPS.values())
        query = f"SELECT ?item {variables} WHERE {{ VALUES ?item {{ {values} }} {optional} }}"
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
            raise
        for b in r.json()["results"]["bindings"]:
            q = b["item"]["value"].rsplit("/", 1)[1]
            d = files.setdefault(q, {})
            for key in IMAGE_PROPS.values():
                if key in b and key not in d:
                    d[key] = unquote(b[key]["value"].rsplit("/", 1)[1])
    for e in reg.entities.values():
        f = files.get(e.wikidataId or "", {})
        order = ("img",) if e.type == "PER" else ("logo", "icon", "seal")
        e.imageUrl = next((u for k in order if k in f and (u := commons_thumb(f[k]))), None)
        e.imageKind = ("portrait" if e.type == "PER" else "logo") if e.imageUrl else None


# --------------------------------------------------------------------------- run


def seed_entity_requests(fs: Firestore) -> dict:
    """Process accepted entity requests (never fails the analysis run)."""
    from .entity_requests import process_entity_requests

    try:
        return process_entity_requests(fs)
    except Exception as e:
        log.warning("entity requests failed: %s", e)
        return {"error": str(e)[:200]}


def fetch_unanalyzed(fs: Firestore, limit: int, newest_first: bool = True) -> list[dict]:
    """Unanalysed comments, newest first so the dashboard reflects recent days first.

    Needs the composite index comments(analyzed ASC, crawledAt DESC); falls back to oldest first.
    """
    direction = "DESCENDING" if newest_first else "ASCENDING"
    try:
        docs = list(
            fs.query(
                "comments",
                where=[("analyzed", "EQUAL", False)],
                order_by=[("crawledAt", direction)],
                limit=limit,
                select=COMMENT_FIELDS,
            )
        )
    except RuntimeError as e:
        if not newest_first or "index" not in str(e).lower():
            raise
        log.warning("index for newest-first missing, falling back to oldest first")
        return fetch_unanalyzed(fs, limit, newest_first=False)
    return [{"id": d.id, **d.data} for d in docs]


def run(
    fs: Firestore,
    limit: int = 20000,
    batch: int = 250,
    ner_backend: str = "bert",
    sentiment_key: str = "cal",
    use_wikidata: bool = True,
    max_wikidata_lookups: int = 150,
    agg_every: int = 2000,
) -> dict:
    from .ner import load_ner
    from .sentiment import load_sentiment

    t0 = time.time()
    requests_info = seed_entity_requests(fs)
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
    sentiment = load_sentiment(sentiment_key)
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
        "entityRequests": requests_info,
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
    community: dict[tuple[str, str], list[dict]] = {}
    for e, day in sorted(pairs):
        mentions = [
            x.data
            for x in fs.query(
                "mentions", where=[("entityId", "EQUAL", e), ("day", "EQUAL", day)], select=MENTION_FIELDS
            )
        ]
        community[(e, day)] = [m for m in mentions if agg.is_community(m)]
        mentions = [m for m in mentions if not agg.is_community(m)]
        nd = agg.build_day_docs(mentions).get((e, day))
        o = old.get(f"entities/{e}/days/{day}")
        od = o.data if o else None
        if (od is None and nd is None) or agg.same_day_doc(od, nd):
            continue
        changes.setdefault(e, []).append((day, od, nd))
        if nd is not None:
            new_docs[(e, day)] = nd
    _commit_entity_changes(fs, changes, new_docs, entity_docs, today)
    n_comm = aggregate_community(fs, community, entity_docs, today)
    refresh_rolling(fs, entity_docs, set(changes))
    return {
        "changedEntities": len(changes),
        "changedDays": sum(len(v) for v in changes.values()),
        "changedCommunityDays": n_comm,
    }


def aggregate_community(
    fs: Firestore, community: dict[tuple[str, str], list[dict]], entity_docs: dict[str, dict], today: str
) -> int:
    """Write entities/{id}/community_text/{day} for user-submitted texts (kept out of the main numbers).

    `community`: {(entityId, day): community mentions of that pair} (empty list = none left).
    """
    if not community:
        return 0
    old = fs.batch_get([f"entities/{e}/community_text/{d}" for e, d in sorted(community)])
    new = agg.community_day_docs([m for ms in community.values() for m in ms])
    writes: list[Write] = []
    touched: set[str] = set()
    for e, day in sorted(community):
        path = f"entities/{e}/community_text/{day}"
        o = old.get(path)
        od = {k: v for k, v in (o.data if o else {}).items() if k != "updatedAt"} or None
        nd = new.get((e, day))
        if od == nd or (od is None and nd is None):
            continue
        writes.append(Write(path, None) if nd is None else Write(path, nd))
        rec = dict(entity_docs[e].get("communityRecentDays") or {})
        if nd is None:
            rec.pop(day, None)
        else:
            rec[day] = {k: nd[k] for k in ("mentions", "pos", "neu", "neg", "scoreSum")}
        entity_docs[e]["communityRecentDays"] = agg.prune_recent(rec, today)
        touched.add(e)
    for e in touched:
        d = entity_docs[e]
        d["communityText7d"] = agg.community_window(d["communityRecentDays"], today)
        writes.append(
            Write(
                f"entities/{e}",
                {"communityRecentDays": d["communityRecentDays"], "communityText7d": d["communityText7d"]},
                mask=["communityRecentDays", "communityText7d"],
            )
        )
    fs.commit(writes)
    return len(writes) - len(touched)


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
    pending: dict[str, set[str]] = {}
    # 1. slide windows
    for eid, d in entity_docs.items():
        if eid in already or (not d.get("recentDays") and not d.get("mentions7d") and not d.get("mentionsPrev7d")):
            continue
        recent = agg.prune_recent(d.get("recentDays") or {}, today)
        win = agg.window_stats(recent, today)
        cur = {k: d.get(k) for k in win}
        if cur != win or recent != (d.get("recentDays") or {}):
            d.update({**win, "recentDays": recent})
            pending.setdefault(eid, set()).update([*win, "recentDays"])
    for eid, d in entity_docs.items():
        if d.get("communityRecentDays") or d.get("communityText7d"):
            rec = agg.prune_recent(d.get("communityRecentDays") or {}, today)
            win = agg.community_window(rec, today)
            if win != d.get("communityText7d") or rec != (d.get("communityRecentDays") or {}):
                d.update({"communityRecentDays": rec, "communityText7d": win})
                pending.setdefault(eid, set()).update(["communityRecentDays", "communityText7d"])
    # 2. global baseline and relative sentiment per entity
    baseline, by_source = agg.baselines(entity_docs, today)
    for eid, d in entity_docs.items():
        rel = agg.relative(d, baseline)
        if d.get("relative7d") != rel and (rel is not None or "relative7d" in d):
            d["relative7d"] = rel
            pending.setdefault(eid, set()).add("relative7d")
    # 3. peer ranks within type and overall
    ranks, strip = agg.peer_ranks(entity_docs)
    for eid, d in entity_docs.items():
        want = ranks.get(eid, dict.fromkeys(PEER_FIELDS))
        if any(d.get(k) != want[k] for k in PEER_FIELDS) and (eid in ranks or any(k in d for k in PEER_FIELDS)):
            d.update(want)
            pending.setdefault(eid, set()).update(PEER_FIELDS)
    for eid, fields in pending.items():
        upd = {f: entity_docs[eid].get(f) for f in fields}
        upd["updatedAt"] = ts
        writes.append(Write(f"entities/{eid}", upd, mask=list(upd)))
    fs.commit(writes)
    trend = agg.trending(entity_docs, today)
    trend["updatedAt"] = ts
    trend["baseline7d"] = baseline
    trend["baselineBySource7d"] = by_source
    total_mentions = fs.count("mentions") - fs.count("mentions", [("sourceId", "EQUAL", agg.COMMUNITY)])
    tracked = sum(1 for d in entity_docs.values() if d.get("allowlisted") and (d.get("mentions") or 0) > 0)
    peers = {"updatedAt": ts, "baseline7d": baseline, "byType": strip}
    # meta/trending and meta/peers are only written when their content changed
    cur = fs.batch_get(["meta/trending", "meta/peers"])
    meta_writes = []
    for path, doc in (("meta/trending", trend), ("meta/peers", peers)):
        old = cur.get(path)
        if old is None or _strip_ts(old.data) != _strip_ts(doc):
            meta_writes.append(Write(path, doc))
    fs.commit(
        [
            *meta_writes,
            Write(
                "meta/stats",
                {
                    "mentions": total_mentions,
                    "entities": tracked,
                    "lastAnalysisAt": ts,
                    "baseline7d": baseline,
                    "baselineBySource7d": by_source,
                },
                mask=["mentions", "entities", "lastAnalysisAt", "baseline7d", "baselineBySource7d"],
            ),
        ]
    )


def rescore(
    fs: Firestore,
    sentiment_key: str = "cal",
    ner_version: str | None = None,
    batch: int = 400,
    **model_kw,
) -> dict:
    """Re-run only the sentiment step for all existing mentions (NER/linking results are kept).

    The mention positions are recovered with the gazetteer (or the stored surface); labels,
    scores and confidences are rewritten where they change, the comments get the new
    analysisVersion, and the affected (entity, day) pairs are re-aggregated.
    """
    from .analyze import mention_window
    from .linker import Match
    from .sentiment import combine, load_sentiment
    from .textutil import clean_text, split_sentences

    t0 = time.time()
    entity_docs = load_entity_docs(fs)
    registry = build_registry(entity_docs)
    countries = source_countries(fs)
    model = load_sentiment(sentiment_key, **model_kw)
    by_comment: dict[str, list] = {}
    for d in fs.query("mentions", select=["commentId", "entityId", "surface", "label", "score", "confidence", "day"]):
        by_comment.setdefault(d.data["commentId"], []).append(d)
    cids = sorted(by_comment)
    changed_pairs: set[tuple[str, str]] = set()
    n_changed = n_mentions = 0
    for i in range(0, len(cids), batch):
        chunk = cids[i : i + batch]
        docs = fs.batch_get([f"comments/{c}" for c in chunk], mask=["text", "sourceId", "analysisVersion"])
        items: list[tuple[str, str]] = []
        plan: list[tuple[str, object, list[int]]] = []
        for cid in chunk:
            cd = docs.get(f"comments/{cid}")
            if cd is None:
                continue
            text = clean_text(cd.data.get("text") or "")
            sents = split_sentences(text)
            matches = registry.gazetteer(text, [], "", countries.get(cd.data.get("sourceId") or ""))
            for md in by_comment[cid]:
                eid = md.data["entityId"]
                mine = [m for m in matches if m.entity_id == eid]
                if not mine:
                    surf = md.data.get("surface") or ""
                    j = text.casefold().find(surf.casefold()) if surf else -1
                    if j < 0:
                        continue
                    mine = [Match(j, j + len(surf), surf, eid, "surface")]
                    matches.append(mine[0])
                name = registry.entities[eid].name if eid in registry.entities else md.data.get("surface", "")
                idxs = []
                for m in mine[:3]:
                    items.append((mention_window(text, sents, m, matches, model.window_mode), name))
                    idxs.append(len(items) - 1)
                plan.append((cid, md, idxs))
        scored = model.predict(items) if items else []
        groups: list[list[Write]] = []
        version_of: dict[str, str] = {}
        for cid, md, idxs in plan:
            label, score, conf = combine([scored[k] for k in idxs])
            n_mentions += 1
            old = (md.data.get("label"), md.data.get("score"), md.data.get("confidence"))
            if old != (label, score, conf):
                n_changed += 1
                groups.append(
                    [
                        Write(
                            md.path,
                            {"label": label, "score": score, "confidence": conf},
                            mask=["label", "score", "confidence"],
                        )
                    ]
                )
                changed_pairs.add((md.data["entityId"], md.data["day"]))
            cd = docs[f"comments/{cid}"]
            old_v = cd.data.get("analysisVersion") or ""
            ner_part = ner_version or (old_v.split("+", 1)[0] if old_v else "unknown")
            version_of[cid] = f"{ner_part}+{model.version}+link@1"
        for cid, v in version_of.items():
            groups.append([Write(f"comments/{cid}", {"analysisVersion": v}, mask=["analysisVersion"])])
        fs.commit_groups(groups)
        log.info("rescore %d/%d comments, %d/%d mentions changed", i + len(chunk), len(cids), n_changed, n_mentions)
    info = aggregate_pairs(fs, changed_pairs, entity_docs)
    return {
        "comments": len(cids),
        "mentions": n_mentions,
        "changed": n_changed,
        "version": model.version,
        "seconds": round(time.time() - t0, 1),
        **info,
    }


def _strip_ts(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "updatedAt"}


class _OnlyFor:
    """Sentiment proxy: runs the real model only for windows about `names` (others get a dummy)."""

    def __init__(self, model, names: set[str]):
        self.model, self.names = model, names
        self.window_mode = getattr(model, "window_mode", "clause")
        self.version = model.version

    def predict(self, items: list) -> list[tuple[str, float, float]]:
        idx = [i for i, x in enumerate(items) if isinstance(x, tuple) and x[1] in self.names]
        out: list[tuple[str, float, float]] = [("neu", 0.0, 0.0)] * len(items)
        for i, r in zip(idx, self.model.predict([items[i] for i in idx]) if idx else [], strict=True):
            out[i] = r
        return out


def relink(
    fs: Firestore,
    days: int = 14,
    entity_ids: set[str] | None = None,
    ner_backend: str = "bert",
    sentiment_key: str = "cal",
    batch: int = 250,
) -> dict:
    """Add mentions of newly added entities to already analysed comments (last `days` days).

    Only comments whose text contains an alias of a new entity are re-run (NER + linking);
    sentiment is computed only for the new (comment, entity) pairs, existing mentions keep their
    labels (their coEntityIds are extended). Afterwards the affected pairs are aggregated.
    """
    from datetime import date, timedelta

    from .ner import load_ner
    from .sentiment import load_sentiment
    from .textutil import clean_text

    t0 = time.time()
    entity_docs = load_entity_docs(fs)
    registry = build_registry(entity_docs)
    if entity_ids is None:  # default: generated entities without any mention yet
        entity_ids = {
            eid
            for eid, e in registry.entities.items()
            if e.source == "generated" and not (entity_docs.get(eid) or {}).get("mentions")
        }
    new = {eid for eid in entity_ids if eid in registry.entities}
    names = {registry.entities[e].name for e in new}
    alias_texts = sorted({a.text.casefold() for a in registry.aliases if a.entity_id in new}, key=len, reverse=True)
    if not alias_texts:
        return {"newEntities": 0}
    prefilter = re.compile(r"(?<![\w])(" + "|".join(re.escape(a) for a in alias_texts) + r")", re.IGNORECASE)
    cutoff = (date.fromisoformat(today_berlin()) - timedelta(days=days - 1)).isoformat()
    candidates = []
    scanned = 0
    for d in fs.query("comments", where=[("day", "GREATER_THAN_OR_EQUAL", cutoff)], select=COMMENT_FIELDS):
        scanned += 1
        if d.data.get("analysisVersion") and prefilter.search(clean_text(d.data.get("text") or "")):
            candidates.append({"id": d.id, **d.data})
    log.info("relink: %d new entities, %d comments scanned, %d candidates", len(new), scanned, len(candidates))
    ner = load_ner(ner_backend)
    sentiment = _OnlyFor(load_sentiment(sentiment_key), names)
    countries = source_countries(fs)
    pairs: set[tuple[str, str]] = set()
    added = 0
    for i in range(0, len(candidates), batch):
        chunk = candidates[i : i + batch]
        ctx = article_context(fs, {c.get("articleId") for c in chunk if c.get("articleId")})
        result = analyze_comments(chunk, registry, ner, sentiment, source_country=countries, article_context=ctx)
        groups: list[list[Write]] = []
        for c in chunk:
            ms = result[c["id"]]
            fresh = [m for m in ms if m["entityId"] in new]
            if not fresh:
                continue
            all_ids = {m["entityId"] for m in ms}
            g = [Write(f"mentions/{m['commentId']}_{m['entityId']}", m) for m in fresh]
            for d in fs.query(
                "mentions", where=[("commentId", "EQUAL", c["id"])], select=["entityId", "coEntityIds", "day"]
            ):
                if d.data.get("entityId") in new:
                    continue
                co = sorted(set(d.data.get("coEntityIds") or []) | ({m["entityId"] for m in fresh} & all_ids))
                if co != sorted(d.data.get("coEntityIds") or []):
                    g.append(Write(d.path, {"coEntityIds": co}, mask=["coEntityIds"]))
                    pairs.add((d.data["entityId"], d.data["day"]))
            pairs |= {(m["entityId"], m["day"]) for m in fresh}
            added += len(fresh)
            groups.append(g)
        fs.commit_groups(groups)
        log.info("relink %d/%d comments, %d new mentions", i + len(chunk), len(candidates), added)
    entity_docs = load_entity_docs(fs)
    info = aggregate_pairs(fs, pairs, entity_docs)
    return {
        "newEntities": len(new),
        "scanned": scanned,
        "candidates": len(candidates),
        "newMentions": added,
        "seconds": round(time.time() - t0, 1),
        **info,
    }


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
    community_mentions = [m for m in mentions if agg.is_community(m)]
    mentions = [m for m in mentions if not agg.is_community(m)]
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
    community: dict[tuple[str, str], list[dict]] = {}
    for d in fs.run_query(
        {
            "from": [{"collectionId": "community_text", "allDescendants": True}],
            "select": {"fields": [{"fieldPath": "__name__"}]},
        }
    ):
        parts = d.path.split("/")
        community[(parts[1], parts[3])] = []
    for m in community_mentions:
        community.setdefault((m["entityId"], m["day"]), []).append(m)
    aggregate_community(fs, community, entity_docs, today)
    refresh_rolling(fs, entity_docs, set(entity_docs))
    return {
        "mentions": len(mentions),
        "dayDocs": len(docs),
        "entitiesWithMentions": len(by_entity),
        "deletedDayDocs": len(existing_days) - len(wanted & set(existing_days)),
    }
