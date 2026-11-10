"""Firestore orchestration: run (analyze new comments), reaggregate, seed."""

from __future__ import annotations

import logging
import os
import re
import time
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from urllib.parse import unquote

import yaml

from . import aggregate as agg
from . import textindex, timebudget
from .analyze import Stats, analyze_comments
from .budget import Budget
from .firestore import Doc, Firestore, Write
from .linker import DATA_DIR, Entity, Registry
from .registry_store import (
    ACTIVE_DOC,
    ensure_docs,
    load_active_docs,
    load_registry,
    needs_upkeep,
    registry_add_writes,
    registry_entries_from_docs,
    write_registry_doc,
)
from .sentiment import DEFAULT_SENTIMENT
from .textutil import BERLIN, clean_text, day_key, normalize_surface, today_berlin

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


def source_countries(fs: Firestore) -> dict[str, str]:
    try:
        return {d.id: d.data.get("country") or "" for d in fs.query("sources", select=["country"])}
    except RuntimeError as e:
        log.warning("could not read sources: %s", e)
        return {}


def article_context(fs: Firestore, article_ids: set[str], cache: dict[str, str] | None = None) -> dict[str, str]:
    """Title and topics of the articles (context for contextual aliases), 1 read per article.

    `cache` (article id -> context, "" for a missing article) is kept by the caller for the whole
    run, so an article whose comments span several batches is read once."""
    cache = {} if cache is None else cache
    missing = sorted(a for a in article_ids if a and a not in cache)
    if missing:
        docs = fs.batch_get([f"articles/{a}" for a in missing], mask=["title", "topics"])
        for path, d in docs.items():
            topics = (d.data.get("topics") or []) if d else []
            title = (d.data.get("title") or "") if d else ""
            cache[path.split("/", 1)[1]] = " ".join([title, *[str(t) for t in topics]]) if d else ""
    return {a: cache[a] for a in article_ids if cache.get(a)}


def new_entity_write(e: Entity) -> Write:
    return Write(f"entities/{e.id}", {**e.metadata(), **ZERO_STATS, "updatedAt": now()})


# --------------------------------------------------------------------------- seed


SEED_RELINK_DAYS = 14  # new entities are relinked over this many days of analysed comments


def seed(fs: Firestore, with_images: bool = True, budget: Budget | None = None, queue_followups: bool = True) -> dict:
    """Write curated entities and events; only documents whose content changed are written.

    With a budget, the writes stop at the remaining daily budget; running seed again later
    continues (it only writes what still differs).

    Follow-up jobs (`queue_followups`), continued by the regular `run` within the budget:
    * a `relink` of the entities this seed created (last SEED_RELINK_DAYS days), merged into an
      already queued relink;
    * a `prune` of existing entities that lost an alias ("Facebook" moved from Meta Platforms to
      the Facebook platform; "Die Katze" dropped from Daniela Katzenberger): their mentions are
      re-checked, and the new owners of the alias, if any, may take them over."""
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
    stopped = None
    created: set[str] = set()
    for i in range(0, len(writes), 400):
        chunk = writes[i : i + 400]
        if budget is not None and not budget.allows(writes=len(chunk) + 2):
            stopped = f"daily Firestore budget reached after {i} of {len(writes)} writes; run seed again"
            log.warning("seed: %s", stopped)
            break
        fs.commit(chunk)
        created |= {
            w.path.split("/", 1)[1]
            for w in chunk
            if w.path.startswith("entities/") and w.path.split("/", 1)[1] not in existing
        }
    # meta/registry: the Firestore-only (discovered / requested) entities, for runs without a full scan
    entries = registry_entries_from_docs(existing, reg)
    cur = fs.get("meta/registry")
    if cur is None or (cur.data.get("entities") or {}) != entries:
        write_registry_doc(fs, entries)
    new = len(reg.entities.keys() - existing.keys())
    queued = _queue_seed_followups(fs, reg, existing, created) if queue_followups else {}
    out = {
        "entities": len(reg.entities),
        "newEntities": new,
        "updatedEntities": sum(1 for w in writes if w.path.startswith("entities/")) - new,
        "changedImages": changed_images,
        "withImage": sum(1 for e in reg.entities.values() if e.imageUrl),
        "events": len(events),
        "stopped": stopped,
        **({"followups": queued} if queued else {}),
    }
    log.info("seed: %s", out)
    return out


def moved_aliases(reg: Registry, existing: dict[str, dict]) -> dict[str, set[str]]:
    """Existing entities that lost an alias -> the entities that now own it (empty if none does:
    a dropped alias such as "Die Katze" or "der Boss" leaves mentions that must be re-checked)."""
    out: dict[str, set[str]] = {}
    for eid, cur in existing.items():
        e = reg.entities.get(eid)
        if e is None or not cur.get("aliases"):
            continue
        now_aliases = {normalize_surface(a) for a in e.aliases}
        for a in cur["aliases"]:
            key = normalize_surface(a)
            if key in now_aliases:
                continue
            owners = {o for o in reg.alias_index.get(key, []) if o != eid}
            out.setdefault(eid, set()).update(owners)
    return out


def _queue_seed_followups(fs: Firestore, reg: Registry, existing: dict[str, dict], created: set[str]) -> dict:
    out: dict = {}
    if created:
        cutoff = (date.fromisoformat(today_berlin()) - timedelta(days=SEED_RELINK_DAYS - 1)).isoformat()
        out["relink"] = queue_job(fs, "relink", {"cutoff": cutoff, "entityIds": sorted(created)})
    moved = moved_aliases(reg, existing)
    if moved:
        out["prune"] = queue_job(fs, "prune", {"ids": sorted(moved), "add": sorted(set().union(*moved.values()))})
    return out


IMAGE_PROPS = {"P18": "img", "P154": "logo", "P8972": "icon", "P2910": "appicon", "P158": "seal"}


def _fill_images(reg: Registry) -> None:
    """Thumbnail per entity from Wikidata (one SPARQL query per 150 items).

    Persons: portrait (P18). Organisations, parties, clubs, media, products: logo only (P154, then
    P8972 small logo/icon, then P2910 icon (apps), then P158 seal); never P18, which is often a
    photo of a building. No logo
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
        for attempt in range(6):
            try:
                r = requests.post(
                    "https://query.wikidata.org/sparql",
                    data={"query": query},
                    headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT},
                    timeout=120,
                )
                if r.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                    # rate limited: honour Retry-After, otherwise back off exponentially
                    wait = float(r.headers.get("Retry-After") or 0) or 5 * 2**attempt
                    log.warning("Wikidata %s, retrying in %.0fs", r.status_code, wait)
                    time.sleep(min(wait, 120))
                    continue
                r.raise_for_status()
                break
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
        order = ("img",) if e.type == "PER" else ("logo", "icon", "appicon", "seal")
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
    sentiment_key: str = DEFAULT_SENTIMENT,
    use_wikidata: bool = True,
    max_wikidata_lookups: int = 150,
    agg_every: int = 0,
    budget: Budget | None = None,
) -> dict:
    """Analyse unanalysed comments within the daily Firestore budget (see analysis/budget.py).

    `agg_every`: aggregate after this many comments as well as at the end (0, the default: only
    at the end). Every aggregation re-reads all mentions of the touched (entity, day) pairs and
    rewrites their day and entity docs, so intermediate rounds cost reads and writes again."""
    mark = fs.ops_mark()
    budget = budget or Budget.load(fs)
    why = budget.exhausted()
    if why:
        log.warning("daily Firestore budget: %s - skipping this run", why)
        return {"comments": 0, "skipped": why, "budget": budget.summary()}
    try:
        out, entity_docs = _run(
            fs, budget, limit, batch, ner_backend, sentiment_key, use_wikidata, max_wikidata_lookups, agg_every
        )
        if out.get("backlogDone") and not out.get("stopped"):
            # backlog cleared: spend what is left of today's budget on unfinished one-off jobs
            out["jobs"] = continue_jobs(
                fs, budget, ner_backend=ner_backend, sentiment_key=sentiment_key, entity_docs=entity_docs
            )
    finally:
        budget.record()
    out["budget"] = budget.summary()
    out["timeBudget"] = timebudget.get().summary()
    out["firestoreOps"] = fs.ops_summary(mark)  # where the reads and writes went (per collection)
    log.info("run summary: %s", out)
    return out


# estimated Firestore operations per analysed comment (refined from the run's own numbers)
EST_WRITES_PER_COMMENT = 1.5
EST_READS_PER_COMMENT = 2.0
# pairs kept in meta/analysis.pendingPairs before an aggregation is forced (bounds the size of that doc)
MAX_PENDING_PAIRS = 8000
# comment index (analysis/textindex.py): documents older than this are deleted; a relink never looks
# further back than its `--days` (14 by default)
INDEX_KEEP_DAYS = 45


def index_enabled() -> bool:
    """ANALYSIS_TEXT_INDEX=0 stops writing the comment index (relinks then scan all comments again)."""
    return os.environ.get("ANALYSIS_TEXT_INDEX", "1").strip().lower() not in ("0", "false", "off", "no")


def _index_since(state: Doc | None) -> str | None:
    """First day whose analysed comments are all in the comment index (None: no usable index)."""
    since = state.data.get("indexSince") if state else None
    return since if index_enabled() and isinstance(since, str) and since else None


def _index_writes(chunk: list[dict], ts: datetime) -> list[Write]:
    """Index documents for a batch of comments (one per day of the batch)."""
    rows = []
    for c in chunk:
        published = c.get("publishedAt")
        day = c.get("day") or (day_key(published) if published else "")
        rows.append((c["id"], day, clean_text(c.get("text") or "")))
    return [Write(path, {**data, "createdAt": ts}) for path, data in textindex.index_docs(rows).items()]


def prune_index(fs: Firestore, limit: int = 300) -> int:
    """Delete comment index documents older than INDEX_KEEP_DAYS (at least 1 read per call)."""
    cutoff = (date.fromisoformat(today_berlin()) - timedelta(days=INDEX_KEEP_DAYS)).isoformat()
    old = [
        d.path
        for d in fs.query(
            textindex.COLLECTION,
            where=[("day", "LESS_THAN", cutoff)],
            order_by=[("day", "ASCENDING")],  # a range filter needs its field first in the order
            limit=limit,
            select=[],
        )
    ]
    fs.commit([Write(p, None) for p in old])
    return len(old)


def _run(fs, budget, limit, batch, ner_backend, sentiment_key, use_wikidata, max_wikidata_lookups, agg_every):
    from .ner import load_ner
    from .sentiment import load_sentiment

    t0 = time.time()
    requests_info = seed_entity_requests(fs)
    # only fetch what the remaining budget can process (fetching costs 1 read per comment)
    fit = int(
        min(
            (budget.remaining_writes() - budget.reserve_writes) / EST_WRITES_PER_COMMENT,
            (budget.remaining_reads() - budget.reserve_reads) / EST_READS_PER_COMMENT,
        )
    )
    if fit < limit:
        log.warning("daily Firestore budget allows about %d comments this run (limit %d)", max(0, fit), limit)
        limit = max(0, fit)
    comments = fetch_unanalyzed(fs, limit) if limit > 0 else []
    log.info("%d unanalyzed comments", len(comments))
    # (entity, day) pairs whose mentions were written by a run that stopped before aggregating.
    state = fs.get(STATE_DOC)
    pending = {tuple(p.split("|", 1)) for p in ((state.data.get("pendingPairs") or []) if state else [])}
    if not comments and not pending and _refreshed_today(fs):
        # nothing new and the rolling windows were already moved to today: a run of a few reads
        return {"comments": 0, "recoveredPairs": 0, "backlogDone": limit > 0, "refresh": "not needed"}, None
    # entity docs: only those still needing upkeep (meta/active); touched ones are read on demand
    entity_docs, _active = load_active_docs(fs, ENTITY_READ_FIELDS)
    if not comments:
        info = aggregate_pairs(fs, pending, entity_docs)
        if pending:
            fs.commit([Write(STATE_DOC, {"pendingPairs": []}, mask=["pendingPairs"])])
        return {"comments": 0, "recoveredPairs": len(pending), "backlogDone": limit > 0, **info}, entity_docs
    registry = load_registry(fs)
    ner = load_ner(ner_backend)
    sentiment = load_sentiment(sentiment_key)
    version = f"{ner.version}+{sentiment.version}+link@1"
    _note_sentiment(fs, state, sentiment_key, sentiment.version)
    wd = None
    if use_wikidata:
        from .wikidata import WikidataLinker

        wd = WikidataLinker(max_lookups=max_wikidata_lookups)
    countries = source_countries(fs)
    stats = Stats()
    # Comment index for later relinks. `indexSince` is the first day all of whose analysed comments
    # are in the index: two days after the first run that writes it (a comment published "tomorrow"
    # by a source's clock may have been analysed before).
    index_on = index_enabled()
    stored_since = state.data.get("indexSince") if state else None
    index_state: list[Write] = []
    if index_on and not stored_since:
        since = (date.fromisoformat(today_berlin()) + timedelta(days=2)).isoformat()
        index_state = [Write(STATE_DOC, {"indexSince": since}, mask=["indexSince"])]
    elif not index_on and stored_since:
        index_state = [Write(STATE_DOC, {}, mask=["indexSince"])]  # switched off: the index has gaps from now on
    to_aggregate: set[tuple[str, str]] = set(pending)
    all_days: set[str] = set()
    changed_entities = 0
    last_agg = 0
    known_entities = set(registry.entities)
    stopped = None
    w_start, r_start = fs.writes, fs.reads
    tb = timebudget.get()
    t_batch, batch_s = time.time(), 0.0
    ctx_cache: dict[str, str] = {}

    for i in range(0, len(comments), batch):
        chunk = comments[i : i + batch]
        if i:
            batch_s, t_batch = time.time() - t_batch, time.time()
        # time: the next batch must end before the CI job timeout (see analysis/timebudget.py)
        if tb.reached("the analysis run", reserve=1.2 * batch_s):
            stopped = f"time budget reached after {i} comments"
            log.warning("%s; the remaining %d wait for the next run", stopped, len(comments) - i)
            break
        # budget: estimate the batch from the run's own rates so far; finish the run gracefully
        per_w = max(EST_WRITES_PER_COMMENT, (fs.writes - w_start) / i) if i else EST_WRITES_PER_COMMENT
        per_r = max(1.0, (fs.reads - r_start) / i) if i else EST_READS_PER_COMMENT
        if not budget.allows(writes=int(per_w * len(chunk)), reads=int(per_r * len(chunk))):
            stopped = f"daily Firestore budget reached after {i} comments"
            log.warning("%s; the remaining %d wait for the next run", stopped, len(comments) - i)
            break
        ctx = article_context(fs, {c.get("articleId") for c in chunk if c.get("articleId")}, ctx_cache)
        result = analyze_comments(
            chunk, registry, ner, sentiment, source_country=countries, article_context=ctx, wikidata=wd, stats=stats
        )
        # the index of this batch is written before its comments are marked as analysed, so every
        # analysed comment is in it (a batch that is repeated after a crash is simply indexed twice)
        pre: list[Write] = [*index_state]
        index_state = []
        if index_on:
            try:
                fs.commit(_index_writes(chunk, now()))
            except RuntimeError as e:
                # the index is an optimisation: never fail the run for it. Without `indexSince`
                # relinks scan all comments again, so the gap this leaves cannot hide a comment.
                log.warning("comment index not written, relinks scan again: %s", str(e)[:300])
                index_on = False
                pre = [Write(STATE_DOC, {}, mask=["indexSince"])]
        # New entities discovered through Wikidata must exist before mentions reference them.
        found = [e for eid, e in registry.entities.items() if eid not in known_entities]
        if found:
            ensure_docs(fs, entity_docs, {e.id for e in found}, ENTITY_READ_FIELDS)
            fresh_entities = []
            for e in found:
                known_entities.add(e.id)
                if e.id in entity_docs:  # a doc exists already (e.g. not allowlisted): never overwrite it
                    continue
                pre.append(new_entity_write(e))
                entity_docs[e.id] = {**e.metadata(), **ZERO_STATS}
                fresh_entities.append(e)
            pre += registry_add_writes(fresh_entities)
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
        # Aggregate at the end of the run (and every `agg_every` comments, if set: the dashboard then
        # fills progressively, at the price of reading the mentions of the same pairs again).
        done = i + len(chunk)
        due = (agg_every and done - last_agg >= agg_every) or len(to_aggregate) >= MAX_PENDING_PAIRS
        if done < len(comments) and not due:
            continue
        last_agg = done
        info = aggregate_pairs(fs, to_aggregate, entity_docs)
        changed_entities += info["changedEntities"]
        all_days |= {d for _, d in to_aggregate}
        to_aggregate = set()
        fs.commit([Write(STATE_DOC, {"pendingPairs": []}, mask=["pendingPairs"])])
        if wd is not None:
            wd.save()
    if to_aggregate:
        # stopped early: always aggregate what was processed (the reserve is kept for this)
        info = aggregate_pairs(fs, to_aggregate, entity_docs)
        changed_entities += info["changedEntities"]
        all_days |= {d for _, d in to_aggregate}
        fs.commit([Write(STATE_DOC, {"pendingPairs": []}, mask=["pendingPairs"])])
        if wd is not None:
            wd.save()
    if index_on and stopped is None:
        try:
            prune_index(fs)
        except RuntimeError as e:
            log.warning("old comment index documents not deleted: %s", str(e)[:300])
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
        "stopped": stopped,
        "backlogDone": stopped is None and len(comments) < limit,
    }
    return summary, entity_docs


def _refreshed_today(fs: Firestore) -> bool:
    """Whether an aggregation already ran on this Berlin day (the rolling windows only move with
    the day, so a run without new mentions has nothing to refresh then)."""
    stats = fs.get("meta/stats")
    last = stats.data.get("lastAnalysisAt") if stats else None
    return isinstance(last, datetime) and today_berlin(last) == today_berlin()


# --------------------------------------------------------------------------- aggregation


def aggregate_pairs(
    fs: Firestore, pairs: set[tuple[str, str]], entity_docs: dict[str, dict], refresh: bool = True
) -> dict:
    """Recompute the day docs of the affected (entity, day) pairs from their mentions.

    Reads only the mentions of those pairs (query entityId == X AND day == D) and the old day
    docs; entity totals change by (new - old) in the same atomic commit as the day docs.
    Then slide rolling windows, rewrite trending and stats (skipped with refresh=False, for the
    intermediate steps of long jobs).
    """
    today = today_berlin()
    ensure_docs(fs, entity_docs, {e for e, _ in pairs}, ENTITY_READ_FIELDS)
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
        comm = [m for m in mentions if agg.is_community(m)]
        if comm or _may_have_community_doc(entity_docs[e], day, today):
            community[(e, day)] = comm
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
    if refresh:
        refresh_rolling(fs, entity_docs, set(changes))
    return {
        "changedEntities": len(changes),
        "changedDays": sum(len(v) for v in changes.values()),
        "changedCommunityDays": n_comm,
    }


def _may_have_community_doc(entity: dict, day: str, today: str) -> bool:
    """Whether entities/{id}/community_text/{day} can exist. The entity's `communityRecentDays`
    lists every such doc of the last RECENT_DAYS days (both are written by aggregate_community), so
    for recent days the doc need not be read just to find that it is missing; older days: unknown."""
    if day in (entity.get("communityRecentDays") or {}):
        return True
    return day < (date.fromisoformat(today) - timedelta(days=agg.RECENT_DAYS - 1)).isoformat() or day > today


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
    pending_delta = 0
    ts = now()

    def flush(writes: list[Write], delta: int) -> None:
        # meta/stats.mentions moves by the same difference, in the same atomic commit
        if delta:
            writes = [
                *writes,
                Write(
                    "meta/stats", {"mentionsUpdatedAt": ts}, mask=["mentionsUpdatedAt"], increments={"mentions": delta}
                ),
            ]
        fs.commit_atomic(writes)

    for eid, ch in sorted(changes.items()):
        group: list[Write] = []
        for day, _old, new in ch:
            path = f"entities/{eid}/days/{day}"
            group.append(Write(path, None) if new is None else Write(path, new_docs[(eid, day)]))
        delta = sum(agg.day_counts(new)["mentions"] - agg.day_counts(old)["mentions"] for _day, old, new in ch)
        upd = agg.apply_day_changes(entity_docs.get(eid, {}), ch, today)
        upd["updatedAt"] = ts
        entity_docs.setdefault(eid, {}).update(upd)
        group.append(Write(f"entities/{eid}", upd, mask=ENTITY_STAT_FIELDS))
        if len(group) > 499:  # one entity with > 498 changed days: only possible on reaggregate
            for i in range(0, len(group), 499):
                flush(group[i : i + 499], delta if i + 499 >= len(group) else 0)
            continue
        if len(pending) + len(group) > 100:
            flush(pending, pending_delta)
            pending, pending_delta = [], 0
        pending.extend(group)
        pending_delta += delta
    if pending:
        flush(pending, pending_delta)


# meta/stats.mentions follows the day docs by increments; this often it is counted from scratch
MENTIONS_RECOUNT_DAYS = 7


def refresh_rolling(
    fs: Firestore, entity_docs: dict[str, dict], already: set[str], recount_mentions: bool = False
) -> None:
    """Slide 7-day windows for all entities (windows move even without new mentions), then trending + stats.

    `meta/stats.mentions` is not counted here: it changes by the difference of every day doc, in
    the commit that writes the day doc (`_commit_entity_changes`). A count query costs one read
    per 1000 mentions, on every aggregation; it runs only every MENTIONS_RECOUNT_DAYS days (or
    with `recount_mentions`, after a full rebuild) to correct any drift."""
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
    # entities with any mention: a count query (1 read per 1000 index entries), not a scan
    tracked = fs.count("entities", [("mentions", "GREATER_THAN", 0)])
    peers = {"updatedAt": ts, "baseline7d": baseline, "byType": strip}
    # meta/trending, meta/peers and meta/active are only written when their content changed
    cur = fs.batch_get(["meta/trending", "meta/peers", ACTIVE_DOC, "meta/stats"])
    stats_doc = cur.get("meta/stats")
    counted = stats_doc.data.get("mentionsCountedAt") if stats_doc else None
    stats = {"entities": tracked, "lastAnalysisAt": ts, "baseline7d": baseline, "baselineBySource7d": by_source}
    if (
        recount_mentions
        or not isinstance(counted, datetime)
        or not isinstance(stats_doc.data.get("mentions"), int)
        or ts - counted >= timedelta(days=MENTIONS_RECOUNT_DAYS)
    ):
        stats["mentions"] = fs.count("mentions") - fs.count("mentions", [("sourceId", "EQUAL", agg.COMMUNITY)])
        stats["mentionsCountedAt"] = ts
    meta_writes = []
    for path, doc in (("meta/trending", trend), ("meta/peers", peers)):
        old = cur.get(path)
        if old is None or _strip_ts(old.data) != _strip_ts(doc):
            meta_writes.append(Write(path, doc))
    old_active = cur.get(ACTIVE_DOC)
    # entities not loaded in this run keep their state; loaded ones are in or out by their fields
    before = set(old_active.data.get("ids") or []) if old_active else set()
    active = sorted((before - set(entity_docs)) | {eid for eid, d in entity_docs.items() if needs_upkeep(d)})
    if old_active is None or sorted(before) != active:
        meta_writes.append(Write(ACTIVE_DOC, {"ids": active, "count": len(active), "updatedAt": ts}))
    fs.commit([*meta_writes, Write("meta/stats", stats, mask=list(stats))])


# --------------------------------------------------------------------------- one-off jobs
# rescore, relink and reaggregate respect the daily budget: they stop at the remaining budget,
# aggregate what they processed, and keep a cursor in meta/analysis.jobs.<name>, so the next
# invocation (or the regular `run`, once its backlog is done) continues where they stopped.

JOB_AGG_PAIRS = 400  # long jobs aggregate their (entity, day) pairs in steps of this size
# order matters: prune and relink need the entities that a queued seed creates
JOB_NAMES = ("seed", "prune", "relink", "rescore", "reaggregate_since", "reaggregate")


def _jobs(fs: Firestore) -> dict:
    st = fs.get(STATE_DOC)
    return dict((st.data.get("jobs") or {}) if st else {})


def _save_job(fs: Firestore, name: str, data: dict | None) -> None:
    """Persist a job's state in meta/analysis.jobs.<name>; None removes it (job finished)."""
    fs.commit([Write(STATE_DOC, {"jobs": {name: data}} if data is not None else {}, mask=[f"jobs.{name}"])])


def _pairs_in(job: dict) -> set[tuple[str, str]]:
    return {tuple(p.split("|", 1)) for p in job.get("pairs") or []}  # type: ignore[misc]


def _pairs_out(pairs: set[tuple[str, str]]) -> list[str]:
    return sorted(f"{e}|{d}" for e, d in pairs)


def _entity_docs(fs: Firestore, shared: dict[str, dict] | None) -> dict[str, dict]:
    """The entity docs a job works on: those the caller already holds in memory (`run` hands its
    own on to the jobs it continues), else the active set from Firestore (1 read per active entity)."""
    return shared if shared is not None else load_active_docs(fs, ENTITY_READ_FIELDS)[0]


def _budgeted(fs: Firestore, budget: Budget | None, fn, *args) -> dict:
    """Run a job with the daily budget; loads and records it unless the caller (run) owns it."""
    own = budget is None
    if own:
        budget = Budget.load(fs)
        why = budget.exhausted()
        if why:
            log.warning("daily Firestore budget: %s - job not started", why)
            return {"skipped": why, "budget": budget.summary()}
    try:
        out = fn(fs, budget, *args)
    finally:
        if own:
            budget.record()
    out["budget"] = budget.summary()
    return out


def continue_jobs(
    fs: Firestore,
    budget: Budget,
    ner_backend: str = "bert",
    sentiment_key: str = DEFAULT_SENTIMENT,
    entity_docs: dict[str, dict] | None = None,
) -> dict:
    """Continue unfinished one-off jobs with what is left of today's budget (called by `run`).

    `entity_docs`: the entity docs the run holds in memory (kept current by every aggregation), so
    the jobs do not read the active set again; without them the first job loads them once."""
    jobs = _jobs(fs)
    out: dict = {}
    for name in JOB_NAMES:
        job = jobs.get(name)
        if not job:
            continue
        if not budget.allows(writes=200, reads=1000):
            out[name] = "waiting for budget"
            continue
        if timebudget.get().reached(f"job {name}"):
            out[name] = "waiting for time"
            continue
        log.info("continuing unfinished job %s", name)
        if entity_docs is None and name not in ("seed", "reaggregate"):
            entity_docs = _entity_docs(fs, None)  # once for all jobs of this run
        if name == "seed":
            res = seed(fs, with_images=True, budget=budget)
            out[name] = res
            if res.get("stopped"):
                break  # later jobs (relink) need the seeded entities
            _save_job(fs, "seed", None)
            if (res.get("newEntities") or res.get("updatedEntities")) and entity_docs is not None:
                # seed changed names, aliases or images: the docs in memory are stale, read them again
                entity_docs.clear()
                entity_docs.update(_entity_docs(fs, None))
        elif name == "prune":
            res = prune(
                fs,
                set(job.get("ids") or []),
                add_ids=set(job.get("add") or []),
                ner_backend=ner_backend,
                sentiment_key=sentiment_key,
                budget=budget,
                entity_docs=entity_docs,
            )
            out[name] = res
            if not res.get("stopped"):
                _save_job(fs, "prune", None)
        elif name == "relink":
            out[name] = relink(
                fs,
                ner_backend=ner_backend,
                sentiment_key=job.get("sentiment") or sentiment_key,
                budget=budget,
                entity_docs=entity_docs,
            )
        elif name == "rescore":
            out[name] = rescore(
                fs,
                sentiment_key=job.get("sentiment") or sentiment_key,
                budget=budget,
                entity_docs=entity_docs,
                leave_for_runs=True,
            )
        elif name == "reaggregate_since":
            out[name] = reaggregate_since(fs, job["since"], budget=budget, entity_docs=entity_docs)
        else:
            out[name] = reaggregate(fs, budget=budget)
    return out


LEGACY_SENTIMENT = "sent-cal@2"  # scored the mentions before meta/analysis.sentiment existed


def _note_sentiment(fs: Firestore, state: Doc | None, key: str, version: str) -> bool:
    """Keep meta/analysis.sentiment = the model that scores new mentions. When that model changes,
    queue a rescore of the existing mentions (newest day first, continued by the regular runs
    within the budget), so the timelines do not stay on two models. Returns whether it queued one."""
    cur = ((state.data.get("sentiment") if state else None) or {}).get("version")
    if cur == version:
        return False
    if cur is None:
        # first run that records the model: an existing state doc means existing mentions, scored
        # by the model before; without one this is a fresh database
        cur = LEGACY_SENTIMENT if state is not None else version
    since = {"version": version, "key": key, "since": now()}
    fs.commit([Write(STATE_DOC, {"sentiment": since}, mask=[f"sentiment.{k}" for k in since])])
    if cur == version:
        return False
    log.warning("sentiment model changed from %s to %s: rescore of the existing mentions queued", cur, version)
    queue_job(fs, "rescore", {"sentiment": key, "from": today_berlin(), "replaces": cur})
    return True


def rescore(
    fs: Firestore,
    sentiment_key: str = DEFAULT_SENTIMENT,
    batch: int = 400,
    budget: Budget | None = None,
    entity_docs: dict[str, dict] | None = None,
    leave_for_runs: bool = False,
    **model_kw,
) -> dict:
    """Re-run only the sentiment step for the existing mentions (NER/linking results are kept).

    Day by day, newest first: the mentions of one day are read (query day == D, in pages), their
    comments fetched, the positions recovered with the gazetteer (or the stored surface), and
    label, score and confidence rewritten where they change. When a day is complete its
    (entity, day) pairs are re-aggregated and the job moves to the next older day with mentions.
    So an unfinished rescore leaves every day on one model, except the day it stopped in, and the
    recent days (the 7-day windows, trending) are converted first.

    Resumable (meta/analysis.jobs.rescore: day, cursor within the day). Continued by a regular
    run (`leave_for_runs`), it leaves the budget that the later runs of the day need
    (`_keep_for_later_runs`). The comments'
    `analysisVersion` is not rewritten: it records the first analysis, the job state and
    meta/analysis.sentiment record the rescore (one write per comment saved).
    """
    return _budgeted(fs, budget, _rescore, sentiment_key, batch, model_kw, entity_docs, leave_for_runs)


def _keep_for_later_runs(budget: Budget, at: datetime | None = None) -> tuple[int, int]:
    """(writes, reads) of today's budget that a long job leaves for the regular runs still to
    come this Berlin day: 1.3 x what this run used before the job started (at least a small
    run), per expected later run (ANALYSIS_RUN_INTERVAL_H, default 12 h between runs)."""
    try:
        interval = float(os.environ.get("ANALYSIS_RUN_INTERVAL_H") or 12)
    except ValueError:
        interval = 12.0
    t = (at or now()).astimezone(BERLIN)
    later = round((24 - t.hour - t.minute / 60) / interval) if interval > 0 else 0
    if later <= 0:
        return 0, 0
    return int(later * 1.3 * max(budget.run_writes, 4000)), int(later * 1.3 * max(budget.run_reads, 10000))


def _older_day(fs: Firestore, day: str) -> str | None:
    """The newest day before `day` that has mentions (1 read)."""
    older = list(
        fs.query(
            "mentions", where=[("day", "LESS_THAN", day)], order_by=[("day", "DESCENDING")], select=["day"], limit=1
        )
    )
    return older[0].data.get("day") if older else None


def _rescore(fs, budget, sentiment_key, batch, model_kw, entity_docs=None, leave_for_runs=False) -> dict:
    from .analyze import mention_window
    from .linker import Match
    from .sentiment import combine, load_sentiment
    from .textutil import split_sentences

    t0 = time.time()
    job = _jobs(fs).get("rescore") or {}
    if job and job.get("sentiment") != sentiment_key:
        log.warning("unfinished rescore with %s replaced by a rescore with %s", job.get("sentiment"), sentiment_key)
        job = {}
    if job.get("cursor") and "day" not in job:
        log.warning("rescore state of the comment-ordered job dropped: starting again by days")
        job = {k: v for k, v in job.items() if k in ("sentiment", "from", "replaces", "startedAt")}
    job = {"sentiment": sentiment_key, "cursor": "", "changed": 0, "mentions": 0, "days": 0, "startedAt": now(), **job}
    job["day"] = job.get("day") or job.get("from") or today_berlin()
    entity_docs = _entity_docs(fs, entity_docs)
    registry = load_registry(fs)
    countries = source_countries(fs)
    model = load_sentiment(sentiment_key, **model_kw)
    keep_w, keep_r = _keep_for_later_runs(budget) if leave_for_runs else (0, 0)
    pending = _pairs_in(job)  # changed, not aggregated yet (also after a crash)
    done, stopped = True, None
    tb = timebudget.get()
    t_step, step_s, worked = time.time(), 0.0, False

    def fits(writes: int, reads: int) -> bool:
        return budget.allows(writes=writes + keep_w, reads=reads + keep_r)

    while job["day"]:
        day = job["day"]
        if not job["cursor"] and worked:
            # a new day is only started if all of it fits (mentions, their day docs, their comments)
            n_day = fs.count("mentions", where=[("day", "EQUAL", day)])
            if not fits(writes=3 * n_day + 50, reads=3 * n_day + 50):
                done, stopped = False, f"daily Firestore budget: {day} ({n_day} mentions) waits for the next run"
                break
        while True:
            # a page: its mentions and comments, the day docs of its pairs when they are aggregated
            if not fits(writes=3 * batch + 2 * len(pending) + 2, reads=4 * batch + 2 * len(pending)):
                done, stopped = False, f"daily Firestore budget reached in {day}"
                break
            if worked:
                step_s, t_step = time.time() - t_step, time.time()
            if tb.reached("rescore", reserve=1.2 * step_s):
                done, stopped = False, f"time budget reached in {day}"
                break
            page = list(
                fs.query(
                    "mentions",
                    where=[("day", "EQUAL", day)],
                    select=["commentId", "entityId", "surface", "label", "score", "confidence", "day"],
                    limit=batch,
                    start_after=[job["cursor"]] if job["cursor"] else None,
                )
            )
            if not page:
                break
            worked = True
            by_comment: dict[str, list] = {}
            for d in page:
                by_comment.setdefault(d.data["commentId"], []).append(d)
            docs = fs.batch_get([f"comments/{c}" for c in by_comment], mask=["text", "sourceId"])
            items: list[tuple[str, str]] = []
            plan: list[tuple[object, list[int]]] = []
            for cid, mds in by_comment.items():
                cd = docs.get(f"comments/{cid}")
                if cd is None:
                    continue
                text = clean_text(cd.data.get("text") or "")
                sents = split_sentences(text)
                matches = registry.gazetteer(text, [], "", countries.get(cd.data.get("sourceId") or ""))
                for md in mds:
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
                    plan.append((md, idxs))
            scored = model.predict(items) if items else []
            groups: list[list[Write]] = []
            page_pairs: set[tuple[str, str]] = set()
            for md, idxs in plan:
                label, score, conf = combine([scored[k] for k in idxs])
                job["mentions"] += 1
                if (md.data.get("label"), md.data.get("score"), md.data.get("confidence")) != (label, score, conf):
                    job["changed"] += 1
                    new = {"label": label, "score": score, "confidence": conf}
                    groups.append([Write(md.path, new, mask=list(new))])
                    page_pairs.add((md.data["entityId"], md.data["day"]))
            if page_pairs - pending:  # remember the pairs before the labels change (crash safety)
                pending |= page_pairs
                _save_job(fs, "rescore", {**job, "pairs": _pairs_out(pending)})
            fs.commit_groups(groups)
            job["cursor"] = page[-1].id
            _save_job(fs, "rescore", {**job, "pairs": _pairs_out(pending)})
            log.info("rescore %s: %d mentions, %d changed so far", day, job["mentions"], job["changed"])
            if len(page) < batch:
                break
        if stopped:
            log.warning("rescore: %s", stopped)
            break
        # the day is complete: its day docs follow at once, then the next older day
        aggregate_pairs(fs, pending, entity_docs, refresh=False)
        pending = set()
        job.update(day=_older_day(fs, day), cursor="", days=job["days"] + 1)
        _save_job(fs, "rescore", {**job, "pairs": []})
    info = aggregate_pairs(fs, pending, entity_docs)
    if done:
        rescored = {"version": model.version, "at": now(), "mentions": job["mentions"]}
        fs.commit(
            [
                Write(
                    STATE_DOC,
                    {"jobs": {}, "sentiment": {"rescored": rescored}},
                    mask=["jobs.rescore", "sentiment.rescored"],
                )
            ]
        )
    else:
        _save_job(fs, "rescore", {**job, "pairs": []})
    return {
        "days": job["days"],
        "nextDay": job["day"],
        "mentions": job["mentions"],
        "changed": job["changed"],
        "version": model.version,
        "seconds": round(time.time() - t0, 1),
        "done": done,
        "stopped": stopped,
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
    sentiment_key: str = DEFAULT_SENTIMENT,
    batch: int = 250,
    page: int = 1000,
    budget: Budget | None = None,
    entity_docs: dict[str, dict] | None = None,
) -> dict:
    """Add mentions of newly added entities to already analysed comments (last `days` days).

    Only comments whose text contains an alias of a new entity are re-run (NER + linking);
    sentiment is computed only for the new (comment, entity) pairs, existing mentions keep their
    labels (their coEntityIds are extended). Afterwards the affected pairs are aggregated.
    Resumable: the comments are scanned in pages (day, id) and the cursor, the entity set and
    the cutoff day are kept in meta/analysis.jobs.relink until the scan is complete.
    """
    return _budgeted(fs, budget, _relink, days, entity_ids, ner_backend, sentiment_key, batch, page, entity_docs)


def queue_job(fs: Firestore, name: str, data: dict) -> dict:
    """Register a one-off job without running it (1 write): the regular `run` continues it once
    its own backlog is done, within the daily budget.

    relink data: {cutoff, entityIds}. A relink that is queued but not started yet takes the new
    entities into the same scan; one that is already scanning keeps its entity set, and the new
    entities wait in `relink_next`, which becomes the relink when the running one is done.
    prune data: {ids, add}; merged into an already queued prune."""
    jobs = _jobs(fs)
    job = {"startedAt": now(), **data}
    slot = name
    if name == "relink":
        job = {"sentiment": DEFAULT_SENTIMENT, "cursor": None, "scanned": 0, "candidates": 0, "newMentions": 0, **job}
        cur = jobs.get("relink")
        if cur and cur.get("cursor") is not None:
            new_ids = set(job.get("entityIds") or []) - set(cur.get("entityIds") or [])
            if not new_ids:
                return {"queued": name, "alreadyQueued": True}
            slot, cur, job["entityIds"] = "relink_next", jobs.get("relink_next"), sorted(new_ids)
        if cur:
            job = {
                **cur,
                "entityIds": sorted(set(cur.get("entityIds") or []) | set(job.get("entityIds") or [])),
                "cutoff": min(c for c in (cur.get("cutoff"), job.get("cutoff")) if c),
            }
    elif name == "prune" and jobs.get("prune"):
        cur = jobs["prune"]
        job = {
            **cur,
            "ids": sorted(set(cur.get("ids") or []) | set(job.get("ids") or [])),
            "add": sorted(set(cur.get("add") or []) | set(job.get("add") or [])),
        }
    _save_job(fs, slot, job)
    return {"queued": slot, **{k: (len(v) if isinstance(v, list) else v) for k, v in job.items()}}


def _finish_relink(fs: Firestore) -> None:
    """Remove the finished relink; entities queued while it was scanning get their own scan next."""
    nxt = _jobs(fs).get("relink_next")
    fs.commit([Write(STATE_DOC, {"jobs": {"relink": nxt} if nxt else {}}, mask=["jobs.relink", "jobs.relink_next"])])


def _relink(fs, budget, days, entity_ids, ner_backend, sentiment_key, batch, page, entity_docs=None) -> dict:
    from .ner import load_ner
    from .sentiment import load_sentiment

    t0 = time.time()
    entity_docs = _entity_docs(fs, entity_docs)
    registry = load_registry(fs)
    state = fs.get(STATE_DOC)
    job = dict(((state.data.get("jobs") or {}) if state else {}).get("relink") or {})
    if job:
        log.info("relink: resuming (cutoff %s, cursor %s)", job.get("cutoff"), job.get("icursor") or job.get("cursor"))
    else:
        if entity_ids is None:  # default: generated entities without any mention yet (reads their docs once)
            gen = [eid for eid, e in registry.entities.items() if e.source == "generated"]
            got = fs.batch_get([f"entities/{e}" for e in gen], mask=["mentions"])
            entity_ids = {e for e in gen if not ((got.get(f"entities/{e}") or Doc("", "", {})).data.get("mentions"))}
        cutoff = (date.fromisoformat(today_berlin()) - timedelta(days=days - 1)).isoformat()
        job = {
            "cutoff": cutoff,
            "entityIds": sorted(e for e in entity_ids if e in registry.entities),
            "sentiment": sentiment_key,
            "cursor": None,
            "scanned": 0,
            "candidates": 0,
            "newMentions": 0,
            "startedAt": now(),
        }
    new = {e for e in job["entityIds"] if e in registry.entities}
    names = {registry.entities[e].name for e in new}
    # the gazetteer matches lower-cased aliases, the lookup of NER spans case-folded ones ("Strauß" / "strauss")
    own = [a.text for a in registry.aliases if a.entity_id in new]
    alias_texts = sorted({a.casefold() for a in own} | {a.lower() for a in own}, key=len, reverse=True)
    if not alias_texts:
        _finish_relink(fs)
        return {"newEntities": 0}
    prefilter = re.compile(r"(?<![\w])(" + "|".join(re.escape(a) for a in alias_texts) + r")", re.IGNORECASE)
    # Days from `split` on are looked up in the comment index (analysis/textindex.py): only the
    # comments whose words can contain an alias are read. Earlier days are scanned comment by comment.
    matcher = textindex.Matcher(alias_texts)
    since = _index_since(state)
    split = since if since and matcher.complete and not (job.get("cursor") and job["cursor"][0] >= since) else None
    if job.get("indexSince") and job["indexSince"] != split:
        # the index was reset since this job started: scan the days that were left to it
        if job.get("scanDone"):
            job.update(scanFrom=job["indexSince"], cursor=None)
        job.update(scanDone=False, icursor=None)
    job["indexSince"] = split
    ner = sentiment = None
    countries = source_countries(fs)
    pairs = _pairs_in(job)
    done, stopped = False, None
    tb = timebudget.get()
    ctx_cache: dict[str, str] = {}

    def process(candidates: list[dict]) -> None:
        """NER + linking for the comments that contain an alias; writes the new mentions."""
        nonlocal ner, sentiment, pairs
        if candidates and ner is None:
            ner = load_ner(ner_backend)
            sentiment = _OnlyFor(load_sentiment(job.get("sentiment") or sentiment_key), names)
        page_pairs: set[tuple[str, str]] = set()
        groups: list[list[Write]] = []
        for i in range(0, len(candidates), batch):
            chunk = candidates[i : i + batch]
            ctx = article_context(fs, {c.get("articleId") for c in chunk if c.get("articleId")}, ctx_cache)
            result = analyze_comments(chunk, registry, ner, sentiment, source_country=countries, article_context=ctx)
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
                        page_pairs.add((d.data["entityId"], d.data["day"]))
                page_pairs |= {(m["entityId"], m["day"]) for m in fresh}
                job["newMentions"] += len(fresh)
                groups.append(g)
        if page_pairs - pairs:  # remember the pairs before the mentions land (crash safety)
            pairs |= page_pairs
            _save_job(fs, "relink", {**job, "pairs": _pairs_out(pairs)})
        fs.commit_groups(groups)
        job["candidates"] += len(candidates)

    def step_done(n: int) -> None:
        nonlocal pairs
        job["scanned"] += n
        if len(pairs) >= JOB_AGG_PAIRS:
            aggregate_pairs(fs, pairs, entity_docs, refresh=False)
            pairs = set()
        log.info(
            "relink: %d comments scanned, %d candidates, %d new mentions",
            job["scanned"],
            job["candidates"],
            job["newMentions"],
        )

    def may_continue(per_w: float, per_r: float, n: int, step_s: float) -> bool:
        nonlocal stopped
        if tb.reached("relink", reserve=1.2 * step_s):
            stopped = f"time budget reached after {job['scanned']} comments"
        elif not budget.allows(writes=int(per_w * n) + 3, reads=int(per_r * n) + 1):
            stopped = f"daily Firestore budget reached after {job['scanned']} comments"
        if stopped:
            log.warning("relink: %s; continuing on the next run", stopped)
        return stopped is None

    # ---- 1. scan: every comment of the days before the index (all days without one)
    w0, r0, seen = fs.writes, fs.reads, 0
    t_step, step_s = time.time(), 0.0
    where = [("day", "GREATER_THAN_OR_EQUAL", max(job["cutoff"], job.get("scanFrom") or ""))]
    if split:
        where.append(("day", "LESS_THAN", split))
    while not job.get("scanDone"):
        if seen:
            step_s, t_step = time.time() - t_step, time.time()
        # estimate a page from this invocation's own rates (first page: 1 read per comment + 10 %)
        per_w = (fs.writes - w0) / seen if seen else 0.1
        per_r = (fs.reads - r0) / seen if seen else 1.2
        if not may_continue(per_w, per_r, page, step_s):
            break
        docs = list(
            fs.query(
                "comments",
                where=where,
                order_by=[("day", "ASCENDING")],
                select=COMMENT_FIELDS,
                limit=page,
                start_after=job["cursor"],
            )
        )
        seen += len(docs)
        process(
            [
                {"id": d.id, **d.data}
                for d in docs
                if d.data.get("analysisVersion") and prefilter.search(clean_text(d.data.get("text") or ""))
            ]
        )
        if docs:
            job["cursor"] = [docs[-1].data.get("day"), docs[-1].id]
        step_done(len(docs))
        if len(docs) < page:
            job["scanDone"] = True
            done = not split
        if not done:
            _save_job(fs, "relink", {**job, "pairs": _pairs_out(pairs)})

    # ---- 2. index: only the comments whose words can contain an alias are read
    w0, r0, seen = fs.writes, fs.reads, 0
    t_step, step_s = time.time(), 0.0
    ipage = max(1, page // 250)  # index documents per step (one holds a batch of up to 250 comments)
    while split and job.get("scanDone") and not done and stopped is None:
        if seen:
            step_s, t_step = time.time() - t_step, time.time()
        # first step: the index documents + about 5 % of their comments
        per_w = (fs.writes - w0) / seen if seen else 0.05
        per_r = (fs.reads - r0) / seen if seen else 0.06
        if not may_continue(per_w, per_r, 250 * ipage, step_s):
            break
        idocs = list(
            fs.query(
                textindex.COLLECTION,
                where=[("day", "GREATER_THAN_OR_EQUAL", max(job["cutoff"], split))],
                order_by=[("day", "ASCENDING")],
                limit=ipage,
                start_after=job.get("icursor"),
            )
        )
        entries = [e for d in idocs for e in textindex.decode(d.data.get("blob") or b"")]
        seen += max(1, len(entries))
        ids = list(dict.fromkeys(matcher.select(entries)))
        got = fs.batch_get([f"comments/{i}" for i in ids], mask=COMMENT_FIELDS) if ids else {}
        process(
            [
                {"id": path.split("/", 1)[1], **d.data}
                for path, d in got.items()
                if d is not None
                and d.data.get("analysisVersion")
                and prefilter.search(clean_text(d.data.get("text") or ""))
            ]
        )
        if idocs:
            job["icursor"] = [idocs[-1].data.get("day"), idocs[-1].id]
        step_done(len(entries))
        if len(idocs) < ipage:
            done = True
        else:
            _save_job(fs, "relink", {**job, "pairs": _pairs_out(pairs)})

    info = aggregate_pairs(fs, pairs, entity_docs)
    if done:
        _finish_relink(fs)
    else:
        _save_job(fs, "relink", {**job, "pairs": []})
    return {
        "newEntities": len(new),
        "scanned": job["scanned"],
        "candidates": job["candidates"],
        "newMentions": job["newMentions"],
        "seconds": round(time.time() - t0, 1),
        "done": done,
        "stopped": stopped,
        **info,
    }


def reaggregate_since(
    fs: Firestore, since: str, budget: Budget | None = None, entity_docs: dict[str, dict] | None = None
) -> dict:
    """Recompute only the (entity, day) pairs with mentions on or after `since` (cheap repair).

    Resumable: pairs are processed in sorted order, the last done pair is the cursor."""
    return _budgeted(fs, budget, _reaggregate_since, since, entity_docs)


def _reaggregate_since(fs, budget, since, entity_docs=None) -> dict:
    job = _jobs(fs).get("reaggregate_since") or {}
    if job.get("since") != since:
        job = {"since": since, "cursor": "", "startedAt": now()}
    entity_docs = _entity_docs(fs, entity_docs)
    keys = sorted(
        {
            f"{d.data.get('entityId', '')}|{d.data.get('day', '')}"
            for d in fs.query(
                "mentions",
                where=[("day", "GREATER_THAN_OR_EQUAL", since)],
                order_by=[("day", "ASCENDING")],  # Firestore rejects a range filter without its field in the order
                select=["entityId", "day"],
            )
        }
    )
    todo = [k for k in keys if k > job["cursor"]]
    done, stopped, changed = True, None, 0
    for i in range(0, len(todo), JOB_AGG_PAIRS):
        chunk = todo[i : i + JOB_AGG_PAIRS]
        if not budget.allows(writes=2 * len(chunk) + 2, reads=3 * len(chunk)):
            done, stopped = False, f"daily Firestore budget reached after {i} of {len(todo)} pairs"
            log.warning("reaggregate --since: %s", stopped)
            break
        if timebudget.get().reached("reaggregate --since"):
            done, stopped = False, f"time budget reached after {i} of {len(todo)} pairs"
            log.warning("reaggregate --since: %s", stopped)
            break
        info = aggregate_pairs(fs, {tuple(k.split("|", 1)) for k in chunk}, entity_docs, refresh=False)  # type: ignore[misc]
        changed += info["changedDays"]
        job["cursor"] = chunk[-1]
        _save_job(fs, "reaggregate_since", job)
    refresh_rolling(fs, entity_docs, set())
    _save_job(fs, "reaggregate_since", None if done else job)
    return {"pairs": len(todo), "changedDays": changed, "done": done, "stopped": stopped}


def reaggregate(fs: Firestore, budget: Budget | None = None) -> dict:
    """Rebuild every day doc and all entity counters from scratch from `mentions` (full scan).

    Writes go per entity (its day docs and totals together); resumable with the last finished
    entity as cursor. Community docs, rolling windows and meta are rebuilt once all are done.
    """
    return _budgeted(fs, budget, _reaggregate)


def _reaggregate(fs, budget) -> dict:
    today = today_berlin()
    job = _jobs(fs).get("reaggregate") or {"cursor": "", "startedAt": now()}
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
    wanted = {f"entities/{e}/days/{d}" for e, d in docs}
    stale: dict[str, list[str]] = {}
    for path in existing_days:
        if path not in wanted:
            stale.setdefault(path.split("/")[1], []).append(path)
    ts = now()
    groups: list[tuple[str, list[Write]]] = []
    for eid in sorted(set(entity_docs) | set(stale)):
        if eid <= job["cursor"]:
            continue
        g = [Write(p, None) for p in stale.get(eid, [])]
        g += [Write(f"entities/{eid}/days/{day}", doc) for day, doc in sorted(by_entity.get(eid, {}).items())]
        d = entity_docs.get(eid)
        if d is not None and (eid in by_entity or d.get("mentions") or d.get("recentDays")):
            upd = agg.totals_from_days(by_entity[eid], today) if eid in by_entity else {**ZERO_STATS}
            upd["updatedAt"] = ts
            d.update(upd)
            g.append(Write(f"entities/{eid}", upd, mask=ENTITY_STAT_FIELDS))
        if g:
            groups.append((eid, g))
    done, stopped, n_writes = True, None, 0
    pack: list[Write] = []
    last = job["cursor"]

    def flush() -> None:
        nonlocal pack
        for i in range(0, len(pack), 100):  # a full rebuild does not need cross-entity atomicity
            fs.commit_atomic(pack[i : i + 100])
        pack = []
        job["cursor"] = last
        _save_job(fs, "reaggregate", job)

    for eid, g in groups:
        if not budget.allows(writes=len(pack) + len(g) + 2):
            done, stopped = False, f"daily Firestore budget reached before entity {eid}"
            log.warning("reaggregate: %s", stopped)
            break
        if timebudget.get().reached("reaggregate"):
            done, stopped = False, f"time budget reached before entity {eid}"
            log.warning("reaggregate: %s", stopped)
            break
        pack.extend(g)
        n_writes += len(g)
        last = eid
        if len(pack) >= 400:
            flush()
    if pack or not done:
        flush()
    if not done:
        return {"dayDocs": len(docs), "writes": n_writes, "done": False, "stopped": stopped}
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
    refresh_rolling(fs, entity_docs, set(entity_docs), recount_mentions=True)  # the day docs were rewritten directly
    _save_job(fs, "reaggregate", None)
    return {
        "mentions": len(mentions),
        "dayDocs": len(docs),
        "entitiesWithMentions": len(by_entity),
        "deletedDayDocs": sum(len(v) for v in stale.values()),
        "done": True,
    }


class _NoSentiment:
    """Sentiment stand-in for re-linking only (no model is run)."""

    version = "none"
    window_mode = "clause"

    def predict(self, items: list) -> list[tuple[str, float, float]]:
        return [("neu", 0.0, 0.0)] * len(items)


def prune(
    fs: Firestore,
    entity_ids: set[str],
    add_ids: set[str] | None = None,
    ner_backend: str = "bert",
    sentiment_key: str = DEFAULT_SENTIMENT,
    budget: Budget | None = None,
    entity_docs: dict[str, dict] | None = None,
) -> dict:
    """Re-check the existing mentions of `entity_ids` against the current entity lists.

    Mentions that the current registry no longer links (after an alias was tightened) are
    deleted, entities that were removed from the lists are deleted with their mentions and day
    docs, the coEntityIds of the other mentions in those comments are updated, and the affected
    (entity, day) pairs are re-aggregated. Mentions of `add_ids` (e.g. an alias that moved to
    a new entity) found in the re-checked comments are added, with sentiment for them only.
    """
    return _budgeted(fs, budget, _prune, set(entity_ids), set(add_ids or ()), ner_backend, sentiment_key, entity_docs)


def _prune(fs, budget, entity_ids, add_ids, ner_backend, sentiment_key, entity_docs=None) -> dict:
    from .ner import load_ner
    from .sentiment import load_sentiment

    if timebudget.get().reached("prune"):
        return {"stopped": "time budget reached before the prune"}

    entity_docs = _entity_docs(fs, entity_docs)
    ensure_docs(fs, entity_docs, set(entity_ids) | set(add_ids), ENTITY_READ_FIELDS)
    reg = Registry.from_yaml()
    targets = {e for e in entity_ids if e in entity_docs}
    removed = {e for e in targets if e not in reg.entities}
    mentions: list = []
    for eid in sorted(targets):
        mentions += list(
            fs.query("mentions", where=[("entityId", "EQUAL", eid)], select=["commentId", "entityId", "day"])
        )
    recheck = sorted({m.data["commentId"] for m in mentions if m.data["entityId"] not in removed})
    est = len(mentions) * 3 + len(recheck) * 2
    if not budget.allows(writes=len(mentions) * 2 + 10, reads=est):
        return {"stopped": "daily Firestore budget too small for this prune", "mentions": len(mentions)}
    still: dict[str, set[str]] = {}
    fresh: dict[str, list[dict]] = {}  # commentId -> new mentions of add_ids
    add_ids = {e for e in add_ids if e in reg.entities and e in entity_docs}
    if recheck:
        ner = load_ner(ner_backend)
        sentiment = (
            _OnlyFor(load_sentiment(sentiment_key), {reg.entities[e].name for e in add_ids})
            if add_ids
            else _NoSentiment()
        )
        countries = source_countries(fs)
        for i in range(0, len(recheck), 250):
            chunk = recheck[i : i + 250]
            docs = fs.batch_get([f"comments/{c}" for c in chunk], mask=COMMENT_FIELDS)
            comments = [{"id": p.rsplit("/", 1)[1], **d.data} for p, d in docs.items() if d is not None]
            ctx = article_context(fs, {c.get("articleId") for c in comments if c.get("articleId")})
            result = analyze_comments(comments, reg, ner, sentiment, source_country=countries, article_context=ctx)
            for cid, ms in result.items():
                still[cid] = {m["entityId"] for m in ms}
                if any(m["entityId"] in add_ids for m in ms):
                    fresh[cid] = [m for m in ms if m["entityId"] in add_ids]
    drop = [
        m
        for m in mentions
        if m.data["entityId"] in removed or m.data["entityId"] not in still.get(m.data["commentId"], set())
    ]
    dropped_by_comment: dict[str, set[str]] = {}
    for m in drop:
        dropped_by_comment.setdefault(m.data["commentId"], set()).add(m.data["entityId"])
    pairs = {(m.data["entityId"], m.data["day"]) for m in drop}
    writes: list[Write] = []
    for _cid, ms in sorted(fresh.items()):
        writes += [Write(f"mentions/{m['commentId']}_{m['entityId']}", m) for m in ms]
        pairs |= {(m["entityId"], m["day"]) for m in ms}
    for cid in sorted(set(dropped_by_comment) | set(fresh)):
        gone = dropped_by_comment.get(cid, set())
        new = {m["entityId"] for m in fresh.get(cid, [])}
        for d in fs.query("mentions", where=[("commentId", "EQUAL", cid)], select=["entityId", "coEntityIds", "day"]):
            if d.data.get("entityId") in gone | new:
                continue
            old = list(d.data.get("coEntityIds") or [])
            co = sorted({x for x in old if x not in gone} | (new & still.get(cid, set())))
            if co != sorted(old):
                writes.append(Write(d.path, {"coEntityIds": co}, mask=["coEntityIds"]))
                pairs.add((d.data["entityId"], d.data["day"]))
    writes += [Write(m.path, None) for m in drop]
    # crash safety: the pairs go to pendingPairs (aggregated by the next run) before the deletes
    state = fs.get(STATE_DOC)
    pending = set((state.data.get("pendingPairs") or []) if state else []) | set(_pairs_out(pairs))
    fs.commit([Write(STATE_DOC, {"pendingPairs": sorted(pending)}, mask=["pendingPairs"])])
    fs.commit(writes)
    info = aggregate_pairs(fs, pairs, entity_docs, refresh=False)
    for eid in sorted(removed):
        rest = [d.path for d in fs.query("days", parent=f"entities/{eid}", select=[])]
        rest += [d.path for d in fs.query("community_text", parent=f"entities/{eid}", select=[])]
        fs.commit([*(Write(p, None) for p in rest), Write(f"entities/{eid}", None)])
        entity_docs.pop(eid, None)
    refresh_rolling(fs, entity_docs, set())
    fs.commit([Write(STATE_DOC, {"pendingPairs": sorted(pending - set(_pairs_out(pairs)))}, mask=["pendingPairs"])])
    return {
        "checkedEntities": len(targets),
        "removedEntities": sorted(removed),
        "mentionsChecked": len(mentions),
        "mentionsDeleted": len(drop),
        "mentionsAdded": sum(len(v) for v in fresh.values()),
        "deletedByEntity": dict(sorted(Counter(m.data["entityId"] for m in drop).items())),
        **info,
    }
