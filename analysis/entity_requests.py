"""Entity requests: user proposals accepted by the API become allowlisted entities.

The API writes `entity_requests/{qid}` with `status: "accepted"`. Every analysis run (and
`python -m analysis seed-requests`) re-validates each request against Wikidata:

* the item exists and is not a disambiguation page, and has >= MIN_SITELINKS sitelinks,
* persons: instance of human (Q5) with an occupation (P106) or a position held (P39),
* organisations: instance of one of the organisation classes in `wikidata.ORG_CLASSES`,
* apps & digital services (PRODUCT): instance of one of `wikidata.PRODUCT_CLASSES` (software,
  mobile app, website, online service, video game ...).

Valid requests are seeded as allowlisted entities (name, aliases from German/English labels,
description, portrait or logo thumbnail, type PER / PARTY / ORG / PRODUCT) and marked `done`. Invalid
ones are marked `failed` with a reason. If Wikidata is unavailable or rate-limited, the
requests stay `accepted` and are retried in the next run.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from .firestore import Firestore, Write
from .linker import Entity, Registry
from .textutil import slugify
from .wikidata import DISAMBIGUATION, HUMAN, WikidataClient, all_names, commons_thumb, description, entity_type, label

log = logging.getLogger(__name__)

MIN_SITELINKS = 10
COLLECTION = "entity_requests"


def validate(item: dict) -> tuple[str | None, str]:
    """Return (entity type, "") for a valid public figure/organisation, else (None, reason)."""
    p31 = set(WikidataClient.claim_ids(item, "P31"))
    if DISAMBIGUATION in p31:
        return None, "disambiguation page"
    sitelinks = len(item.get("sitelinks", {}))
    if sitelinks < MIN_SITELINKS:
        return None, f"only {sitelinks} sitelinks (need {MIN_SITELINKS})"
    if HUMAN in p31:
        if not (WikidataClient.claim_ids(item, "P106") or WikidataClient.claim_ids(item, "P39")):
            return None, "person without occupation or position"
        return "PER", ""
    etype = entity_type(item)
    if etype:
        return etype, ""
    return None, "not a person, a known organisation class or an app/digital service"


def entity_from_item(qid: str, item: dict, etype: str, taken: set[str]) -> Entity:
    name = label(item) or qid
    eid = slugify(name)
    if eid in taken:
        eid = f"{eid}-{qid.lower()}"
    props = ("P18",) if etype == "PER" else ("P154", "P8972", "P2910", "P158")
    url = next((u for p in props for f in WikidataClient.claim_strings(item, p) if (u := commons_thumb(f))), None)
    return Entity(
        id=eid,
        name=name,
        type=etype,
        description=description(item)[:120],
        wikidataId=qid,
        aliases=list(dict.fromkeys([name, *all_names(item)]))[:12],
        imageUrl=url,
        imageKind=("portrait" if etype == "PER" else "logo") if url else None,
        source="wikidata",
    )


def process_entity_requests(fs: Firestore, client: WikidataClient | None = None, limit: int = 50) -> dict:
    reqs = list(fs.query(COLLECTION, where=[("status", "EQUAL", "accepted")], limit=limit))
    if not reqs:
        return {"requests": 0}
    # only when requests are pending (rare): a scan of the ids, to avoid id/QID clashes
    existing = {d.id: d.data for d in fs.query("entities", select=["wikidataId"])}
    by_qid = {d.get("wikidataId"): eid for eid, d in existing.items() if d.get("wikidataId")}
    taken = set(existing) | set(Registry.from_yaml().entities)
    qids = [r.data.get("qid") or r.id for r in reqs]
    client = client or WikidataClient(max_wait=30)
    try:
        items = client.get_entities([q for q in qids if q not in by_qid])
    except RuntimeError as e:  # rate limited / unavailable: retry next run
        log.warning("entity requests postponed: %s", e)
        return {"requests": len(reqs), "postponed": len(reqs), "reason": str(e)}
    ts = datetime.now(UTC)
    writes: list[Write] = []
    done = failed = 0
    for r, qid in zip(reqs, qids, strict=True):
        status, reason, eid = "failed", "", None
        if qid in by_qid:
            status, reason, eid = "done", "already tracked", by_qid[qid]
        elif qid not in items:
            reason = "Wikidata item not found"
        else:
            etype, reason = validate(items[qid])
            if etype:
                ent = entity_from_item(qid, items[qid], etype, taken)
                taken.add(ent.id)
                by_qid[qid] = ent.id
                from .pipeline import ZERO_STATS  # local import: pipeline imports this module

                writes.append(
                    Write(
                        f"entities/{ent.id}",
                        {**ent.metadata(), **ZERO_STATS, "updatedAt": ts},
                    )
                )
                from .registry_store import registry_add_writes

                writes += registry_add_writes([ent])  # so runs link it without scanning all entities
                status, reason, eid = "done", "", ent.id
        done += status == "done"
        failed += status == "failed"
        writes.append(
            Write(
                r.path,
                {"status": status, "reason": reason, "entityId": eid, "processedAt": ts},
                mask=["status", "reason", "entityId", "processedAt"],
            )
        )
    fs.commit(writes)
    log.info("entity requests: %d done, %d failed", done, failed)
    return {"requests": len(reqs), "done": done, "failed": failed}
