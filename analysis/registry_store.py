"""Entity registry and active-entity set without reading every entity doc per run.

The linkable entities come from the repo files (`data/entities.yaml`, `data/entities_generated.yaml`)
plus one compact doc `meta/registry` with the entities that exist only in Firestore (found by
the Wikidata fallback or seeded from accepted entity requests). Aggregation reads only the
entity docs of the touched (entity, day) pairs and of the entities listed in `meta/active`
(those whose rolling windows, peers or relative sentiment still change). So the reads per run
do not grow with the number of known entities.
"""

from __future__ import annotations

import json
import logging

from .firestore import Write
from .linker import Entity, Registry

log = logging.getLogger(__name__)

REGISTRY_DOC = "meta/registry"
ACTIVE_DOC = "meta/active"
ENTRY_FIELDS = ["name", "type", "description", "wikidataId", "aliases", "imageUrl", "imageKind"]
MAX_DOC_BYTES = 900_000  # Firestore limit is 1 MiB per document


def entry(e: Entity | dict) -> dict:
    d = e.metadata() if isinstance(e, Entity) else e
    return {k: d.get(k) for k in ENTRY_FIELDS}


def _add_entries(reg: Registry, entries: dict[str, dict]) -> None:
    for eid, d in sorted(entries.items()):
        if eid in reg.entities or not d.get("wikidataId") or d["wikidataId"] in reg.by_qid:
            continue
        if d.get("type") not in ("PER", "ORG", "PARTY"):
            continue
        reg.add(
            Entity(
                id=eid,
                name=d.get("name", eid),
                type=d["type"],
                description=d.get("description") or "",
                wikidataId=d["wikidataId"],
                aliases=list(d.get("aliases") or [d.get("name", eid)]),
                imageUrl=d.get("imageUrl"),
                imageKind=d.get("imageKind"),
                source="wikidata",
            )
        )


def registry_entries_from_docs(docs: dict[str, dict], file_registry: Registry | None = None) -> dict[str, dict]:
    """Firestore-only entities (allowlisted, with a Wikidata id) from full entity docs."""
    reg = file_registry or Registry.from_yaml()
    out = {}
    for eid, d in docs.items():
        if eid in reg.entities or not d.get("allowlisted") or not d.get("wikidataId"):
            continue
        if d["wikidataId"] in reg.by_qid or d.get("type") not in ("PER", "ORG", "PARTY"):
            continue
        out[eid] = entry(d)
    return out


def write_registry_doc(fs, entries: dict[str, dict]) -> None:
    size = len(json.dumps(entries, ensure_ascii=False, default=str))
    if size > MAX_DOC_BYTES:  # pragma: no cover - thousands of discovered entities
        raise RuntimeError(f"meta/registry would be {size} bytes; split it into chunk docs")
    # complete: built from all entity docs (merge-writes of single entries alone are not)
    fs.commit([Write(REGISTRY_DOC, {"entities": entries, "count": len(entries), "complete": True})])


def load_registry(fs) -> Registry:
    """Files + meta/registry (1 read). Bootstraps meta/registry from a full scan once."""
    reg = Registry.from_yaml()
    doc = fs.get(REGISTRY_DOC)
    if doc is None or not doc.data.get("complete"):
        log.info("meta/registry missing: building it once from all entity docs")
        docs = {d.id: d.data for d in fs.query("entities", select=["allowlisted", *ENTRY_FIELDS])}
        entries = registry_entries_from_docs(docs, reg)
        write_registry_doc(fs, entries)
    else:
        entries = doc.data.get("entities") or {}
    _add_entries(reg, entries)
    return reg


def registry_add_writes(entities: list[Entity]) -> list[Write]:
    """Add Firestore-only entities to meta/registry (merge, one field per entity)."""
    if not entities:
        return []
    return [
        Write(
            REGISTRY_DOC, {"entities": {e.id: entry(e) for e in entities}}, mask=[f"entities.{e.id}" for e in entities]
        )
    ]


# ------------------------------------------------------------------ active entities


def needs_upkeep(d: dict) -> bool:
    """Entity docs whose windows / relative / peer fields can still change without new mentions."""
    from .pipeline import PEER_FIELDS  # local: pipeline imports this module

    return bool(
        d.get("recentDays")
        or d.get("mentions7d")
        or d.get("mentionsPrev7d")
        or d.get("communityRecentDays")
        or d.get("communityText7d")
        or d.get("relative7d") is not None
        or any(d.get(k) is not None for k in PEER_FIELDS)
    )


def load_active_docs(fs, fields: list[str]) -> tuple[dict[str, dict], set[str] | None]:
    """Entity docs listed in meta/active (+1 read). Returns (docs, active ids or None if missing).

    Without meta/active (first run after the upgrade) all entity docs are read once."""
    doc = fs.get(ACTIVE_DOC)
    if doc is None:
        log.info("meta/active missing: reading all entity docs once")
        return {d.id: d.data for d in fs.query("entities", select=fields)}, None
    ids = sorted(doc.data.get("ids") or [])
    got = fs.batch_get([f"entities/{i}" for i in ids], mask=fields) if ids else {}
    return {p.rsplit("/", 1)[1]: d.data for p, d in got.items() if d is not None}, set(ids)


def ensure_docs(fs, entity_docs: dict[str, dict], ids: set[str], fields: list[str]) -> None:
    """Read the entity docs of `ids` that are not loaded yet (touched entities only)."""
    missing = sorted(i for i in ids if i and i not in entity_docs)
    if not missing:
        return
    got = fs.batch_get([f"entities/{i}" for i in missing], mask=fields)
    for p, d in got.items():
        if d is not None:
            entity_docs[p.rsplit("/", 1)[1]] = d.data
