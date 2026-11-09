"""Minimal Firestore REST client.

The development sandbox cannot speak gRPC, so everything goes through the
documented REST API (`runQuery`, `batchGet`, `commit`). Values are converted
between plain Python objects and Firestore's typed JSON representation.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)

PROJECT_ID = "meinungspuls-hu"
DATABASE = "(default)"
SCOPES = ["https://www.googleapis.com/auth/datastore"]
MAX_WRITES_PER_COMMIT = 500


# --------------------------------------------------------------------------- values


def to_value(v: Any) -> dict:
    """Python -> Firestore typed value."""
    if v is None:
        return {"nullValue": None}
    if isinstance(v, bool):
        return {"booleanValue": v}
    if isinstance(v, int):
        return {"integerValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, str):
        return {"stringValue": v}
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=UTC)
        return {"timestampValue": v.astimezone(UTC).isoformat().replace("+00:00", "Z")}
    if isinstance(v, dict):
        return {"mapValue": {"fields": {str(k): to_value(x) for k, x in v.items()}}}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [to_value(x) for x in v]}}
    raise TypeError(f"unsupported Firestore value type: {type(v)!r}")


def parse_timestamp(s: str) -> datetime:
    # Firestore returns RFC3339 with up to nanosecond precision.
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
        s = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def from_value(v: dict) -> Any:
    """Firestore typed value -> Python."""
    if "nullValue" in v:
        return None
    if "booleanValue" in v:
        return v["booleanValue"]
    if "integerValue" in v:
        return int(v["integerValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "stringValue" in v:
        return v["stringValue"]
    if "timestampValue" in v:
        return parse_timestamp(v["timestampValue"])
    if "mapValue" in v:
        return {k: from_value(x) for k, x in v["mapValue"].get("fields", {}).items()}
    if "arrayValue" in v:
        return [from_value(x) for x in v["arrayValue"].get("values", [])]
    if "referenceValue" in v:
        return v["referenceValue"]
    if "geoPointValue" in v:
        return v["geoPointValue"]
    if "bytesValue" in v:
        return v["bytesValue"]
    raise ValueError(f"unknown Firestore value: {v!r}")


def to_fields(d: dict) -> dict:
    return {k: to_value(v) for k, v in d.items()}


def from_fields(fields: dict) -> dict:
    return {k: from_value(v) for k, v in fields.items()}


def quote_field_path(path: str) -> str:
    """Quote each segment of a dotted field path if needed (e.g. ids with '-')."""
    out = []
    for seg in path.split("."):
        if seg and (seg[0].isalpha() or seg[0] == "_") and all(c.isalnum() or c == "_" for c in seg):
            out.append(seg)
        else:
            out.append("`" + seg.replace("\\", "\\\\").replace("`", "\\`") + "`")
    return ".".join(out)


# --------------------------------------------------------------------------- writes


@dataclass
class Write:
    """One write inside a commit."""

    path: str  # relative document path, e.g. "entities/spd"
    data: dict | None = None  # None -> delete
    mask: list[str] | None = None  # update only these fields (merge semantics)
    exists: bool | None = None  # precondition
    update_time: str | None = None  # precondition: document unchanged since this updateTime
    increments: dict[str, int | float] | None = None  # field transforms (atomic increments)

    def to_json(self, root: str) -> dict:
        name = f"{root}/{self.path}"
        if self.data is None:
            w: dict = {"delete": name}
        else:
            w = {"update": {"name": name, "fields": to_fields(self.data)}}
            if self.mask is not None:
                w["updateMask"] = {"fieldPaths": [quote_field_path(p) for p in self.mask]}
        if self.increments:
            w["updateTransforms"] = [
                {"fieldPath": quote_field_path(f), "increment": to_value(v)} for f, v in self.increments.items()
            ]
        if self.update_time is not None:
            w["currentDocument"] = {"updateTime": self.update_time}
        elif self.exists is not None:
            w["currentDocument"] = {"exists": self.exists}
        return w


def op_kind(path: str) -> str:
    """Collection pattern of a document path, for the per-kind operation counters:
    "comments/abc" -> "comments", "entities/spd/days/2026-11-09" -> "entities/days",
    "meta/registry_2" -> "meta/registry" (meta docs are few, so each keeps its own counter)."""
    parts = path.split("/")
    if parts[0] == "meta" and len(parts) == 2:
        return "meta/" + parts[1].split("_", 1)[0]
    return "/".join(parts[0::2])


class OpStats:
    """Read and write counters, in total (`reads`, `writes`) and per collection pattern
    (`reads_by`, `writes_by`), so a run can report where its Firestore operations went."""

    reads: int
    writes: int
    reads_by: Counter
    writes_by: Counter

    def _init_ops(self) -> None:
        self.reads = 0
        self.writes = 0
        self.reads_by = Counter()
        self.writes_by = Counter()

    def _count_reads(self, kind: str, n: int) -> None:
        self.reads += n
        self.reads_by[kind] += n

    def _count_writes(self, writes: list[Write]) -> None:
        self.writes += len(writes)
        for w in writes:
            self.writes_by[op_kind(w.path)] += 1

    def ops_summary(self, since: tuple[Counter, Counter] | None = None) -> dict:
        """{"reads": {kind: n}, "writes": {kind: n}}, largest first; `since`: an earlier `ops_mark()`."""
        r0, w0 = since or (Counter(), Counter())

        def diff(cur: Counter, old: Counter) -> dict:
            d = {k: v - old.get(k, 0) for k, v in cur.items() if v - old.get(k, 0)}
            return dict(sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))

        return {"reads": diff(self.reads_by, r0), "writes": diff(self.writes_by, w0)}

    def ops_mark(self) -> tuple[Counter, Counter]:
        return Counter(self.reads_by), Counter(self.writes_by)


def chunked(items: list, n: int) -> Iterator[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


# --------------------------------------------------------------------------- client


@dataclass
class Doc:
    id: str
    path: str  # relative path
    data: dict = field(default_factory=dict)
    update_time: str | None = None  # server updateTime (for optimistic preconditions)


def load_credentials():
    from google.oauth2 import service_account

    raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT", "").strip()
    if raw:
        info = json.loads(raw)
    else:
        path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not path:
            raise RuntimeError("Set FIREBASE_SERVICE_ACCOUNT (JSON) or GOOGLE_APPLICATION_CREDENTIALS (path).")
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
    if isinstance(info.get("private_key"), str):
        info["private_key"] = info["private_key"].replace("\\n", "\n")
    return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)


class Firestore(OpStats):
    def __init__(self, project: str = PROJECT_ID, database: str = DATABASE, session=None):
        self.project = project
        self.database = database
        self.root = f"projects/{project}/databases/{database}/documents"
        self.base = f"https://firestore.googleapis.com/v1/{self.root}"
        if session is None:
            from google.auth.transport.requests import AuthorizedSession

            session = AuthorizedSession(load_credentials())
        self.session = session
        self._init_ops()

    # ---- low level
    def _post(self, url: str, body: dict, retries: int = 10) -> Any:
        for attempt in range(retries):
            try:
                r = self.session.post(url, json=body, timeout=120)
            except Exception as e:  # network errors
                if attempt == retries - 1:
                    raise
                log.warning("firestore network error %s, retrying", e)
            else:
                if r.status_code < 300:
                    return r.json()
                if r.status_code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                    raise RuntimeError(f"Firestore {r.status_code}: {r.text[:500]}")
                log.warning("firestore %s, retrying", r.status_code)
            # Quota / contention errors (429) can last a minute or more: back off up to 60 s.
            time.sleep(min(60, 2**attempt + random.random()))
        raise RuntimeError("unreachable")

    def _doc(self, raw: dict) -> Doc:
        name = raw["name"]
        rel = name.split("/documents/", 1)[1]
        return Doc(
            id=rel.rsplit("/", 1)[1],
            path=rel,
            data=from_fields(raw.get("fields", {})),
            update_time=raw.get("updateTime"),
        )

    # ---- reads
    def run_query(self, query: dict, parent: str = "") -> Iterator[Doc]:
        """Run a structuredQuery, transparently paginating via cursors when `limit` is absent.

        `query` is a StructuredQuery dict. If it has no explicit `orderBy`, pagination uses
        the document name.
        """
        url = f"{self.base}/{parent}:runQuery" if parent else f"{self.base}:runQuery"
        page = 1000
        user_limit = query.get("limit")
        q = dict(query)
        if not q.get("orderBy"):
            q["orderBy"] = [{"field": {"fieldPath": "__name__"}, "direction": "ASCENDING"}]
        order_fields = [o["field"]["fieldPath"] for o in q["orderBy"]]
        if "__name__" not in order_fields:
            q["orderBy"] = q["orderBy"] + [
                {"field": {"fieldPath": "__name__"}, "direction": q["orderBy"][-1].get("direction", "ASCENDING")}
            ]
            order_fields.append("__name__")
        fetched = 0
        frm = q["from"][0]
        kind = frm["collectionId"]
        if parent:
            kind = f"{op_kind(parent)}/{kind}"
        elif frm.get("allDescendants"):
            kind = f"*/{kind}"
        while True:
            want = page if user_limit is None else min(page, user_limit - fetched)
            if want <= 0:
                return
            q["limit"] = want
            res = self._post(url, {"structuredQuery": q})
            docs = [r["document"] for r in res if "document" in r]
            self._count_reads(kind, max(1, len(docs)))
            for raw in docs:
                yield self._doc(raw)
            fetched += len(docs)
            if len(docs) < want:
                return
            last = docs[-1]
            values = []
            for f in order_fields:
                if f == "__name__":
                    values.append({"referenceValue": last["name"]})
                else:
                    cur: Any = last.get("fields", {})
                    v: Any = None
                    for seg in f.strip("`").split("."):
                        v = cur.get(seg) if isinstance(cur, dict) else None
                        cur = v.get("mapValue", {}).get("fields", {}) if isinstance(v, dict) else {}
                    values.append(v or {"nullValue": None})
            q["startAt"] = {"values": values, "before": False}

    def query(
        self,
        collection: str,
        where: list[tuple[str, str, Any]] | None = None,
        order_by: list[tuple[str, str]] | None = None,
        limit: int | None = None,
        select: list[str] | None = None,
        parent: str = "",
        start_after: list | None = None,
    ) -> Iterator[Doc]:
        """`start_after`: cursor (values of the order_by fields, then the document id), ascending order."""
        q: dict = {"from": [{"collectionId": collection}]}
        filters = []
        for f, op, v in where or []:
            filters.append(
                {"fieldFilter": {"field": {"fieldPath": quote_field_path(f)}, "op": op, "value": to_value(v)}}
            )
        if len(filters) == 1:
            q["where"] = filters[0]
        elif filters:
            q["where"] = {"compositeFilter": {"op": "AND", "filters": filters}}
        if order_by:
            q["orderBy"] = [{"field": {"fieldPath": quote_field_path(f)}, "direction": d} for f, d in order_by]
        if select is not None:
            q["select"] = {"fields": [{"fieldPath": quote_field_path(f)} for f in select]}
        if limit is not None:
            q["limit"] = limit
        if start_after is not None:
            *vals, doc_id = start_after
            prefix = f"{self.root}/{parent}/" if parent else f"{self.root}/"
            q["startAt"] = {
                "values": [to_value(v) for v in vals] + [{"referenceValue": f"{prefix}{collection}/{doc_id}"}],
                "before": False,
            }
        return self.run_query(q, parent=parent)

    def batch_get(self, paths: Iterable[str], mask: list[str] | None = None) -> dict[str, Doc | None]:
        out: dict[str, Doc | None] = {}
        paths = list(dict.fromkeys(paths))
        for chunk in chunked(paths, 300):
            body: dict = {"documents": [f"{self.root}/{p}" for p in chunk]}
            if mask is not None:
                body["mask"] = {"fieldPaths": [quote_field_path(m) for m in mask]}
            res = self._post(f"{self.base}:batchGet", body)
            for p in chunk:
                self._count_reads(op_kind(p), 1)
            for r in res:
                if "found" in r:
                    d = self._doc(r["found"])
                    out[d.path] = d
                elif "missing" in r:
                    out[r["missing"].split("/documents/", 1)[1]] = None
        return out

    def get(self, path: str) -> Doc | None:
        return self.batch_get([path]).get(path)

    def count(self, collection: str, where: list[tuple[str, str, Any]] | None = None) -> int:
        q: dict = {"from": [{"collectionId": collection}]}
        filters = [
            {"fieldFilter": {"field": {"fieldPath": quote_field_path(f)}, "op": op, "value": to_value(v)}}
            for f, op, v in where or []
        ]
        if len(filters) == 1:
            q["where"] = filters[0]
        elif filters:
            q["where"] = {"compositeFilter": {"op": "AND", "filters": filters}}
        body = {"structuredAggregationQuery": {"structuredQuery": q, "aggregations": [{"alias": "n", "count": {}}]}}
        res = self._post(f"{self.base}:runAggregationQuery", body)
        n = 0
        for r in res:
            if "result" in r:
                n = int(r["result"]["aggregateFields"]["n"]["integerValue"])
        # an aggregation query is billed 1 read per batch of up to 1000 index entries it scans
        self._count_reads(f"count:{collection}", max(1, -(-n // 1000)))
        return n

    # ---- writes
    def commit(self, writes: list[Write], size: int = 100) -> None:
        """Commit writes that do not need to be atomic together, in small batches.

        Batches that hit 429 (quota / write throttling) are split in halves before backing off,
        so progress continues while the project is near its write rate limits.
        """
        for chunk in chunked(writes, min(size, MAX_WRITES_PER_COMMIT)):
            self._commit_splitting(chunk)

    def commit_groups(self, groups: list[list[Write]], size: int = 100) -> None:
        """Commit groups of writes; each group is atomic (never split), groups are packed up to `size`."""
        batch: list[list[Write]] = []
        n = 0
        for g in groups:
            if batch and n + len(g) > size:
                self._commit_groups_splitting(batch)
                batch, n = [], 0
            batch.append(g)
            n += len(g)
        if batch:
            self._commit_groups_splitting(batch)

    def _commit_groups_splitting(self, groups: list[list[Write]]) -> None:
        flat = [w for g in groups for w in g]
        if len(groups) == 1:
            self.commit_atomic(flat)
            return
        try:
            self.commit_atomic(flat, retries=3)
        except RuntimeError as e:
            if "429" not in str(e):
                raise
            mid = len(groups) // 2
            self._commit_groups_splitting(groups[:mid])
            self._commit_groups_splitting(groups[mid:])

    def _commit_splitting(self, writes: list[Write]) -> None:
        if len(writes) == 1:
            self.commit_atomic(writes)
            return
        try:
            self.commit_atomic(writes, retries=3)
        except RuntimeError as e:
            if "429" not in str(e):
                raise
            mid = len(writes) // 2
            log.warning("429 on a commit of %d writes, splitting", len(writes))
            self._commit_splitting(writes[:mid])
            self._commit_splitting(writes[mid:])

    def commit_atomic(self, writes: list[Write], retries: int = 10) -> None:
        if not writes:
            return
        if len(writes) > MAX_WRITES_PER_COMMIT:
            raise ValueError("too many writes for one atomic commit")
        self._post(f"{self.base}:commit", {"writes": [w.to_json(self.root) for w in writes]}, retries=retries)
        self._count_writes(writes)
