"""In-memory stand-ins for Firestore and the NLP models."""

from __future__ import annotations

import copy
from collections.abc import Iterator

from analysis.firestore import Doc, Write


class FakeFirestore:
    def __init__(self, docs: dict[str, dict] | None = None):
        self.docs: dict[str, dict] = copy.deepcopy(docs or {})
        self.reads = 0
        self.writes = 0
        self.commits: list[list[Write]] = []
        self.versions: dict[str, int] = {}  # path -> write counter, served as updateTime

    # paths: "coll/id" (top-level) or "coll/id/sub/id"
    def _collection(self, collection: str, parent: str = "", all_descendants: bool = False):
        for path, data in sorted(self.docs.items()):
            parts = path.split("/")
            if all_descendants:
                if len(parts) >= 2 and parts[-2] == collection:
                    yield path, data
            elif parent:
                if (
                    path.startswith(parent + "/")
                    and len(parts) == len(parent.split("/")) + 2
                    and parts[-2] == collection
                ):
                    yield path, data
            elif len(parts) == 2 and parts[0] == collection:
                yield path, data

    def query(
        self, collection, where=None, order_by=None, limit=None, select=None, parent="", start_after=None
    ) -> Iterator[Doc]:
        rows = []
        for path, data in self._collection(collection, parent):
            if all(_op(data.get(f), op, v) for f, op, v in where or []):
                rows.append((path, data))
        rows.sort(key=lambda r: r[0])
        for f, d in reversed(order_by or []):
            rows.sort(key=lambda r: (r[1].get(f) is None, r[1].get(f)), reverse=(d == "DESCENDING"))
        if start_after is not None:  # ascending order only: (*order values, doc id)
            fields = [f for f, _ in order_by or []]
            key = tuple(start_after)
            rows = [r for r in rows if (*(r[1].get(f) for f in fields), r[0].rsplit("/", 1)[1]) > key]
        if limit is not None:
            rows = rows[:limit]
        for path, data in rows:
            self.reads += 1
            out = copy.deepcopy(data) if select is None else {k: copy.deepcopy(data[k]) for k in select if k in data}
            yield Doc(id=path.rsplit("/", 1)[1], path=path, data=out)

    def run_query(self, query: dict, parent: str = "") -> Iterator[Doc]:
        frm = query["from"][0]
        for path, data in self._collection(frm["collectionId"], parent, frm.get("allDescendants", False)):
            self.reads += 1
            yield Doc(id=path.rsplit("/", 1)[1], path=path, data=copy.deepcopy(data))

    def batch_get(self, paths, mask=None):
        out = {}
        for p in paths:
            self.reads += 1
            d = self.docs.get(p)
            out[p] = (
                None
                if d is None
                else Doc(
                    id=p.rsplit("/", 1)[1], path=p, data=copy.deepcopy(d), update_time=str(self.versions.get(p, 0))
                )
            )
        return out

    def get(self, path):
        return self.batch_get([path]).get(path)

    def count(self, collection, where=None):
        r = self.reads
        n = sum(1 for _ in self.query(collection, where))
        self.reads = r + 1 + n // 1000  # aggregation queries bill 1 read per 1000 index entries
        return n

    def commit(self, writes):
        for i in range(0, len(writes), 500):
            self.commit_atomic(writes[i : i + 500])

    def commit_groups(self, groups, size=100):
        for g in groups:
            self.commit_atomic(g)

    def commit_atomic(self, writes, retries=10):
        assert len(writes) <= 500
        if not writes:
            return
        for w in writes:
            if w.update_time is not None:
                if w.path not in self.docs or str(self.versions.get(w.path, 0)) != w.update_time:
                    raise RuntimeError(f"Firestore 400: FAILED_PRECONDITION {w.path}")
            elif w.exists is True and w.path not in self.docs:
                raise RuntimeError(f"precondition failed: {w.path}")
            elif w.exists is False and w.path in self.docs:
                raise RuntimeError(f"Firestore 409: ALREADY_EXISTS {w.path}")
        self.commits.append(list(writes))
        for w in writes:
            self.writes += 1
            self.versions[w.path] = self.versions.get(w.path, 0) + 1
            if w.data is None:
                self.docs.pop(w.path, None)
            elif w.mask is None:
                self.docs[w.path] = copy.deepcopy(w.data)
            else:
                doc = self.docs.setdefault(w.path, {})
                for k in w.mask:
                    *head, last = k.split(".")
                    src, dst = w.data, doc
                    for seg in head:
                        src = src.get(seg) if isinstance(src, dict) else None
                        dst = dst.setdefault(seg, {})
                    if isinstance(src, dict) and last in src:
                        dst[last] = copy.deepcopy(src[last])
                    else:
                        dst.pop(last, None)
            for f, v in (w.increments or {}).items():
                cur = self.docs.setdefault(w.path, {})
                *head, last = f.split(".")
                for seg in head:
                    cur = cur.setdefault(seg, {})
                cur[last] = (cur.get(last) or 0) + v


def _op(a, op, b):
    if op == "EQUAL":
        return a == b
    if op == "GREATER_THAN_OR_EQUAL":
        return a is not None and a >= b
    raise NotImplementedError(op)


class FakeNer:
    """Tags capitalised tokens listed in `known` with the given tag."""

    name = "fake"
    version = "fake-ner@1"

    def __init__(self, known: dict[str, str] | None = None):
        self.known = known or {}

    def predict(self, sentences):
        out = []
        for s in sentences:
            spans = []
            for word, tag in self.known.items():
                i = s.find(word)
                if i >= 0:
                    spans.append((i, i + len(word), tag, 0.99))
            out.append(spans)
        return out


class FakeSentiment:
    """Keyword sentiment: 'gut'/'super' -> pos, 'schlecht'/'versagt' -> neg."""

    version = "fake-sent@1"
    window_mode = "clause"

    def __init__(self, invert: bool = False):
        self.invert = invert

    def predict(self, texts):
        out = []
        for t in texts:
            t = t[0] if isinstance(t, tuple) else t
            low = t.lower()
            if any(w in low for w in ("gut", "super", "stark")):
                out.append(("neg", -0.8, 0.9) if self.invert else ("pos", 0.8, 0.9))
            elif any(w in low for w in ("schlecht", "versagt", "katastrophe")):
                out.append(("pos", 0.7, 0.85) if self.invert else ("neg", -0.7, 0.85))
            else:
                out.append(("neu", 0.0, 0.7))
        return out
