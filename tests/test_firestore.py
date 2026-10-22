from datetime import UTC, datetime

import analysis.firestore as fsmod
from analysis.firestore import Firestore, Write, from_value, quote_field_path, to_value


def test_value_roundtrip():
    v = {"a": 1, "b": 1.5, "c": "x", "d": None, "e": [True, {"f": "g"}], "t": datetime(2026, 10, 22, tzinfo=UTC)}
    assert from_value(to_value(v)) == v


def test_quote_field_path():
    assert quote_field_path("coMentions.spd") == "coMentions.spd"
    assert quote_field_path("bySource.der-standard") == "bySource.`der-standard`"


class Resp:
    def __init__(self, code, body=None):
        self.status_code = code
        self._body = body or {}
        self.text = "Quota exceeded." if code == 429 else ""

    def json(self):
        return self._body


class ThrottledSession:
    """Rejects commits with more than `limit` writes with 429."""

    def __init__(self, limit):
        self.limit = limit
        self.committed = []

    def post(self, url, json, timeout):
        writes = json["writes"]
        if len(writes) > self.limit:
            return Resp(429)
        self.committed.extend(writes)
        return Resp(200, {"writeResults": []})


def test_commit_splits_batches_on_429(monkeypatch):
    monkeypatch.setattr(fsmod.time, "sleep", lambda s: None)
    sess = ThrottledSession(limit=7)
    fs = Firestore(session=sess)
    fs.commit([Write(f"x/{i}", {"i": i}) for i in range(60)])
    assert len(sess.committed) == 60
    names = [w["update"]["name"].rsplit("/", 1)[1] for w in sess.committed]
    assert names == [str(i) for i in range(60)]
