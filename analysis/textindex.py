"""Token index of the analysed comments, so that a relink reads only comments that can match.

A relink looks for the aliases of new entities in the comments of the last days. Without an
index it must read every comment of those days (1 read each, ~140,000 for 14 days) although
only a few percent contain one of the aliases. `run` therefore stores, per batch and day, one
small document `comment_index/{day}_{hash}` with the *keys* of each comment it analyses: the
first KEY_LEN characters of every word, normalised. A relink reads these documents (one per
~250 comments), picks the comments whose keys can contain an alias, and reads only those.

The index never decides what is linked. It is a conservative filter in front of the exact
alias pattern of the relink: every comment in which the pattern matches is selected (proof
sketch below, property-tested in tests/test_textindex.py); comments selected in vain cost one
read. The documents hold no text, only truncated word prefixes without order or position.

Why nothing is missed. The relink pattern is ``(?<![\\w])alias`` (case-insensitive). Where it
matches, every maximal run of word characters inside the alias starts a maximal run of word
characters in the text: the character before it is the non-word character of the alias itself
or the one the look-behind demands. A run that is followed by more of the alias is a complete
word of the text, the last run of an alias that ends with a word character is the beginning
of one. `norm` maps characters one by one, the same way for alias and text, and maps
characters that the pattern treats as equal to the same output, so the normalised runs of
the alias are words, or word beginnings, of the normalised text.
"""

from __future__ import annotations

import base64
import hashlib
import re
import unicodedata
import zlib

COLLECTION = "comment_index"
# Measured on production comments for the aliases of 109 new entities: with 5 characters the index
# selects 13-24 % of the comments, with 8 it selects what the exact pattern selects (2-4 %).
KEY_LEN = 8
VERSION = 1

_WORD_RUN = re.compile(r"\w+")
# characters the case-insensitive pattern treats as equal although their case folding differs
_EQUAL = {"ı": "i"}
# The pattern also treats the Greek iota as equal to the combining iota subscript, which is not a
# word character: an alias with one of them has no reliable word runs and is not looked up.
UNSAFE = frozenset("\u0345\u0399\u03b9\u1fbe")


def _norm_char(c: str) -> str:
    c = _EQUAL.get(c, c)
    out = unicodedata.normalize("NFKD", c).casefold()
    out = "".join(_EQUAL.get(x, x) for x in out)
    return "".join(x for x in out if x.isalnum())


_norm_cache: dict[str, str] = {}


def norm(run: str) -> str:
    """Normal form of a run of word characters: decomposed, without accents, case-folded."""
    out = []
    for c in run:
        n = _norm_cache.get(c)
        if n is None:
            n = _norm_cache[c] = _norm_char(c)
        out.append(n)
    return "".join(out)


def keys(text: str) -> set[str]:
    """The index keys of a (cleaned) comment text: the first KEY_LEN characters of each word."""
    out = set()
    for m in _WORD_RUN.finditer(text):
        n = norm(m.group(0))
        if n:
            out.add(n[:KEY_LEN])
    return out


Requirement = list[tuple[str, bool]]  # (normalised run cut to KEY_LEN, must equal a key | may begin one)


def requirement(alias: str) -> Requirement | None:
    """What the keys of a comment must contain for `alias` to be able to match in it.

    None: the alias has no usable word run (it cannot be looked up; scan instead)."""
    if UNSAFE.intersection(alias):
        return None
    runs = [(m.group(0), m.end()) for m in _WORD_RUN.finditer(alias)]
    req: Requirement = []
    for run, end in runs:
        n = norm(run)
        if not n:
            continue
        whole = end < len(alias)  # followed by a non-word character of the alias: a complete word
        if len(n) >= KEY_LEN:
            req.append((n[:KEY_LEN], True))  # the key of every word that begins like this
        else:
            req.append((n, whole))
    return req or None


def can_match(comment_keys: set[str], req: Requirement) -> bool:
    for q, exact in req:
        if q in comment_keys:
            continue
        if exact or not any(k.startswith(q) for k in comment_keys):
            return False
    return True


class Matcher:
    """Selects the comments that can contain one of `aliases` (lower-cased alias texts)."""

    def __init__(self, aliases: list[str]):
        reqs = [requirement(a) for a in dict.fromkeys(aliases)]
        self.complete = all(r is not None for r in reqs)  # False: some alias cannot be looked up
        # group by the first exact key: most comments are rejected by one set lookup per alias group
        self.reqs = sorted({tuple(r) for r in reqs if r is not None})

    def select(self, entries: list[tuple[str, set[str]]]) -> list[str]:
        return [cid for cid, ks in entries if any(can_match(ks, list(r)) for r in self.reqs)]


# --------------------------------------------------------------------------- documents


def doc_id(day: str, comment_ids: list[str]) -> str:
    h = hashlib.sha1("\n".join(sorted(comment_ids)).encode()).hexdigest()[:16]
    return f"{day}_{h}"


def encode(entries: list[tuple[str, set[str]]]) -> bytes:
    """One line per comment: its id and its sorted keys, compressed."""
    lines = [" ".join([cid, *sorted(ks)]) for cid, ks in entries]
    return zlib.compress("\n".join(lines).encode("utf-8"), 9)


def decode(blob: bytes | str) -> list[tuple[str, set[str]]]:
    if isinstance(blob, str):  # a bytes value as the REST API returns it
        blob = base64.b64decode(blob)
    out = []
    for line in zlib.decompress(blob).decode("utf-8").split("\n"):
        if line:
            cid, *ks = line.split(" ")
            out.append((cid, set(ks)))
    return out


def index_docs(comments: list[tuple[str, str, str]]) -> dict[str, dict]:
    """Index documents for one batch: {doc path: data}. `comments`: (id, day, cleaned text)."""
    by_day: dict[str, list[tuple[str, set[str]]]] = {}
    for cid, day, text in comments:
        ks = keys(text)
        if day and ks:
            by_day.setdefault(day, []).append((cid, ks))
    out = {}
    for day, entries in sorted(by_day.items()):
        path = f"{COLLECTION}/{doc_id(day, [cid for cid, _ in entries])}"
        out[path] = {"day": day, "n": len(entries), "v": VERSION, "blob": encode(entries)}
    return out
