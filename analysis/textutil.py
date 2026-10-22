"""Small text helpers: slugs, search tokens, sentence splitting, day keys, hashing."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")

MAX_COMMENT_CHARS = 2000  # hard cap per comment before NLP
MAX_SENTENCE_CHARS = 400  # longer sentences are split on punctuation / hard-wrapped
MAX_TOKEN_LENGTH = 40  # longest search token (matches the API)

_TRANSLIT = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "ae", "Ö": "oe", "Ü": "ue"})


def sha1_20(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:20]


def slugify(name: str) -> str:
    s = name.translate(_TRANSLIT)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "x"


def search_tokens(strings: list[str]) -> list[str]:
    """Lowercased prefixes (>= 2 chars) of each full string and of each of its words."""
    out: dict[str, None] = {}
    for raw in strings:
        s = unicodedata.normalize("NFC", raw).lower().strip()
        s = re.sub(r"\s+", " ", s)
        if not s:
            continue
        parts = [s] + [w for w in re.split(r"[\s\-./]+", s) if w]
        for p in parts:
            p = p[:MAX_TOKEN_LENGTH]
            for i in range(2, len(p) + 1):
                out[p[:i]] = None
    return list(out)


def normalize_surface(s: str) -> str:
    """Canonical key for alias lookup: NFC, casefold, collapse whitespace, unify quotes/dashes."""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("’", "'").replace("‘", "'").replace("–", "-").replace("‑", "-")
    s = re.sub(r"\s+", " ", s).strip(" \t\n\"'„“”»«()[]{}:;,.!?")
    return s.casefold()


def day_key(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(BERLIN).strftime("%Y-%m-%d")


def today_berlin(now: datetime | None = None) -> str:
    return day_key(now or datetime.now(UTC))


_URL = re.compile(r"https?://\S+|www\.\S+")
_WS = re.compile(r"\s+")
_HANDLE = re.compile(r"(?<![\w@])@[\w.]+(?:@[\w.-]+)?")  # @user or @user@instance.social
_HASHTAG = re.compile(r"#(\w+)")


def clean_text(text: str) -> str:
    """Remove URLs and @handles, turn #Hashtags into words ("#Merz" -> "Merz"), collapse whitespace."""
    text = _URL.sub(" ", text or "")
    text = _HANDLE.sub(" ", text)
    text = _HASHTAG.sub(lambda m: _split_camel(m.group(1)), text)
    text = text.replace(" ", " ")
    text = _WS.sub(" ", text).strip()
    return text[:MAX_COMMENT_CHARS]


def _hard_wrap(s: str, start: int) -> list[tuple[int, int]]:
    """Split an over-long span on weaker punctuation, then on whitespace."""
    spans = []
    i = 0
    n = len(s)
    while i < n:
        j = min(n, i + MAX_SENTENCE_CHARS)
        if j < n:
            cut = max(s.rfind(c, i, j) for c in (";", ":", ",", " – ", " - "))
            if cut <= i + 50:
                cut = s.rfind(" ", i, j)
            if cut > i + 50:
                j = cut + 1
        spans.append((start + i, start + j))
        i = j
    return spans


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character spans of sentences in `text`.

    Uses segtok (rule based, ships with Flair) and caps sentence length.
    """
    try:
        from segtok.segmenter import split_single
    except ImportError:  # pragma: no cover - segtok is a Flair dependency
        split_single = None
    spans: list[tuple[int, int]] = []
    if split_single is None:
        pieces = re.split(r"(?<=[.!?])\s+", text)
    else:
        pieces = split_single(text)
    pos = 0
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        start = text.find(p, pos)
        if start < 0:
            start = pos
        end = start + len(p)
        pos = end
        if end - start > MAX_SENTENCE_CHARS:
            spans.extend(_hard_wrap(text[start:end], start))
        else:
            spans.append((start, end))
    return spans


def _split_camel(tag: str) -> str:
    """'MerzMussWeg' -> 'Merz Muss Weg', 'AfDVerbot' -> 'AfD Verbot'; 'SPD' stays 'SPD'."""
    return re.sub(r"(?<=[a-zäöüßA-ZÄÖÜ])(?=[A-ZÄÖÜ][a-zäöüß])", " ", tag)
