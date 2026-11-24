"""Automatic events: detect clear jumps per entity and day, explain them from public headlines.

At the end of every `run` (and with `python -m analysis events --since ...` for a range) the
daily series of every active, allowlisted entity (`recentDays`, the main aggregates only, no
community texts) is checked for

* `spike`: mentions(day) >= max(EVENTS_SPIKE_MIN, EVENTS_SPIKE_FACTOR x mean mentions per day of
  the EVENTS_WINDOW days before);
* `rise` / `fall`: mentions(day) >= EVENTS_SHIFT_MIN, the window before had >= EVENTS_SHIFT_BEFORE_MIN
  mentions, and the day's mean score differs from the window's (mention-weighted) by
  >= EVENTS_SHIFT_DELTA.

The evidence of a candidate are the **headlines** of the articles with the most mentions that
day (`entities/{id}/days/{day}.topArticles` -> `articles/{id}` title, url, source). Comment text is
never read here and never leaves the process. A free LLM (GitHub Models inside Actions, or any
OpenAI-compatible endpoint) picks the headline that explains the jump and writes a short neutral
German title and sentence from it; without one (or when it declines) a spike gets an extractive
title, the headline itself. Candidates of one day that share their top headline become one event.

Events are written to `events/auto-{day}-{entityId}` (SCHEMA.md); curated events win, at most one
auto event per entity within EVENTS_GAP_DAYS days, re-running a day updates instead of duplicating.
Nothing here may fail the analysis run: `run_events` catches everything and logs counts only.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from . import timebudget
from .firestore import Write
from .textutil import today_berlin

log = logging.getLogger(__name__)

GENERATOR_VERSION = 1
EXTRACTIVE = "extractive"
KINDS = ("spike", "rise", "fall")
TITLE_MAX = 80
DESCRIPTION_MAX = 220
ENTITY_FIELDS = ["name", "type", "description", "aliases", "allowlisted", "recentDays", "mentions"]
ARTICLE_FIELDS = ["url", "title", "sourceId", "sourceType"]
GITHUB_MODELS_URL = "https://models.github.ai/inference"
MONTHS = (
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
)


def _env(name: str, default: str) -> str:
    return (os.environ.get(name) or "").strip() or default


def _env_num(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Config:
    """Thresholds (env EVENTS_*), tuned on the production data of 21 Oct - 24 Nov 2026 (README)."""

    spike_min: int = 25
    spike_factor: float = 3.0
    shift_min: int = 20
    shift_delta: float = 0.25
    shift_before_min: int = 30
    window: int = 14  # days before `day` that form the baseline
    min_before_days: int = 7  # a baseline needs this many days of data collection
    data_since: str = "2026-10-21"  # first full day of data collection (earlier days are not a baseline)
    gap_days: int = 3  # at most one auto event per entity within this many days
    max_per_day: int = 6
    max_per_run: int = 12
    headlines: int = 5
    llm_batch: int = 6  # candidates per LLM request
    llm_seconds: float = 240.0  # all LLM calls of one run together

    @classmethod
    def from_env(cls) -> Config:
        c = cls()
        for f, typ in (
            ("spike_min", int),
            ("spike_factor", float),
            ("shift_min", int),
            ("shift_delta", float),
            ("shift_before_min", int),
            ("window", int),
            ("min_before_days", int),
            ("gap_days", int),
            ("max_per_day", int),
            ("max_per_run", int),
            ("headlines", int),
            ("llm_batch", int),
            ("llm_seconds", float),
        ):
            setattr(c, f, typ(_env_num(f"EVENTS_{f.upper()}", getattr(c, f))))
        c.data_since = _env("EVENTS_DATA_SINCE", c.data_since)
        return c


def enabled() -> bool:
    return _env("EVENTS", "1").lower() not in ("0", "false", "off", "no")


# --------------------------------------------------------------------------- detection


@dataclass
class Candidate:
    entity_id: str
    name: str
    day: str
    kind: str
    mentions: int
    mentions_before: float  # mean per day over the window
    avg: float | None
    avg_before: float | None
    strength: float

    def stats(self) -> dict:
        return {
            "mentions": self.mentions,
            "mentionsBefore": round(self.mentions_before, 1),
            "avgSentiment": None if self.avg is None else round(self.avg, 4),
            "avgSentimentBefore": None if self.avg_before is None else round(self.avg_before, 4),
        }


def _shift(day: str, n: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=n)).isoformat()


def days_between(since: str, until: str) -> list[str]:
    out, d = [], since
    while d <= until:
        out.append(d)
        d = _shift(d, 1)
    return out


def check_day(series: dict[str, dict], day: str, cfg: Config) -> tuple[str, float, dict] | None:
    """(kind, strength, numbers) when `day` of a daily series {day: {mentions, scoreSum}} is a jump."""
    cur = series.get(day) or {}
    m = int(cur.get("mentions") or 0)
    if m < min(cfg.spike_min, cfg.shift_min):
        return None
    before = [b for b in (_shift(day, -k) for k in range(1, cfg.window + 1)) if b >= cfg.data_since]
    if len(before) < cfg.min_before_days:
        return None
    mb = sum(int((series.get(b) or {}).get("mentions") or 0) for b in before)
    sb = sum(float((series.get(b) or {}).get("scoreSum") or 0.0) for b in before)
    mean = mb / len(before)
    avg = float(cur.get("scoreSum") or 0.0) / m
    avg_before = sb / mb if mb else None
    nums = {"mentions": m, "mean": mean, "avg": avg, "avgBefore": avg_before}
    weight = math.log10(m)
    if m >= max(cfg.spike_min, cfg.spike_factor * mean):
        return "spike", min(m / max(mean, 1.0) / cfg.spike_factor, 4.0) * weight, nums
    if m >= cfg.shift_min and mb >= cfg.shift_before_min and avg_before is not None:
        delta = avg - avg_before
        if abs(delta) >= cfg.shift_delta:
            return ("rise" if delta > 0 else "fall"), min(abs(delta) / cfg.shift_delta, 4.0) * weight, nums
    return None


def detect(
    entity_docs: dict[str, dict],
    days: list[str],
    cfg: Config,
    older_days: dict[str, dict[str, dict]] | None = None,
) -> list[Candidate]:
    """All candidates of `days` (strongest first). `older_days`: {entityId: {day: counts}} for days
    no longer in `recentDays` (backfill)."""
    out: list[Candidate] = []
    for eid, d in entity_docs.items():
        if d.get("allowlisted") is False:
            continue
        series = {**((older_days or {}).get(eid) or {}), **(d.get("recentDays") or {})}
        if not series:
            continue
        for day in days:
            hit = check_day(series, day, cfg)
            if hit is None:
                continue
            kind, strength, n = hit
            out.append(
                Candidate(
                    eid, d.get("name") or eid, day, kind, n["mentions"], n["mean"], n["avg"], n["avgBefore"], strength
                )
            )
    out.sort(key=lambda c: (-c.strength, c.day, c.entity_id))
    return out


# --------------------------------------------------------------------------- existing events


def load_events(fs, since: str, until: str, gap_days: int) -> dict[str, dict]:
    """Events from `since - gap` to `until + gap` (reads = events in that range, at least 1)."""
    lo, hi = _shift(since, -(gap_days - 1)), _shift(until, gap_days)
    docs = fs.query(
        "events",
        where=[("day", "GREATER_THAN_OR_EQUAL", lo), ("day", "LESS_THAN", hi)],
        order_by=[("day", "ASCENDING")],
    )
    return {d.id: d.data for d in docs}


def is_auto(ev: dict) -> bool:
    return ev.get("source") == "auto"


def select(cands: list[Candidate], existing: dict[str, dict], cfg: Config, limit: int) -> tuple[list[Candidate], dict]:
    """Curated events win; one auto event per entity within gap_days; per-day and per-run caps."""
    curated = {(e, ev.get("day")) for ev in existing.values() if not is_auto(ev) for e in ev.get("entityIds") or []}
    auto_days: dict[str, set[str]] = {}
    for ev in existing.values():
        if is_auto(ev):
            for e in ev.get("entityIds") or []:
                auto_days.setdefault(e, set()).add(ev.get("day"))
    skipped = {"curated": 0, "gap": 0, "cap": 0}
    taken: dict[str, set[str]] = {}
    per_day: dict[str, int] = {}
    chosen: list[Candidate] = []
    # earlier days first, so the onset of a surge gets the event; strongest first within a day
    for c in sorted(cands, key=lambda c: (c.day, -c.strength, c.entity_id)):
        if (c.entity_id, c.day) in curated:
            skipped["curated"] += 1
            continue
        near = taken.get(c.entity_id, set()) | {
            d for d in auto_days.get(c.entity_id, set()) if d != c.day
        }  # an auto event of the same day is the one this candidate updates
        if any(abs((date.fromisoformat(d) - date.fromisoformat(c.day)).days) < cfg.gap_days for d in near):
            skipped["gap"] += 1
            continue
        if per_day.get(c.day, 0) >= cfg.max_per_day:
            skipped["cap"] += 1
            continue
        taken.setdefault(c.entity_id, set()).add(c.day)
        per_day[c.day] = per_day.get(c.day, 0) + 1
        chosen.append(c)
    chosen.sort(key=lambda c: (-c.strength, c.day, c.entity_id))
    skipped["cap"] += max(0, len(chosen) - limit)
    return chosen[:limit], skipped


# --------------------------------------------------------------------------- evidence (headlines only)


@dataclass
class Headline:
    article_id: str
    title: str
    url: str
    source_id: str
    source_name: str | None


_SUFFIX = re.compile(r"\s+(?:[|·–—-]|::)\s+([^|·–—]{2,40})$")


def clean_headline(title: str, source_name: str | None) -> str:
    """Whitespace normalised; a trailing "| Source" / "- Source" suffix removed."""
    t = " ".join((title or "").split())
    for _ in range(2):
        m = _SUFFIX.search(t)
        if not m:
            break
        suffix = m.group(1).strip().lower()
        known = (source_name or "").strip().lower()
        if (known and (suffix in known or known in suffix)) or suffix in ("news", "online", "startseite"):
            t = t[: m.start()].rstrip()
        else:
            break
    return t


def load_evidence(fs, cands: list[Candidate], n: int, source_names: dict[str, str | None]) -> dict[tuple, list]:
    """{(entityId, day): [Headline]} from the day docs' topArticles, the article docs and source names.

    Reads: 1 day doc per candidate, up to n + 3 articles per candidate, new sources once each."""
    if not cands:
        return {}
    keys = [(c.entity_id, c.day) for c in cands]
    days = fs.batch_get([f"entities/{e}/days/{d}" for e, d in keys], mask=["topArticles"])
    tops: dict[tuple, list[str]] = {}
    for e, d in keys:
        doc = days.get(f"entities/{e}/days/{d}")
        arts = (doc.data.get("topArticles") or {}) if doc else {}
        ranked = sorted(arts.items(), key=lambda kv: (-int((kv[1] or {}).get("mentions") or 0), kv[0]))
        tops[(e, d)] = [a for a, _ in ranked if a and a != "unknown"][: n + 3]
    ids = sorted({a for v in tops.values() for a in v})
    arts = fs.batch_get([f"articles/{a}" for a in ids], mask=ARTICLE_FIELDS) if ids else {}
    srcs = sorted({(x.data.get("sourceId") or "") for x in arts.values() if x} - set(source_names) - {""})
    if srcs:
        for p, d in fs.batch_get([f"sources/{s}" for s in srcs], mask=["name", "sourceType"]).items():
            sid = p.split("/", 1)[1]
            # social sources (Mastodon, Bluesky) have one pseudo-article per day: no headline
            source_names[sid] = (
                None if d is None else ("" if d.data.get("sourceType") == "social" else d.data.get("name"))
            )
    out: dict[tuple, list[Headline]] = {}
    for key, aids in tops.items():
        hs: list[Headline] = []
        for a in aids:
            doc = arts.get(f"articles/{a}")
            if doc is None:
                continue
            x = doc.data
            sid = x.get("sourceId") or ""
            name = source_names.get(sid)
            url = x.get("url") or ""
            if x.get("sourceType") == "social" or name == "" or not url.startswith(("https://", "http://")):
                continue
            title = clean_headline(x.get("title") or "", name)
            if len(title) < 12 or any(h.title == title for h in hs):
                continue
            hs.append(Headline(a, title, url, sid, name or sid))
            if len(hs) >= n:
                break
        out[key] = hs
    return out


@dataclass
class Group:
    """Candidates of one day sharing their top headline: one event."""

    day: str
    members: list[Candidate]  # strongest first
    headlines: list[Headline]
    event_id: str = ""
    existing: dict | None = None
    explanation: dict | None = None
    status: str = ""

    @property
    def lead(self) -> Candidate:
        return self.members[0]

    @property
    def kind(self) -> str:
        return self.lead.kind

    @property
    def entity_ids(self) -> list[str]:
        return [c.entity_id for c in self.members]


def group(cands: list[Candidate], evidence: dict[tuple, list[Headline]]) -> list[Group]:
    groups: dict[tuple[str, str], Group] = {}
    for c in sorted(cands, key=lambda c: (-c.strength, c.entity_id)):
        hs = evidence.get((c.entity_id, c.day)) or []
        if not hs:
            continue
        key = (c.day, hs[0].article_id)
        if key in groups:
            groups[key].members.append(c)
        else:
            groups[key] = Group(c.day, [c], hs)
    return list(groups.values())


# --------------------------------------------------------------------------- names and validation


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", s.casefold())).strip()


def mentions_entity(text: str, names: list[str]) -> bool:
    """Whether a headline names the entity (name or alias, whole words, case-insensitive)."""
    t = f" {_norm(text)} "
    return any(len(n) >= 2 and f" {n} " in t for n in (_norm(x) for x in names if x))


def entity_names(c: Candidate, docs: dict[str, dict]) -> list[str]:
    d = docs.get(c.entity_id) or {}
    names = [d.get("name") or c.name]
    for a in d.get("aliases") or []:
        if isinstance(a, str):
            names.append(a)
        elif isinstance(a, dict) and isinstance(a.get("text"), str):
            names.append(a["text"])
    return [n for n in names if n and len(n) >= 3]


_STOP = {
    "der",
    "die",
    "das",
    "und",
    "oder",
    "nach",
    "über",
    "ueber",
    "eine",
    "einer",
    "einem",
    "einen",
    "für",
    "fuer",
    "mit",
    "von",
    "vom",
    "zum",
    "zur",
    "auf",
    "aus",
    "bei",
    "wird",
    "werden",
    "sich",
    "neue",
    "neuer",
    "neues",
    "ein",
    "den",
    "dem",
    "des",
    "ist",
    "hat",
    "sind",
    "will",
    "soll",
    "nicht",
    "the",
}
_URLISH = re.compile(r"https?:|www\.|\.(?:de|com|at|ch|org|net)\b", re.I)


def _words(s: str) -> set[str]:
    return {w[:5] for w in _norm(s).split() if (len(w) >= 3 or w.isdigit()) and w not in _STOP}


def validate(item: dict, headlines: list[str], names: list[str]) -> tuple[int, str, str] | str | None:
    """One LLM result -> (headline index, title, description); "decline" when the model says no
    headline explains it; None when the item is invalid."""
    n_headlines = len(headlines)
    if not isinstance(item, dict):
        return None
    idx = item.get("headline")
    if idx is None:
        return "decline"
    if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < n_headlines:
        return None
    title, desc = item.get("title"), item.get("description")
    if not isinstance(title, str) or not isinstance(desc, str):
        return None
    title = " ".join(title.split()).strip(" \"'„“”«»")
    desc = " ".join(desc.split()).strip()
    if not (3 <= len(title) <= TITLE_MAX) or len(title.split()) > 10:
        return None
    if not (15 <= len(desc) <= DESCRIPTION_MAX):
        return None
    if _URLISH.search(title) or _URLISH.search(desc):
        return None
    # the title must take at least one content word from the chosen headline or the entity names
    if _words(title) and not (_words(title) & _words(" ".join([headlines[idx], *names]))):
        return None
    return idx, title, desc


# --------------------------------------------------------------------------- LLM providers


class ProviderError(Exception):
    def __init__(self, msg: str, fatal: bool = False):
        super().__init__(msg)
        self.fatal = fatal  # do not use this provider again in this run


Transport = Callable[[str, dict, dict, float], tuple[int, dict, str]]


def requests_transport(url: str, headers: dict, body: dict, timeout: float) -> tuple[int, dict, str]:
    import requests

    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    return r.status_code, dict(r.headers), r.text


@dataclass
class ChatProvider:
    """An OpenAI-compatible chat completions endpoint (GitHub Models is one)."""

    name: str
    base_url: str
    api_key: str = field(repr=False)
    model: str
    transport: Transport = requests_transport
    timeout: float = 45.0
    retries: int = 3
    json_mode: bool = True
    sleep: Callable[[float], None] = time.sleep
    min_interval: float = 0.0  # seconds between requests (rate limits)
    _last: float = 0.0

    def complete(self, messages: list[dict], max_tokens: int = 1500) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.name == "github":
            headers["Accept"] = "application/vnd.github+json"
            headers["X-GitHub-Api-Version"] = "2022-11-28"
        last = "no attempt"
        for attempt in range(self.retries):
            body: dict = {"model": self.model, "messages": messages, "temperature": 0.1, "max_tokens": max_tokens}
            if self.json_mode:
                body["response_format"] = {"type": "json_object"}
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.time()
            try:
                status, rh, text = self.transport(url, headers, body, self.timeout)
            except Exception as e:  # timeouts, connection errors
                last = type(e).__name__
                self.sleep(2.0 * 2**attempt)
                continue
            if status == 200:
                try:
                    return _content(text)
                except (ValueError, KeyError, IndexError, TypeError) as e:
                    ctype = next((v for k, v in (rh or {}).items() if k.lower() == "content-type"), "?")
                    # the body of a model endpoint, never our request: safe to show its start
                    head = " ".join(text[:100].split())
                    raise ProviderError(
                        f"unexpected response ({type(e).__name__}, {ctype}, {len(text)} bytes: {head!r})"
                    ) from None
            last = f"HTTP {status}: {_error_text(text)}"
            if status in (401, 403, 404):
                raise ProviderError(last, fatal=True)
            if status == 400 and self.json_mode and "response_format" in text:
                self.json_mode = False  # model without JSON mode: the prompt asks for JSON anyway
                continue
            if status == 429 or status >= 500:
                retry_after = _retry_after(rh)
                if status == 429 and retry_after > 90:  # daily quota: give up for this run
                    raise ProviderError(f"{last} (retry after {retry_after:.0f} s)", fatal=True)
                self.sleep(retry_after or 3.0 * 2**attempt)
                continue
            raise ProviderError(last)
        raise ProviderError(f"gave up after {self.retries} attempts ({last})")


def _content(text: str) -> str:
    """The message of a chat completion; also accepts a server-sent event stream of chunks."""
    text = text.lstrip("\ufeff").strip()
    if text.startswith("data:"):
        parts = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                ch = json.loads(line[5:])["choices"]
                if ch:
                    parts.append((ch[0].get("delta") or ch[0].get("message") or {}).get("content") or "")
        return "".join(parts)
    return json.loads(text)["choices"][0]["message"]["content"] or ""


def _retry_after(headers: dict) -> float:
    for k, v in (headers or {}).items():
        if k.lower() in ("retry-after", "x-ratelimit-reset-after"):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _error_text(text: str) -> str:
    try:
        err = json.loads(text).get("error")
        msg = err.get("message") if isinstance(err, dict) else err
    except (ValueError, AttributeError):
        msg = text
    return " ".join(str(msg or "").split())[:160]


def providers_from_env(transport: Transport | None = None) -> list[ChatProvider]:
    """Provider chain from env: EVENTS_LLM_PROVIDERS (default "github,openai"; "none" = extractive only).

    * github: GitHub Models with GITHUB_TOKEN (Actions: `permissions: models: read`), model
      EVENTS_GITHUB_MODEL (default openai/gpt-4.1-mini);
    * openai: any OpenAI-compatible endpoint, EVENTS_LLM_BASE_URL + EVENTS_LLM_API_KEY + EVENTS_LLM_MODEL."""
    out: list[ChatProvider] = []
    kw = {"transport": transport} if transport else {}
    for name in _env("EVENTS_LLM_PROVIDERS", "github,openai").split(","):
        name = name.strip().lower()
        if name == "github" and os.environ.get("GITHUB_TOKEN"):
            out.append(
                ChatProvider(
                    "github",
                    _env("EVENTS_GITHUB_MODELS_URL", GITHUB_MODELS_URL),
                    os.environ["GITHUB_TOKEN"],
                    _env("EVENTS_GITHUB_MODEL", "openai/gpt-4.1-mini"),
                    min_interval=4.5,  # free tier: 15 requests per minute
                    **kw,
                )
            )
        elif name == "openai" and os.environ.get("EVENTS_LLM_BASE_URL") and os.environ.get("EVENTS_LLM_API_KEY"):
            out.append(
                ChatProvider(
                    "openai",
                    os.environ["EVENTS_LLM_BASE_URL"],
                    os.environ["EVENTS_LLM_API_KEY"],
                    _env("EVENTS_LLM_MODEL", "gpt-4o-mini"),
                    **kw,
                )
            )
    return out


SYSTEM_PROMPT = """Du erklärst auffällige Ausschläge in einer Statistik: wie oft Personen, Parteien, \
Organisationen und Produkte in Leserkommentaren deutschsprachiger Nachrichtenseiten an einem Tag \
erwähnt werden und wie positiv oder negativ. Zu jedem Fall bekommst du die Entität(en), den Tag, \
die Art des Ausschlags (spike = viel mehr Erwähnungen als üblich, rise = Stimmung deutlich \
positiver, fall = deutlich negativer) und nummerierte Schlagzeilen der an diesem Tag \
meistkommentierten Artikel zu dieser Entität.

Regeln:
- Wähle die Schlagzeile, die den Ausschlag erklärt, und gib ihren Index an. Betrifft keine \
Schlagzeile die Entität erkennbar, gib "headline": null.
- Verwende ausschließlich Fakten aus den Schlagzeilen, kein eigenes Wissen, keine Vermutungen.
- title: Deutsch, höchstens 8 Wörter, sachlich und neutral, ohne Wertung, ohne Datum, ohne \
Anführungszeichen, z. B. "Release von GTA 6".
- description: genau ein neutraler deutscher Satz, höchstens 200 Zeichen, ohne Wertung, ohne URL; \
bei rise/fall beschreibe nur das Ereignis, nicht die Stimmung der Kommentare.
- Antworte nur mit JSON: {"results": [{"id": "...", "headline": 0, "title": "...", "description": "..."}]}"""


def build_request(groups: list[Group], names: dict[str, str]) -> list[dict]:
    """Chat messages for a batch of groups. Contains entity names, numbers and headlines only."""
    cases = []
    for i, g in enumerate(groups):
        lead = g.lead
        cases.append(
            {
                "id": f"c{i}",
                "entities": [names.get(c.entity_id) or c.name for c in g.members],
                "day": g.day,
                "kind": g.kind,
                "mentions": lead.mentions,
                "mentionsPerDayBefore": round(lead.mentions_before, 1),
                "headlines": [{"i": k, "title": h.title, "source": h.source_name} for k, h in enumerate(g.headlines)],
            }
        )
    user = json.dumps({"cases": cases}, ensure_ascii=False)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_results(text: str) -> dict[str, dict]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ProviderError("no JSON in the answer")
    try:
        data = json.loads(m.group(0))
    except ValueError:
        raise ProviderError("invalid JSON in the answer") from None
    items = data.get("results") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ProviderError("answer without results")
    return {str(x.get("id")): x for x in items if isinstance(x, dict)}


def explain(groups: list[Group], providers: list[ChatProvider], names: dict[str, list[str]], cfg: Config) -> dict:
    """Fill `g.explanation` ({title, description, headline, model}) for each group; returns counts."""
    stats = {"llmRequests": 0, "llmErrors": 0, "llm": 0, "declined": 0, "invalid": 0, "extractive": 0, "none": 0}
    deadline = time.time() + cfg.llm_seconds
    alive = list(providers)
    failures: dict[str, int] = {}  # consecutive failed requests per provider
    todo = [g for g in groups if g.explanation is None]
    for i in range(0, len(todo), max(1, cfg.llm_batch)):
        batch = todo[i : i + cfg.llm_batch]
        answers: dict[str, dict] = {}
        used = None
        for p in list(alive):
            if time.time() > deadline:
                log.warning("events: LLM time budget used up, the rest is extractive")
                break
            stats["llmRequests"] += 1
            try:
                text = p.complete(build_request(batch, {k: v[0] for k, v in names.items() if v}))
                answers, used = parse_results(text), p
                failures[p.name] = 0
                break
            except ProviderError as e:
                stats["llmErrors"] += 1
                failures[p.name] = failures.get(p.name, 0) + 1
                log.warning("events: LLM provider %s failed: %s", p.name, e)
                if e.fatal or failures[p.name] >= 2:  # do not ask a broken provider for every batch
                    alive.remove(p)
        for k, g in enumerate(batch):
            lead_names = [n for c in g.members for n in names.get(c.entity_id, [c.name])]
            res = None
            if used is not None:
                item = answers.get(f"c{k}")
                res = validate(item, [h.title for h in g.headlines], lead_names) if item is not None else None
                if res == "decline":
                    stats["declined"] += 1
                elif res is None:
                    stats["invalid"] += 1
                else:
                    idx, title, desc = res
                    g.explanation = {"title": title, "description": desc, "headline": idx, "model": used.model}
                    g.status = "llm"
                    stats["llm"] += 1
                    continue
            # extractive fallback: spikes only, and only from a headline that names the entity; a
            # sentiment shift is never explained without a model, nor when the model declined it
            if g.kind == "spike":
                ex = extractive(g, lead_names)
                if ex is not None:
                    g.explanation, g.status = ex, EXTRACTIVE
                    stats["extractive"] += 1
                    continue
            g.status = "none"
            stats["none"] += 1
    return stats


def _de_day(day: str) -> str:
    d = date.fromisoformat(day)
    return f"{d.day}. {MONTHS[d.month - 1]}"


def shorten(text: str, n: int) -> str:
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    cut = text[: n - 1].rsplit(" ", 1)[0].rstrip(" ,;:-–")
    return cut + "…"


def extractive(g: Group, names: list[str]) -> dict | None:
    for idx, h in enumerate(g.headlines):
        if not mentions_entity(h.title, names):
            continue
        lead = f"Viele Erwähnungen am {_de_day(g.day)} rund um: „"
        tail = f"“ ({h.source_name})."
        quote = shorten(h.title, DESCRIPTION_MAX - len(lead) - len(tail))
        return {
            "title": shorten(h.title, TITLE_MAX),
            "description": lead + quote + tail,
            "headline": idx,
            "model": EXTRACTIVE,
        }
    return None


# --------------------------------------------------------------------------- writing


def assign_ids(groups: list[Group], existing: dict[str, dict]) -> list[str]:
    """Event id per group: an auto event of that day that already covers one of its entities
    keeps its id (re-runs update, never duplicate); others it fully replaces are returned for deletion."""
    delete: list[str] = []
    by_day: dict[str, list[tuple[str, dict]]] = {}
    for eid, ev in sorted(existing.items()):
        if is_auto(ev):
            by_day.setdefault(ev.get("day"), []).append((eid, ev))
    claimed: set[str] = set()
    for g in groups:
        own = f"auto-{g.day}-{g.lead.entity_id}"
        overlap = [(i, ev) for i, ev in by_day.get(g.day, []) if set(ev.get("entityIds") or []) & set(g.entity_ids)]
        overlap = [x for x in overlap if x[0] not in claimed]
        overlap.sort(key=lambda x: (x[0] != own, x[0]))
        if overlap:
            g.event_id, g.existing = overlap[0]
            for i, ev in overlap[1:]:
                if set(ev.get("entityIds") or []) <= set(g.entity_ids):
                    delete.append(i)
                    claimed.add(i)
        else:
            g.event_id = own
        claimed.add(g.event_id)
    return delete


def reuse_existing(g: Group) -> bool:
    """A model-written explanation of this event whose headline is still among the evidence is kept
    (no new LLM request on every run)."""
    ev = g.existing or {}
    gen = ev.get("generator") or {}
    if not gen.get("model") or gen.get("model") == EXTRACTIVE or gen.get("version") != GENERATOR_VERSION:
        return False
    for idx, h in enumerate(g.headlines):
        if h.url == ev.get("url"):
            g.explanation = {
                "title": ev.get("title"),
                "description": ev.get("description"),
                "headline": idx,
                "model": gen["model"],
            }
            g.status = "kept"
            return True
    return False


def event_doc(g: Group, ts: datetime) -> dict:
    ex = g.explanation or {}
    h = g.headlines[ex["headline"]]
    old = g.existing or {}
    doc = {
        "day": g.day,
        "title": ex["title"][:TITLE_MAX],
        "description": ex["description"][:DESCRIPTION_MAX],
        "entityIds": g.entity_ids,
        "url": h.url,
        "source": "auto",
        "kind": g.kind,
        "sourceName": h.source_name,
        "stats": {c.entity_id: c.stats() for c in g.members},
        "generator": {"model": ex["model"], "version": GENERATOR_VERSION},
        "createdAt": old.get("createdAt") if isinstance(old.get("createdAt"), datetime) else ts,
        "updatedAt": ts,
    }
    if ex["model"] == EXTRACTIVE and (old.get("generator") or {}).get("model") not in (None, EXTRACTIVE):
        # never replace a model-written text with an extractive one (e.g. the LLM was unavailable)
        for k in ("title", "description", "url", "sourceName", "generator"):
            doc[k] = old.get(k)
    return doc


def _same(a: dict | None, b: dict) -> bool:
    if a is None:
        return False
    skip = ("updatedAt", "createdAt")
    return {k: v for k, v in a.items() if k not in skip} == {k: v for k, v in b.items() if k not in skip}


# --------------------------------------------------------------------------- orchestration


def process(
    fs,
    entity_docs: dict[str, dict],
    since: str,
    until: str,
    *,
    cfg: Config | None = None,
    budget=None,
    dry_run: bool = False,
    providers: list[ChatProvider] | None = None,
    older_days: dict[str, dict[str, dict]] | None = None,
    source_names: dict[str, str | None] | None = None,
    max_events: int | None = None,
) -> dict:
    cfg = cfg or Config.from_env()
    providers = providers_from_env() if providers is None else providers
    r0, w0 = fs.reads, fs.writes
    days = days_between(since, until)
    cands = detect(entity_docs, days, cfg, older_days)
    out: dict = {"days": f"{since}..{until}", "candidates": len(cands), "providers": [p.name for p in providers]}
    if not cands:
        out.update(events=0, reads=fs.reads - r0, writes=fs.writes - w0)
        return out
    existing = load_events(fs, since, until, cfg.gap_days)
    limit = max_events if max_events is not None else cfg.max_per_run
    chosen, skipped = select(cands, existing, cfg, limit)
    out["skipped"] = skipped
    est_reads = len(chosen) * (cfg.headlines + 5)
    if budget is not None and not budget.allows(writes=len(chosen) + 2, reads=est_reads):
        log.warning("events: daily Firestore budget too low for %d candidates, skipped", len(chosen))
        out.update(events=0, stopped="budget", reads=fs.reads - r0, writes=fs.writes - w0)
        return out
    evidence = load_evidence(fs, chosen, cfg.headlines, source_names if source_names is not None else {})
    groups = group(chosen, evidence)
    out["withoutHeadlines"] = sum(1 for c in chosen if not evidence.get((c.entity_id, c.day)))
    deletes = assign_ids(groups, existing)
    names = {c.entity_id: entity_names(c, entity_docs) for g in groups for c in g.members}
    kept = sum(reuse_existing(g) for g in groups)
    out["llm"] = explain(groups, providers, names, cfg)
    out["llm"]["kept"] = kept
    ts = datetime.now(UTC)
    writes: list[Write] = []
    events: list[dict] = []
    for g in sorted(groups, key=lambda g: (g.day, -g.lead.strength)):
        if g.explanation is None:
            continue
        doc = event_doc(g, ts)
        events.append(
            {
                "id": g.event_id,
                "day": g.day,
                "entityIds": g.entity_ids,
                "kind": g.kind,
                "title": doc["title"],
                "model": doc["generator"]["model"],
                "headlines": len(g.headlines),
                "mentions": g.lead.mentions,
                "before": round(g.lead.mentions_before, 1),
            }
        )
        if not _same(g.existing, doc):
            writes.append(Write(f"events/{g.event_id}", doc))
    writes += [Write(f"events/{i}", None) for i in deletes]
    for e in events:
        log.info(
            "event %s %s %s (%d vs %.1f/day, %d headlines, %s): %s",
            e["day"],
            ",".join(e["entityIds"]),
            e["kind"],
            e["mentions"],
            e["before"],
            e["headlines"],
            e["model"],
            e["title"],
        )
    if dry_run:
        out["wouldWrite"] = len(writes)
    elif writes:
        if budget is not None and not budget.allows(writes=len(writes)):
            log.warning("events: daily Firestore budget too low for %d writes, skipped", len(writes))
            out["stopped"] = "budget"
        else:
            fs.commit(writes)
    out.update(
        events=len(events),
        written=0 if dry_run or out.get("stopped") else len(writes),
        list=events,
        reads=fs.reads - r0,
        writes=fs.writes - w0,
    )
    return out


def run_events(fs, entity_docs: dict[str, dict] | None, budget=None) -> dict:
    """The events step of `run`: yesterday and today (Berlin). Never raises."""
    if not enabled():
        return {"skipped": "disabled"}
    if not entity_docs:
        return {"skipped": "no entity docs in memory"}
    try:
        if timebudget.get().reached("events", reserve=60 + Config.from_env().llm_seconds):
            return {"skipped": "time budget"}
        today = today_berlin()
        out = process(fs, entity_docs, _shift(today, -1), today, budget=budget)
        out.pop("list", None)
        log.info("events: %s", out)
        return out
    except Exception as e:  # never fail the analysis run because of events
        log.warning("events failed: %s", type(e).__name__ + ": " + str(e)[:200])
        return {"error": type(e).__name__}


# --------------------------------------------------------------------------- backfill


def load_older_days(fs, entity_docs: dict[str, dict], frm: str, to: str, cfg: Config, budget=None) -> dict:
    """Day counts in [frm, to) (before `recentDays`) for the entities that can have a candidate,
    via one query per entity on its `days` subcollection. Reads ~ existing day docs."""
    need = []
    for eid, d in entity_docs.items():
        if d.get("allowlisted") is False:
            continue
        recent = d.get("recentDays") or {}
        older = int(d.get("mentions") or 0) - sum(int(v.get("mentions") or 0) for v in recent.values())
        if older >= min(cfg.spike_min, cfg.shift_min) or (
            older > 0 and any(int(v.get("mentions") or 0) >= min(cfg.spike_min, cfg.shift_min) for v in recent.values())
        ):
            need.append(eid)
    n_days = len(days_between(frm, _shift(to, -1))) if frm < to else 0
    est = len(need) * max(1, n_days)
    log.info("events backfill: older day docs of %d entities, %d days, at most ~%d reads", len(need), n_days, est)
    if budget is not None and not budget.allows(reads=est):
        raise RuntimeError(f"daily read budget too low for the backfill (~{est} reads)")
    out: dict[str, dict[str, dict]] = {}
    for eid in need:
        for d in fs.query(
            "days",
            parent=f"entities/{eid}",
            where=[("day", "GREATER_THAN_OR_EQUAL", frm), ("day", "LESS_THAN", to)],
            order_by=[("day", "ASCENDING")],
            select=["mentions", "scoreSum"],
        ):
            out.setdefault(eid, {})[d.id] = d.data
    return out


def backfill(fs, since: str, until: str | None = None, dry_run: bool = False, providers=None) -> dict:
    """`python -m analysis events --since ...`: detect and write events for a range of days."""
    from .budget import Budget
    from .registry_store import load_active_docs

    cfg = Config.from_env()
    until = until or today_berlin()
    budget = Budget.load(fs)
    why = budget.exhausted()
    if why:
        log.warning("daily Firestore budget: %s - events not started", why)
        return {"skipped": why, "budget": budget.summary()}
    try:
        entity_docs, _ = load_active_docs(fs, ENTITY_FIELDS)
        recent_from = _shift(today_berlin(), -29)
        frm = max(cfg.data_since, _shift(since, -cfg.window))
        older = load_older_days(fs, entity_docs, frm, recent_from, cfg, budget) if frm < recent_from else {}
        n_days = len(days_between(since, until))
        if dry_run:
            log.info("events: threshold grid (candidates before caps): %s", grid(entity_docs, since, until, cfg, older))
        out = process(
            fs,
            entity_docs,
            since,
            until,
            cfg=cfg,
            budget=budget,
            dry_run=dry_run,
            providers=providers,
            older_days=older,
            max_events=cfg.max_per_day * n_days,
        )
        out["entities"] = len(entity_docs)
    finally:
        budget.record()
    out["budget"] = budget.summary()
    return out


def grid(entity_docs: dict, since: str, until: str, cfg: Config, older: dict) -> dict:
    """Candidates per variant of the thresholds (dry runs, for tuning)."""
    from dataclasses import replace

    days = days_between(since, until)
    out = {}
    for name, kw in (
        ("default", {}),
        ("spike x4", {"spike_factor": 4.0}),
        ("spike min 40", {"spike_min": 40}),
        ("delta 0.20", {"shift_delta": 0.20}),
        ("delta 0.30", {"shift_delta": 0.30}),
        ("shift min 30", {"shift_min": 30}),
    ):
        cs = detect(entity_docs, days, replace(cfg, **kw), older)
        kinds = {k: sum(1 for c in cs if c.kind == k) for k in KINDS}
        out[name] = {"total": len(cs), "perDay": round(len(cs) / max(1, len(days)), 1), **kinds}
    return out
