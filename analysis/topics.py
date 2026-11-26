"""Worüber gesprochen wird: the words that dominate the comments about an entity (pure, no I/O).

1. Extraction (during the analysis, while the comment text is in memory): for every mention the
   nouns of the window the sentiment model reads (the mention's sentence +- one sentence). A cheap
   German noun heuristic, no tagger: capitalised words that do not start a sentence, >= 4 letters,
   letters and hyphens only (no digits, URLs, handles; hashtags arrive as words, see clean_text),
   not a stopword, not inside a linked entity or an NER person span, not an alias of a known entity and
   not a first name of a known person, not right after a form of address or a first name
   ("Herr Müller", "Klaus Meier") (entities are covered by coMentions; this also drops the names
   of private persons). At most MAX_TERMS_PER_MENTION per mention, stored on the mention as
   `terms` (single words, never text), so day docs stay recomputable from the mentions.
2. Day docs (`build_day_docs`): `terms` = {term: [count, scoreSum, sourceId, ...]} for the
   TERMS_PER_DAY most frequent terms of the (entity, day), with up to MAX_TERM_SOURCES sources.
3. Entity docs: `recentTerms` = the day doc terms of the last TERMS_DAYS days (internal, so the
   rolling window needs no reads), `topics7d` = the published topics of the last 7 days:
   a term needs >= MIN_MENTIONS mentions from >= MIN_SOURCES sources (privacy: no single
   commenter's catchphrase) and a lift >= MIN_LIFT against all mentions of all entities in the
   same 7 days; ranked by mentions x ln(lift).
"""

from __future__ import annotations

import math
import re
from datetime import date, timedelta

from .textutil import normalize_surface

MAX_TERMS_PER_MENTION = 6
MIN_TERM_LETTERS = 4
MAX_TERM_CHARS = 40
TERMS_PER_DAY = 40  # terms kept per day doc (by count)
MAX_TERM_SOURCES = 3  # sources kept per term and day; enough to decide ">= MIN_SOURCES distinct" over days
TERMS_DAYS = 14  # recentTerms: the 7-day window and the 7 days before (for `prev`)
TOPICS_TOP = 10
MIN_MENTIONS = 5
MIN_SOURCES = 2
MIN_LIFT = 1.5

# a word: letters and inner hyphens ("E-Auto", "Ampel-Aus"); digits make the whole run invalid
_RUN = re.compile(r"[\w-]+")
_WORD = re.compile(r"[A-ZÄÖÜ][a-zäöüßA-ZÄÖÜ]*(?:-[A-Za-zÄÖÜäöüß]+)*")
_LOWER = re.compile(r"[a-zäöüß]")
_OPENERS = " \t\n\"'„“”»«‚‘’()[]-–—*"
_ENDS = ".!?:;…"

# capitalised words that say nothing about a topic: formal address, greetings, fillers, time
# words, comment-section vocabulary, idioms and insults (compared lower-cased)
_STOPWORDS = """
    ihre ihren ihrem ihrer ihres ihnen euer eure euch deren dessen
    herr herrn herren frau damen leute mensch menschen person personen kinder
    hallo moin servus danke bitte gruß grüße dank prost sorry
    gott himmel teufel hölle
    jahr jahre jahren jahres jahrzehnt jahrzehnte zeit zeiten tag tage tagen tages woche wochen
    monat monate monaten stunde stunden minute minuten sekunden morgen abend nacht heute
    montag dienstag mittwoch donnerstag freitag samstag sonntag wochenende
    januar februar märz april juni juli august september oktober november dezember
    artikel kommentar kommentare kommentaren kommentator kommentatoren beitrag beiträge forum
    foristen forist user leser autor autorin redaktion text überschrift link quelle quellen
    frage fragen antwort antworten problem probleme sache sachen ding dinge fall falle fälle
    grund gründe grunde ende anfang teil teile seite seiten weg wege art weise punkt punkte
    thema themen lage situation beispiel prinzip zusammenhang unterschied ergebnis folge folgen
    fakt fakten recht unrecht ahnung meinung meinungen ansicht eindruck gefühl idee hinsicht
    bezug rahmen sinne stelle stellen mal male zahl zahlen prozent millionen milliarden
    land länder landes welt deutschland leben hand hände augen auge kopf mund nase
    bild bilder hinblick blick wort worte wörter satz sätze geschichte
    quatsch unsinn blödsinn schwachsinn mist käse murks lol haha
    katastrophe witz schande lüge lügen versager idiot idioten frechheit unfug dummheit
"""
STOPWORDS = frozenset(_STOPWORDS.split())


def _sentence_initial(text: str, s0: int, start: int) -> bool:
    """The word at `start` is the first word of its sentence (only quotes, dashes or brackets
    before it), or follows a colon, semicolon or another sentence end inside the sentence span."""
    before = text[s0:start].rstrip(_OPENERS)
    return not before or before[-1] in _ENDS


# a word right after these is a name ("Herr Müller", "Frau Dr. Meier"), whoever it is
_ADDRESS = re.compile(r"(?:\b(?:Herrn?|Frau|Dr|Prof|Hr|Fr)\.?)\s+$")


def _after_name(text: str, start: int, first_names: set[str]) -> bool:
    """The word follows a form of address or a first name of a known person ("Klaus Meier")."""
    before = text[max(0, start - 30) : start]
    if _ADDRESS.search(before):
        return True
    prev = re.search(r"([A-ZÄÖÜ][a-zäöüß]+)\s+$", before)
    return bool(prev and prev.group(1).casefold() in first_names)


def _candidate(word: str) -> bool:
    letters = word.replace("-", "")
    if len(letters) < MIN_TERM_LETTERS or len(word) > MAX_TERM_CHARS or not letters.isalpha():
        return False
    if not _WORD.fullmatch(word):
        return False
    last = word.rsplit("-", 1)[-1]
    if not _LOWER.search(last[1:]):  # all-caps word or acronym ("SPD", "KI-GESETZ", "NATO")
        return False
    return word.casefold() not in STOPWORDS


_WORD_ALIAS_MODES = {"ctx", "cue", "local-ctx", "local-cue"}


class NameCheck:
    """Words that are names: aliases of known entities and first names of known persons.

    Contextual and cue-only aliases are words unless their context links them ("Bayern" the
    state, "Sinn", "Nordsee"); where they are linked, the match span excludes them anyway."""

    def __init__(self, registry):
        self.aliases = {key for (key, _eid), modes in registry.alias_modes.items() if modes - _WORD_ALIAS_MODES}
        self.first_names = {
            e.name.split()[0].casefold()
            for e in registry.entities.values()
            if e.type == "PER" and len(e.name.split()) >= 2
        }

    def __call__(self, word: str) -> bool:
        if normalize_surface(word) in self.aliases or word.casefold() in self.first_names:
            return True
        # inflected alias ("Trumps", "Grünen"): the linker accepts these suffixes, so do we
        for suf in ("s", "es", "n", "en", "'s"):
            if word.endswith(suf) and normalize_surface(word[: -len(suf)]) in self.aliases:
                return True
        return any(normalize_surface(p) in self.aliases for p in word.split("-") if len(p) >= 3)


_name_checks: dict[int, tuple[int, NameCheck]] = {}


def name_check(registry) -> NameCheck:
    """One NameCheck per registry (rebuilt when entities were added during the run)."""
    key = id(registry)
    cached = _name_checks.get(key)
    if cached is None or cached[0] != len(registry.entities):
        cached = _name_checks[key] = (len(registry.entities), NameCheck(registry))
    return cached[1]


def mention_terms(
    text: str,
    sents: list[tuple[int, int]],
    targets: list[tuple[int, int]],
    exclude: list[tuple[int, int]],
    is_name,
    limit: int = MAX_TERMS_PER_MENTION,
) -> list[str]:
    """Topic words of one mention: nouns of the sentence of each target span +- one sentence.

    `targets`: (start, end) of the entity's own matches; `exclude`: spans of all linked entities
    and NER spans (no word overlapping them is a term); `is_name(word)`: aliases and first names.
    The words nearest to the mention win when there are more than `limit`; returned sorted."""
    if not sents or not targets:
        return []
    idx: set[int] = set()
    for a, _b in targets:
        k = next((j for j, (s0, s1) in enumerate(sents) if s0 <= a < s1), None)
        if k is not None:
            idx |= {j for j in (k - 1, k, k + 1) if 0 <= j < len(sents)}
    found: dict[str, int] = {}  # word -> distance to the nearest target
    for j in sorted(idx):
        s0, s1 = sents[j]
        for m in _RUN.finditer(text, s0, s1):
            word = m.group(0).strip("-")
            start = m.start() + (len(m.group(0)) - len(m.group(0).lstrip("-")))
            end = start + len(word)
            if not word or not _candidate(word) or _sentence_initial(text, s0, start):
                continue
            if any(start < b and a < end for a, b in exclude) or is_name(word):
                continue
            if _after_name(text, start, getattr(is_name, "first_names", set())):
                continue
            dist = min(abs(start - a) for a, _b in targets)
            if dist < found.get(word, 1 << 30):
                found[word] = dist
    # light inflection: "Renten" next to "Rente" in the same window is one term (the shorter)
    for w in list(found):
        for cut in (1, 2):
            base = w[:-cut]
            if w[-cut:] in ("s", "n", "es", "en") and base in found and base != w:
                found[base] = min(found[base], found.pop(w))
                break
    best = sorted(found, key=lambda w: (found[w], w))[:limit]
    return sorted(best)


# --------------------------------------------------------------------------- day docs


def day_terms(mentions: list[dict]) -> dict[str, list]:
    """{term: [count, scoreSum, sourceId, ...]} of one (entity, day): the TERMS_PER_DAY most
    frequent terms, each with up to MAX_TERM_SOURCES distinct sources (sorted)."""
    acc: dict[str, list] = {}
    for m in mentions:
        for t in dict.fromkeys(m.get("terms") or []):
            a = acc.setdefault(t, [0, 0.0, set()])
            a[0] += 1
            a[1] += float(m.get("score") or 0.0)
            a[2].add(m.get("sourceId") or "unknown")
    top = sorted(acc.items(), key=lambda kv: (-kv[1][0], kv[0]))[:TERMS_PER_DAY]
    return {t: [n, round(s, 4), *sorted(src)[:MAX_TERM_SOURCES]] for t, (n, s, src) in top}


def prune_terms(recent: dict[str, dict], today: str) -> dict[str, dict]:
    cutoff = (date.fromisoformat(today) - timedelta(days=TERMS_DAYS - 1)).isoformat()
    return {d: v for d, v in sorted(recent.items()) if cutoff <= d <= today and v}


# --------------------------------------------------------------------------- topics7d


def _windows(today: str) -> tuple[str, str, str]:
    t = date.fromisoformat(today)
    return (
        (t - timedelta(days=6)).isoformat(),
        (t - timedelta(days=13)).isoformat(),
        (t - timedelta(days=7)).isoformat(),
    )


def _variants(term: str) -> list[str]:
    """Shorter forms a term may be an inflection of: genitive -s/-es, plural -n/-en."""
    out = []
    for suf in ("es", "en", "s", "n"):
        if term.endswith(suf) and len(term) - len(suf) >= MIN_TERM_LETTERS:
            out.append(term[: -len(suf)])
    return out


def global_terms(entities: dict[str, dict], today: str) -> tuple[dict[str, int], int]:
    """Mentions per term over all allowlisted entities in the 7-day window, and all their mentions."""
    cur_from, _p0, _p1 = _windows(today)
    counts: dict[str, int] = {}
    total = 0
    for e in entities.values():
        if not e.get("allowlisted"):
            continue
        for d, v in (e.get("recentDays") or {}).items():
            if cur_from <= d <= today:
                total += int(v.get("mentions") or 0)
        for d, terms in (e.get("recentTerms") or {}).items():
            if cur_from <= d <= today:
                for t, row in terms.items():
                    counts[t] = counts.get(t, 0) + int(row[0])
    return counts, total


def canonical_map(counts: dict[str, int]) -> dict[str, str]:
    """term -> display form: inflected variants ("Renten", "Krieges") join their base form when it
    occurs too; the group is shown in its most frequent spelling."""
    parent = {t: t for t in counts}

    def root(t: str) -> str:
        while parent[t] != t:
            t = parent[t]
        return t

    for t in sorted(counts, key=len):
        for base in _variants(t):
            if base in counts:
                parent[root(t)] = root(base)
                break
    groups: dict[str, list[str]] = {}
    for t in counts:
        groups.setdefault(root(t), []).append(t)
    out = {}
    for members in groups.values():
        shown = max(members, key=lambda t: (counts[t], -len(t), t))
        for t in members:
            out[t] = shown
    return out


def topics(
    entity: dict, global_counts: dict[str, int], global_total: int, today: str, canon: dict[str, str] | None = None
) -> list[dict]:
    """topics7d of one entity: [{term, mentions, share, avgSentiment, lift, prev}], at most TOPICS_TOP.

    Published only with >= MIN_MENTIONS mentions from >= MIN_SOURCES distinct sources in the
    7 days and lift >= MIN_LIFT; ranked by mentions x ln(lift)."""
    cur_from, prev_from, prev_to = _windows(today)
    m7 = sum(int(v.get("mentions") or 0) for d, v in (entity.get("recentDays") or {}).items() if cur_from <= d <= today)
    if m7 < MIN_MENTIONS or not global_total:
        return []
    canon = canon if canon is not None else canonical_map(global_counts)
    cur: dict[str, list] = {}
    prev: dict[str, int] = {}
    gcount: dict[str, int] = {}
    for t, n in global_counts.items():
        k = canon.get(t, t)
        gcount[k] = gcount.get(k, 0) + n
    for d, terms in (entity.get("recentTerms") or {}).items():
        for t, row in terms.items():
            k = canon.get(t, t)
            if cur_from <= d <= today:
                a = cur.setdefault(k, [0, 0.0, set()])
                a[0] += int(row[0])
                a[1] += float(row[1])
                a[2].update(row[2:])
            elif prev_from <= d <= prev_to:
                prev[k] = prev.get(k, 0) + int(row[0])
    out = []
    for k, (n, s, src) in cur.items():
        n = min(n, m7)
        if n < MIN_MENTIONS or len(src) < MIN_SOURCES or not gcount.get(k):
            continue
        lift = (n / m7) / (max(gcount[k], n) / global_total)
        if lift < MIN_LIFT:
            continue
        out.append(
            {
                "term": k,
                "mentions": n,
                "share": round(n / m7, 3),
                "avgSentiment": round(s / n, 3),
                "lift": round(lift, 1),
                "prev": int(prev.get(k, 0)),
                "_rank": n * math.log(lift),
            }
        )
    out.sort(key=lambda x: (-x["_rank"], x["term"]))
    return [{k: v for k, v in x.items() if k != "_rank"} for x in out[:TOPICS_TOP]]
