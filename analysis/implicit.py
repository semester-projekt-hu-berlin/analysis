"""Implicit mentions: a comment that refers to an entity without naming it.

Under the article "Merz kündigt Rücktritt an" the comment "Endlich tritt er zurück" names no
entity, so the gazetteer finds nothing. These rules use the comment's context instead:

* A (`via: "headline"` / `"article"`): a comment under a real article (not a social pseudo-article)
  with no linked entity, a referring expression ("er", "der Kanzler", "die Partei", "die App", a
  role word from the entity's description), and exactly one compatible target in the headline
  (or, failing that, one dominant entity among the article's other comments).
* B (`via: "parent"`): a reply with no linked entity whose parent comment names exactly one
  compatible entity.
* C (`via: "role"`): a title that names one person at the time of the comment ("der Kanzler" on a
  German source -> the federal chancellor).

`find()` returns every candidate with its rule and cue kind; `ACCEPTED` holds the (rule, cue)
combinations that passed the precision check in the README ("Context: implicit mentions"), and
only those become mentions. Nothing here keeps text: a candidate carries the entity, the rule,
the kind of cue and the cue's character span.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field

from .linker import Entity, Registry

TYPES = ("PER", "PARTY", "ORG", "PRODUCT")
TITLE_CHARS = 200  # the same cap as the headline experiment of the sentiment model
WINDOW_CHARS = 600
PARENT_CHARS = 300
SHORT_WORDS = 10  # "short top-level reaction"
DOMINANT_MIN = 3  # mentions of the article's dominant entity among its other comments
DOMINANT_SHARE = 0.5

# (rule, cue kind) pairs that become mentions in production. Set from the measured sample: see
# the README section "Context: implicit mentions". Everything else is only counted.
ACCEPTED: frozenset[tuple[str, str]] = frozenset()


def enabled() -> bool:
    return os.environ.get("ANALYSIS_IMPLICIT", "1") != "0"


# --------------------------------------------------------------------------- cues

_DET = r"(?:der|die|das|den|dem|des|dieser|diese|dieses|diesen|diesem|unser|unsere|unseren|unserem|unserer|unseres)"
# no "nächste", "frühere", "ehemalige" ...: those are not the person who holds the title now
_NOW_ADJ = r"(?:(?:jetzige|aktuelle|amtierende|derzeitige|gute|liebe|werte)[nmrs]?)"
_ADJ = r"(?:[a-zäöüß]+(?:e|en|er|es|em)\s+)"
_NOUN = r"([A-ZÄÖÜ][a-zäöüß]{2,}(?:-[A-ZÄÖÜ]?[a-zäöüß]+)*)"
DET_NOUN = re.compile(rf"(?<![\w-])(?i:{_DET}|Herr|Herrn|Frau)\s+{_ADJ}?{_NOUN}(?![\w-])")

# bare "sein" only before a noun ("sein Rücktritt"; "kann sein" is the verb)
PRON_M = re.compile(r"(?<![\w-])(?:[Ee]r|[Ii]hn|[Ii]hm|[Ss]eine[mnrs]?|[Ss]ein(?=\s+[A-ZÄÖÜ]))(?![\w-])")
# feminine singular, often also plural ("sie" = she / they); "Sie" mid-sentence is the polite you
PRON_F = re.compile(r"(?<![\w-])(?:sie|ihre[mnrs]?)(?![\w-])")
PRON_F_START = re.compile(r"(?:^|[.!?]\s+)(Sie|Ihre[mnrs]?)(?![\w-])")
# demonstrative at the start of a sentence: "Der muss weg", "Den kann man nicht ernst nehmen"
# (not "Der neue Kanzler": a lower-case word with an adjective ending before a noun)
_NOT_ADJ_NOUN = r"(?=[a-zäöüß])(?![a-zäöüß]+(?:e|en|er|es|em)\s+[A-ZÄÖÜ])"
DEM_M = re.compile(rf"(?:^|[.!?:]\s+)(Der|Den|Dem|Dessen)\s+{_NOT_ADJ_NOUN}")
DEM_F = re.compile(rf"(?:^|[.!?:]\s+)(Die|Deren)\s+{_NOT_ADJ_NOUN}")

# generic nouns that refer to a person / a party / an organisation of a kind
PERSON_M = {"mann", "typ", "kerl", "herr", "herrn", "politiker", "mensch", "junge", "bursche", "kollege"}
PERSON_F = {"frau", "dame", "politikerin", "kollegin"}
PARTY_NOUNS = {"partei", "truppe", "fraktion", "genossen", "laden", "verein"}
KIND_NOUNS = {
    "club": {"verein", "club", "klub", "mannschaft", "team", "truppe", "elf"},
    "company": {"unternehmen", "konzern", "firma", "laden", "hersteller", "betrieb"},
    "broadcaster": {"sender", "anstalt"},
    "press": {"zeitung", "blatt", "magazin", "verlag", "medium", "käseblatt"},
    "show": {"sendung", "show", "talkshow", "format"},
    "product": {"app", "dienst", "plattform", "programm", "software", "netzwerk", "messenger", "spiel", "seite"},
}
# words after a determiner that never refer to a person or organisation of the headline
STOP_NOUNS = {
    "deutsche",
    "deutschen",
    "leute",
    "menschen",
    "bürger",
    "wähler",
    "artikel",
    "kommentar",
    "kommentare",
    "frage",
    "sache",
    "welt",
    "politik",
    "regierung",
    "staat",
    "land",
    "ende",
    "zeit",
    "jahr",
    "jahre",
    "problem",
    "thema",
    "medien",
    "presse",
}

INFLECTION = ("innen", "ern", "en", "es", "er", "n", "s", "e")


def _stem(w: str) -> str:
    w = w.lower()
    for suf in INFLECTION:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


_FEM_WORD = re.compile(r"(?:erin|entin|antin|istin|eurin|orin|ärin|ffin|änin|ündin|in)$")


def gender(e: Entity) -> str | None:
    """'m' / 'f' from the German description ("deutsche Schauspielerin", "Bundeskanzler"), else None."""
    d = (e.description or "").split("(")[0].strip()
    if not d:
        return None
    words = re.findall(r"[A-Za-zÄÖÜäöüß-]+", d)
    fem = any(len(w) > 5 and _FEM_WORD.search(w) for w in words)
    fem = fem or bool(re.match(r"(?:deutsche|österreichische|schweizer(?:ische)?|US-amerikanische)\s", d))
    fem = fem or bool(re.search(r"(?:abgeordnete|vorsitzende|beauftragte)\b(?!r)", d, re.IGNORECASE))
    masc = bool(re.match(r"(?:deutscher|österreichischer|schweizerischer|US-amerikanischer)\s", d))
    masc = masc or any(
        re.search(r"(?:er|ent|ant|ist|eur|or|är|präsident|kanzler|chef|papst|oberhaupt|ministerpräsident)$", w.lower())
        for w in words[:1]
    )
    masc = masc or bool(re.search(r"(?:abgeordneter|vorsitzender|beauftragter)\b", d, re.IGNORECASE))
    if fem and not masc:
        return "f"
    if masc and not fem:
        return "m"
    return None


def kind(e: Entity) -> set[str]:
    """Kinds of an organisation, for generic nouns ("der Verein", "die App")."""
    d = (e.description or "").lower()
    out = set()
    if e.type == "PRODUCT":
        out.add("product")
    if e.type == "ORG":
        if ("fußball" in d and any(k in d for k in ("verein", "club", "klub", "mannschaft"))) or "sportverein" in d:
            out.add("club")
        if any(k in d for k in ("unternehmen", "konzern", "hersteller", "gesellschaft", "bank", "händler", "kette")):
            out.add("company")
        if any(k in d for k in ("sender", "rundfunk", "fernseh")):
            out.add("broadcaster")
        if any(k in d for k in ("zeitung", "magazin", "verlag", "nachrichten")):
            out.add("press")
        if "sendung" in d or "show" in d:
            out.add("show")
    return out


def description_nouns(e: Entity) -> set[str]:
    """Stems of the description's nouns and their parts: "Bundestagsabgeordneter (CDU)" ->
    {"bundestagsabgeordnet"}; a cue noun matches when one of them ends with its stem."""
    d = (e.description or "").split("(")[0]
    out = set()
    for w in re.findall(r"[A-ZÄÖÜ][\w-]+", d):
        for part in w.split("-"):
            if len(part) >= 3:
                out.add(_stem(part))
    return out


@dataclass
class Cue:
    kind: str  # pron-m | pron-f | dem-m | dem-f | person-m | person-f | role | party | kind | short | title
    start: int
    end: int
    noun: str = ""  # stem of the cue noun


def cues(text: str, top_level: bool) -> list[Cue]:
    """Referring expressions in a comment, in text order."""
    out: list[Cue] = []
    for m in DET_NOUN.finditer(text):
        noun = m.group(1)
        stem = _stem(noun)
        low = noun.lower()
        if low in STOP_NOUNS:
            continue
        if low in PERSON_M or stem in {_stem(x) for x in PERSON_M}:
            out.append(Cue("person-m", m.start(), m.end(), stem))
        elif low in PERSON_F or stem in {_stem(x) for x in PERSON_F}:
            out.append(Cue("person-f", m.start(), m.end(), stem))
        else:
            out.append(Cue("noun", m.start(), m.end(), stem))
    for m in PRON_M.finditer(text):
        out.append(Cue("pron-m", m.start(), m.end()))
    for m in PRON_F.finditer(text):
        out.append(Cue("pron-f", m.start(), m.end()))
    for m in PRON_F_START.finditer(text):
        out.append(Cue("pron-f", m.start(1), m.end(1)))
    for m in DEM_M.finditer(text):
        out.append(Cue("dem-m", m.start(1), m.end(1)))
    for m in DEM_F.finditer(text):
        out.append(Cue("dem-f", m.start(1), m.end(1)))
    out.sort(key=lambda c: c.start)
    if not out and top_level and 0 < len(text.split()) <= SHORT_WORDS:
        out.append(Cue("short", 0, 0))
    return out


def compatible(cue: Cue, e: Entity) -> str | None:
    """The cue kind under which `cue` can refer to entity `e` (None if it cannot)."""
    if e.type not in TYPES:
        return None
    g = gender(e) if e.type == "PER" else None
    k = cue.kind
    if k in ("pron-m", "dem-m", "person-m"):
        return k if e.type == "PER" and g == "m" else None
    if k in ("pron-f", "dem-f", "person-f"):
        if e.type == "PER" and g == "f":
            return k
        if k in ("pron-f", "dem-f") and e.type == "PARTY":
            return k + "-party"  # plural "sie" / "die" for a party
        return None
    if k == "noun":
        if any(d.endswith(cue.noun) for d in description_nouns(e)):
            return "role"
        if e.type == "PARTY" and cue.noun in {_stem(x) for x in PARTY_NOUNS}:
            return "party"
        if any(cue.noun in {_stem(x) for x in KIND_NOUNS[kd]} for kd in kind(e)):
            return "kind"
        return None
    if k == "short":
        return "short"
    return None


# --------------------------------------------------------------------------- titles (rule C)


@dataclass(frozen=True)
class Title:
    pattern: str
    country: str | None  # source country, None = any
    entity_id: str
    since: str  # first day (YYYY-MM-DD)
    until: str = "9999-12-31"


# Titles held by exactly one person; a comment's day must fall into the term. Checked against
# the entity list in tests/test_implicit.py.
TITLES = (
    Title(r"(?:Bundes)?[Kk]anzler", "DE", "friedrich-merz", "2025-05-06"),
    Title(r"(?:Bundes)?[Kk]anzler", "AT", "christian-stocker", "2025-03-03"),
    Title(r"Vizekanzler", "DE", "lars-klingbeil", "2025-05-06"),
    Title(r"Vizekanzler", "AT", "andreas-babler", "2025-03-03"),
    Title(r"Bundespräsident", "DE", "frank-walter-steinmeier", "2017-03-19", "2027-03-18"),
    Title(r"Bundespräsident", "AT", "alexander-van-der-bellen", "2017-01-26", "2029-01-25"),
    Title(r"US-Präsident", None, "donald-trump", "2025-01-20", "2029-01-19"),
)
_TITLE_RX = {
    t: re.compile(
        rf"(?<![\w-])(?i:{_DET}|Herr|Herrn|Frau)\s+(?:{_NOW_ADJ}\s+)?({t.pattern})(?:s|en|n)?(?![\w-])(?!\s+[A-ZÄÖÜ][a-zäöüß]+\b)"
    )
    for t in TITLES
}


def title_mentions(text: str, country: str | None, day: str) -> list[tuple[str, int, int]]:
    """(entity id, start, end) of titles that name one office holder on `day` ("der Kanzler")."""
    out = []
    for t, rx in _TITLE_RX.items():
        if t.country is not None and t.country != country:
            continue
        if not (t.since <= day <= t.until):
            continue
        for m in rx.finditer(text):
            # "Vizekanzler" must not also count as "Kanzler"
            if t.pattern.endswith("[Kk]anzler") and text[m.start(1) - 4 : m.start(1)].lower() == "vize":
                continue
            out.append((t.entity_id, m.start(), m.end()))
    return out


# --------------------------------------------------------------------------- candidates


@dataclass
class Candidate:
    entity_id: str
    rule: str  # A | B | C
    via: str  # headline | article | parent | role
    cue: str  # cue kind (see compatible())
    start: int
    end: int
    n_targets: int = 1  # targets of the context of this kind (headline / parent entities)


@dataclass
class CommentContext:
    """What the rules know about one comment besides its text."""

    title: str = ""  # headline, "" for social pseudo-articles or unknown articles
    title_entities: list[str] = field(default_factory=list)
    article_counts: Counter = field(default_factory=Counter)  # explicit mentions in the article's other comments
    is_reply: bool = False
    parent_entities: list[str] | None = None  # None: no parent known
    country: str | None = None
    day: str = ""


def _targets(ids: list[str], registry: Registry) -> list[str]:
    seen = []
    for i in ids:
        e = registry.entities.get(i)
        if e is not None and e.type in TYPES and i not in seen:
            seen.append(i)
    return seen


def dominant(counts: Counter) -> str | None:
    total = sum(counts.values())
    if not total:
        return None
    (eid, n), *rest = [*counts.most_common(2), (None, 0)]
    if n >= DOMINANT_MIN and n / total >= DOMINANT_SHARE and n > rest[0][1]:
        return eid
    return None


def _pick(cs: list[Cue], targets: list[str], registry: Registry, allow_short: bool):
    """First cue that fits exactly one of the targets: (entity id, cue kind, cue)."""
    for c in cs:
        if c.kind == "short" and not allow_short:
            continue
        fits = [(t, k) for t in targets if (k := compatible(c, registry.entities[t]))]
        if len(fits) == 1:
            return fits[0][0], fits[0][1], c
        if len(fits) > 1:
            return None  # ambiguous: the first referring expression could mean several
    return None


def find(text: str, linked: set[str], ctx: CommentContext, registry: Registry) -> list[Candidate]:
    """All implicit candidates of one cleaned comment text (whose explicit entities are `linked`)."""
    out: list[Candidate] = []
    for eid, a, b in title_mentions(text, ctx.country, ctx.day):
        if eid in registry.entities and eid not in linked and all(c.entity_id != eid for c in out):
            out.append(Candidate(eid, "C", "role", "title", a, b))
    if linked or not ctx.title:
        return out
    cs = cues(text, top_level=not ctx.is_reply)
    if not cs:
        return out
    taken = {c.entity_id for c in out}
    if ctx.is_reply and ctx.parent_entities:
        targets = _targets(ctx.parent_entities, registry)
        got = _pick(cs, targets, registry, allow_short=False)
        if got and got[0] not in taken:
            eid, k, c = got
            out.append(Candidate(eid, "B", "parent", k, c.start, c.end, len(targets)))
            return out
    targets = _targets(ctx.title_entities, registry)
    got = _pick(cs, targets, registry, allow_short=True) if targets else None
    if got and got[0] not in taken:
        eid, k, c = got
        out.append(Candidate(eid, "A", "headline", k, c.start, c.end, len(targets)))
        return out
    if not targets:
        dom = dominant(ctx.article_counts)
        if dom and dom in registry.entities:
            got = _pick(cs, [dom], registry, allow_short=False)
            if got and got[0] not in taken:
                eid, k, c = got
                out.append(Candidate(eid, "A", "article", k, c.start, c.end, 1))
    return out


def may_refer(text: str) -> bool:
    """Whether a reply has a referring expression at all (cheap pre-check before its parent is read)."""
    return bool(cues(text, top_level=False))


def uses(rule: str) -> bool:
    """Whether production accepts candidates of `rule` (else its context is not even fetched)."""
    return any(r == rule for r, _ in ACCEPTED)


def accepted(c: Candidate) -> bool:
    return (c.rule, c.cue) in ACCEPTED or (c.rule, "*") in ACCEPTED


# --------------------------------------------------------------------------- sentiment input


def window(text: str, start: int, end: int) -> str:
    """The comment (capped around the cue) with the referring span in [brackets]."""
    if end > start:
        text = f"{text[:start]}[{text[start:end]}]{text[end:]}"
        end += 2
    if len(text) <= WINDOW_CHARS:
        return text
    a = max(0, min(start - WINDOW_CHARS // 3, len(text) - WINDOW_CHARS))
    return text[a : a + WINDOW_CHARS]


def premise(text: str, title: str, parent: str = "") -> str:
    """Premise for the sentiment model: the headline (and the parent comment of a reply) as context.

    The same "Artikel: … | Kommentar: …" form as the headline experiment of the sentiment model
    (scripts/eval_sentiment_context.py), so a model trained with headlines reads it as it was trained."""
    title = " ".join((title or "").split())[:TITLE_CHARS]
    parts = []
    if title:
        parts.append(f"Artikel: {title}")
    if parent:
        parts.append(f"Antwort auf: {' '.join(parent.split())[:PARENT_CHARS]}")
    parts.append(f"Kommentar: {text}")
    return " | ".join(parts) if len(parts) > 1 else text
