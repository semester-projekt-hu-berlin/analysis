"""Entity registry and linking of surface forms to canonical entities.

Linking combines two signals:

1. a gazetteer over the curated aliases in `data/entities.yaml` (plus aliases of
   entities discovered earlier via Wikidata), matched at word boundaries with German
   inflection suffixes ("Merz'", "Trumps", "SPDler" is an explicit alias);
2. NER spans (Flair) that confirm ambiguous ("weak") aliases and are the only source of
   new entities, which must pass the strict Wikidata public-figure filter.

Ambiguous aliases are resolved by context keywords (comment text + article title) and
by the country of the source (e.g. "Grüne" on derstandard.at -> Austrian Greens).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .textutil import normalize_surface, search_tokens, slugify

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEED_FILE = DATA_DIR / "entities.yaml"
GENERATED_FILE = DATA_DIR / "entities_generated.yaml"  # from scripts/generate_entities.py

TRACKED_TYPES = {"PER", "ORG", "PARTY"}

# Suffixes allowed after an alias (genitive / plural / inflection).
PER_SUFFIXES = ["'s", "’s", "s", "'", "’"]
ORG_SUFFIXES = ["'s", "’s", "s", "es", "en", "n"]

# Titles and role words stripped from NER spans before lookup ("Kanzler Merz" -> "Merz").
TITLES = {
    "herr",
    "frau",
    "dr",
    "dr.",
    "prof",
    "prof.",
    "kanzler",
    "kanzlerin",
    "bundeskanzler",
    "bundeskanzlerin",
    "altkanzler",
    "altkanzlerin",
    "vizekanzler",
    "minister",
    "ministerin",
    "bundesminister",
    "bundesministerin",
    "präsident",
    "präsidentin",
    "bundespräsident",
    "bundespräsidentin",
    "us-präsident",
    "ex-präsident",
    "ministerpräsident",
    "ministerpräsidentin",
    "senator",
    "senatorin",
    "gouverneur",
    "gouverneurin",
    "chef",
    "chefin",
    "ceo",
    "parteichef",
    "parteichefin",
    "genosse",
    "genossin",
    "sir",
    "mr",
    "mr.",
    "mrs",
    "mrs.",
    "ms",
    "ms.",
    "general",
    "papst",
    "könig",
    "königin",
    "prinz",
    "prinzessin",
    "ex-kanzler",
    "finanzminister",
    "innenminister",
    "außenminister",
    "wirtschaftsminister",
    "verteidigungsminister",
    "gesundheitsminister",
    "arbeitsministerin",
    "landeshauptmann",
    "landeshauptfrau",
    "bürgermeister",
    "bürgermeisterin",
    "die",
    "der",
    "den",
    "dem",
    "des",
}


# Words too common in entity names to tell candidates apart.
GENERIC_WORDS = {
    "partei",
    "deutschland",
    "deutsche",
    "deutscher",
    "union",
    "verein",
    "borussia",
    "eintracht",
    "fortuna",
    "sport",
    "club",
    "national",
    "group",
    "holding",
    "schweiz",
    "österreich",
    "gruene",
    "grünen",
    "freie",
    "demokratische",
    "sozialdemokratische",
    "bundesliga",
    "fussball",
    "fußball",
    "stadt",
    "berliner",
}


@dataclass
class Entity:
    id: str
    name: str
    type: str
    description: str = ""
    wikidataId: str | None = None
    country: str | None = None
    aliases: list[str] = field(default_factory=list)  # display aliases (markers stripped)
    context: list[str] = field(default_factory=list)  # context keywords
    imageUrl: str | None = None
    imageKind: str | None = None  # "logo" | "portrait" | None
    source: str = "seed"  # seed | wikidata

    @property
    def search_tokens(self) -> list[str]:
        return search_tokens([self.name, *self.aliases])

    def metadata(self) -> dict:
        """Entity fields owned by the seed/linker (counts are owned by aggregation)."""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "wikidataId": self.wikidataId,
            "imageUrl": self.imageUrl,
            "imageKind": self.imageKind,
            "aliases": self.aliases,
            "searchTokens": self.search_tokens,
            "allowlisted": True,
        }


@dataclass
class Alias:
    text: str
    entity_id: str
    mode: str  # plain | cs | weak | ner | ctx | local


@dataclass
class Match:
    start: int
    end: int
    surface: str
    entity_id: str
    via: str  # gazetteer | ner | wikidata


def parse_alias(raw: str) -> tuple[str, str]:
    raw = str(raw)
    if raw.startswith("~"):
        return raw[1:], "weak"
    if raw.startswith("!"):
        return raw[1:], "ner"
    if raw.startswith("@"):
        return raw[1:], "ctx"
    if raw.startswith("="):
        return raw[1:], "cs"
    if raw.startswith("^"):
        return raw[1:], "local"
    return raw, "plain"


def _word_set(text: str) -> set[str]:
    return {w.casefold() for w in re.findall(r"[\w][\w\-]*", text)}


class Registry:
    """All linkable entities plus compiled gazetteer regexes."""

    def __init__(self, entities: Iterable[Entity] = (), alias_specs: dict[str, list[tuple[str, str]]] | None = None):
        self.entities: dict[str, Entity] = {}
        self.by_qid: dict[str, str] = {}
        self.aliases: list[Alias] = []
        self.alias_index: dict[str, list[str]] = {}  # normalized surface -> entity ids
        self.alias_modes: dict[tuple[str, str], set[str]] = {}  # (normalized surface, entity id) -> modes
        self._compiled = False
        self._name_word_cache: dict[str, set[str]] = {}
        alias_specs = alias_specs or {}
        for e in entities:
            self.add(e, alias_specs.get(e.id))

    # ------------------------------------------------------------------ building
    @classmethod
    def from_yaml(cls, path: Path = SEED_FILE, generated: Path | None = GENERATED_FILE) -> Registry:
        """Curated seed list plus (optionally) the Wikidata-generated list; curated entries win."""
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        contexts = {k: list(v) for k, v in (data.get("contexts") or {}).items()}
        reg = cls()
        lists = [(data["entities"], "seed")]
        if generated is not None and generated.exists():
            gen = yaml.safe_load(generated.read_text(encoding="utf-8")) or {}
            lists.append((gen.get("entities") or [], "generated"))
        for entries, origin in lists:
            for raw in entries:
                if origin == "generated" and (raw["id"] in reg.entities or raw.get("wikidata") in reg.by_qid):
                    continue
                specs = [parse_alias(a) for a in raw.get("aliases", [])]
                ctx = raw.get("context")
                ctx_words = contexts.get(ctx, []) if isinstance(ctx, str) else list(ctx or [])
                display = list(dict.fromkeys([raw["name"], *[t for t, _ in specs]]))
                e = Entity(
                    id=raw["id"],
                    name=raw["name"],
                    type=raw["type"],
                    description=raw.get("description", ""),
                    wikidataId=raw.get("wikidata"),
                    country=raw.get("country"),
                    aliases=display,
                    context=ctx_words,
                    source=origin,
                )
                reg.add(e, specs)
        return reg

    def add(self, e: Entity, specs: list[tuple[str, str]] | None = None) -> None:
        if e.type not in TRACKED_TYPES:
            raise ValueError(f"{e.id}: type {e.type} is not tracked")
        if e.id in self.entities:
            raise ValueError(f"duplicate entity id {e.id}")
        self.entities[e.id] = e
        if e.wikidataId:
            self.by_qid.setdefault(e.wikidataId, e.id)
        if specs is None:
            # Discovered entities: every alias needs NER confirmation.
            specs = [(a, "weak") for a in e.aliases]
        names = {t for t, _ in specs}
        if e.name not in names:
            specs = [(e.name, "plain" if e.source in ("seed", "generated") else "weak"), *specs]
        for text, mode in specs:
            text = text.strip()
            if len(text) < 2:
                continue
            self.aliases.append(Alias(text, e.id, mode))
            self.alias_index.setdefault(normalize_surface(text), [])
            if e.id not in self.alias_index[normalize_surface(text)]:
                self.alias_index[normalize_surface(text)].append(e.id)
            self.alias_modes.setdefault((normalize_surface(text), e.id), set()).add(mode)
        self._compiled = False

    def add_discovered(self, info: dict) -> Entity:
        """Add an entity found through the Wikidata fallback; returns the (possibly existing) entity."""
        qid = info["qid"]
        if qid in self.by_qid:
            return self.entities[self.by_qid[qid]]
        eid = slugify(info["name"])
        if eid in self.entities:
            eid = f"{eid}-{qid.lower()}"
        e = Entity(
            id=eid,
            name=info["name"],
            type=info.get("type", "ORG"),
            description=info.get("description", ""),
            wikidataId=qid,
            aliases=list(dict.fromkeys([info["name"], *info.get("aliases", [])]))[:12],
            imageUrl=info.get("imageUrl"),
            imageKind=info.get("imageKind"),
            source="wikidata",
        )
        self.add(e)
        return e

    def _pattern(self, aliases: list[Alias]) -> re.Pattern | None:
        if not aliases:
            return None
        texts = sorted({a.text for a in aliases}, key=len, reverse=True)
        alt = "|".join(re.escape(t) for t in texts)
        suffix = "|".join(re.escape(s) for s in sorted(set(PER_SUFFIXES + ORG_SUFFIXES), key=len, reverse=True))
        return re.compile(rf"(?<![\w])({alt})(?P<suf>{suffix})?(?![\w])")

    def compile(self) -> None:
        plain = [a for a in self.aliases if a.mode == "plain"]
        strict = [a for a in self.aliases if a.mode != "plain"]
        self._re_plain = self._pattern(plain)
        if self._re_plain is not None:
            self._re_plain = re.compile(self._re_plain.pattern, re.IGNORECASE)
        self._re_strict = self._pattern(strict)
        self._lookup_plain: dict[str, list[Alias]] = {}
        self._lookup_strict: dict[str, list[Alias]] = {}
        for a in plain:
            self._lookup_plain.setdefault(a.text.casefold(), []).append(a)
        for a in strict:
            self._lookup_strict.setdefault(a.text, []).append(a)
        # first names of all tracked persons: "Hermann Merz" is not Friedrich Merz
        self._first_names = {
            e.name.split()[0]
            for e in self.entities.values()
            if e.type == "PER" and len(e.name.split()) >= 2 and len(e.name.split()[0]) >= 3
        }
        self._compiled = True

    # ------------------------------------------------------------------ linking
    def _suffix_ok(self, entity: Entity, suffix: str | None) -> bool:
        if not suffix:
            return True
        if entity.source == "generated":
            # inflected forms only for curated names ("Grünen"); "Norma" in "Norman", "smart" in "smarten"
            return suffix in ("s", "'s", "’s")
        allowed = PER_SUFFIXES if entity.type == "PER" else ORG_SUFFIXES
        return suffix in allowed

    def gazetteer(
        self,
        text: str,
        ner_spans: list[tuple[int, int, str, float]] = (),
        context_text: str = "",
        country: str | None = None,
    ) -> list[Match]:
        """Find alias matches in `text`.

        `ner_spans`: (start, end, tag, score) in `text` coordinates.
        `context_text`: additional context (article title) for contextual aliases.
        """
        if not self._compiled:
            self.compile()
        words = _word_set(text) | _word_set(context_text)
        lowered = f"{text}\n{context_text}".casefold()
        candidates: list[tuple[int, int, str, list[Alias], str | None]] = []
        for rx, lookup, fold in (
            (self._re_plain, self._lookup_plain, True),
            (self._re_strict, self._lookup_strict, False),
        ):
            if rx is None:
                continue
            for m in rx.finditer(text):
                key = m.group(1).casefold() if fold else m.group(1)
                als = lookup.get(key, [])
                if als:
                    candidates.append((m.start(), m.end(), m.group(0), als, m.group("suf")))
        # The same span may match in both regexes (e.g. plain "RWE" the company, contextual "@RWE"
        # the club): merge their aliases so the disambiguation sees all readings.
        merged: dict[tuple[int, int], tuple[int, int, str, list[Alias], str | None]] = {}
        for c in candidates:
            k = (c[0], c[1])
            merged[k] = (c[0], c[1], c[2], merged[k][3] + c[3], c[4]) if k in merged else c
        candidates = list(merged.values())
        # Longest-first, non-overlapping.
        out: list[Match] = []
        for start, end, surface, als, suf in sorted(candidates, key=lambda c: (-(c[1] - c[0]), c[0])):
            if any(not (end <= m.start or start >= m.end) for m in out):
                continue
            eid = self._choose(als, start, end, suf, ner_spans, words, lowered, country)
            if eid and self._other_first_name(text, start, surface, eid):
                continue
            if eid:
                core = surface[: len(surface) - len(suf)] if suf else surface
                out.append(Match(start, end, core, eid, "gazetteer"))
        out.sort(key=lambda m: m.start)
        return out

    def _other_first_name(self, text: str, start: int, surface: str, eid: str) -> bool:
        """A bare surname preceded by a different person's first name ("Hermann Merz")."""
        e = self.entities[eid]
        if e.type != "PER" or " " in surface.strip():
            return False
        prev = re.search(r"([A-ZÄÖÜ][\w-]+) $", text[max(0, start - 40) : start])
        if not prev:
            return False
        first = prev.group(1)
        own = e.name.split()[0]
        if first[:3].casefold() == own[:3].casefold():  # "Olav Scholz" is a typo, not another Scholz
            return False
        return first in self._first_names and first not in e.name.split() and first not in " ".join(e.aliases)

    def _context_ok(self, e: Entity, words: set[str], lowered: str) -> bool:
        for kw in e.context:
            k = kw.casefold()
            if (" " in k and k in lowered) or k in words:
                return True
        return False

    def _choose(self, als, start, end, suf, ner_spans, words, lowered, country) -> str | None:
        ok: list[tuple[Entity, int]] = []  # (entity, evidence: 2 = context/NER confirmed, 1 = plain)
        for a in als:
            e = self.entities[a.entity_id]
            if not self._suffix_ok(e, suf):
                continue
            ev = 1
            if a.mode == "weak":
                ner_ok = any(s < end and start < t and _ner_compatible(tag, e.type) for s, t, tag, _score in ner_spans)
                if not (ner_ok or (e.context and self._context_ok(e, words, lowered))):
                    continue
                ev = 2
            elif a.mode == "ner":
                # NER-only (generated surnames): a context keyword is not enough ("die grüne Wiese"),
                # and only on sources of the person's own country (a "Radwan" on derstandard.at)
                if country and e.country and country != e.country:
                    continue
                if not any(s < end and start < t and _ner_compatible(tag, e.type) for s, t, tag, _score in ner_spans):
                    continue
                ev = 2
            elif a.mode == "ctx":
                if not self._context_ok(e, words, lowered):
                    continue
                ev = 2
            elif a.mode == "local" and country and e.country and country != e.country:
                continue
            ok.append((e, ev))
        if not ok:
            return None
        if len(ok) > 1:
            # 1. distinctive words of a candidate elsewhere in the text ("Eintracht ... Braunschweig")
            others = (lowered[:start] + " " + lowered[end:]) if lowered else ""
            other_words = set(re.findall(r"[\w]+", others))
            named = [(e, ev) for e, ev in ok if self._name_words(e) & other_words]
            if named:
                ok = named
        if len(ok) > 1:
            # 2. context/NER-confirmed readings beat plain ones ("RWE" + Spiel/Tor -> Rot-Weiss Essen)
            best = max(ev for _, ev in ok)
            ok = [(e, ev) for e, ev in ok if ev == best]
        if len(ok) > 1:
            # 3. the source country, then Germany
            same = [x for x in ok if country and x[0].country == country]
            ok = same or [x for x in ok if x[0].country == "DE"] or ok
        return ok[0][0].id

    def _name_words(self, e: Entity) -> set[str]:
        cached = self._name_word_cache.get(e.id)
        if cached is None:
            ws: set[str] = set()
            for n in [e.name, *e.aliases]:
                ws |= {w for w in re.findall(r"[\w]+", n.casefold()) if len(w) >= 5 and w not in GENERIC_WORDS}
            # words shared with many entities are not distinctive
            cached = self._name_word_cache[e.id] = ws
        return cached

    def lookup_surface(self, surface: str, ner_tag: str, country: str | None = None) -> str | None:
        """Exact (normalized) alias lookup for an NER span, with titles stripped."""
        for cand in surface_variants(surface):
            key = normalize_surface(cand)
            ids = self.alias_index.get(key, [])
            ids = [
                i for i in ids if _ner_compatible(ner_tag, self.entities[i].type) and self._lookup_ok(key, i, country)
            ]
            if ids:
                if len(ids) > 1 and country:
                    same = [i for i in ids if self.entities[i].country == country]
                    ids = same or ids
                return ids[0]
        return None

    def _lookup_ok(self, key: str, eid: str, country: str | None) -> bool:
        """An NER span may use an alias only where the gazetteer could: contextual aliases need
        their context (handled by the gazetteer), NER-only and local ones the right country."""
        modes = self.alias_modes.get((key, eid), {"plain"})
        if modes <= {"ctx"}:
            return False
        e = self.entities[eid]
        return not (modes <= {"ner", "local"} and country and e.country and country != e.country)


def _ner_compatible(tag: str, etype: str) -> bool:
    if etype == "PER":
        return tag == "PER"
    return tag in ("ORG", "MISC")


def strip_titles(surface: str) -> str:
    """'Kanzler Friedrich Merz' -> 'Friedrich Merz'."""
    s = re.sub(r"\s+", " ", surface).strip(" \"'„“”»«()[]:;,.!?")
    toks = s.split(" ")
    while len(toks) > 1 and toks[0].casefold() in TITLES:
        toks = toks[1:]
    return " ".join(toks)


def surface_variants(surface: str) -> list[str]:
    """'Kanzler Merzs' -> ['Kanzler Merzs', 'Merzs', 'Merz']."""
    s = re.sub(r"\s+", " ", surface).strip(" \"'„“”»«()[]:;,.!?")
    core = strip_titles(s)
    out = [s, core]
    for suf in ("'s", "’s", "s", "'", "’"):
        if core.endswith(suf) and len(core) > len(suf) + 2:
            out.append(core[: -len(suf)])
    return list(dict.fromkeys(x for x in out if x))
