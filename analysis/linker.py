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

import functools
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .textutil import normalize_surface, search_tokens, slugify

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEED_FILE = DATA_DIR / "entities.yaml"
GENERATED_FILE = DATA_DIR / "entities_generated.yaml"  # from scripts/generate_entities.py

# PRODUCT = "Apps & Dienste": apps, platforms, digital services, operating systems and games
TRACKED_TYPES = {"PER", "ORG", "PARTY", "PRODUCT"}

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
    cue: dict | None = None  # cue set: local evidence for contextual (@) and cue-only (%) aliases

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
    mode: str  # plain | cs | weak | ner | ctx | cue | local, or local-weak / local-ctx / local-cue


@dataclass
class Match:
    start: int
    end: int
    surface: str
    entity_id: str
    via: str  # gazetteer | ner | wikidata


def _load_yaml(path: Path):
    """Parsed YAML, cached per file version (the lists are read several times per run)."""
    st = path.stat()
    return _load_yaml_cached(str(path), st.st_mtime_ns, st.st_size)


@functools.lru_cache(maxsize=8)
def _load_yaml_cached(path: str, _mtime: int, _size: int):
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=loader)


def _trie_pattern(words: set[str]) -> str:
    trie: dict = {}
    for w in words:
        node = trie
        for ch in w:
            node = node.setdefault(ch, {})
        node[""] = {}

    def build(node: dict) -> str:
        alts = [re.escape(ch) + build(child) for ch, child in sorted(node.items()) if ch]
        if not alts:
            return ""
        body = alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"
        if "" in node:
            body = "(?:" + body + ")?"
        return body

    return build(trie)


_MARKERS = {"~": "weak", "!": "ner", "@": "ctx", "=": "cs", "%": "cue"}


def parse_alias(raw: str) -> tuple[str, str]:
    """'~Reiche' -> ('Reiche', 'weak'); '^~BR' -> ('BR', 'local-weak'); see the header of entities.yaml."""
    raw = str(raw)
    if raw.startswith("^"):
        if raw[1:2] in ("~", "@", "%"):
            text, mode = parse_alias(raw[1:])
            return text, f"local-{mode}"
        return raw[1:], "local"
    mode = _MARKERS.get(raw[:1])
    if mode:
        return raw[1:], mode
    return raw, "plain"


def _cs_key(s: str) -> str:
    """Like normalize_surface, but keeps the case (case-sensitive aliases in the NER lookup)."""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("’", "'").replace("‘", "'").replace("–", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", s).strip(" \t\n\"'„“”»«()[]{}:;,.!?")


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
        # (normalized surface, entity id) -> (exact text, mode) of each alias (NER lookup case rules)
        self.alias_entries: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self._compiled = False
        self._name_word_cache: dict[str, set[str]] = {}
        alias_specs = alias_specs or {}
        for e in entities:
            self.add(e, alias_specs.get(e.id))

    # ------------------------------------------------------------------ building
    @classmethod
    def from_yaml(cls, path: Path = SEED_FILE, generated: Path | None = GENERATED_FILE) -> Registry:
        """Curated seed list plus (optionally) the Wikidata-generated list; curated entries win."""
        data = _load_yaml(path)
        contexts = {k: list(v) for k, v in (data.get("contexts") or {}).items()}
        cues = {k: dict(v) for k, v in (data.get("cues") or {}).items()}
        reg = cls()
        lists = [(data["entities"], "seed")]
        if generated is not None and generated.exists():
            gen = _load_yaml(generated) or {}
            lists.append((gen.get("entities") or [], "generated"))
        for entries, origin in lists:
            for raw in entries:
                if origin == "generated" and (raw["id"] in reg.entities or raw.get("wikidata") in reg.by_qid):
                    continue
                specs = [parse_alias(a) for a in raw.get("aliases", [])]
                ctx = raw.get("context")
                ctx_words = contexts.get(ctx, []) if isinstance(ctx, str) else list(ctx or [])
                display = list(dict.fromkeys([raw["name"], *[t for t, _ in specs]]))
                cue = raw.get("cue")
                cue = cues.get(cue) if isinstance(cue, str) else cue
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
                    cue=_compile_cue(cue) if cue else None,
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
        # generated entities list exactly the aliases they want (a dropped name such as "Sandra" or
        # "Haftbefehl" must not come back as a plain alias)
        if e.name not in names and e.source != "generated":
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
            self.alias_entries.setdefault((normalize_surface(text), e.id), []).append((_cs_key(text), mode))
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

    def _pattern(self, aliases: list[Alias], fold: bool = False) -> re.Pattern | None:
        if not aliases:
            return None
        # a trie-shaped pattern: tens of thousands of aliases stay fast (a flat alternation is tried
        # alias by alias at every position); greedy branches keep "longest alias first". For the
        # case-insensitive pattern the trie is built over lower-cased aliases, so "Twitter" and
        # "twitter.com" share one branch and the longer one wins.
        alt = _trie_pattern({a.text.lower() if fold else a.text for a in aliases})
        suffix = "|".join(re.escape(s) for s in sorted(set(PER_SUFFIXES + ORG_SUFFIXES), key=len, reverse=True))
        return re.compile(rf"(?<![\w])({alt})(?P<suf>{suffix})?(?![\w])")

    def compile(self) -> None:
        plain = [a for a in self.aliases if a.mode == "plain"]
        strict = [a for a in self.aliases if a.mode != "plain"]
        self._re_plain = self._pattern(plain, fold=True)
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
            # inflected forms only for curated names ("Grünen"); "Norma" in "Norman", "smart" in "smarten";
            # a bare genitive "s" only for persons ("Kimmichs"), not "Hama" in "Hamas"
            return suffix in (("s", "'s", "’s") if entity.type == "PER" else ("'s", "’s"))
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
            eid = self._choose(als, start, end, suf, ner_spans, words, lowered, country, surface, text)
            if eid and self._other_first_name(text, start, surface, eid):
                continue
            if eid and self._inside_other_name(
                text, start, surface[: len(surface) - len(suf)] if suf else surface, eid
            ):
                continue
            if eid and self.entities[eid].type == "PRODUCT" and _in_url(text, start, end):
                continue  # "de.wikipedia.org/wiki/...", "paulkrugman.substack.com": a link, not a mention
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

    def _inside_other_name(self, text: str, start: int, core: str, eid: str) -> bool:
        """A one-word generated name right after "The" or after another entity's name is part of a
        longer name: "The Focus", "Apple One", "Ford Focus" are not the bike brand or the TV channel."""
        e = self.entities[eid]
        if (
            e.source != "generated"
            or e.type == "PER"
            or " " in core.strip()
            or not core[:1].isupper()
            or core.isupper()
        ):
            return False
        prev = re.search(r"([\w][\w.&+-]*) $", text[max(0, start - 40) : start])
        if not prev:
            return False
        w = prev.group(1)
        if w.casefold() == "the":
            return True
        if not w[:1].isupper() or w.isupper():  # "Apple One", "Ford Focus"; not "ARD ZDF Arte" or "ard zdf ndr"
            return False
        key = normalize_surface(w)
        for oid in self.alias_index.get(key, []):
            if oid == eid:
                continue
            if any(m == "plain" or t == _cs_key(w) for t, m in self.alias_entries.get((key, oid), [])):
                return True
        return False

    @staticmethod
    def _ner_contradicts(e: Entity, start: int, end: int, ner_spans, loc: bool = False) -> bool:
        """An NER span covering exactly this surface says person where the entity is an
        organisation or product, or the other way round (with `loc`: also a place where the
        entity is an organisation: "Fulda", "Salzgitter", "Nordsee")."""
        for s, t, tag, score in ner_spans:
            if s <= start and end <= t and score >= 0.9:
                if tag in ("PER", "ORG") and (tag == "PER") != (e.type == "PER"):
                    return True
                if loc and tag == "LOC" and e.type == "ORG":
                    return True
        return False

    def _context_ok(self, e: Entity, words: set[str], lowered: str, own: set[str] = frozenset()) -> bool:
        """A context keyword in the comment or the article title. For products the alias itself does
        not count ("Claude" is an AI keyword, but "Claude ist mein Nachbar" is no evidence for the
        chatbot); the older context sets rely on it ("Kongress" in the US keywords)."""
        own = own if e.type == "PRODUCT" else frozenset()
        for kw in e.context:
            k = kw.casefold()
            if (" " in k and k in lowered) or (k in words and k not in own):
                return True
        return False

    def _choose(
        self, als, start, end, suf, ner_spans, words, lowered, country, surface: str = "", text: str = ""
    ) -> str | None:
        own = _word_set(surface)
        core = surface[: len(surface) - len(suf)] if suf else surface
        one_word = " " not in core.strip()
        ok: list[tuple[Entity, int]] = []  # (entity, evidence: 2 = context/NER confirmed, 1 = plain)
        for a in als:
            e = self.entities[a.entity_id]
            mode = a.mode
            if mode.startswith("local"):
                # local aliases: only on sources of the entity's own country ("BR" on blick.ch is the
                # Swiss Bundesrat); "local-weak" etc. need the evidence of the other mode as well
                if country and e.country and country != e.country:
                    continue
                mode = mode[len("local-") :] if mode != "local" else "cs"
            if not self._suffix_ok(e, suf):
                continue
            if e.source == "generated" and e.type != "PER":
                # "Lanz" tagged as a person is not the Heinrich Lanz AG, "Fulda" tagged as a place not
                # the tyre maker (one-word names only: "Bank of America" may be tagged LOC); a cue
                # overrules the tag ("„Emma“" is often tagged as a person)
                if self._ner_contradicts(e, start, end, ner_spans, loc=one_word) and not (
                    mode in ("ctx", "cue") and _cue_ok(e.cue, text, start, end, core, suf)
                ):
                    continue
            elif (
                e.source == "generated" and mode in ("plain", "cs") and self._ner_contradicts(e, start, end, ner_spans)
            ):
                continue
            if e.type == "PRODUCT" and self._ner_contradicts(e, start, end, ner_spans):
                continue  # "Claude" / "Bolt" tagged as a person is not the chatbot / the ride app
            if (mode in ("ctx", "cue", "weak") or (mode == "cs" and e.cue)) and _shouting(text, start, end, core):
                continue  # "IN DIESER IRREN ZEIT", "DIE WELT GEHT UNTER": capitals, not the newspaper
            if (
                mode in ("ctx", "cue", "weak")
                and core.isupper()
                and re.search(r"\w-$", text[max(0, start - 2) : start])
            ):
                continue  # the end of a compound: "pt-BR" (a language code), not the broadcaster
            ev = 1
            if mode == "weak":
                ner_ok = any(s < end and start < t and _ner_compatible(tag, e.type) for s, t, tag, _score in ner_spans)
                if not (ner_ok or (e.context and self._context_ok(e, words, lowered, own))):
                    continue
                ev = 2
            elif mode == "ner":
                # NER-only (generated surnames): a context keyword is not enough ("die grüne Wiese"),
                # and only on sources of the person's own country (a "Radwan" on derstandard.at)
                if country and e.country and country != e.country:
                    continue
                if not any(s < end and start < t and _ner_compatible(tag, e.type) for s, t, tag, _score in ner_spans):
                    continue
                # a bare surname is weaker evidence than a contextual reading of the same span
                # ("Lanz" + "Sendung" -> the show, not the presenter)
                ev = 1.5
            elif mode == "ctx":
                article = bool(_ARTICLE_NAME.match(core)) and bool(text)
                if _cue_ok(e.cue, text, start, end, core, suf):
                    pass
                elif article and not _sentence_start(text, start):
                    pass  # "stand in Die Welt": a capital "Die" inside a sentence is the title
                elif article or not self._context_ok(e, words, lowered, own):
                    # "Die Welt steuert auf 2,8 Grad zu": at the start of a sentence "Die Welt" is the
                    # world, even next to "Zeitung"; there the newspaper needs a cue ("Die Welt berichtet")
                    continue
                ev = 2
            elif mode == "cue":
                # cue-only ("%Bild"): a context keyword is not enough ("ein Bild machen" next to
                # "Artikel"), only local evidence counts ("laut Bild", "der Bild", "„Bild“")
                if not _cue_ok(e.cue, text, start, end, core, suf):
                    continue
                ev = 2
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
                i
                for i in ids
                if _ner_compatible(ner_tag, self.entities[i].type) and self._lookup_ok(key, i, country, cand)
            ]
            if ids:
                if len(ids) > 1 and country:
                    same = [i for i in ids if self.entities[i].country == country]
                    ids = same or ids
                return ids[0]
        return None

    def rejected_alias(self, surface: str, qid: str) -> bool:
        """`surface` is an alias of the known entity with this QID, so the gazetteer and the alias
        lookup have already decided on it (e.g. a contextual alias without its context)."""
        eid = self.by_qid.get(qid)
        if not eid:
            return False
        return any(eid in self.alias_index.get(normalize_surface(v), []) for v in surface_variants(surface))

    def _lookup_ok(self, key: str, eid: str, country: str | None, surface: str | None = None) -> bool:
        """An NER span may use an alias only where the gazetteer could: contextual and cue-only
        aliases need their context or cue (handled by the gazetteer), NER-only and local ones the
        right country, and weak and NER-only ones their spelling ("die reiche Elite" is not "Reiche")."""
        modes = self.alias_modes.get((key, eid), {"plain"})
        if modes <= {"ctx", "cue", "local-ctx", "local-cue"}:
            return False
        e = self.entities[eid]
        if (
            modes <= {"ner", "local", "local-weak", "local-ctx", "local-cue"}
            and country
            and e.country
            and country != e.country
        ):
            return False
        if surface is None:
            return True
        cand = _cs_key(surface)
        for text, mode in self.alias_entries.get((key, eid), [("", "plain")]):
            if mode in ("ctx", "cue", "local-ctx", "local-cue"):
                continue
            if mode == "plain" or cand == text:
                return True
            if mode in ("cs", "local"):
                # case-sensitive brand names vary in case ("LIDL", "arte", "Mediamarkt", "Vp"), but a
                # word is not a name written in capitals ("Spiegel" is not "SPIEGEL", "Zeit" not "ZEIT")
                short = len(text) <= 3  # "Vp" for "VP"
                if cand.isupper() or ((short or not text.isupper()) and cand.casefold() == text.casefold()):
                    return True
            elif mode in ("weak", "ner", "local-weak") and cand.isupper() and len(cand) >= 3:
                return True  # "REICHE"; not "die reiche Elite"
        return False


_URLISH = re.compile(r"^@|://|\bwww\.|\w\.(?:com|org|net|de|at|ch|eu|io|app|tv|fm|social|co|uk)\b", re.IGNORECASE)


def _in_url(text: str, start: int, end: int) -> bool:
    """The span is part of a link, domain or handle ("de.wikipedia.org/wiki/...",
    "krugman.substack.com/p/the-starlink", "@spotify"). Aliases that are domains themselves
    ("x.com", "booking.com", "web.de") are mentions of the service and still link."""
    if "." in text[start:end]:
        return False
    a = max(text.rfind(c, 0, start) for c in (" ", "\n", "\t", "(", '"'))
    b = min((i for i in (text.find(c, end) for c in (" ", "\n", "\t", ")", '"')) if i >= 0), default=len(text))
    return _URLISH.search(text[a + 1 : b]) is not None


_ARTICLE_NAME = re.compile(r"^(?:Die|Der|Das) [A-ZÄÖÜ][\w-]*$")  # "Die Welt", "Die Zeit" (not "DIE ZEIT")
_QUOTES_OPEN = "„“\"'‚»«‹›"
_QUOTES_CLOSE = "“”\"'‘«»›‹"
# what may stand between two media in a list: quotes, a separator, an article ("Spiegel und den Focus")
_CUE_SEP = re.compile(
    r"^[\s„“”\"'»«]*(?:(?:,|/|&|\+|und|oder|sowie)[\s„“”\"'»«]*(?:(?:der|die|das|den|dem)\s+)?)?[„“\"'»«]?$",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[\w][\w.+-]*")


def _compile_cue(cfg: dict) -> dict:
    """A cue set from entities.yaml (`cues:`): local evidence that a contextual (`@`) or cue-only
    (`%`) alias means the entity, independent of the context keywords."""
    return {
        "before": {w.casefold() for w in cfg.get("before") or []},
        "after": {w.casefold() for w in cfg.get("after") or []},
        "compound": tuple(w.casefold() for w in cfg.get("compound") or []),
        "neighbors": set(cfg.get("neighbors") or []),  # case-sensitive: "Welt" the paper, "welt" no
        "quoted": bool(cfg.get("quoted")),
        "colon": bool(cfg.get("colon")),
    }


def _cue_ok(cue: dict | None, text: str, start: int, end: int, core: str, suf: str | None) -> bool:
    """`core` at text[start:end] has a cue of the entity's cue set: a word before ("laut Bild",
    "der ZEIT") or after it ("Bild berichtet", "WELT:"), quotes ("„Spiegel“"), a compound
    ("ZEIT-Redakteurin") or another medium next to it ("Spiegel, FAZ und Bild")."""
    if not cue or not text:
        return False
    before = text[max(0, start - 60) : start]
    after = text[end : end + 60]
    prev = re.search(r"([\w.]+)[\s:]*$", before)
    if prev and prev.group(1).casefold() in cue["before"]:
        return True
    if cue["colon"] and core.isupper() and re.match(r"\s?:", after):  # "WELT: ..."; not "Bild: dpa"
        return True
    nxt = re.match(r"[\s,.]*([\w]+)", after)
    if nxt and nxt.group(1).casefold() in cue["after"]:
        return True
    if cue["quoted"] and before[-1:] in _QUOTES_OPEN and after[:1] in _QUOTES_CLOSE:
        return True
    comp = re.match(r"-(\w+)", after)
    if comp and cue["compound"] and comp.group(1).casefold().startswith(cue["compound"]):
        return True
    if cue["neighbors"]:
        # another medium right before or after it ("Spiegel, SZ", "ARD/ZDF/Spiegel/Bild", "Focus und Welt")
        toks = list(_TOKEN.finditer(before))[-3:]
        for t in reversed(toks):
            if t.group(0).rstrip(".") in cue["neighbors"] and _CUE_SEP.match(before[t.end() :]):
                return True
        for t in list(_TOKEN.finditer(after))[:3]:
            if t.group(0).rstrip(".") in cue["neighbors"] and _CUE_SEP.match(after[: t.start()]):
                return True
    return False


def _sentence_start(text: str, start: int) -> bool:
    """`start` begins a sentence: only spaces, quotes or dashes since the last . ! ? : or line break."""
    before = text[max(0, start - 200) : start].rstrip(" \t„“\"'»«-–(")
    return not before or before[-1] in ".!?:;…\n"


def _shouting(text: str, start: int, end: int, core: str) -> bool:
    """An all-caps word in all-caps text ("IN DIESER IRREN ZEIT", "DIE WELT GEHT UNTER"): at least
    two of the three words before and after are capitals too (3+ letters)."""
    if not text or not core.isupper() or len(core) < 3:
        return False
    before = re.findall(r"[^\W\d_]+", text[max(0, start - 60) : start])[-3:]
    after = re.findall(r"[^\W\d_]+", text[end : end + 60])[:3]
    return sum(1 for w in before + after if len(w) >= 3 and w.isupper()) >= 2


def _ner_compatible(tag: str, etype: str) -> bool:
    """NER tags that confirm a weak alias of this entity type. Products (apps, platforms, games)
    are tagged ORG ("WhatsApp") or MISC ("Fortnite", "iOS") by the German NER models."""
    if etype == "PER":
        return tag == "PER"
    if etype == "PRODUCT":
        return tag in ("ORG", "MISC")
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
