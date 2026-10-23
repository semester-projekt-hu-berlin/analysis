"""Wikidata access and the strict public-figure filter.

Used (a) when seeding, to fetch images/check ids, and (b) as an optional fallback
linker for PER/ORG spans that are not in the curated seed list. A fallback candidate
becomes an allowlisted entity only if it passes `public_figure_check`, so private
persons named in comments can never be allowlisted:

* the Wikidata label or an alias must match the surface form exactly,
* persons: instance of human (Q5), full name (>= 2 tokens) in the comment, at least one
  occupation (P106) or position held (P39) and >= MIN_PERSON_SITELINKS Wikipedia/
  sitelinks (a proxy for notability),
* organisations: instance of one of ORG_CLASSES and >= MIN_ORG_SITELINKS sitelinks,
* apps & digital services (type PRODUCT): instance of one of PRODUCT_CLASSES (software, mobile
  app, website, online service, video game ...) and >= MIN_PRODUCT_SITELINKS sitelinks.

The lookup cache is keyed by a SHA-1 of the surface form, so surface strings from
comments (which might be private names) are never written to disk in clear text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import quote

import requests

from .textutil import normalize_surface

log = logging.getLogger(__name__)

API = "https://www.wikidata.org/w/api.php"
USER_AGENT = (
    "MeinungspulsAnalysis/1.0 (HU Berlin student project 'Stimmungsanalyse im Web'; "
    "https://github.com/semester-projekt-hu-berlin/analysis)"
)

MIN_PERSON_SITELINKS = 15
MIN_ORG_SITELINKS = 10
MIN_PRODUCT_SITELINKS = 20  # app and game names are often words: only well-known ones

HUMAN = "Q5"
DISAMBIGUATION = "Q4167410"
ORG_CLASSES = {
    "Q7278": "political party",
    "Q4830453": "business",
    "Q6881511": "enterprise",
    "Q891723": "public company",
    "Q783794": "company",
    "Q786820": "automobile manufacturer",
    "Q46970": "airline",
    "Q22687": "bank",
    "Q66344": "central bank",
    "Q476028": "association football club",
    "Q847017": "sports club",
    "Q43229": "organization",
    "Q484652": "international organization",
    "Q245065": "intergovernmental organization",
    "Q327333": "government agency",
    "Q192350": "ministry",
    "Q178790": "trade union",
    "Q163740": "nonprofit organization",
    "Q79913": "non-governmental organization",
    "Q11032": "newspaper",
    "Q1002697": "periodical",
    "Q192283": "news agency",
    "Q1126006": "public broadcaster",
    "Q15265344": "broadcaster",
    "Q11204": "legislature",
    "Q35749": "parliament",
    "Q41487": "court",
    "Q190752": "supreme court",
    "Q1640824": "constitutional court",
    "Q772547": "armed forces",
    "Q3918": "university",
    "Q7210356": "political organization",
    "Q17149090": "armed organization",
    "Q4438121": "sports organization",
    "Q15911314": "association",
}


# Apps, platforms, digital services, operating systems and games -> entity type PRODUCT.
PRODUCT_CLASSES = {
    "Q7397": "software",
    "Q166142": "application software",
    "Q218616": "proprietary software",
    "Q341": "free software",
    "Q506883": "free and open-source software",
    "Q620615": "mobile app",
    "Q189210": "web application",
    "Q35127": "website",
    "Q19967801": "online service",
    "Q1668024": "service on Internet",
    "Q3220391": "social networking service",
    "Q118140435": "very large online platform",
    "Q122759350": "user-generated content platform",
    "Q92438": "microblogging",
    "Q2462003": "instant messaging client",
    "Q58199": "instant messaging",
    "Q15614021": "VoIP software",
    "Q28051323": "video-conferencing software",
    "Q59152282": "video streaming service",
    "Q15590336": "music streaming service",
    "Q102336293": "live streaming service",
    "Q559856": "online video platform",
    "Q723685": "video on demand",
    "Q841645": "Internet television",
    "Q870780": "chatbot",
    "Q133284163": "generative artificial intelligence chatbot",
    "Q116961403": "conversational AI",
    "Q115305900": "large language model",
    "Q3467906": "virtual assistant",
    "Q125265855": "voice assistant",
    "Q4182287": "search engine",
    "Q6368": "web browser",
    "Q9135": "operating system",
    "Q920890": "mobile operating system",
    "Q131669": "Linux distribution",
    "Q3390477": "online marketplace",
    "Q57590076": "hotel booking website",
    "Q1343205": "file-hosting service",
    "Q914359": "cloud storage",
    "Q207170": "office suite",
    "Q986008": "payment system",
    "Q25401607": "cryptocurrency exchange",
    "Q28134845": "dating app",
    "Q612825": "online dating service",
    "Q28031555": "machine translation software",
    "Q124587498": "internet map service",
    "Q89288125": "COVID-19 app",
    "Q7889": "video game",
    "Q7058673": "video game series",
    "Q81989119": "video game distribution platform",
    "Q19307174": "digital distribution platform",
}
# media organisations are often also "website": they stay organisations
MEDIA_ORG_CLASSES = {"Q11032", "Q1002697", "Q192283", "Q1126006", "Q15265344"}


def entity_type(item: dict) -> str | None:
    """PER / PARTY / ORG / PRODUCT from the item's P31 classes (None: not a tracked kind).

    An item that is both a company and a service (Netflix, Uber, Airbnb) is a PRODUCT: the brand
    is discussed as the service. Newspapers and broadcasters with a website stay ORG."""
    p31 = set(WikidataClient.claim_ids(item, "P31"))
    if HUMAN in p31:
        return "PER"
    if "Q7278" in p31:
        return "PARTY"
    if p31 & MEDIA_ORG_CLASSES:
        return "ORG"
    if p31 & PRODUCT_CLASSES.keys():
        return "PRODUCT"
    if p31 & ORG_CLASSES.keys():
        return "ORG"
    return None


def _cache_dir() -> Path:
    return Path(os.environ.get("ANALYSIS_CACHE_DIR", Path.cwd() / ".cache"))


class RateLimited(RuntimeError):
    pass


class WikidataClient:
    def __init__(self, session: requests.Session | None = None, min_interval: float = 1.0, max_wait: float = 120):
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.min_interval = min_interval
        self.max_wait = max_wait  # longest Retry-After we are willing to sleep; beyond that: RateLimited
        self._last = 0.0
        self.calls = 0

    def _get(self, params: dict) -> dict:
        params = {**params, "format": "json"}
        for attempt in range(8):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.session.get(API, params=params, timeout=30)
                self.calls += 1
                if r.status_code == 200:
                    return r.json()
                log.warning("wikidata %s", r.status_code)
                if r.status_code == 429:
                    # Respect the server's Retry-After (Wikimedia rate limits per UA+IP).
                    wait = float(r.headers.get("retry-after") or 10)
                    if wait > self.max_wait:
                        raise RateLimited(f"rate limited for {wait:.0f}s")
                    time.sleep(wait)
                    continue
            except requests.RequestException as e:
                log.warning("wikidata error %s", e)
            time.sleep(2**attempt)
        raise RuntimeError("Wikidata API unavailable")

    def search(self, text: str, limit: int = 5, language: str = "de") -> list[dict]:
        res = self._get(
            {
                "action": "wbsearchentities",
                "search": text,
                "language": language,
                "uselang": language,
                "type": "item",
                "limit": limit,
            }
        )
        return res.get("search", [])

    def get_entities(self, ids: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for i in range(0, len(ids), 50):
            res = self._get(
                {
                    "action": "wbgetentities",
                    "ids": "|".join(ids[i : i + 50]),
                    "props": "labels|aliases|descriptions|claims|sitelinks",
                    "languages": "de|en",
                }
            )
            out.update({k: v for k, v in res.get("entities", {}).items() if "missing" not in v})
        return out

    @staticmethod
    def claim_ids(item: dict, prop: str) -> list[str]:
        out = []
        for c in item.get("claims", {}).get(prop, []):
            v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
            if isinstance(v, dict) and "id" in v:
                out.append(v["id"])
        return out

    @staticmethod
    def claim_strings(item: dict, prop: str) -> list[str]:
        out = []
        for c in item.get("claims", {}).get(prop, []):
            v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
            if isinstance(v, str):
                out.append(v)
        return out


def label(item: dict, lang: str = "de") -> str | None:
    for lg in (lang, "en"):
        v = item.get("labels", {}).get(lg, {}).get("value")
        if v:
            return v
    return None


def description(item: dict, lang: str = "de") -> str:
    for lg in (lang, "en"):
        v = item.get("descriptions", {}).get(lg, {}).get("value")
        if v:
            return v
    return ""


def all_names(item: dict) -> list[str]:
    names = []
    for lg in ("de", "en"):
        v = item.get("labels", {}).get(lg, {}).get("value")
        if v:
            names.append(v)
        names.extend(a["value"] for a in item.get("aliases", {}).get(lg, []))
    return list(dict.fromkeys(names))


THUMB_WIDTH = 250  # Wikimedia only serves standard thumbnail widths; 160px returns HTTP 400


def commons_thumb(filename: str, width: int = THUMB_WIDTH) -> str | None:
    """Direct upload.wikimedia.org thumbnail URL for a Commons file (no redirect).

    Commons stores files under /<md5[0]>/<md5[0:2]>/ of the underscored file name; SVGs are
    rendered to PNG. Formats without a simple thumbnail path (TIFF, PDF, video) return None.
    """
    name = filename.strip().replace(" ", "_")
    if not name:
        return None
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ("jpg", "jpeg", "png", "gif", "svg", "webp"):
        return None
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    q = quote(name)
    thumb = f"{width}px-{q}" + (".png" if ext == "svg" else "")
    return f"https://upload.wikimedia.org/wikipedia/commons/thumb/{h[0]}/{h[:2]}/{q}/{thumb}"


def image_url(item: dict, person: bool) -> tuple[str | None, str | None]:
    """(thumbnail URL, kind): persons -> portrait (P18); organisations and products -> logo
    (P154/P8972/P2910/P158) or None."""
    props = ("P18",) if person else ("P154", "P8972", "P2910", "P158")
    for prop in props:
        for f in WikidataClient.claim_strings(item, prop):
            url = commons_thumb(f)
            if url:
                return url, ("portrait" if person else "logo")
    return None, None


def public_figure_check(item: dict, surface: str, ner_type: str) -> tuple[bool, str]:
    """Strict filter deciding whether a Wikidata item may be allowlisted for `surface`."""
    p31 = set(WikidataClient.claim_ids(item, "P31"))
    if DISAMBIGUATION in p31:
        return False, "disambiguation"
    names = {normalize_surface(n) for n in all_names(item)}
    if normalize_surface(surface) not in names:
        return False, "label mismatch"
    sitelinks = len(item.get("sitelinks", {}))
    if ner_type == "PER":
        if HUMAN not in p31:
            return False, "not human"
        if len(surface.split()) < 2:
            return False, "surname only"
        if not (WikidataClient.claim_ids(item, "P106") or WikidataClient.claim_ids(item, "P39")):
            return False, "no occupation/position"
        if sitelinks < MIN_PERSON_SITELINKS:
            return False, f"only {sitelinks} sitelinks"
        return True, "public person"
    if ner_type == "ORG":
        if HUMAN in p31:
            return False, "is a human"
        etype = entity_type(item)
        if etype not in ("ORG", "PARTY", "PRODUCT"):
            return False, "not an organisation or product class"
        if sitelinks < (MIN_PRODUCT_SITELINKS if etype == "PRODUCT" else MIN_ORG_SITELINKS):
            return False, f"only {sitelinks} sitelinks"
        if len(surface) < 3:
            return False, "too short"
        return True, "app or digital service" if etype == "PRODUCT" else "public organisation"
    return False, f"type {ner_type} not tracked"


class WikidataLinker:
    """Cached fallback linker: surface (+NER type) -> public Wikidata item or None."""

    def __init__(self, client: WikidataClient | None = None, max_lookups: int = 150, path: Path | None = None):
        self.client = client or WikidataClient(max_wait=5)
        self.max_lookups = max_lookups
        self.disabled = False
        self.lookups = 0
        self.path = path or (_cache_dir() / "wikidata.json")
        self.cache: dict[str, str] = {}  # sha1(surface|type) -> qid or ""
        self.items: dict[str, dict] = {}  # qid -> compact public item info
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.cache = raw.get("lookups", {})
            self.items = raw.get("items", {})
        except (OSError, ValueError):
            pass

    @staticmethod
    def key(surface: str, ner_type: str) -> str:
        return hashlib.sha1(f"{normalize_surface(surface)}|{ner_type}".encode()).hexdigest()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"lookups": self.cache, "items": self.items}), encoding="utf-8")
        tmp.replace(self.path)

    def lookup(self, surface: str, ner_type: str) -> dict | None:
        k = self.key(surface, ner_type)
        if k in self.cache:
            qid = self.cache[k]
            return self.items.get(qid) if qid else None
        if self.lookups >= self.max_lookups or self.disabled:
            return None
        self.lookups += 1
        result = None
        try:
            hits = self.client.search(surface, limit=5)
            ids = [h["id"] for h in hits]
            items = self.client.get_entities(ids) if ids else {}
            for qid in ids:
                it = items.get(qid)
                if not it:
                    continue
                ok, _why = public_figure_check(it, surface, ner_type)
                if ok:
                    result = {
                        "qid": qid,
                        "name": label(it) or surface,
                        "description": description(it)[:120],
                        "aliases": all_names(it)[:12],
                        "type": "PER" if ner_type == "PER" else (entity_type(it) or "ORG"),
                        "imageUrl": image_url(it, person=ner_type == "PER")[0],
                        "imageKind": image_url(it, person=ner_type == "PER")[1],
                    }
                    break
        except RuntimeError as e:
            # Rate limited / unavailable: stop using the fallback for the rest of this run
            # (uncached surfaces are retried next run).
            log.warning("wikidata fallback disabled for this run: %s", e)
            self.lookups -= 1
            self.disabled = True
            return None
        self.cache[k] = result["qid"] if result else ""
        if result:
            self.items[result["qid"]] = result
        return result
