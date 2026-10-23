"""Generate data/entities_generated.yaml from Wikidata: German politicians, companies, universities.

Raw SPARQL results are cached in data/wikidata_cache/*.json (use --refresh to re-query).
Everything here is a public figure or organisation by role and notability:

Politicians (PER), alive:
  * members of the 21st Bundestag, current German MEPs,
  * current holders of federal or state offices (federal/state ministers, Ministerpräsident:innen,
    Bundespräsident ...; parliament memberships excluded),
  * current federal and state party leaders (P488 of German parties and their branches),
  * former chancellors, federal ministers and heads of state governments,
  * any German politician with >= 15 sitelinks (born after 1920, alive or died after 2000).
Companies (ORG): current DAX / MDAX / SDAX / TecDAX members, German companies with > 10,000
  employees, and German companies, retailers, banks, insurers, airlines, media with >= 25 sitelinks.
Universities (ORG): German universities (P31/P279* Q3918); universities of applied sciences only
  with >= 10 sitelinks.

Alias policy (linking quality):
  * full names / full company names: plain aliases;
  * person surnames: NER-only ("!": needs an NER PER span, context keywords are not enough) and
    only if unique among all entities,
    not a very common surname and not a frequent German word;
  * one-word company names that are frequent German words (Post, Bahn, Bild, ...): weak (needs
    NER ORG or business context); acronyms case-sensitive;
  * no alias that already belongs to a curated entity in data/entities.yaml (curated wins).

    python scripts/generate_entities.py [--refresh]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.linker import parse_alias  # noqa: E402
from analysis.textutil import normalize_surface, slugify  # noqa: E402
from analysis.wikidata import USER_AGENT  # noqa: E402

CACHE = ROOT / "data" / "wikidata_cache"
OUT = ROOT / "data" / "entities_generated.yaml"
SEED = ROOT / "data" / "entities.yaml"
SPARQL = "https://query.wikidata.org/sparql"

BUNDESTAG_21 = "Q124661964"
INDICES = {"Q155718": "DAX", "Q595622": "MDAX", "Q448445": "SDAX", "Q378967": "TecDAX"}
FH_CLASSES = {"Q17028020", "Q1365560"}  # Fachhochschule / University of applied sciences
FEMALE = "Q6581072"
# executive offices that make someone "well known" (parliament memberships and county offices are not enough)
OFFICE = re.compile(
    r"(minister|Ministerpräsident|Senator|Erster Bürgermeister|Regierender Bürgermeister|Bürgermeister von Bremen|"
    r"Bundespräsident|Bundeskanzler|Bundestagspräsident|Landtagspräsident|Präsident des|Staatssekretär|"
    r"Beauftragte|Wehrbeauftragte|Bundesratspräsident)",
    re.IGNORECASE,
)

COMMON_SURNAMES = set(
    [
        "müller",
        "schmidt",
        "schneider",
        "fischer",
        "weber",
        "meyer",
        "meier",
        "wagner",
        "becker",
        "schulz",
        "hoffmann",
        "hofmann",
        "schäfer",
        "koch",
        "bauer",
        "richter",
        "klein",
        "wolf",
        "schröder",
        "neumann",
        "schwarz",
        "zimmermann",
        "braun",
        "krüger",
        "hofmann",
        "hartmann",
        "lange",
        "schmitt",
        "werner",
        "schmitz",
        "krause",
        "meier",
        "lehmann",
        "schmid",
        "schulze",
        "maier",
        "köhler",
        "herrmann",
        "könig",
        "walter",
        "mayer",
        "huber",
        "kaiser",
        "fuchs",
        "peters",
        "lang",
        "scholz",
        "möller",
        "weiß",
        "jung",
        "hahn",
        "schubert",
        "vogel",
        "friedrich",
        "keller",
        "günther",
        "frank",
        "berger",
        "winkler",
        "roth",
        "beck",
        "lorenz",
        "baumann",
        "franke",
        "albrecht",
        "schuster",
        "simon",
        "ludwig",
        "böhm",
        "winter",
        "kraus",
        "martin",
        "schumacher",
        "krämer",
        "vogt",
        "stein",
        "jäger",
        "otto",
        "sommer",
        "groß",
        "seidel",
        "heinrich",
        "brandt",
        "haas",
        "schreiber",
        "graf",
        "schulte",
        "dietrich",
        "ziegler",
        "kuhn",
        "kühn",
        "pohl",
        "engel",
        "horn",
        "busch",
        "bergmann",
        "thomas",
        "voigt",
        "sauer",
        "arnold",
        "wolff",
        "pfeiffer",
        "lindner",
        "merz",
        "linke",
        "grün",
        "schwarz",
        "roth",
        "weiss",
        "weiß",
        "rot",
    ]
)


def sparql(name: str, query: str, refresh: bool) -> list[dict]:
    path = CACHE / f"{name}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(8):
        r = requests.post(
            SPARQL,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT},
            timeout=300,
        )
        if r.status_code == 200:
            try:
                rows = [{k: v["value"] for k, v in b.items()} for b in r.json()["results"]["bindings"]]
            except ValueError:  # a response cut off by the server's timeout
                print(f"  {name}: truncated response, retrying", file=sys.stderr)
                time.sleep(10 * (attempt + 1))
                continue
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            print(f"  {name}: {len(rows)} rows", file=sys.stderr)
            return rows
        wait = float(r.headers.get("retry-after") or 10 * (attempt + 1))
        print(f"  {name}: HTTP {r.status_code}, waiting {wait:.0f}s", file=sys.stderr)
        time.sleep(min(wait, 120))
    raise RuntimeError(f"SPARQL failed for {name}")


def q(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


ALIVE = "FILTER NOT EXISTS { ?p wdt:P570 ?dod }"

COMPANY_CLASSES = [
    "Q4830453",
    "Q6881511",
    "Q891723",
    "Q783794",
    "Q167037",
    "Q1589009",
    "Q219577",
    "Q22687",
    "Q46970",
    "Q2085381",
    "Q1320047",
    "Q1137109",
    "Q507619",
    "Q786820",
    "Q1331793",
    "Q1616075",
    "Q1631129",
    "Q18388277",
    "Q431289",
    "Q2659904",
]

# --------------------------------------------------------------------------- German media & tech
# Hand-picked QIDs (resolved by label + description and checked by hand), with the context set
# their weak/contextual aliases need and extra aliases that Wikidata lacks.
MEDIA_TECH: dict[str, tuple[str, list[str]]] = {
    # international and German tech
    "Q20800404": ("business", []),  # Alphabet Inc.
    "Q117617480": ("business", ["X Corp"]),  # X Corp.
    "Q1390577": ("business", []),  # Twitter, Inc. (the platform "Twitter" / X is the curated PRODUCT x-twitter)
    "Q55606242": ("business", ["Bytedance"]),
    "Q907311": ("business", []),  # Netflix
    "Q689141": ("business", []),  # Spotify
    "Q20718": ("business", ["Samsung"]),
    "Q160120": ("business", []),  # Huawei
    "Q23384493": ("business", []),  # Delivery Hero
    "Q27479372": ("business", []),  # N26
    "Q63725648": ("business", []),  # Celonis
    "Q322579": ("business", []),  # United Internet
    "Q1259028": ("business", ["1&1"]),
    "Q61740994": ("business", ["TeamViewer"]),
    "Q127380521": ("business", []),  # Helsing
    "Q123368857": ("business", []),  # Aleph Alpha
    "Q43968444": ("business", ["DeepL"]),
    "Q114258161": ("business", []),  # Temu
    "Q107377209": ("business", []),  # Shein
    "Q1359568": ("business", ["Alibaba"]),
    "Q128896": ("business", []),  # AMD
    "Q37156": ("business", []),  # IBM
    "Q19900": ("business", ["Oracle"]),
    "Q780442": ("business", []),  # Uber
    "Q63327": ("business", []),  # Airbnb
    "Q483959": ("business", []),  # PayPal
    "Q1636958": ("business", []),  # Xiaomi
    "Q2608329": ("business", []),  # BYD
    "Q116758847": ("business", []),  # Anthropic
    "Q119718658": ("business", []),  # Mistral AI
    "Q2047336": ("business", ["Palantir"]),
    "Q105475811": ("business", []),  # Trade Republic
    "Q47457075": ("business", ["Auto1"]),
    "Q1630943": ("business", []),  # idealo
    "Q1068745": ("business", []),  # Check24
    "Q265286": ("business", []),  # Trivago
    "Q30024526": ("business", []),  # Wolt
    "Q122141": ("business", []),  # Vodafone
    "Q2401561": ("business", ["O2", "Telefónica"]),
    # broadcasters
    "Q203453": ("media", ["WDR"]),
    "Q201275": ("media", ["NDR"]),
    "Q278637": ("media", ["~BR"]),
    "Q700133": ("media", ["SWR"]),
    "Q158948": ("media", ["MDR"]),
    "Q19899": ("media", ["RBB", "rbb"]),
    "Q23565": ("media", []),  # Hessischer Rundfunk
    "Q691047": ("media", []),  # Saarländischer Rundfunk
    "Q694837": ("media", []),  # Radio Bremen
    "Q153770": ("media", ["~DW"]),
    "Q695328": ("media", ["=DLF", "Dlf"]),
    "Q702314": ("media", []),  # Deutschlandradio
    "Q155702": ("media", ["~RTL"]),
    "Q3304009": ("media", []),  # RTL Deutschland
    "Q313332": ("media", ["Sat1", "Sat 1"]),
    "Q684981": ("media", ["~VOX", "~Vox"]),
    "Q265027": ("media", ["ntv", "n-tv"]),
    "Q1959672": ("media", ["Welt TV", "WELT TV"]),
    "Q229805": ("media", []),  # 3sat
    "Q262627": ("media", ["~Kika"]),
    "Q265612": ("media", ["Kabel Eins", "Kabel 1"]),
    "Q136012": ("media", []),  # ZDFneo
    "Q594177": ("media", ["RTL2", "RTL II"]),
    "Q652282": ("media", ["Sky"]),
    "Q30133334": ("media", []),  # DAZN
    # newspapers, magazines, agencies, online media
    "Q158870": ("media", ["~SZ", "Süddeutsche"]),
    "Q10184": ("media", ["FAZ", "F.A.Z."]),
    "Q157142": ("media", ["=ZEIT", "Zeit Online", "ZEIT Online", "zeit.de"]),
    "Q29872": ("media", ["=WELT", "Welt Online", "welt.de"]),
    "Q819697": ("media", []),  # Handelsblatt
    "Q158322": ("media", ["=STERN", "stern.de"]),
    "Q10189": ("media", ["Tagesspiegel"]),
    "Q326068": ("media", []),  # Frankfurter Rundschau
    "Q10185": ("media", []),  # Berliner Zeitung
    "Q658945": ("media", ["NZZ"]),
    "Q157948": ("media", []),  # Junge Freiheit
    "Q15803741": ("media", ["@Compact"]),
    "Q781987": ("media", ["@Cicero"]),
    "Q27902651": ("media", []),  # Tichys Einblick
    "Q312653": ("media", ["=dpa", "=DPA"]),
    "Q20181258": ("media", []),  # Correctiv
    "Q1601295": ("media", ["heise"]),
    "Q590940": ("media", ["t-online.de"]),
}

# TV shows: ORG entities "TV-Sendung (<broadcaster>)". Shows named after their presenter only link
# with a TV context word ("Sendung", "Talkshow", ...) or as "bei Lanz"; otherwise the person wins.
SHOWS: dict[str, tuple[str, str, list[str]]] = {
    "Q703907": ("Tagesschau", "Das Erste", ["tagesschau"]),
    "Q19916": ("Tagesthemen", "Das Erste", []),
    "Q885345": ("heute journal", "ZDF", ["heute-journal"]),
    "Q1499182": ("Markus Lanz", "ZDF", ["@Markus Lanz", "@Lanz", "bei Lanz", "bei Markus Lanz"]),
    "Q1330010": ("maischberger", "Das Erste", ["@Maischberger", "bei Maischberger", "bei Sandra Maischberger"]),
    "Q1587023": ("hart aber fair", "Das Erste", ["@hart aber fair", "@Hart aber fair", "bei hart aber fair"]),
    "Q122961070": ("Caren Miosga", "Das Erste", ["@Caren Miosga", "@Miosga", "bei Miosga", "bei Caren Miosga"]),
    "Q1914407": ("Maybrit Illner", "ZDF", ["@Maybrit Illner", "@Illner", "bei Illner", "bei Maybrit Illner"]),
    "Q689438": ("Tatort", "Das Erste", ["@Tatort", "Tatort-Krimi"]),
    "Q698522": ("Polizeiruf 110", "Das Erste", []),
    "Q546099": ("Germany’s Next Topmodel", "ProSieben", ["Germany's Next Topmodel", "=GNTM"]),
    "Q79715": ("Wer wird Millionär?", "RTL", ["Wer wird Millionär", "=WWM"]),
    "Q875331": ("Let’s Dance", "RTL", ["@Let's Dance", "@Let’s Dance"]),
    "Q885167": ("Ich bin ein Star – Holt mich hier raus!", "RTL", ["Dschungelcamp", "Ich bin ein Star"]),
    "Q20827221": ("Die Höhle der Löwen", "VOX", ["Höhle der Löwen", "=DHDL"]),
    "Q265519": ("heute-show", "ZDF", ["heute show"]),
    "Q98826340": ("ZDF Magazin Royale", "ZDF", ["Magazin Royale"]),
    "Q33059": ("Sportschau", "Das Erste", []),
    "Q43632": ("das aktuelle sportstudio", "ZDF", ["aktuelle Sportstudio", "aktuelles Sportstudio"]),
    "Q622314": ("The Voice of Germany", "ProSieben/Sat.1", []),
    "Q22236152": ("Bauer sucht Frau", "RTL", ["@Bauer sucht Frau"]),
    "Q695478": ("Gute Zeiten, schlechte Zeiten", "RTL", ["=GZSZ"]),
    "Q1234602": ("Monitor", "Das Erste", ["@Monitor"]),
    "Q1713244": ("Panorama", "Das Erste", ["@Panorama"]),
    "Q1470117": ("Frontal", "ZDF", ["@Frontal", "Frontal 21", "Frontal21"]),
    "Q2145103": ("Report Mainz", "Das Erste", []),
    "Q151788": ("Die Sendung mit der Maus", "Das Erste", ["Sendung mit der Maus"]),
    "Q1783282": ("Kontraste", "Das Erste", ["@Kontraste"]),
}

# names that are everyday phrases ("Die Welt ist schlecht", "Die Zeit drängt"): media context only
CONTEXT_ONLY_NAMES = {"Q29872", "Q157142", "Q1959672", "Q158322"}

ATHLETE_MIN_SITELINKS = 20  # sports bots give athletes many sitelinks
PUBLIC_MIN_SITELINKS = 10
PRESS_MIN_SITELINKS = 5  # TV presenters and journalists
PRESS_OCCS = {"Q947873", "Q1930187", "Q13590141", "Q2722764", "Q1371925"}
BUNDESLIGA = "Q82595"


QUERIES = {
    "mdb21": f"""SELECT DISTINCT ?p WHERE {{ ?p p:P39 ?st . ?st ps:P39 wd:Q1939555 ; pq:P2937 wd:{BUNDESTAG_21} .
        FILTER NOT EXISTS {{ ?st pq:P582 ?end }} }}""",
    "mep_de": f"""SELECT DISTINCT ?p WHERE {{ ?p p:P39 ?st . ?st ps:P39 wd:Q27169 ; pq:P580 ?s .
        FILTER(?s >= "2024-07-01T00:00:00Z"^^xsd:dateTime) FILTER NOT EXISTS {{ ?st pq:P582 ?end }}
        ?p wdt:P27 wd:Q183 . {ALIVE} }}""",
    # federal and state offices (not parliament memberships); current and former
    "offices2": f"""SELECT ?p ?pos ?posLabel ?jur ?jurLabel ?start ?end WHERE {{
        ?p p:P39 ?st . ?st ps:P39 ?pos . ?pos wdt:P1001 ?jur .
        {{ VALUES ?jur {{ wd:Q183 }} }} UNION {{ ?jur wdt:P31 wd:Q1221156 }}
        FILTER NOT EXISTS {{ ?pos wdt:P279* wd:Q486839 }}
        FILTER(?pos != wd:Q1939555)
        OPTIONAL {{ ?st pq:P582 ?end }} OPTIONAL {{ ?st pq:P580 ?start }}
        ?p wdt:P31 wd:Q5 . {ALIVE}
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "de,en". }} }}""",
    "party_leaders": """SELECT DISTINCT ?p ?org ?orgLabel WHERE {
        ?org p:P488 ?st . ?st ps:P488 ?p . FILTER NOT EXISTS { ?st pq:P582 ?end }
        { ?org wdt:P31 wd:Q7278 ; wdt:P17 wd:Q183 } UNION
        { ?org wdt:P749 ?party . ?party wdt:P31 wd:Q7278 ; wdt:P17 wd:Q183 }
        ?p wdt:P31 wd:Q5 .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "de,en". } }""",
    "notable_politicians": """SELECT DISTINCT ?p WHERE {
        ?p wdt:P27 wd:Q183 ; wdt:P106 wd:Q82955 ; wikibase:sitelinks ?sl ; wdt:P569 ?dob .
        FILTER(?sl >= 15) FILTER(YEAR(?dob) > 1920)
        OPTIONAL { ?p wdt:P570 ?dod } FILTER(!BOUND(?dod) || YEAR(?dod) >= 2000) }""",
    "index_members": """SELECT DISTINCT ?c ?idx WHERE {
        VALUES ?idx { wd:Q155718 wd:Q595622 wd:Q448445 wd:Q378967 }
        { ?c p:P361 ?st . ?st ps:P361 ?idx . FILTER NOT EXISTS { ?st pq:P582 ?end } }
        UNION { ?idx p:P527 ?st2 . ?st2 ps:P527 ?c . FILTER NOT EXISTS { ?st2 pq:P582 ?end2 } } }""",
    "companies_employees": """SELECT DISTINCT ?c WHERE {
        ?c wdt:P1128 ?emp . FILTER(?emp > 10000) ?c wdt:P17 wd:Q183 .
        FILTER NOT EXISTS { ?c wdt:P576 ?dissolved } FILTER NOT EXISTS { ?c wdt:P31 wd:Q5 } }""",
    # well-known German brands/retailers/media that the generic queries miss
    "brands": """SELECT DISTINCT ?c WHERE {
        VALUES ?name { "Aldi Nord"@de "Aldi Süd"@de "Robert Bosch GmbH"@de "Sparkassen-Finanzgruppe"@de "Kaufland"@de
                       "Penny"@de "Netto Marken-Discount"@de "Dirk Rossmann GmbH"@de "MediaMarkt"@de "Bertelsmann"@de
                       "ProSiebenSat.1 Media"@de "RTL Deutschland"@de "Schwarz Gruppe"@de "Rewe Group"@de
                       "Hornbach"@de "Obi"@de "Deutsche Post"@de "Otto Group"@de "Flixbus"@de "Burda"@de }
        ?c rdfs:label ?name . FILTER NOT EXISTS { ?c wdt:P31 wd:Q4167410 } }""",
    # politicians: lower threshold, Landtag members, parliamentary group leaders
    "notable_politicians10": """SELECT DISTINCT ?p WHERE {
        ?p wdt:P27 wd:Q183 ; wdt:P106 wd:Q82955 ; wikibase:sitelinks ?sl ; wdt:P569 ?dob .
        FILTER(?sl >= 10) FILTER(YEAR(?dob) > 1920)
        OPTIONAL { ?p wdt:P570 ?dod } FILTER(!BOUND(?dod) || YEAR(?dod) >= 2000) }""",
    "landtag_members": """SELECT DISTINCT ?p ?l WHERE {
        ?jur wdt:P31 wd:Q1221156 . ?pos wdt:P1001 ?jur .
        ?pos rdfs:label ?l . FILTER(LANG(?l) = "de" && STRSTARTS(?l, "Mitglied"))
        ?p p:P39 ?st . ?st ps:P39 ?pos . FILTER NOT EXISTS { ?st pq:P582 ?end }
        OPTIONAL { ?st pq:P580 ?s } OPTIONAL { ?st pq:P2937 ?term . ?term wdt:P580 ?ts }
        FILTER((BOUND(?s) && ?s >= "2016-01-01T00:00:00Z"^^xsd:dateTime)
               || (BOUND(?ts) && ?ts >= "2016-01-01T00:00:00Z"^^xsd:dateTime))
        ?p wikibase:sitelinks ?sl . FILTER(?sl >= 5) FILTER NOT EXISTS { ?p wdt:P570 ?dod } }""",
    "faction_leaders": """SELECT DISTINCT ?p ?gLabel WHERE {
        ?g wdt:P31/wdt:P279* wd:Q848197 ; wdt:P17 wd:Q183 ; p:P488 ?st . ?st ps:P488 ?p .
        FILTER NOT EXISTS { ?st pq:P582 ?end } ?p wdt:P31 wd:Q5 .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "de". } }""",
    # public persons: living German citizens (thresholds by occupation are applied in code)
    "public_persons": """SELECT ?p ?sl (GROUP_CONCAT(DISTINCT STRAFTER(STR(?occ), "entity/"); separator="|") AS ?occs)
        WHERE { ?p wdt:P27 wd:Q183 ; wdt:P31 wd:Q5 ; wikibase:sitelinks ?sl . FILTER(?sl >= 5)
        FILTER NOT EXISTS { ?p wdt:P570 ?d } OPTIONAL { ?p wdt:P106 ?occ } } GROUP BY ?p ?sl""",
    # Bundesliga players who joined a club since 2022 (and have not left), and current head coaches
    "bundesliga_players": f"""SELECT DISTINCT ?p WHERE {{ ?club wdt:P118 wd:{BUNDESLIGA} .
        ?p p:P54 ?st . ?st ps:P54 ?club ; pq:P580 ?s . FILTER(?s >= "2022-01-01T00:00:00Z"^^xsd:dateTime)
        FILTER NOT EXISTS {{ ?st pq:P582 ?e }} ?p wikibase:sitelinks ?sl . FILTER(?sl >= 20) }}""",
    "bundesliga_coaches": f"""SELECT DISTINCT ?p WHERE {{ ?club wdt:P118 wd:{BUNDESLIGA} ; p:P286 ?st .
        ?st ps:P286 ?p . FILTER NOT EXISTS {{ ?st pq:P582 ?e }} }}""",
    "dax_ceos": """SELECT DISTINCT ?p ?cLabel WHERE {
        VALUES ?idx { wd:Q155718 wd:Q595622 }
        { ?c p:P361 ?s0 . ?s0 ps:P361 ?idx . FILTER NOT EXISTS { ?s0 pq:P582 ?e0 } }
        UNION { ?idx p:P527 ?s1 . ?s1 ps:P527 ?c . FILTER NOT EXISTS { ?s1 pq:P582 ?e1 } }
        ?c p:P169 ?st . ?st ps:P169 ?p . FILTER NOT EXISTS { ?st pq:P582 ?end } ?p wdt:P31 wd:Q5 .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "de". } }""",
    "public_offices": """SELECT DISTINCT ?p ?l WHERE {
        VALUES ?l { "Richter des Bundesverfassungsgerichts"@de "Richterin des Bundesverfassungsgerichts"@de
                    "Präsident des Bundesverfassungsgerichts"@de "Vizepräsident des Bundesverfassungsgerichts"@de
                    "Ratsvorsitzender der Evangelischen Kirche in Deutschland"@de
                    "Vorsitzender der Deutschen Bischofskonferenz"@de "Präsident des Robert Koch-Instituts"@de
                    "Präsident der Deutschen Bundesbank"@de }
        ?pos rdfs:label ?l . ?p p:P39 ?st . ?st ps:P39 ?pos . FILTER NOT EXISTS { ?st pq:P582 ?end }
        FILTER NOT EXISTS { ?p wdt:P570 ?d } }""",
    **{
        f"companies8_{cls}": f"""SELECT DISTINCT ?c WHERE {{
        ?c wdt:P31 wd:{cls} ; wdt:P17 wd:Q183 ; wikibase:sitelinks ?sl . FILTER(?sl >= 8)
        FILTER NOT EXISTS {{ ?c wdt:P576 ?dissolved }} }}"""
        for cls in COMPANY_CLASSES
    },
    "universities": """SELECT DISTINCT ?u ?cls ?sl WHERE {
        ?u wdt:P31 ?cls . ?cls wdt:P279* wd:Q3918 . ?u wdt:P17 wd:Q183 ; wikibase:sitelinks ?sl .
        FILTER NOT EXISTS { ?u wdt:P576 ?dissolved } }""",
}

BUSINESS = re.compile(
    r"unternehmen|konzern|hersteller|kette|gruppe|händler|bank|kreditinstitut|versicher|dienstleist|holding|"
    r"zulieferer|logistik|marke|discounter|drogerie|markt|anbieter|spedition|firma|reisekonzern|verkehrsbetrieb"
)
NOT_BUSINESS = re.compile(
    r"ehemalig|kirch|katholisch|diakon|caritas|stiftung|wohlfahrt|bundeswehr|polizei|behörde|ministerium|"
    r"bundesamt|bundesanstalt|landesamt|verwaltung"
)
# acronyms of companies/universities only for well-known ones: "EUV" (Viadrina) is mostly the
# lithography, "MBS" (Munich Business School) mostly Mohammed bin Salman
ACRONYM_MIN_SITELINKS = 30
# company names that are ordinary words in comments ("3000 Netto", "beim Globus", "der Puma")
AMBIGUOUS_WORDS = {
    "netto",
    "globus",
    "puma",
    "penny",
    "otto",
    "real",
    "plus",
    "merkur",
    "adler",
    "bauhaus",
    "continental",
    "signal",
    "phoenix",
    "union",
    "allianz",
    "hansa",
    "condor",
    "aurora",
    "atlas",
    "vision",
    "concordia",
    "basler",
    "aida",
    "smart",
    "norma",
}
# ... and some that are almost never the company ("mehr Netto vom Brutto")
DROP_WORDS = {"netto"}

DETAILS = """SELECT ?i ?labelDe ?labelEn ?desc ?sl ?gender ?party
    (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases)
    (GROUP_CONCAT(DISTINCT ?short; separator="|") AS ?shorts)
    (GROUP_CONCAT(DISTINCT STRAFTER(STR(?cls), "entity/"); separator="|") AS ?classes) WHERE {
  VALUES ?i { %s }
  ?i wikibase:sitelinks ?sl .
  OPTIONAL { ?i rdfs:label ?labelDe FILTER(LANG(?labelDe) = "de") }
  OPTIONAL { ?i rdfs:label ?labelEn FILTER(LANG(?labelEn) = "en") }
  OPTIONAL { ?i schema:description ?desc FILTER(LANG(?desc) = "de") }
  OPTIONAL { ?i skos:altLabel ?alias FILTER(LANG(?alias) = "de") }
  OPTIONAL { ?i wdt:P1813 ?short FILTER(LANG(?short) = "de" || LANG(?short) = "en" || LANG(?short) = "mul") }
  OPTIONAL { ?i wdt:P21 ?gender }
  OPTIONAL { ?i p:P102 ?pst . ?pst ps:P102 ?party . FILTER NOT EXISTS { ?pst pq:P582 ?pend } }
  OPTIONAL { ?i wdt:P31 ?cls }
} GROUP BY ?i ?labelDe ?labelEn ?desc ?sl ?gender ?party"""


def details(qids: list[str], refresh: bool) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for k in range(0, len(qids), 200):
        chunk = qids[k : k + 200]
        key = hashlib.sha1(" ".join(chunk).encode()).hexdigest()[:12]
        rows = sparql(f"details_{key}", DETAILS % " ".join(f"wd:{x}" for x in chunk), refresh)
        for r in rows:
            d = out.setdefault(q(r["i"]), {"parties": set()})
            for f in ("labelDe", "labelEn", "desc", "gender", "aliases", "shorts", "classes"):
                if r.get(f) and not d.get(f):
                    d[f] = r[f]
            d["sl"] = int(r.get("sl", 0))
            if r.get("party"):
                d["parties"].add(q(r["party"]))
    return out


def load_seed() -> tuple[dict[str, dict], set[str], set[str]]:
    data = yaml.safe_load(SEED.read_text(encoding="utf-8"))
    by_qid = {e["wikidata"]: e for e in data["entities"] if e.get("wikidata")}
    ids = {e["id"] for e in data["entities"]}
    aliases = {
        normalize_surface(parse_alias(a)[0]) for e in data["entities"] for a in [e["name"], *e.get("aliases", [])]
    }
    return by_qid, ids, aliases


LEGAL = re.compile(
    r"\s*(,?\s*(AG|SE|GmbH|KGaA|AG & Co\.? KGaA|SE & Co\.? KGaA|& Co\.? KG|KG|e\.\s?V\.|plc|Inc\.?|N\.V\.))+$"
)


def word_freq(w: str, lang: str = "de") -> float:
    try:
        from wordfreq import zipf_frequency
    except ImportError:  # pragma: no cover
        return 0.0
    return zipf_frequency(w.lower(), lang)


def athlete_occupations(occs: list[str], refresh: bool) -> set[str]:
    """Occupations that are athletes or sports coaches (subclasses of athlete / coach)."""
    key = hashlib.sha1(" ".join(occs).encode()).hexdigest()[:12]
    rows = sparql(
        f"athlete_occs_{key}",
        "SELECT DISTINCT ?o WHERE { VALUES ?o { "
        + " ".join(f"wd:{o}" for o in occs)
        + " } { ?o wdt:P279* wd:Q2066131 } UNION { ?o wdt:P279* wd:Q41583 } }",
        refresh,
    )
    return {q(r["o"]) for r in rows} | {"Q2066131", "Q41583", "Q628099"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    seed_by_qid, seed_ids, seed_aliases = load_seed()
    res = {name: sparql(name, query, args.refresh) for name, query in QUERIES.items()}

    groups: dict[str, set[str]] = defaultdict(set)
    roles: dict[str, list[str]] = defaultdict(list)  # qid -> role descriptions (most specific first)
    for r in res["mdb21"]:
        groups["Bundestag (21. WP)"].add(q(r["p"]))
        roles[q(r["p"])].append("MDB")
    for r in res["mep_de"]:
        groups["Europaabgeordnete (DE)"].add(q(r["p"]))
        roles[q(r["p"])].append("MEP")
    for r in res["offices2"]:
        p, pos = q(r["p"]), r.get("posLabel", "")
        # "current" = no end date and started 2017 or later (old statements often lack end dates)
        current = "end" not in r and (r.get("start", "")[:4] >= "2017" or "start" not in r)
        head = re.match(r"(Ministerpräsident|Regierender Bürgermeister|Erster Bürgermeister|Präsident des Senats)", pos)
        federal = q(r["jur"]) == "Q183"
        if current and OFFICE.search(pos) and not re.match(r"(Stellvertretendes )?Mitglied", pos):
            groups["Amtsträger:innen Bund/Länder (aktuell)"].add(p)
            roles[p].insert(0, re.sub(r"\s*\(\d{4}[–-]\d{0,4}\)", "", pos))
        elif (federal and re.search(r"(Bundeskanzler|Bundesminister|Bundespräsident)", pos)) or head:
            groups["ehemalige Kanzler, Bundesminister, Regierungschefs"].add(p)
            roles[p].append("ehem. " + pos)
    for r in res["party_leaders"]:
        groups["Partei-Vorsitzende (Bund/Land)"].add(q(r["p"]))
        roles[q(r["p"])].append("Vorsitz " + r.get("orgLabel", ""))
    for r in res["notable_politicians"]:
        groups["Politiker:innen mit >= 15 Sitelinks"].add(q(r["p"]))
    for r in res["notable_politicians10"]:
        groups["Politiker:innen mit >= 10 Sitelinks"].add(q(r["p"]))
    for r in res["landtag_members"]:
        groups["Landtagsabgeordnete (>= 5 Sitelinks)"].add(q(r["p"]))
        roles[q(r["p"])].append("MDL")
    for r in res["faction_leaders"]:
        groups["Fraktionsvorsitzende"].add(q(r["p"]))
        roles[q(r["p"])].insert(0, "Vorsitz " + r.get("gLabel", ""))
    politicians = set().union(*groups.values())

    # public persons (not politicians): thresholds by occupation
    pgroups: dict[str, set[str]] = defaultdict(set)
    pub_rows = res["public_persons"]
    occ_all = sorted({o for r in pub_rows for o in (r.get("occs") or "").split("|") if o})
    ath = athlete_occupations(occ_all, args.refresh)
    for r in pub_rows:
        p, sl = q(r["p"]), int(r["sl"])
        occs = set((r.get("occs") or "").split("|"))
        if p in politicians:
            continue
        if occs & ath:
            if sl >= ATHLETE_MIN_SITELINKS:
                pgroups["Sportler:innen (>= 20 Sitelinks)"].add(p)
        elif occs & PRESS_OCCS and sl >= PRESS_MIN_SITELINKS:
            pgroups["Moderation/Journalismus (>= 5 Sitelinks)"].add(p)
        elif sl >= PUBLIC_MIN_SITELINKS:
            pgroups["weitere öffentliche Personen (>= 10 Sitelinks)"].add(p)
    for r in res["bundesliga_players"]:
        pgroups["Bundesliga (aktuelle Spieler, Trainer)"].add(q(r["p"]))
    for r in res["bundesliga_coaches"]:
        pgroups["Bundesliga (aktuelle Spieler, Trainer)"].add(q(r["p"]))
    for r in res["dax_ceos"]:
        pgroups["Vorstandsvorsitzende (DAX/MDAX)"].add(q(r["p"]))
        roles[q(r["p"])].insert(0, "Vorstandsvorsitz " + LEGAL.sub("", r.get("cLabel", "")).strip())
    for r in res["public_offices"]:
        pgroups["BVerfG, Kirchen, Bundesbank, RKI"].add(q(r["p"]))
        roles[q(r["p"])].insert(0, r.get("l", ""))
    public = set().union(*pgroups.values()) - politicians
    persons = politicians | public

    cgroups: dict[str, set[str]] = defaultdict(set)
    index_of: dict[str, list[str]] = defaultdict(list)
    for r in res["index_members"]:
        cgroups["Indexwerte (DAX/MDAX/SDAX/TecDAX)"].add(q(r["c"]))
        index_of[q(r["c"])].append(INDICES[q(r["idx"])])
    for r in res["brands"]:
        cgroups["bekannte Marken (Liste)"].add(q(r["c"]))
    for r in res["companies_employees"]:
        cgroups["Unternehmen > 10.000 Beschäftigte"].add(q(r["c"]))
    for name, rows in res.items():
        if name.startswith("companies8_"):
            for r in rows:
                cgroups["Unternehmen/Marken (>= 8 Sitelinks)"].add(q(r["c"]))
    cgroups["Tech & Medien (Liste)"] = set(MEDIA_TECH)
    companies = set().union(*cgroups.values())
    unis_all: dict[str, dict] = {}
    for r in res["universities"]:
        d = unis_all.setdefault(q(r["u"]), {"classes": set(), "sl": int(r["sl"])})
        d["classes"].add(q(r["cls"]))
    universities = {u for u, d in unis_all.items() if not (d["classes"] & FH_CLASSES) or d["sl"] >= 10}

    brand_only = cgroups["bekannte Marken (Liste)"] - set().union(
        *(v for k, v in cgroups.items() if k != "bekannte Marken (Liste)")
    )
    # employee-count-only items: also public bodies (Bundeswehr, Polizei, Kirchen, ...)
    emp_only = cgroups["Unternehmen > 10.000 Beschäftigte"] - set().union(
        *(v for k, v in cgroups.items() if k != "Unternehmen > 10.000 Beschäftigte")
    )
    allq = sorted(persons | companies | universities)
    det = details(allq, args.refresh)
    # party labels
    party_q = sorted({p for d in det.values() for p in d["parties"]})
    party_rows = sparql(
        "party_labels",
        "SELECT ?i ?l ?s WHERE { VALUES ?i { "
        + " ".join(f"wd:{x}" for x in party_q)
        + " } OPTIONAL { ?i rdfs:label ?l FILTER(LANG(?l)='de') } OPTIONAL { ?i wdt:P1813 ?s } }",
        args.refresh,
    )
    party_name: dict[str, str] = {}
    for r in party_rows:
        pq = q(r["i"])
        seed = seed_by_qid.get(pq)
        party_name.setdefault(pq, seed["name"] if seed else (r.get("s") or r.get("l") or ""))

    # ---------------------------------------------------------------- build entities
    entries: list[dict] = []
    stats = Counter()
    used_ids = set(seed_ids)
    surname_count = Counter()
    first_names = set()
    for qid in persons:
        lab = det.get(qid, {}).get("labelDe") or det.get(qid, {}).get("labelEn") or ""
        if lab:
            surname_count[normalize_surface(lab.split()[-1])] += 1
            first_names.add(normalize_surface(lab.split()[0]))
    for qid in sorted(allq, key=lambda x: -det.get(x, {}).get("sl", 0)):
        if qid in seed_by_qid:
            stats["skipped: curated"] += 1
            continue
        d = det.get(qid)
        if not d:
            continue
        name = d.get("labelDe") or d.get("labelEn")
        if not name or re.match(r"^Q\d+$", name):
            continue
        name = re.sub(r"\s*\([^)]*\)$", "", name).strip()  # "Rewe (Markt)" -> "Rewe"
        if normalize_surface(name) in seed_aliases:
            # "Mercedes-Benz AG" -> "Mercedes-Benz": the curated entity already owns this name
            stats["skipped: name is a curated alias"] += 1
            continue
        if name.isupper() and len(name) >= 4 and " " not in name:  # "EDEKA" -> also "Edeka"
            extra_case = name.capitalize()
        else:
            extra_case = None
        is_person = qid in persons
        if qid in brand_only and (
            not re.search(
                r"(unternehmen|kette|konzern|discounter|gruppe|händler|verbund|bank|markt|hersteller)",
                (d.get("desc") or "").lower(),
            )
            or "DDR" in (d.get("desc") or "")
        ):
            stats["skipped: brand label not a company"] += 1
            continue
        desc_l = (d.get("desc") or "").lower()
        listed = qid in index_of or qid in cgroups["bekannte Marken (Liste)"] or qid in MEDIA_TECH
        if not is_person and qid not in universities and not listed and NOT_BUSINESS.search(desc_l):
            # ministries, authorities, churches ... are not companies
            stats["skipped: public body / non-profit"] += 1
            continue
        if qid in emp_only and qid not in universities and (not BUSINESS.search(desc_l) or NOT_BUSINESS.search(desc_l)):
            # employee count alone also lists Bundeswehr, Polizei, Kirchen, Verwaltungen, Kliniken ...
            stats["skipped: >10k employees but not a company"] += 1
            continue
        etype = "PER" if is_person else "ORG"
        aliases: list[str] = [name]
        extra = [a for a in (d.get("aliases") or "").split("|") if a] + [
            s for s in (d.get("shorts") or "").split("|") if s
        ]
        if is_person:
            for a in extra:
                if len(a.split()) >= 2 and "(" not in a:
                    aliases.append(a)
            toks = name.split()
            sur = toks[-1]
            key = normalize_surface(sur)
            if (
                len(toks) >= 2
                and len(sur) >= 4
                and surname_count[key] == 1
                and key not in seed_aliases
                and key not in COMMON_SURNAMES
                and word_freq(sur) < 4.0
                and word_freq(sur, "en") < 2.5  # "Thiel", "Kant", "Moser", "Maas": also famous elsewhere
                and key not in first_names  # "Clemens", "Paulus"
            ):
                aliases.append("!" + sur)
                stats["surname aliases (NER-only)"] += 1
            # prefer a current party that is a curated entity (e.g. Die Linke over a historic PDS statement)
            ps = sorted(d["parties"], key=lambda p: (p not in seed_by_qid, p))
            party = next((party_name[p] for p in ps if party_name.get(p)), "")
            female = d.get("gender", "").endswith(FEMALE)
            role = next(iter(roles.get(qid, [])), "")
            if role == "MDB":
                desc = ("Bundestagsabgeordnete" if female else "Bundestagsabgeordneter") + (
                    f" ({party})" if party else ""
                )
            elif role == "MEP":
                desc = ("Europaabgeordnete" if female else "Europaabgeordneter") + (f" ({party})" if party else "")
            elif role == "MDL":
                desc = ("Landtagsabgeordnete" if female else "Landtagsabgeordneter") + (f" ({party})" if party else "")
            elif role and qid in politicians:
                desc = role + (f" ({party})" if party and party not in role else "")
            elif role:
                desc = role
            elif qid in politicians:
                desc = d.get("desc") or "Politiker"
            else:
                desc = d.get("desc") or "Person des öffentlichen Lebens"
        else:
            core = LEGAL.sub("", name).strip()
            if core and core != name:
                aliases.append(core)
            if extra_case:
                aliases.append(extra_case)
            for a in extra:
                aliases.append(a)
            if qid in universities:
                desc = d.get("desc") or "Hochschule"
            else:
                idx = "/".join(sorted(set(index_of.get(qid, []))))
                desc = (d.get("desc") or "Unternehmen") + (f" ({idx})" if idx else "")
        if qid in CONTEXT_ONLY_NAMES:
            aliases = ["@" + a if (a[:1] not in ("~", "!", "@", "=") and " " in a) or a == name else a for a in aliases]
        # alias policy
        final: list[str] = []
        for a in dict.fromkeys(aliases):
            raw = a[1:] if a[:1] in ("~", "!", "@") else a
            key = normalize_surface(raw)
            if key in seed_aliases or len(key) < 2:
                continue
            if a[:1] in ("~", "!", "@"):
                final.append(a)
            elif is_person and " " not in raw.strip():
                # one-word names (artists: "Nena", "Sido", "Cro"): case-sensitive; NER-only if a word
                zf = max(word_freq(raw), word_freq(raw, "en"))
                if len(raw) < 3 or normalize_surface(raw) in first_names or word_freq(raw) >= 3.5:
                    # "Sandra", "Nicole", "Ben" (first names), "Haftbefehl", "Apparat" (words)
                    stats["dropped person aliases"] += 1
                elif zf >= 2.5:
                    final.append("!" + raw)
                else:
                    final.append("=" + raw)
            elif is_person:
                final.append(raw)
            elif " " not in raw.strip():
                zf = word_freq(raw)
                if (
                    len(raw) < 3
                    or (zf >= 4.5 and raw != name)
                    or (raw.isupper() and raw != name and d.get("sl", 0) < ACRONYM_MIN_SITELINKS)
                ):
                    # "CL", "Deutsche", "EUV": too short / too frequent / too obscure to be reliable
                    stats["dropped company aliases"] += 1
                elif raw.casefold() in DROP_WORDS and raw != name:
                    stats["dropped company aliases"] += 1
                elif raw.casefold() in AMBIGUOUS_WORDS:
                    # "3000 Netto", "beim Globus": only with a business keyword nearby
                    final.append("@" + raw)
                    stats["contextual company aliases"] += 1
                elif raw.isupper() or zf >= 4.0:
                    # acronyms (EWG, KIT) and common words (Post, Bahn): need NER ORG or context
                    final.append("~" + raw)
                    stats["weak company aliases"] += 1
                elif zf >= 2.5:
                    # a word in its own right ("rational"): case-sensitive
                    final.append("=" + raw)
                else:
                    final.append(raw)
            elif all(word_freq(t) >= 3.0 for t in re.findall(r"[\w]+", raw)):
                final.append("=" + raw)  # "Das Erste", "Deutsche Marine": case-sensitive
            else:
                final.append(raw)
        for a in MEDIA_TECH.get(qid, ("", []))[1]:  # hand-written aliases (WDR, FAZ, ntv, ...)
            if normalize_surface(parse_alias(a)[0]) not in seed_aliases and a not in final:
                final.append(a)
        if not final:
            continue
        eid = slugify(name)
        if eid in used_ids:
            eid = f"{eid}-{qid.lower()}"
        used_ids.add(eid)
        entry = {
            "id": eid,
            "wikidata": qid,
            "name": name,
            "type": etype,
            "country": "DE",
            "description": desc[:120],
            "aliases": final[:12],
        }
        if qid in MEDIA_TECH:
            entry["context"] = MEDIA_TECH[qid][0]
        elif is_person:
            if qid in politicians:
                entry["context"] = "politics"
        else:
            entry["context"] = "education" if qid in universities else "business"
        entries.append(entry)
        stats[
            ("politician" if qid in politicians else "public person")
            if is_person
            else ("university" if qid in universities else ("media/tech" if qid in MEDIA_TECH else "company"))
        ] += 1
    # an alias shared by several generated entities is ambiguous: keep it only for the most notable one
    owner: dict[str, str] = {}
    for e in entries:  # already sorted by sitelinks desc
        keep = []
        for a in e["aliases"]:
            key = normalize_surface(parse_alias(a)[0])
            if key in owner and owner[key] != e["id"]:
                continue
            owner[key] = e["id"]
            keep.append(a)
        e["aliases"] = keep
    # a second entity with exactly the same name (two "Michael Müller") would get its name back as
    # a plain alias in the registry and steal mentions: drop it, the more notable namesake wins
    named = []
    for e in entries:
        nkey = normalize_surface(e["name"])
        own = {normalize_surface(parse_alias(a)[0]) for a in e["aliases"]}
        if nkey not in own and owner.get(nkey, e["id"]) != e["id"]:
            stats["skipped: namesake of a more notable entity"] += 1
            continue
        if e["aliases"]:
            named.append(e)
    entries = named
    # TV shows (after the dedupe: "Markus Lanz" the show shares its name with the presenter)
    taken = {e["id"] for e in entries} | used_ids
    for qid, (name, broadcaster, extra) in SHOWS.items():
        if qid in seed_by_qid:
            continue
        aliases = [a for a in extra if normalize_surface(parse_alias(a)[0]) not in seed_aliases]
        if normalize_surface(name) not in {normalize_surface(parse_alias(a)[0]) for a in aliases}:
            aliases.insert(0, name)
        eid = slugify(name)
        if eid in taken:
            eid = f"{eid}-sendung"
        taken.add(eid)
        entries.append(
            {
                "id": eid,
                "wikidata": qid,
                "name": name,
                "type": "ORG",
                "country": "DE",
                "description": f"TV-Sendung ({broadcaster})",
                "aliases": aliases,
                "context": "tv",
            }
        )
        stats["tv show"] += 1
    header = (
        "# GENERATED by scripts/generate_entities.py from Wikidata - do not edit by hand.\n"
        "# Curated entities (data/entities.yaml) take precedence: same QIDs and their aliases are skipped.\n"
    )
    OUT.write_text(
        header + yaml.safe_dump({"entities": entries}, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8"
    )
    print("groups (politicians):", {k: len(v) for k, v in groups.items()})
    print("groups (public persons):", {k: len(v) for k, v in pgroups.items()})
    print(
        "groups (companies):",
        {k: len(v) for k, v in cgroups.items()},
        "universities:",
        len(universities),
        "of",
        len(unis_all),
    )
    print("generated:", dict(stats), "total", len(entries))


if __name__ == "__main__":
    main()
