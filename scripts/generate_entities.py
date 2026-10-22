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
            rows = [{k: v["value"] for k, v in b.items()} for b in r.json()["results"]["bindings"]]
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
    **{
        f"companies_notable_{cls}": f"""SELECT DISTINCT ?c WHERE {{
        ?c wdt:P31 wd:{cls} ; wdt:P17 wd:Q183 ; wikibase:sitelinks ?sl . FILTER(?sl >= 25)
        FILTER NOT EXISTS {{ ?c wdt:P576 ?dissolved }} }}"""
        for cls in COMPANY_CLASSES
    },
    # well-known German brands/retailers/media that the generic queries miss
    "brands": """SELECT DISTINCT ?c WHERE {
        VALUES ?name { "Aldi Nord"@de "Aldi Süd"@de "Robert Bosch GmbH"@de "Sparkassen-Finanzgruppe"@de "Kaufland"@de
                       "Penny"@de "Netto Marken-Discount"@de "Dirk Rossmann GmbH"@de "MediaMarkt"@de "Bertelsmann"@de
                       "ProSiebenSat.1 Media"@de "RTL Deutschland"@de "Schwarz Gruppe"@de "Rewe Group"@de
                       "Hornbach"@de "Obi"@de "Deutsche Post"@de "Otto Group"@de "Flixbus"@de "Burda"@de }
        ?c rdfs:label ?name . FILTER NOT EXISTS { ?c wdt:P31 wd:Q4167410 } }""",
    "universities": """SELECT DISTINCT ?u ?cls ?sl WHERE {
        ?u wdt:P31 ?cls . ?cls wdt:P279* wd:Q3918 . ?u wdt:P17 wd:Q183 ; wikibase:sitelinks ?sl .
        FILTER NOT EXISTS { ?u wdt:P576 ?dissolved } }""",
}

BUSINESS = re.compile(
    r"unternehmen|konzern|hersteller|kette|gruppe|händler|bank|kreditinstitut|versicher|dienstleist|holding|"
    r"zulieferer|logistik|marke|discounter|drogerie|markt|anbieter|spedition|firma|reisekonzern|verkehrsbetrieb"
)
NOT_BUSINESS = re.compile(r"ehemalig|kirch|katholisch|diakon|caritas|stiftung|wohlfahrt|bundeswehr|polizei|behörde")

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


def word_freq(w: str) -> float:
    try:
        from wordfreq import zipf_frequency
    except ImportError:  # pragma: no cover
        return 0.0
    return zipf_frequency(w.lower(), "de")


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
    persons = set().union(*groups.values())

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
        if name.startswith("companies_notable_"):
            for r in rows:
                cgroups["bekannte Unternehmen/Marken (>= 25 Sitelinks)"].add(q(r["c"]))
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
    for qid in persons:
        lab = det.get(qid, {}).get("labelDe") or det.get(qid, {}).get("labelEn") or ""
        if lab:
            surname_count[normalize_surface(lab.split()[-1])] += 1
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
            elif role:
                desc = role + (f" ({party})" if party and party not in role else "")
            else:
                desc = d.get("desc") or "Politiker"
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
        # alias policy
        final: list[str] = []
        for a in dict.fromkeys(aliases):
            raw = a[1:] if a[:1] in ("~", "!") else a
            key = normalize_surface(raw)
            if key in seed_aliases or len(key) < 2:
                continue
            if a[:1] in ("~", "!"):
                final.append(a)
            elif is_person:
                final.append(raw)
            elif " " not in raw.strip():
                zf = word_freq(raw)
                if len(raw) < 3 or (zf >= 4.5 and raw != name):
                    # "CL", "Deutsche": too short / too frequent to ever be a reliable alias
                    stats["dropped company aliases"] += 1
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
        entry["context"] = "politics" if is_person else ("education" if qid in universities else "business")
        entries.append(entry)
        stats["PER" if is_person else ("university" if qid in universities else "company")] += 1
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
    header = (
        "# GENERATED by scripts/generate_entities.py from Wikidata - do not edit by hand.\n"
        "# Curated entities (data/entities.yaml) take precedence: same QIDs and their aliases are skipped.\n"
    )
    OUT.write_text(
        header + yaml.safe_dump({"entities": entries}, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8"
    )
    print("groups (politicians):", {k: len(v) for k, v in groups.items()})
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
