"""Fill in missing `wikidata:` ids in data/entities.yaml via the Wikidata SPARQL endpoint.

For every seed entity without an id, all items whose German/English label equals the
entity name (or one of its multi-word aliases) are fetched together with their classes
and sitelink counts. Persons must be humans (Q5); others must not be humans or
disambiguation pages. Among the remaining candidates the item with the most sitelinks
wins. Choices are printed for manual review; --write inserts them into the YAML in place.

    python scripts/resolve_wikidata.py            # dry run
    python scripts/resolve_wikidata.py --write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.linker import parse_alias
from analysis.wikidata import USER_AGENT

SEED = Path(__file__).resolve().parents[1] / "data" / "entities.yaml"
SPARQL = "https://query.wikidata.org/sparql"
EXCLUDE_CLASSES = {
    "Q476028",  # the class "association football club" itself
    "Q4167410",
    "Q4167836",
    "Q13406463",
    "Q11266439",
    "Q17633526",
}  # disambig, category, list, template, article

# Manual overrides where the heuristic picks the wrong item (checked by hand).
OVERRIDES: dict[str, str] = {
    "gruene-at": "Q193178",
    "michael-ludwig": "Q1560091",
    "us-supreme-court": "Q11201",
    "ice": "Q943091",
    "verdi": "Q547951",
    "ard": "Q49653",
    "bild": "Q156203",
    "der-spiegel": "Q131478",
    "continental": "Q163241",
    "henkel": "Q276507",
    "apple": "Q312",
    "sartorius": "Q175835",
    "dhl-group": "Q157645",
    "mercedes-benz-group": "Q27530",
    "volkswagen": "Q156578",
    "1-fc-heidenheim": "Q162251",
    "holstein-kiel": "Q157828",
    "sv-darmstadt-98": "Q479351",
    "sv-elversberg": "Q692691",
    "1-fc-magdeburg": "Q155730",
    "sk-sturm-graz": "Q124007617",
    "bsw": "Q123121346",
    "neos": "Q13564543",
    "tino-chrupalla": "Q40895143",
    "hubert-aiwanger": "Q1633044",
    "kamala-harris": "Q10853588",
    "fed": "Q53536",
    "emmanuel-macron": "Q3052772",
    "jordan-bardella": "Q40655898",
    "sergej-lawrow": "Q58217",
    "ezb": "Q8901",
    "allianz": "Q487292",
    "hannover-rueck": "Q657359",
    "porsche-se": "Q2104551",
    "openai": "Q21708200",
    "thyssenkrupp": "Q137910",
    "glp": "Q545900",
    "bundesrat-schweiz": "Q30917",
    "nationalrat-schweiz": "Q676078",
    "roche": "Q212646",
    "fc-basel": "Q189671",
    "bsc-young-boys": "Q190526",
    "fc-zuerich": "Q206692",
    "dfl": "Q28062486",
    # further clubs were resolved with a football-specific search (description must mention
    # Fußball/Sportverein, women's/youth teams excluded) and checked by hand
}


def sparql(query: str) -> list[dict]:
    for attempt in range(6):
        r = requests.get(
            SPARQL,
            params={"query": query},
            headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT},
            timeout=90,
        )
        if r.status_code == 200:
            return r.json()["results"]["bindings"]
        wait = float(r.headers.get("retry-after") or 2**attempt * 5)
        print(f"sparql {r.status_code}, waiting {wait}s", file=sys.stderr)
        time.sleep(min(wait, 120))
    raise RuntimeError("SPARQL unavailable")


def lit(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def candidates(names: list[str]) -> dict[str, list[dict]]:
    values = " ".join(f"{lit(n)}@de {lit(n)}@en" for n in names)
    q = f"""
    SELECT ?name ?item ?sl (GROUP_CONCAT(DISTINCT STRAFTER(STR(?cls), "entity/"); separator=" ") AS ?classes)
           (SAMPLE(?d) AS ?desc) WHERE {{
      VALUES ?name {{ {values} }}
      ?item rdfs:label ?name ; wikibase:sitelinks ?sl .
      OPTIONAL {{ ?item wdt:P31 ?cls }}
      OPTIONAL {{ ?item schema:description ?d FILTER(LANG(?d) = "de") }}
    }} GROUP BY ?name ?item ?sl
    """
    out: dict[str, list[dict]] = {}
    for b in sparql(q):
        out.setdefault(b["name"]["value"], []).append(
            {
                "qid": b["item"]["value"].rsplit("/", 1)[1],
                "sl": int(b["sl"]["value"]),
                "classes": set(b.get("classes", {}).get("value", "").split()),
                "desc": b.get("desc", {}).get("value", ""),
            }
        )
    return out


CLUB_WORDS = ("fußball", "sportverein", "sportclub", "fußballklub", "fußballunternehmen")
COMPANY_WORDS = (
    "unternehmen",
    "konzern",
    "hersteller",
    "bank",
    "gesellschaft",
    "versicher",
    "händler",
    "holding",
    "aktiengesellschaft",
    "distributor",
    "portal",
    "marke",
    "rückversicher",
)
POLITICS_WORDS = (
    "politiker",
    "präsident",
    "minister",
    "kanzler",
    "bürgermeister",
    "gouverneur",
    "senator",
    "abgeordnete",
    "führer",
    "machthaber",
    "papst",
    "generalsekretär",
    "regierungschef",
    "vertreter",
    "sprecher",
    "landeshauptfrau",
    "jurist",
    "richter",
    "vizepräsident",
    "staatsoberhaupt",
)


def plausible(e: dict, desc: str) -> bool:
    """Type-specific sanity check of a candidate's German Wikidata description."""
    d = desc.lower()
    mine = (e.get("description") or "").lower()
    if e.get("context") == "football" and e["type"] == "ORG":
        return any(w in d for w in CLUB_WORDS)
    if e["type"] == "PARTY":
        return "partei" in d or "fraktion" in d or "zusammenschluss" in d
    if e["type"] == "PER":
        if any(
            w in mine
            for w in (
                "politik",
                "minister",
                "kanzler",
                "präsident",
                "bürgermeister",
                "gouverneur",
                "senator",
                "vorsitz",
                "fraktion",
                "sprecher",
                "führer",
                "landeshaupt",
            )
        ):
            return any(w in d for w in POLITICS_WORDS)
        return True
    if e["type"] == "ORG" and any(
        w in mine
        for w in (
            "konzern",
            "hersteller",
            "unternehmen",
            "bank",
            "versicher",
            "händler",
            "betreiber",
            "fluggesellschaft",
            "anbieter",
        )
    ):
        return any(w in d for w in COMPANY_WORDS)
    return bool(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    text = SEED.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    todo = [e for e in data["entities"] if not e.get("wikidata")]
    names_of = {}
    for e in todo:
        plain = [parse_alias(a)[0] for a in e.get("aliases", []) if parse_alias(a)[1] in ("plain", "cs")]
        extra = [e["description"]] if e["type"] in ("PARTY", "ORG") else []
        names_of[e["id"]] = list(dict.fromkeys([e["name"], *extra, *plain[:4]]))
    chosen: dict[str, str] = {}
    for i in range(0, len(todo), 20):
        chunk = todo[i : i + 20]
        cands = candidates(sorted({n for e in chunk for n in names_of[e["id"]]}))
        for e in chunk:
            if e["id"] in OVERRIDES:  # checked by hand
                chosen[e["id"]] = OVERRIDES[e["id"]]
                print(f"{e['id']:32s} {OVERRIDES[e['id']]:12s} (override)")
                continue
            pool: dict[str, dict] = {}
            for n in names_of[e["id"]]:
                for c in cands.get(n, []):
                    pool[c["qid"]] = c
            ok = []
            for c in pool.values():
                if c["classes"] & EXCLUDE_CLASSES:
                    continue
                is_human = "Q5" in c["classes"]
                if (e["type"] == "PER") != is_human:
                    continue
                if not plausible(e, c["desc"]):
                    continue
                ok.append(c)
            ok.sort(key=lambda c: -c["sl"])
            if ok:
                best = ok[0]
                chosen[e["id"]] = best["qid"]
                alt = f" (next: {ok[1]['qid']} sl={ok[1]['sl']})" if len(ok) > 1 else ""
                print(f"{e['id']:32s} {best['qid']:12s} sl={best['sl']:<4d} {best['desc'][:70]}{alt}")
            else:
                print(f"{e['id']:32s} -            no candidate")
        time.sleep(1)
    if args.write:
        for eid, qid in chosen.items():
            pat = re.compile(r"\{id: " + re.escape(eid) + r", ")
            text, n = pat.subn(lambda m, q=qid: m.group(0) + f"wikidata: {q}, ", text, count=1)
            assert n == 1, eid
        SEED.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
