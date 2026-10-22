"""Linking precision on the hand-labelled gold pairs, seed-only vs. seed + generated registry.

Reads the local (gitignored) eval/sample*.local.jsonl texts and eval/labels.csv (linkOk).
Prints aggregate numbers and writes the new (generated-entity) matches to
eval/new_matches.local.jsonl for a manual spot check. No comment text is printed.

    python scripts/eval_linking.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze import MIN_NER_CHARS, link_comment  # noqa: E402
from analysis.linker import Registry  # noqa: E402
from analysis.ner import load_ner  # noqa: E402
from analysis.textutil import clean_text, split_sentences  # noqa: E402

EVAL = ROOT / "eval"
COUNTRY = {"derstandard": "AT", "blick": "CH", "nzz": "CH"}


def main() -> None:
    labels = {r["mentionId"]: r for r in csv.DictReader((EVAL / "labels.csv").open(encoding="utf-8"))}
    rows = {}
    for f in sorted(EVAL.glob("sample*.local.jsonl")):
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            rows[r["mentionId"]] = r
    comments: dict[str, dict] = {}
    for mid, r in rows.items():
        cid = mid.split("_", 1)[0]
        src = r.get("source") or (labels.get(mid) or {}).get("source") or ""
        comments.setdefault(cid, {"text": r["text"], "source": src})
    ner = load_ner("bert")
    old, new = Registry.from_yaml(generated=None), Registry.from_yaml()
    prep = []
    sents_all = []
    for cid, c in comments.items():
        text = clean_text(c["text"])
        sents = split_sentences(text)
        prep.append((cid, c, text, sents, len(sents_all)))
        sents_all.extend(text[a:b] for a, b in sents)
    todo = [i for i, s in enumerate(sents_all) if len(s) >= MIN_NER_CHARS and re.search(r"[A-ZÄÖÜ]", s)]
    ner_out = [[] for _ in sents_all]
    for i, sp in zip(todo, ner.predict([sents_all[i] for i in todo]), strict=True):
        ner_out[i] = sp
    links = {"old": set(), "new": set()}
    new_matches = []
    for cid, c, text, sents, off in prep:
        country = COUNTRY.get(c["source"], "DE")
        for name, reg in (("old", old), ("new", new)):
            for m in link_comment(text, sents, ner_out[off : off + len(sents)], reg, country=country):
                links[name].add(f"{cid}_{m.entity_id}")
                if name == "new" and new.entities[m.entity_id].source == "generated":
                    new_matches.append(
                        {
                            "mentionId": f"{cid}_{m.entity_id}",
                            "surface": m.surface,
                            "source": c["source"],
                            "description": new.entities[m.entity_id].description,
                        }
                    )
    gold = {mid for mid in rows if mid in labels}
    ok = {mid for mid in gold if labels[mid].get("linkOk", "1") != "0"}
    for name in ("old", "new"):
        found = links[name] & gold
        tp = len(found & ok)
        print(
            f"{name}: gold pairs still linked {len(found)}/{len(gold)}; correct {tp}; "
            f"precision on gold {tp / max(1, len(found)):.3f}; correct pairs lost {len(ok - links[name])}"
        )
    lost = ok - links["new"]
    print("correct pairs lost by the new registry:", sorted(m.split("_", 1)[1] for m in lost))
    changed = (links["old"] & gold) - links["new"]
    print("gold pairs (any) no longer linked:", sorted(m.split("_", 1)[1] for m in changed))
    print("new generated-entity matches:", len(new_matches))
    with (EVAL / "new_matches.local.jsonl").open("w", encoding="utf-8") as f:
        for m in new_matches:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
