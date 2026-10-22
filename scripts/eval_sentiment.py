"""Evaluate target sentiment on hand-labelled (comment, entity) pairs.

Privacy: the committed gold file `eval/labels.csv` only contains mention ids
(commentId_entityId) and labels. Comment text is fetched from Firestore at evaluation
time and is only written to the git-ignored `eval/sample.local.jsonl` for labelling.

    # 1. draw a sample to label (writes eval/sample.local.jsonl, git-ignored)
    python scripts/eval_sentiment.py sample --n 60
    # 2. label: add {"mentionId": ..., "gold": "pos|neu|neg"} rows to eval/labels.csv
    # 3. score all model variants
    python scripts/eval_sentiment.py score
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze import analyze_comments  # noqa: E402
from analysis.firestore import Firestore  # noqa: E402
from analysis.pipeline import COMMENT_FIELDS, build_registry, load_entity_docs, source_countries  # noqa: E402
from analysis.textutil import clean_text  # noqa: E402

LABELS_FILE = ROOT / "eval" / "labels.csv"
SAMPLE_FILE = ROOT / "eval" / "sample.local.jsonl"


class CachedNer:
    def __init__(self, inner):
        self.inner = inner
        self.cache: dict[str, list] = {}
        self.version = inner.version

    def predict(self, sentences):
        todo = [s for s in dict.fromkeys(sentences) if s not in self.cache]
        for s, r in zip(todo, self.inner.predict(todo), strict=True):
            self.cache[s] = r
        return [self.cache[s] for s in sentences]


def fetch_comments(fs: Firestore, ids: list[str]) -> dict[str, dict]:
    docs = fs.batch_get([f"comments/{i}" for i in ids], mask=COMMENT_FIELDS)
    return {p.split("/", 1)[1]: {"id": p.split("/", 1)[1], **d.data} for p, d in docs.items() if d}


def cmd_sample(args) -> None:
    fs = Firestore()
    mentions = [d for d in fs.query("mentions", select=["commentId", "entityId", "surface", "label"])]
    random.Random(args.seed).shuffle(mentions)
    # at most one pair per comment, spread over entities
    seen_c, per_entity, chosen = set(), Counter(), []
    for d in mentions:
        c, e = d.data["commentId"], d.data["entityId"]
        if c in seen_c or per_entity[e] >= 4:
            continue
        seen_c.add(c)
        per_entity[e] += 1
        chosen.append(d)
        if len(chosen) >= args.n:
            break
    comments = fetch_comments(fs, [d.data["commentId"] for d in chosen])
    SAMPLE_FILE.parent.mkdir(exist_ok=True)
    with SAMPLE_FILE.open("w", encoding="utf-8") as f:
        for d in chosen:
            c = comments.get(d.data["commentId"])
            if not c:
                continue
            row = {"mentionId": d.id, "entity": d.data["entityId"], "surface": d.data["surface"]}
            row["text"] = clean_text(c.get("text", ""))[:1000]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(chosen)} pairs to {SAMPLE_FILE}")


def macro_f1(gold, pred) -> float:
    f1s = []
    for lab in ("pos", "neu", "neg"):
        tp = sum(g == p == lab for g, p in zip(gold, pred, strict=True))
        fp = sum(p == lab and g != lab for g, p in zip(gold, pred, strict=True))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred, strict=True))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / 3


def cmd_score(args) -> None:
    from analysis.ner import load_ner
    from analysis.sentiment import SentimentModel

    gold = {}
    with LABELS_FILE.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold[row["mentionId"]] = row["gold"]
    fs = Firestore()
    comments = fetch_comments(fs, sorted({m.rsplit("_", 1)[0] for m in gold}))
    registry = build_registry(load_entity_docs(fs))
    countries = source_countries(fs)
    ner = CachedNer(load_ner(args.ner))
    dist = Counter(gold.values())
    print(f"{len(gold)} labelled pairs, gold distribution: {dict(dist)}")
    print(f"majority baseline accuracy: {max(dist.values()) / len(gold):.2f}")
    for key in args.models:
        model = SentimentModel(key)
        out = analyze_comments(list(comments.values()), registry, ner, model, source_country=countries)
        pred = {f"{m['commentId']}_{m['entityId']}": m["label"] for ms in out.values() for m in ms}
        # whole-comment baseline (same model, no target window)
        whole = dict(
            zip(
                comments,
                [r[0] for r in model.predict([clean_text(c.get("text", "")) for c in comments.values()])],
                strict=True,
            )
        )
        keys = [k for k in gold if k in pred]
        g = [gold[k] for k in keys]
        p = [pred[k] for k in keys]
        w = [whole[k.rsplit("_", 1)[0]] for k in keys]
        acc = sum(a == b for a, b in zip(g, p, strict=True)) / len(keys)
        acc_w = sum(a == b for a, b in zip(g, w, strict=True)) / len(keys)
        print(
            f"{key:9s} target-window acc={acc:.2f} macroF1={macro_f1(g, p):.2f} | "
            f"whole-comment acc={acc_w:.2f} macroF1={macro_f1(g, w):.2f} | n={len(keys)} "
            f"(missing {len(gold) - len(keys)})"
        )
        conf = Counter(zip(g, p, strict=True))
        print("   confusion gold->pred:", {f"{a}->{b}": n for (a, b), n in sorted(conf.items())})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--seed", type=int, default=7)
    sc = sub.add_parser("score")
    sc.add_argument("--models", nargs="+", default=["ens", "guhr", "guhr-raw", "news"])
    sc.add_argument("--ner", default="flair")
    args = ap.parse_args()
    {"sample": cmd_sample, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
