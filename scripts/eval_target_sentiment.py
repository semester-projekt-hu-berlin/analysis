"""Compare target-sentiment candidates on the hand-labelled pairs in eval/labels.csv.

Comment texts are fetched from Firestore and cached locally in eval/*.local.* (git-ignored).
Pairs are split deterministically into dev / test (by hash of the mention id); thresholds are
tuned on dev only, and test scores are reported separately.

    python scripts/eval_target_sentiment.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze import mention_window  # noqa: E402
from analysis.linker import Match, Registry  # noqa: E402
from analysis.sentiment import SentimentModel, to_label  # noqa: E402
from analysis.textutil import clean_text, split_sentences  # noqa: E402

LABELS = ROOT / "eval" / "labels.csv"
TEXTS = ROOT / "eval" / "texts.local.json"
PROBS = ROOT / "eval" / "probs.local.json"
COUNTRY = {"derstandard": "AT", "blick": "CH", "taz": "DE"}
THRESHOLDS = [None, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]


def macro_f1(gold: list[str], pred: list[str]) -> float:
    f1s = []
    for lab in ("pos", "neu", "neg"):
        tp = sum(g == p == lab for g, p in zip(gold, pred, strict=True))
        fp = sum(p == lab != g for g, p in zip(gold, pred, strict=True))
        fn = sum(g == lab != p for g, p in zip(gold, pred, strict=True))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1s) / 3


def load_items() -> list[dict]:
    rows = [r for r in csv.DictReader(LABELS.open(encoding="utf-8")) if r.get("linkOk", "1") != "0"]
    try:
        texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    except OSError:
        texts = {}
    missing = sorted({r["mentionId"].rsplit("_", 1)[0] for r in rows} - texts.keys())
    if missing:
        from analysis.firestore import Firestore

        fs = Firestore()
        docs = fs.batch_get([f"comments/{c}" for c in missing], mask=["text", "sourceId"])
        for p, d in docs.items():
            if d:
                texts[p.split("/", 1)[1]] = {"text": d.data.get("text", ""), "source": d.data.get("sourceId", "")}
        TEXTS.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    reg = Registry.from_yaml()
    items = []
    for r in rows:
        cid, eid = r["mentionId"].rsplit("_", 1)
        t = texts.get(cid)
        if not t or eid not in reg.entities:
            continue
        text = clean_text(t["text"])
        sents = split_sentences(text)
        matches = reg.gazetteer(text, [], "", COUNTRY.get(t["source"]))
        target = next((m for m in matches if m.entity_id == eid), None)
        if target is None:  # weak alias confirmed by NER at analysis time: locate the surface
            surf = next((a for a in reg.entities[eid].aliases if a.casefold() in text.casefold()), None)
            if surf is None:
                continue
            i = text.casefold().index(surf.casefold())
            target = Match(i, i + len(surf), text[i : i + len(surf)], eid, "surface")
            matches = [*matches, target]
        items.append(
            {
                "id": r["mentionId"],
                "gold": r["gold"],
                "source": t["source"],
                "name": reg.entities[eid].name,
                "clause": mention_window(text, sents, target, matches, "clause"),
                "entity": mention_window(text, sents, target, matches, "entity"),
                "split": "dev" if int(hashlib.sha1(r["mentionId"].encode()).hexdigest(), 16) % 2 == 0 else "test",
            }
        )
    return items


def main() -> None:
    items = load_items()
    print(
        f"{len(items)} pairs: {dict(Counter(i['gold'] for i in items))};",
        f"by source {dict(Counter(i['source'] for i in items))}",
    )
    print(f"dev {sum(i['split'] == 'dev' for i in items)} / test {sum(i['split'] == 'test' for i in items)}")
    try:
        cache = json.loads(PROBS.read_text(encoding="utf-8"))
    except OSError:
        cache = {}
    speed: dict[str, float] = {}
    for key in ("guhr-raw", "news", "nli", "nli-small"):
        model = None
        for mode in ("clause", "entity"):
            ck = f"{key}|{mode}"
            if ck in cache and len(cache[ck]) == len(items):
                continue
            model = model or SentimentModel(key)
            t = time.time()
            cache[ck] = dict(
                zip([i["id"] for i in items], model.probs([(i[mode], i["name"]) for i in items]), strict=True)
            )
            speed[key] = len(items) / (time.time() - t)
            PROBS.write_text(json.dumps(cache), encoding="utf-8")
    if speed:
        print("throughput (mentions/s, this machine):", {k: round(v, 1) for k, v in speed.items()})

    def probs(combo: list[tuple[str, float]], mode: str, it: dict) -> tuple[float, float, float]:
        tot = sum(w for _, w in combo)
        return tuple(sum(w * cache[f"{k}|{mode}"][it["id"]][c] for k, w in combo) / tot for c in range(3))

    candidates = {
        "a  ens (news+guhr-raw), clause window [baseline]": ([("news", 1), ("guhr-raw", 1)], "clause"),
        "b  NLI mDeBERTa, clause window": ([("nli", 1)], "clause"),
        "b  NLI mDeBERTa, entity window (+-1 sentence, others masked)": ([("nli", 1)], "entity"),
        "b  NLI MiniLM, entity window": ([("nli-small", 1)], "entity"),
        "c  ens, entity window (others masked)": ([("news", 1), ("guhr-raw", 1)], "entity"),
        "d  NLI + ens (1:1), entity window": ([("nli", 2), ("news", 1), ("guhr-raw", 1)], "entity"),
        "d  NLI + ens (2:1), entity window": ([("nli", 4), ("news", 1), ("guhr-raw", 1)], "entity"),
        "d  NLI + news (1:1), entity window": ([("nli", 1), ("news", 1)], "entity"),
        "d  NLI + ens (1:1), clause window": ([("nli", 2), ("news", 1), ("guhr-raw", 1)], "clause"),
    }
    dev = [i for i in items if i["split"] == "dev"]
    test = [i for i in items if i["split"] == "test"]
    print(f"\n{'candidate':64s} {'thr':>5s} {'dev F1':>7s} {'test F1':>8s} {'test acc':>9s}")
    best = None
    for name, (combo, mode) in candidates.items():
        scored = []
        for t in THRESHOLDS:
            pd = [to_label(*probs(combo, mode, i), t)[0] for i in dev]
            scored.append((macro_f1([i["gold"] for i in dev], pd), t))
        f1_dev, thr = max(scored, key=lambda x: (x[0], -(x[1] or 0)))
        pt = [to_label(*probs(combo, mode, i), thr)[0] for i in test]
        gt = [i["gold"] for i in test]
        f1_test = macro_f1(gt, pt)
        acc = sum(a == b for a, b in zip(gt, pt, strict=True)) / len(gt)
        print(f"{name:64s} {thr if thr is not None else 'argm':>5} {f1_dev:7.3f} {f1_test:8.3f} {acc:9.3f}")
        if best is None or f1_dev > best[0]:
            best = (f1_dev, name, combo, mode, thr, pt, gt)
    f1_dev, name, combo, mode, thr, pt, gt = best
    print(f"\nbest on dev: {name} (threshold {thr})")
    print(
        "  test confusion gold->pred:", dict(sorted(Counter(f"{g}->{p}" for g, p in zip(gt, pt, strict=True)).items()))
    )
    print("  test predicted distribution:", dict(Counter(pt)), "gold:", dict(Counter(gt)))
    allp = [to_label(*probs(combo, mode, i), thr)[0] for i in items]
    for src in sorted({i["source"] for i in items}):
        g = [i["gold"] for i in items if i["source"] == src]
        p = [x for x, i in zip(allp, items, strict=True) if i["source"] == src]
        print(f"  {src or 'first sample':14s} n={len(g):3d} macroF1={macro_f1(g, p):.3f}")


if __name__ == "__main__":
    main()
