"""Target sentiment under one clean protocol: the calibrated model ("cal") and the fine-tuned one ("ft").

Protocol (scripts/eval_protocol.py): the gold pairs (eval/labels.csv, labels_extra.csv,
labels_extra2.csv) come from a few hundred comments, so the folds are stratified over comments
(a comment's pairs are never split between training and test), everything is fitted inside the
training folds, the confidence intervals come from a cluster bootstrap over comments, and two
models are compared on the same folds with a paired bootstrap.

* cal: the calibration (scripts/train_sentiment_calibration.py) is refitted inside every
  training fold; 10 fold seeds give the mean. Its score is reported twice: with the class
  weights left in (sent-cal@2) and taken out again (sent-cal@3, the label is the same).
* ft:  out-of-fold predictions of scripts/train_sentiment_finetune.py cv (fold seed 0), if present.

Besides macro-F1, the score is checked where the dashboard uses it: the mean score per entity,
per source and per entity type against the mean gold polarity (+1 / 0 / -1).

    python scripts/eval_sentiment_cv.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_protocol import (  # noqa: E402
    CLASSES,
    bootstrap_ci,
    bootstrap_diff,
    confusion,
    grouped_folds,
    macro_f1,
    pearson,
    per_class,
)
from train_sentiment_finetune import CV_FILE, FOLD_SEED, FOLDS, load_pairs  # noqa: E402

LABEL_FILES = [ROOT / "eval" / f for f in ("labels.csv", "labels_extra.csv", "labels_extra2.csv")]
FEATURES = ROOT / "eval" / "calfeatures-all.local.pt"
REPS = 10
POLARITY = {"pos": 1.0, "neu": 0.0, "neg": -1.0}
GROUPS = {"entity": ("entity", 5), "source": ("source", 10), "type": ("type", 10)}  # key, minimum pairs


def load_gold() -> list[dict]:
    from analysis.linker import Registry

    reg = Registry.from_yaml()
    return [p for f in LABEL_FILES for p in load_pairs(f, reg)]


def calibration_module():
    spec = importlib.util.spec_from_file_location("cal", ROOT / "scripts" / "train_sentiment_calibration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cal_features(pairs: list[dict]) -> torch.Tensor:
    """Features of the calibrated model through the production code path (cached locally)."""
    ids = [p["id"] for p in pairs]
    if FEATURES.exists():
        saved = torch.load(FEATURES)
        if saved["ids"] == ids:
            return saved["x"]
    from analysis.sentiment import CalibratedSentiment

    x = torch.tensor(CalibratedSentiment().features([(p["clause"], p["name"]) for p in pairs]))
    torch.save({"ids": ids, "x": x}, FEATURES)
    return x


def cal_cv(cal, x: torch.Tensor, y: list[int], groups: list[str], seed: int) -> dict:
    """Out-of-fold labels and both scores (with / without the class weights) for one fold seed."""
    yt = torch.tensor(y)
    pol = torch.tensor([1.0, 0.0, -1.0])
    out = {"pred": [0] * len(y), "score_weighted": [0.0] * len(y), "score": [0.0] * len(y)}
    folds = grouped_folds(y, groups, FOLDS, seed)
    for k, test in enumerate(folds):
        train = torch.tensor([i for j, f in enumerate(folds) if j != k for i in f])
        params = cal.train(x[train], yt[train])
        lg = cal.label_logits(params, x[torch.tensor(test)])
        corr = torch.tensor(cal.score_correction(yt[train]))
        for name, p in (("score_weighted", torch.softmax(lg, 1)), ("score", torch.softmax(lg - corr, 1))):
            for i, s in zip(test, (p @ pol).tolist(), strict=True):
                out[name][i] = s
        for i, lab in zip(test, lg.argmax(1).tolist(), strict=True):
            out["pred"][i] = lab
    return out


def group_means(pairs: list[dict], score: list[float], key: str, min_n: int) -> tuple[float, float, int]:
    """Pearson r and mean absolute error between the mean score and the mean gold polarity per
    group (groups with at least `min_n` pairs)."""
    by: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        by[p[key]].append(i)
    idx = [v for v in by.values() if len(v) >= min_n]
    gold = [sum(POLARITY[pairs[i]["gold"]] for i in v) / len(v) for v in idx]
    pred = [sum(score[i] for i in v) / len(v) for v in idx]
    mae = sum(abs(a - b) for a, b in zip(pred, gold, strict=True)) / len(idx)
    return pearson(pred, gold), mae, len(idx)


def score_report(pairs: list[dict], scores: list[list[float]]) -> dict:
    """Score checks, averaged over fold seeds."""
    gold = [POLARITY[p["gold"]] for p in pairs]
    n = len(scores)
    rep = {
        "bias": sum(sum(s) / len(s) for s in scores) / n - sum(gold) / len(gold),
        "r_mention": sum(pearson(s, gold) for s in scores) / n,
    }
    for name, (key, min_n) in GROUPS.items():
        vals = [group_means(pairs, s, key, min_n) for s in scores]
        rep[f"r_{name}"] = sum(v[0] for v in vals) / n
        rep[f"mae_{name}"] = sum(v[1] for v in vals) / n
        rep[f"k_{name}"] = vals[0][2]
    return rep


def print_scores(name: str, rep: dict) -> None:
    groups = "; ".join(
        f"{g} means (k={rep[f'k_{g}']}): r {rep[f'r_{g}']:.3f}, MAE {rep[f'mae_{g}']:.3f}" for g in GROUPS
    )
    print(f"  {name}: bias of the mean score {rep['bias']:+.3f}, r per mention {rep['r_mention']:.3f}; {groups}")


def describe(name: str, pairs: list[dict], y: list[int], preds: list[list[int]]) -> dict:
    groups = [p["comment"] for p in pairs]
    f1s = [macro_f1(y, p) for p in preds]
    f1, lo, hi = bootstrap_ci(y, preds[0], groups)
    pc = [per_class(confusion(y, p)) for p in preds]
    cls = [sum(c[k][2] for c in pc) / len(pc) for k in range(3)]
    acc = sum(sum(a == b for a, b in zip(y, p, strict=True)) / len(y) for p in preds) / len(preds)
    print(f"\n{name}")
    print(
        f"  macro-F1 {sum(f1s) / len(f1s):.3f} (mean of {len(f1s)} fold seeds, {min(f1s):.3f} to {max(f1s):.3f}; "
        f"seed 0: {f1:.3f}, 95 % CI {lo:.3f} to {hi:.3f}), accuracy {acc:.3f}"
    )
    print("  F1 pos / neu / neg " + " / ".join(f"{c:.3f}" for c in cls))
    print(f"  predicted (seed 0): {dict(Counter(CLASSES[k] for k in preds[0]))}")
    return {"f1": sum(f1s) / len(f1s), "per_class": cls}


def breakdown(pairs: list[dict], y: list[int], pred: list[int], key: str) -> None:
    by: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        by[p[key]].append(i)
    print(f"\nby {key} (seed 0):")
    for g, idx in sorted(by.items(), key=lambda kv: -len(kv[1]))[:12]:
        f1 = macro_f1([y[i] for i in idx], [pred[i] for i in idx])
        print(f"  {g or '-':16s} n={len(idx):4d}  macro-F1 {f1:.3f}")


def main() -> None:
    pairs = load_gold()
    y = [CLASSES.index(p["gold"]) for p in pairs]
    groups = [p["comment"] for p in pairs]
    counts = ", ".join(f"{c} {y.count(k)}" for k, c in enumerate(CLASSES))
    print(f"{len(pairs)} gold pairs from {len(set(groups))} comments: {counts}")

    cal = calibration_module()
    x = cal_features(pairs)
    runs = [cal_cv(cal, x, y, groups, seed) for seed in range(REPS)]
    base = describe("cal (label of sent-cal@2 and @3)", pairs, y, [r["pred"] for r in runs])
    weighted = score_report(pairs, [r["score_weighted"] for r in runs])
    corrected = score_report(pairs, [r["score"] for r in runs])
    print_scores("score with class weights (sent-cal@2)", weighted)
    print_scores("score without class weights (sent-cal@3)", corrected)

    if CV_FILE.exists():
        oof = json.loads(CV_FILE.read_text(encoding="utf-8"))
        idx = [i for i, p in enumerate(pairs) if p["id"] in oof]
        if idx:
            sub = [pairs[i] for i in idx]
            ys, gs = [y[i] for i in idx], [groups[i] for i in idx]
            ft = [CLASSES.index(oof[p["id"]]["label"]) for p in sub]
            cal0 = [runs[FOLD_SEED]["pred"][i] for i in idx]
            print(f"\nft: out-of-fold predictions for {len(idx)} of {len(pairs)} pairs (fold seed {FOLD_SEED})")
            new = describe("ft", sub, ys, [ft])
            old = describe("cal on the same pairs and folds", sub, ys, [cal0])
            d, lo, hi, p0 = bootstrap_diff(ys, ft, cal0, gs)
            print(f"\nft - cal: {d:+.3f} macro-F1 (paired 95 % CI {lo:+.3f} to {hi:+.3f}; draws <= 0: {p0:.3f})")
            change = [round(b - a, 3) for a, b in zip(old["per_class"], new["per_class"], strict=True)]
            print(f"  change of F1 per class: {change}")
            print_scores("ft score", score_report(sub, [[oof[p["id"]]["score"] for p in sub]]))
            print_scores("cal score on the same pairs", score_report(sub, [[runs[FOLD_SEED]["score"][i] for i in idx]]))
    print(f"\ncal summary: macro-F1 {base['f1']:.3f}")
    for key in ("source", "type"):
        breakdown(pairs, y, runs[0]["pred"], key)


if __name__ == "__main__":
    main()
