"""Compare the calibrated model ("cal") and the fine-tuned model ("ft") under one clean protocol.

Protocol (scripts/eval_protocol.py): the 455 gold pairs come from 363 comments, so the folds are
stratified over comments (a comment's pairs are never split between training and test), the
confidence intervals come from a cluster bootstrap over comments, and the two models are
compared on the same folds with a paired bootstrap.

* cal: the calibration is refitted inside every training fold (same settings as
  scripts/train_sentiment_calibration.py); 10 repetitions with different folds give the mean.
* ft:  out-of-fold predictions of scripts/train_sentiment_finetune.py cv (fold seed 0).

    python scripts/eval_sentiment_cv.py            # needs eval/ft_cv.local.json for "ft"
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
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
from train_sentiment_finetune import CV_FILE, FOLD_SEED, FOLDS, LABELS, load_pairs  # noqa: E402

FEATURES = ROOT / "eval" / "calfeatures-cv.local.pt"
REPS = 10
POLARITY = {"pos": 1.0, "neu": 0.0, "neg": -1.0}
MIN_PER_ENTITY = 5


def _calibration_module():
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


def cal_cv(cal, x: torch.Tensor, y: list[int], groups: list[str], seed: int) -> tuple[list[int], list[float]]:
    """Out-of-fold labels and scores (P(pos) - P(neg)) of the calibrated model for one fold seed."""
    yt = torch.tensor(y)
    pred, score = [0] * len(y), [0.0] * len(y)
    folds = grouped_folds(y, groups, FOLDS, seed)
    for k, test in enumerate(folds):
        train = torch.tensor([i for j, f in enumerate(folds) if j != k for i in f])
        te = torch.tensor(test)
        mu, sd, w, b = cal.train(x[train], yt[train])
        p = torch.softmax((x[te] - mu) / sd @ w + b + torch.tensor(cal.BIAS), dim=1)
        for i, row in zip(test, p.tolist(), strict=True):
            pred[i] = max(range(3), key=row.__getitem__)
            score[i] = row[0] - row[2]
    return pred, score


def entity_level(pairs: list[dict], score: list[float]) -> tuple[float, float, int]:
    """Pearson r and mean absolute error between the mean predicted score and the mean gold
    polarity per entity (entities with at least MIN_PER_ENTITY gold pairs)."""
    by: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        by[p["entity"]].append(i)
    idx = [v for v in by.values() if len(v) >= MIN_PER_ENTITY]
    gold = [sum(POLARITY[pairs[i]["gold"]] for i in v) / len(v) for v in idx]
    pred = [sum(score[i] for i in v) / len(v) for v in idx]
    mae = sum(abs(a - b) for a, b in zip(pred, gold, strict=True)) / len(idx)
    return pearson(pred, gold), mae, len(idx)


def describe(name: str, pairs: list[dict], y: list[int], pred: list[int], score: list[float]) -> dict:
    groups = [p["comment"] for p in pairs]
    f1, lo, hi = bootstrap_ci(y, pred, groups)
    conf = confusion(y, pred)
    pc = per_class(conf)
    gold_pol = [POLARITY[p["gold"]] for p in pairs]
    r_ent, mae_ent, n_ent = entity_level(pairs, score)
    print(f"\n{name}")
    acc = sum(a == b for a, b in zip(y, pred, strict=True)) / len(y)
    print(f"  macro-F1 {f1:.3f}  (95 % CI {lo:.3f} to {hi:.3f}), accuracy {acc:.3f}")
    for k, c in enumerate(CLASSES):
        print(f"  {c}: precision {pc[k][0]:.2f}  recall {pc[k][1]:.2f}  F1 {pc[k][2]:.2f}")
    print(f"  confusion (rows gold pos/neu/neg, columns predicted): {conf}")
    print(
        f"  score vs gold polarity: r {pearson(score, gold_pol):.3f} per mention, "
        f"mean {sum(score) / len(score):+.3f} (gold {sum(gold_pol) / len(gold_pol):+.3f}); "
        f"entity means ({n_ent} entities with >= {MIN_PER_ENTITY} pairs): r {r_ent:.3f}, MAE {mae_ent:.3f}"
    )
    return {"f1": f1, "per_class": [c[2] for c in pc], "r_entity": r_ent, "mae_entity": mae_ent}


def breakdown(pairs: list[dict], y: list[int], preds: dict[str, list[int]], key: str) -> None:
    by: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        by[p[key]].append(i)
    print(f"\nby {key}: " + "  ".join(f"{n} macro-F1" for n in preds))
    for g, idx in sorted(by.items(), key=lambda kv: -len(kv[1])):
        cells = "  ".join(f"{macro_f1([y[i] for i in idx], [p[i] for i in idx]):.3f}" for p in preds.values())
        print(f"  {g or '-':16s} n={len(idx):3d}  {cells}")


def main() -> None:
    pairs = load_pairs(LABELS)
    y = [CLASSES.index(p["gold"]) for p in pairs]
    groups = [p["comment"] for p in pairs]
    n_comments = len(set(groups))
    print(
        f"{len(pairs)} gold pairs from {n_comments} comments: "
        + ", ".join(f"{c} {y.count(k)}" for k, c in enumerate(CLASSES))
    )

    cal = _calibration_module()
    x = cal_features(pairs)
    runs = [cal_cv(cal, x, y, groups, seed) for seed in range(REPS)]
    f1s = [macro_f1(y, p) for p, _ in runs]
    print(
        f"\ncal, grouped {FOLDS}-fold CV, {REPS} fold seeds: macro-F1 mean {sum(f1s) / REPS:.3f} "
        f"(min {min(f1s):.3f}, max {max(f1s):.3f})"
    )
    cal_pred, cal_score = runs[FOLD_SEED]
    base = describe(f"cal (sent-cal@2), fold seed {FOLD_SEED}", pairs, y, cal_pred, cal_score)
    preds = {"cal": cal_pred}

    if CV_FILE.exists():
        oof = json.loads(CV_FILE.read_text(encoding="utf-8"))
        if all(p["id"] in oof for p in pairs):
            ft_pred = [CLASSES.index(oof[p["id"]]["label"]) for p in pairs]
            ft_score = [oof[p["id"]]["score"] for p in pairs]
            new = describe(f"ft (fine-tuned), fold seed {FOLD_SEED}", pairs, y, ft_pred, ft_score)
            preds["ft"] = ft_pred
            d, lo, hi, p0 = bootstrap_diff(y, ft_pred, cal_pred, groups)
            print(f"\nft - cal on the same folds: {d:+.3f} macro-F1 (95 % CI {lo:+.3f} to {hi:+.3f})")
            print(f"  share of bootstrap draws with ft <= cal: {p0:.4f}")
            drops = [round(b - a, 3) for a, b in zip(base["per_class"], new["per_class"], strict=True)]
            print(f"  change of F1 per class (pos, neu, neg): {drops}")
            print(
                f"  entity means: r {base['r_entity']:.3f} -> {new['r_entity']:.3f}, "
                f"MAE {base['mae_entity']:.3f} -> {new['mae_entity']:.3f}"
            )
        else:
            print(f"\n{CV_FILE.name} is incomplete: run scripts/train_sentiment_finetune.py cv --resume")
    for key in ("source", "type"):
        breakdown(pairs, y, preds, key)


if __name__ == "__main__":
    main()
