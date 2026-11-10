"""Evaluation protocol for target sentiment on the hand-labelled pairs (pure Python, no model).

Why a protocol of its own: the gold set has 455 pairs from 363 comments, so

* a pair must never be scored by a model that saw another pair of the same comment in training
  (`grouped_folds`: stratified k-fold over comments, not over pairs), and
* a macro-F1 on a few hundred pairs has a confidence interval of about +-0.05, so two candidates
  are compared with a paired cluster bootstrap over comments (`bootstrap_ci`, `bootstrap_diff`).

Used by scripts/eval_sentiment_cv.py and tests/test_eval_protocol.py.
"""

from __future__ import annotations

import random
from collections import defaultdict

CLASSES = ("pos", "neu", "neg")


def confusion(gold: list[int], pred: list[int]) -> list[list[int]]:
    c = [[0, 0, 0] for _ in range(3)]
    for g, p in zip(gold, pred, strict=True):
        c[g][p] += 1
    return c


def per_class(conf: list[list[float]]) -> list[tuple[float, float, float]]:
    """(precision, recall, F1) per class from a confusion matrix (rows gold, columns predicted)."""
    out = []
    for k in range(3):
        tp = conf[k][k]
        fp = sum(conf[j][k] for j in range(3)) - tp
        fn = sum(conf[k]) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out.append((p, r, 2 * p * r / (p + r) if p + r else 0.0))
    return out


def macro_f1_conf(conf: list[list[float]]) -> float:
    return sum(f for _p, _r, f in per_class(conf)) / 3


def macro_f1(gold: list[int], pred: list[int]) -> float:
    return macro_f1_conf(confusion(gold, pred))


def grouped_folds(labels: list[int], groups: list[str], k: int = 5, seed: int = 0) -> list[list[int]]:
    """Split item indices into k folds: all items of a group share a fold, classes are balanced.

    Greedy: groups in random order, largest first; each group goes to the fold where it keeps the
    per-class counts closest to an even split (ties: the smaller fold).
    """
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    names = sorted(by_group)
    random.Random(seed).shuffle(names)
    names.sort(key=lambda g: -len(by_group[g]))
    total = [sum(1 for y in labels if y == c) for c in range(3)]
    counts = [[0, 0, 0] for _ in range(k)]
    folds: list[list[int]] = [[] for _ in range(k)]
    for g in names:
        add = [sum(1 for i in by_group[g] if labels[i] == c) for c in range(3)]

        def cost(f: int, add: list[int] = add) -> tuple[float, int]:
            dev = 0.0
            for j in range(k):
                for c in range(3):
                    n = counts[j][c] + (add[c] if j == f else 0)
                    dev += (n - total[c] / k) ** 2 / max(1, total[c])
            return dev, len(folds[f])

        best = min(range(k), key=cost)
        folds[best].extend(by_group[g])
        for c in range(3):
            counts[best][c] += add[c]
    return [sorted(f) for f in folds]


def _group_confusions(gold: list[int], pred: list[int], groups: list[str]) -> list[list[list[int]]]:
    idx: dict[str, int] = {}
    out: list[list[list[int]]] = []
    for g, y, p in zip(groups, gold, pred, strict=True):
        if g not in idx:
            idx[g] = len(out)
            out.append([[0, 0, 0] for _ in range(3)])
        out[idx[g]][y][p] += 1
    return out


def _resample(confs: list[list[list[list[int]]]], rng: random.Random) -> list[list[list[int]]]:
    """One bootstrap draw of groups; the same draw is applied to every system in `confs`."""
    n = len(confs[0])
    draw = [rng.randrange(n) for _ in range(n)]
    sums = [[[0, 0, 0] for _ in range(3)] for _ in confs]
    for s, gc in enumerate(confs):
        for d in draw:
            for a in range(3):
                for b in range(3):
                    sums[s][a][b] += gc[d][a][b]
    return sums


def bootstrap_ci(
    gold: list[int], pred: list[int], groups: list[str], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """(macro-F1, lower, upper) with a 95 % cluster bootstrap over groups (comments)."""
    rng = random.Random(seed)
    gc = _group_confusions(gold, pred, groups)
    vals = sorted(macro_f1_conf(_resample([gc], rng)[0]) for _ in range(n_boot))
    return macro_f1(gold, pred), vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def bootstrap_diff(
    gold: list[int], pred_a: list[int], pred_b: list[int], groups: list[str], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float, float]:
    """Paired difference macro-F1(a) - macro-F1(b): (diff, lower, upper, share of draws with diff <= 0)."""
    rng = random.Random(seed)
    ga, gb = _group_confusions(gold, pred_a, groups), _group_confusions(gold, pred_b, groups)
    diffs = []
    for _ in range(n_boot):
        sa, sb = _resample([ga, gb], rng)
        diffs.append(macro_f1_conf(sa) - macro_f1_conf(sb))
    diffs.sort()
    return (
        macro_f1(gold, pred_a) - macro_f1(gold, pred_b),
        diffs[int(0.025 * n_boot)],
        diffs[int(0.975 * n_boot) - 1],
        sum(d <= 0 for d in diffs) / n_boot,
    )


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else 0.0
