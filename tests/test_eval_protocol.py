"""Evaluation protocol for the sentiment gold set: grouped folds, cluster bootstrap."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_protocol import (
    bootstrap_ci,
    bootstrap_diff,
    confusion,
    grouped_folds,
    macro_f1,
    pearson,
    per_class,
)


def _toy(n_groups: int = 120, seed: int = 1):
    rng = random.Random(seed)
    labels, groups = [], []
    for g in range(n_groups):
        for _ in range(rng.choice([1, 1, 1, 2, 3])):
            labels.append(rng.choices([0, 1, 2], weights=[13, 36, 51])[0])
            groups.append(f"c{g}")
    return labels, groups


def test_macro_f1_and_per_class():
    gold = [0, 0, 1, 1, 2, 2]
    assert macro_f1(gold, gold) == 1.0
    pred = [0, 1, 1, 1, 2, 0]
    conf = confusion(gold, pred)
    assert conf == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    p, r, f = per_class(conf)[1]
    assert (round(p, 3), r, round(f, 3)) == (0.667, 1.0, 0.8)
    assert macro_f1(gold, [2] * 6) == (2 * (1 / 3) * 1 / (1 / 3 + 1)) / 3  # only the majority class scores


def test_grouped_folds_keep_comments_together_and_cover_everything():
    labels, groups = _toy()
    folds = grouped_folds(labels, groups, k=5, seed=3)
    assert sorted(i for f in folds for i in f) == list(range(len(labels)))
    seen: dict[str, int] = {}
    for k, f in enumerate(folds):
        for i in f:
            assert seen.setdefault(groups[i], k) == k  # one fold per comment
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 3


def test_grouped_folds_are_stratified_and_depend_on_the_seed():
    labels, groups = _toy()
    folds = grouped_folds(labels, groups, k=5, seed=0)
    for c in range(3):
        per_fold = [sum(labels[i] == c for i in f) for f in folds]
        assert max(per_fold) - min(per_fold) <= 3
    assert folds != grouped_folds(labels, groups, k=5, seed=1)
    assert folds == grouped_folds(labels, groups, k=5, seed=0)


def test_bootstrap_ci_contains_the_point_estimate_and_narrows_with_more_data():
    rng = random.Random(0)
    labels, groups = _toy(100)
    pred = [y if rng.random() < 0.6 else rng.randrange(3) for y in labels]
    f, lo, hi = bootstrap_ci(labels, pred, groups, n_boot=400)
    assert lo < f < hi
    big_l, big_g = _toy(1200, seed=2)
    big_p = [y if rng.random() < 0.6 else rng.randrange(3) for y in big_l]
    _f, lo2, hi2 = bootstrap_ci(big_l, big_p, big_g, n_boot=400)
    assert hi2 - lo2 < (hi - lo) / 2


def test_bootstrap_diff_is_paired():
    rng = random.Random(4)
    labels, groups = _toy(300)
    noisy = [y if rng.random() < 0.5 else rng.randrange(3) for y in labels]
    better = [y if rng.random() < 0.5 else p for y, p in zip(labels, noisy, strict=True)]  # fixes half of the errors
    d, lo, hi, p_le0 = bootstrap_diff(labels, better, noisy, groups, n_boot=400)
    assert d > 0 and lo > 0 and p_le0 == 0.0
    d0, lo0, hi0, _ = bootstrap_diff(labels, noisy, noisy, groups, n_boot=200)
    assert (d0, lo0, hi0) == (0.0, 0.0, 0.0)
    assert hi - lo < 0.2


def test_pearson():
    assert round(pearson([1, 2, 3], [2, 4, 6.5]), 3) > 0.99
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0
