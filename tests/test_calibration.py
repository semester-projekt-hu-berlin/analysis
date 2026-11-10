"""Calibrated sentiment (sent-cal@3): the label keeps the class weights, the score does not."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from analysis.sentiment import CALIBRATION_FILE, CalibratedSentiment

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def _trainer():
    spec = importlib.util.spec_from_file_location("cal", ROOT / "scripts" / "train_sentiment_calibration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _model(features: list[list[float]], correction: list[float]) -> CalibratedSentiment:
    """A CalibratedSentiment whose features are given (no model is loaded): logits = features."""
    m = object.__new__(CalibratedSentiment)
    m.torch = torch
    m.mean, m.std = torch.zeros(3), torch.ones(3)
    m.W, m.b = torch.eye(3), torch.zeros(3)
    m.correction = torch.tensor(correction)
    m.features = lambda items: features[: len(items)]
    return m


def test_label_from_the_weighted_logits_score_and_confidence_from_the_corrected_ones():
    # logits favour "pos" only because of its weight: corrected, "neg" is more likely
    corr = [math.log(2.0), 0.0, math.log(0.5)]
    m = _model([[1.0, 0.0, 0.5]], corr)
    ((label, score, conf),) = m.predict([("Merz ist gut.", "Merz")])
    p = torch.softmax(torch.tensor([1.0, 0.0, 0.5]) - torch.tensor(corr), 0).tolist()
    assert label == "pos"  # unchanged by the correction
    assert score == round(p[0] - p[2], 4) and score < 0
    assert conf == round(p[0], 4)
    assert m.probs([("x", "y")]) == pytest.approx([tuple(p)], abs=1e-6)
    assert m.predict([]) == [] and m.probs([]) == []


def test_without_correction_score_and_label_agree_as_before():
    m = _model([[0.2, 1.5, -0.3]], [0.0, 0.0, 0.0])
    ((label, score, conf),) = m.predict([("a", "b")])
    p = torch.softmax(torch.tensor([0.2, 1.5, -0.3]), 0).tolist()
    assert (label, score, conf) == ("neu", round(p[0] - p[2], 4), round(p[1], 4))


def test_score_correction_undoes_class_weights_and_bias():
    cal = _trainer()
    y = torch.tensor([0] * 10 + [1] * 30 + [2] * 60)
    corr = cal.score_correction(y)
    want = [
        math.log((100 / (3 * n)) ** cal.CLASS_WEIGHT_POWER) + b for n, b in zip((10, 30, 60), cal.BIAS, strict=True)
    ]
    assert corr == pytest.approx(want)
    assert corr[0] > corr[1] > corr[2]  # the rare class was boosted most


def test_correction_makes_the_mean_score_match_the_class_rates():
    """A model fitted with class weights over-states rare classes; the correction removes that."""
    cal = _trainer()
    g = torch.Generator().manual_seed(0)
    y = torch.tensor([0] * 60 + [1] * 200 + [2] * 340)
    x = torch.randn(len(y), 4, generator=g) * 2.0 + torch.nn.functional.one_hot(y, 3).float() @ torch.randn(
        3, 4, generator=g
    )
    params = cal.train(x, y)
    lg = cal.label_logits(params, x)
    gold = float((y == 0).float().mean() - (y == 2).float().mean())
    weighted = torch.softmax(lg, 1)
    corrected = torch.softmax(lg - torch.tensor(cal.score_correction(y)), 1)
    bias_w = float((weighted[:, 0] - weighted[:, 2]).mean()) - gold
    bias_c = float((corrected[:, 0] - corrected[:, 2]).mean()) - gold
    assert bias_w > 0.1 and abs(bias_c) < abs(bias_w) / 3


def test_shipped_calibration_file():
    cal = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    assert cal["version"] == 3 and len(cal["scoreCorrection"]) == 3
    assert len(cal["W"]) == len(cal["mean"]) == len(cal["features"]) == 12
    assert cal["trainedOn"] >= 1000
