"""Train the calibrated target-sentiment model (data/sentiment_calibration.json).

Features per (comment, entity) pair, on the clause window (CalibratedSentiment.features):
  * NLI logits for "Der Kommentar äußert sich positiv / negativ über {name}." and
    "Der Kommentar bewertet {name}." (3 passes)
  * probabilities of mdraw/german-news-sentiment-bert (1 pass)
A multinomial logistic regression (L2, class weights = balanced^0.75, small neutral bias) maps
them to the label. Class weights and bias give the best labels but over-state the rarer classes;
`scoreCorrection` takes them out again for the score (version 3), so the mean score is unbiased.

Training pairs: every labelled pair (scripts/eval_sentiment_cv.py LABEL_FILES). The settings
were fixed on the first 455 pairs; scripts/eval_sentiment_cv.py reports the cross-validation.

    python scripts/train_sentiment_calibration.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.sentiment import CALIBRATION_FILE  # noqa: E402

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
L2 = 0.1
CLASS_WEIGHT_POWER = 0.75
HYPOTHESES = ("pos", "neg", "judge")
# additive logit bias (pos, neu, neg); the neutral bias was chosen by 5-fold CV inside the dev split
BIAS = (0.0, 0.1, 0.0)
VERSION = 3
CLASSES = ["pos", "neu", "neg"]


def class_weights(y: torch.Tensor) -> list[float]:
    return [(len(y) / (3 * max(1, int((y == k).sum())))) ** CLASS_WEIGHT_POWER for k in range(3)]


def train(x: torch.Tensor, y: torch.Tensor):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xn = (x - mu) / sd
    w = torch.zeros(x.shape[1], 3, requires_grad=True)
    b = torch.zeros(3, requires_grad=True)
    cw = torch.tensor(class_weights(y))
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(xn @ w + b, y, weight=cw) + L2 * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return mu, sd, w.detach(), b.detach()


def label_logits(params, x: torch.Tensor) -> torch.Tensor:
    """Logits whose argmax is the label (class weights and bias included)."""
    mu, sd, w, b = params
    return (x - mu) / sd @ w + b + torch.tensor(BIAS)


def score_correction(y: torch.Tensor) -> list[float]:
    """Subtracted from the label logits to undo the class weights and the bias: a model fitted
    with weight w_k estimates probabilities proportional to w_k * p_k."""
    return [math.log(c) + d for c, d in zip(class_weights(y), BIAS, strict=True)]


def predict(params, x: torch.Tensor) -> torch.Tensor:
    return label_logits(params, x).argmax(1)


def main() -> None:
    import eval_sentiment_cv as ev

    pairs = ev.load_gold()
    x = ev.cal_features(pairs)
    y = torch.tensor([CLASSES.index(p["gold"]) for p in pairs])
    params = train(x, y)
    corr = score_correction(y)
    fit = Counter(CLASSES[v] for v in predict(params, x).tolist())
    p = torch.softmax(label_logits(params, x) - torch.tensor(corr), 1)
    mean_score = float((p[:, 0] - p[:, 2]).mean())
    gold_mean = float(sum({"pos": 1, "neu": 0, "neg": -1}[q["gold"]] for q in pairs) / len(pairs))
    print(f"{len(pairs)} pairs; fitted labels {dict(fit)}, gold {dict(Counter(q['gold'] for q in pairs))}")
    print(f"mean score {mean_score:+.3f} (gold mean polarity {gold_mean:+.3f})")
    mu, sd, w, b = params
    CALIBRATION_FILE.write_text(
        json.dumps(
            {
                "version": VERSION,
                "hypotheses": list(HYPOTHESES),
                "bias": list(BIAS),
                "scoreCorrection": [round(c, 6) for c in corr],
                "features": [f"nli_{h}_{k}" for h in HYPOTHESES for k in ("entail", "neutral", "contradiction")]
                + ["news_pos", "news_neu", "news_neg"],
                "classes": CLASSES,
                "mean": mu.tolist(),
                "std": sd.tolist(),
                "W": w.tolist(),
                "b": b.tolist(),
                "trainedOn": len(pairs),
                "l2": L2,
                "classWeightPower": CLASS_WEIGHT_POWER,
                "nliModel": NLI_MODEL,
                "classifier": "mdraw/german-news-sentiment-bert",
                "window": "clause",
                "evaluation": "scripts/eval_sentiment_cv.py (grouped 5-fold CV, eval/README.md)",
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print("saved", CALIBRATION_FILE)


if __name__ == "__main__":
    main()
