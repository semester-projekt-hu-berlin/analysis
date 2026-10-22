"""Train the calibrated target-sentiment model (data/sentiment_calibration.json).

Features per (comment, entity) pair, on the clause window:
  * NLI logits for "Der Kommentar äußert sich positiv über {name}." and "… negativ …" (2 passes)
  * probabilities of mdraw/german-news-sentiment-bert (1 pass)
A multinomial logistic regression (L2, class weights = balanced^0.75, small neutral bias) maps
them to pos/neu/neg. Features: NLI logits for the hypotheses in HYPOTHESES plus the classifier.
Reports 5-fold cross-validation (3 repetitions) and the fixed dev -> test split.

    python scripts/train_sentiment_calibration.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.sentiment import CALIBRATION_FILE, _Nli, _Single  # noqa: E402

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
L2 = 0.1
CLASS_WEIGHT_POWER = 0.75
HYPOTHESES = ("pos", "neg", "judge")
# additive logit bias (pos, neu, neg); the neutral bias was chosen by 5-fold CV inside the dev split
BIAS = (0.0, 0.1, 0.0)
VERSION = 2
CLASSES = ["pos", "neu", "neg"]


def load_eval():
    spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts" / "eval_target_sentiment.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def features(items: list[dict]) -> torch.Tensor:
    nli, clf = _Nli(NLI_MODEL, 256), _Single("news", 256)
    rows = []
    with torch.inference_mode():
        for b in range(0, len(items), 16):
            batch = [(i["clause"], i["name"]) for i in items[b : b + 16]]
            nl = nli.logits(torch, batch, HYPOTHESES)
            cp = clf.probs(torch, [x[0] for x in batch])
            rows += [a + list(c) for a, c in zip(nl, cp, strict=True)]
    return torch.tensor(rows)


def train(x: torch.Tensor, y: torch.Tensor):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xn = (x - mu) / sd
    w = torch.zeros(x.shape[1], 3, requires_grad=True)
    b = torch.zeros(3, requires_grad=True)
    cw = torch.tensor([(len(y) / (3 * max(1, int((y == k).sum())))) ** CLASS_WEIGHT_POWER for k in range(3)])
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(xn @ w + b, y, weight=cw) + L2 * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return mu, sd, w.detach(), b.detach()


def predict(params, x: torch.Tensor, bias: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> torch.Tensor:
    mu, sd, w, b = params
    return ((x - mu) / sd @ w + b + torch.tensor(bias)).argmax(1)


def per_class_f1(gold: list[str], pred: list[str]) -> dict[str, float]:
    out = {}
    for lab in CLASSES:
        tp = sum(g == p == lab for g, p in zip(gold, pred, strict=True))
        fp = sum(p == lab != g for g, p in zip(gold, pred, strict=True))
        fn = sum(g == lab != p for g, p in zip(gold, pred, strict=True))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[lab] = {"f1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0, "p": round(pr, 3), "r": round(rc, 3)}
    return out


def main() -> None:
    ev = load_eval()
    items = ev.load_items()
    cache = ROOT / "eval" / f"calfeatures-{'-'.join(HYPOTHESES)}-{len(items)}.local.pt"
    if cache.exists():
        x = torch.load(cache)
    else:
        x = features(items)
        torch.save(x, cache)
    y = torch.tensor([CLASSES.index(i["gold"]) for i in items])
    gold = [i["gold"] for i in items]
    scores = []
    for rep in range(3):
        idx = list(range(len(items)))
        random.Random(rep).shuffle(idx)
        folds = [idx[k::5] for k in range(5)]
        p = torch.zeros(len(items), dtype=torch.long)
        for k in range(5):
            te = torch.tensor(folds[k])
            tr = torch.tensor([j for f in range(5) if f != k for j in folds[f]])
            p[te] = predict(train(x[tr], y[tr]), x[te], BIAS)
        scores.append(ev.macro_f1(gold, [CLASSES[v] for v in p.tolist()]))
    cv = sum(scores) / len(scores)
    dev = torch.tensor([int(hashlib.sha1(i["id"].encode()).hexdigest(), 16) % 2 == 0 for i in items])
    dev_params = train(x[dev], y[dev])
    test_gold = [g for g, d in zip(gold, dev.tolist(), strict=True) if not d]
    pt = [CLASSES[v] for v in predict(dev_params, x[~dev], BIAS).tolist()]
    test_f1 = ev.macro_f1(test_gold, pt)
    print(f"5-fold CV macro-F1 {cv:.3f}; dev->test macro-F1 {test_f1:.3f} (n_test={len(test_gold)})")
    print("  test per class:", per_class_f1(test_gold, pt))
    params = train(x, y)
    fit = Counter(CLASSES[v] for v in predict(params, x, BIAS).tolist())
    print("fit distribution", dict(fit), "gold", dict(Counter(gold)))
    mu, sd, w, b = params
    CALIBRATION_FILE.write_text(
        json.dumps(
            {
                "version": VERSION,
                "hypotheses": list(HYPOTHESES),
                "bias": list(BIAS),
                "features": [f"nli_{h}_{k}" for h in HYPOTHESES for k in ("entail", "neutral", "contradiction")]
                + ["news_pos", "news_neu", "news_neg"],
                "classes": CLASSES,
                "mean": mu.tolist(),
                "std": sd.tolist(),
                "W": w.tolist(),
                "b": b.tolist(),
                "trainedOn": len(items),
                "l2": L2,
                "classWeightPower": CLASS_WEIGHT_POWER,
                "nliModel": NLI_MODEL,
                "classifier": "mdraw/german-news-sentiment-bert",
                "window": "clause",
                "cv5MacroF1": round(cv, 3),
                "devTestMacroF1": round(test_f1, 3),
                "devTestPerClass": per_class_f1(test_gold, pt),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print("saved", CALIBRATION_FILE)


if __name__ == "__main__":
    main()
