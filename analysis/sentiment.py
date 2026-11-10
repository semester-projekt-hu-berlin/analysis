"""Sentiment scoring of the text window around an entity mention.

The window is the sentence containing the mention; if the sentence mentions several
entities and contains a contrastive clause boundary ("aber", "während", ";" ...), the
window shrinks to the clause containing the mention (cheap target dependence). The
window is classified by a German 3-class sentiment BERT; the score is
P(pos) - P(neg) in [-1, 1], the label the argmax class, the confidence its probability.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import ClassVar

log = logging.getLogger(__name__)

DEFAULT_SENTIMENT = "cal"  # key of the model that scores new mentions (see load_sentiment)

MODELS = {
    # Trained on ~1.8M German texts (reviews, tweets, Facebook comments, news). Expects
    # the package's normalization (lowercase, letters only).
    "guhr": ("oliverguhr/german-sentiment-bert", True),
    # Same architecture fine-tuned on German news; no special normalization.
    "news": ("mdraw/german-news-sentiment-bert", False),
    # Guhr model on raw text (no lowercasing), evaluated as an alternative.
    "guhr-raw": ("oliverguhr/german-sentiment-bert", False),
}

_URL = re.compile(r"https?://\S+|www\.\S+")
_AT = re.compile(r"@\w+")
_CHARS = re.compile(r"[^a-zA-ZäöüÄÖÜß ]")
_NUM = {
    "0": "null",
    "1": "eins",
    "2": "zwei",
    "3": "drei",
    "4": "vier",
    "5": "fünf",
    "6": "sechs",
    "7": "sieben",
    "8": "acht",
    "9": "neun",
}


def guhr_clean(text: str) -> str:
    """Normalization used in training of oliverguhr/german-sentiment-bert (germansentiment pkg)."""
    text = text.replace("\n", " ")
    text = _URL.sub("", text)
    text = _AT.sub("", text)
    text = re.sub(r"\d", lambda m: " " + _NUM[m.group(0)] + " ", text)
    text = _CHARS.sub("", text)
    text = " ".join(text.split())
    return text.strip().lower()


CONTRAST = re.compile(
    r"(?:\s*;\s*|\s+[–—-]\s+|,?\s+(?:aber|doch|jedoch|sondern|während|wohingegen|allerdings|dagegen|hingegen|trotzdem)\s+)",
    re.IGNORECASE,
)


def target_window(sentence: str, mention_start: int, mention_end: int, other_mentions: list[tuple[int, int]]) -> str:
    """Narrow a sentence to the clause of the mention if other entities sit in other clauses."""
    if not other_mentions:
        return sentence
    cuts = [0]
    for m in CONTRAST.finditer(sentence):
        cuts.append(m.start())
        cuts.append(m.end())
    cuts.append(len(sentence))
    clauses = [(cuts[i], cuts[i + 1]) for i in range(0, len(cuts) - 1, 2)]
    if len(clauses) < 2:
        return sentence
    mine = next((c for c in clauses if c[0] <= mention_start < c[1] + 1), None)
    if mine is None:
        return sentence
    # Only narrow if some other entity is outside our clause.
    if all(mine[0] <= s < mine[1] for s, _ in other_mentions):
        return sentence
    clause = sentence[mine[0] : mine[1]].strip()
    return clause if len(clause) >= 12 else sentence


def _forward(model, tok, name: str, pair: bool = False):
    """encoding -> logits: through ONNX Runtime when it reproduces the PyTorch model (see
    analysis/onnx_backend.py), else through PyTorch."""
    from . import onnx_backend

    session = onnx_backend.accelerate(model, tok, name, pair=pair)
    if session is not None:
        return session.logits
    return lambda enc: model(**enc).logits


class _Single:
    def __init__(self, key: str, max_length: int):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name, self.clean = MODELS[key]
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.forward = _forward(self.model, self.tok, name)
        self.max_length = max_length
        labels = {i: lab.lower() for i, lab in self.model.config.id2label.items()}
        self.idx = {k: next(i for i, lab in labels.items() if lab.startswith(k)) for k in ("pos", "neu", "neg")}

    def probs(self, torch, texts: list[str]) -> list[tuple[float, float, float]]:
        prepared = [(guhr_clean(t) if self.clean else t) or "." for t in texts]
        enc = self.tok(prepared, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        out = torch.softmax(self.forward(enc), dim=-1).tolist()
        return [(p[self.idx["pos"]], p[self.idx["neu"]], p[self.idx["neg"]]) for p in out]


# Ensemble default, chosen on the hand-labelled sample (see eval/README.md).
ENSEMBLE = ("news", "guhr-raw")
THRESHOLD = 0.2  # |P(pos) - P(neg)| above this -> pos/neg, else neu (models are over-neutral on comments)


class _Nli:
    """Zero-shot target sentiment with an NLI model: premise = window, one hypothesis per class."""

    HYPOTHESES: ClassVar[dict[str, str]] = {
        "pos": "Der Kommentar äußert sich positiv über {name}.",
        "neu": "Der Kommentar äußert sich neutral über {name}.",
        "neg": "Der Kommentar äußert sich negativ über {name}.",
        # target vs. context: is the entity judged at all, or only mentioned in passing?
        "judge": "Der Kommentar bewertet {name}.",
    }

    def __init__(self, name: str, max_length: int):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.forward = _forward(self.model, self.tok, name, pair=True)
        self.max_length = max_length
        labels = {i: lab.lower() for i, lab in self.model.config.id2label.items()}
        self.ent = next(i for i, lab in labels.items() if lab.startswith("entail"))

    def logits(self, torch, items: list[tuple[str, str]], keys: tuple[str, ...]) -> list[list[float]]:
        """Raw NLI logits (in model label order) for each item and hypothesis in `keys`, flattened."""
        prem, hyp = [], []
        for text, name in items:
            for k in keys:
                prem.append(text or ".")
                hyp.append(self.HYPOTHESES[k].format(name=name))
        enc = self.tok(
            prem, hyp, padding=True, truncation="only_first", max_length=self.max_length, return_tensors="pt"
        )
        out = self.forward(enc)
        order = [self.ent, *[i for i in range(out.shape[1]) if i != self.ent]]
        lg = out[:, order].reshape(len(items), -1).tolist()
        return lg

    def probs(self, torch, items: list[tuple[str, str]]) -> list[tuple[float, float, float]]:
        prem, hyp = [], []
        for text, name in items:
            for k in ("pos", "neu", "neg"):
                prem.append(text or ".")
                hyp.append(self.HYPOTHESES[k].format(name=name))
        enc = self.tok(
            prem, hyp, padding=True, truncation="only_first", max_length=self.max_length, return_tensors="pt"
        )
        ent = self.forward(enc)[:, self.ent].reshape(-1, 3)
        return [tuple(p) for p in torch.softmax(ent, dim=-1).tolist()]


NLI_MODELS = {
    "nli": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "nli-small": "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
}


CALIBRATION_FILE = Path(__file__).resolve().parents[1] / "data" / "sentiment_calibration.json"


class CalibratedSentiment:
    """Default target sentiment ("cal"): 4 model passes per mention.

    Features: the NLI model's logits for "… positiv über {name}", "… negativ über {name}" and
    "Der Kommentar bewertet {name}" (3 passes, target-aware; the last one separates the judged
    target from entities that are only context: victims, comparisons, places) plus the
    news-sentiment classifier's probabilities (1 pass) on the clause window. A multinomial
    logistic regression trained on the hand-labelled pairs
    (data/sentiment_calibration.json, see eval/README.md) maps them to pos/neu/neg.
    """

    window_mode = "clause"

    def __init__(self, batch_size: int = 16, max_length: int = 256, path: Path = CALIBRATION_FILE):
        import torch

        t = time.time()
        self.torch = torch
        cal = json.loads(path.read_text(encoding="utf-8"))
        self.cal = cal
        self.nli = _Nli(cal["nliModel"], max_length)
        self.clf = _Single("news", max_length)
        self.mean = torch.tensor(cal["mean"])
        self.std = torch.tensor(cal["std"])
        self.W = torch.tensor(cal["W"])
        self.b = torch.tensor(cal["b"]) + torch.tensor(cal.get("bias", [0.0, 0.0, 0.0]))
        self.hypotheses = tuple(cal.get("hypotheses", ["pos", "neg"]))
        self.batch_size = batch_size
        self.version = f"sent-cal@{cal.get('version', 1)}"
        log.info("loaded calibrated sentiment in %.1fs", time.time() - t)

    def features(self, items: list) -> list[list[float]]:
        """The model outputs the calibration is fitted on, per (window, target name): the NLI
        logits of each hypothesis, then the classifier's probabilities. Batches are formed over
        the windows sorted by length, so little is padded."""
        items = [x if isinstance(x, tuple) else (x, "") for x in items]
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        out: dict[int, list[float]] = {}
        with self.torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idxs = order[s : s + self.batch_size]
                batch = [items[i] for i in idxs]
                nl = self.nli.logits(self.torch, batch, self.hypotheses)
                cp = self.clf.probs(self.torch, [x[0] for x in batch])
                for i, a, c in zip(idxs, nl, cp, strict=True):
                    out[i] = a + list(c)
        return [out[i] for i in range(len(items))]

    def probs(self, items: list) -> list[tuple[float, float, float]]:
        if not items:
            return []
        x = self.torch.tensor(self.features(items))
        p = self.torch.softmax((x - self.mean) / self.std @ self.W + self.b, dim=-1).tolist()
        return [(row[0], row[1], row[2]) for row in p]

    def predict(self, items: list) -> list[tuple[str, float, float]]:
        return [to_label(*p) for p in self.probs(items)]


FT_META = Path(__file__).resolve().parents[1] / "data" / "sentiment_ft.json"


class FineTunedSentiment:
    """Fine-tuned target sentiment ("ft"): one model pass per window.

    The NLI model fine-tuned on the hand-labelled pairs (scripts/train_sentiment_finetune.py;
    evaluation in eval/README.md). Premise: the mention's sentence +- one sentence with the
    target in [brackets]; hypothesis: "Der Kommentar äußert sich positiv über {name}."; the NLI
    head then reads entailment as positive, neutral as neutral and contradiction as negative
    toward the entity. The weights are assets of a GitHub release, downloaded once and verified
    against the hashes in data/sentiment_ft.json (analysis/finetune.py).
    """

    window_mode = "target"

    def __init__(self, batch_size: int = 16, meta: Path = FT_META):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from .finetune import ensure_model

        t = time.time()
        self.torch = torch
        cfg = json.loads(meta.read_text(encoding="utf-8"))
        self.cfg = cfg
        path = ensure_model(cfg)
        self.tok = AutoTokenizer.from_pretrained(path)
        # stored in half precision, run in single precision (exact conversion)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.float32).eval()
        digest = cfg["files"]["model.safetensors"][:12]
        self.forward = _forward(self.model, self.tok, f"meinungspuls/{cfg['release']}-{digest}", pair=True)
        self.order = list(cfg["logitOrder"])  # model logits -> (pos, neu, neg)
        self.hypothesis = cfg["hypothesis"]
        self.max_length = int(cfg["maxLength"])
        self.batch_size = batch_size
        self.version = f"sent-ft@{cfg['version']}"
        log.info("loaded fine-tuned sentiment %s in %.1fs", cfg["release"], time.time() - t)

    def probs(self, items: list) -> list[tuple[float, float, float]]:
        """(p_pos, p_neu, p_neg) per (window, target name); batches over the windows sorted by length."""
        items = [x if isinstance(x, tuple) else (x, "") for x in items]
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        out: dict[int, tuple[float, float, float]] = {}
        with self.torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idxs = order[s : s + self.batch_size]
                enc = self.tok(
                    [items[i][0] or "." for i in idxs],
                    [self.hypothesis.format(name=items[i][1]) for i in idxs],
                    padding=True,
                    truncation="only_first",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                p = self.torch.softmax(self.forward(enc)[:, self.order], dim=-1).tolist()
                for i, row in zip(idxs, p, strict=True):
                    out[i] = (row[0], row[1], row[2])
        return [out[i] for i in range(len(items))]

    def predict(self, items: list) -> list[tuple[str, float, float]]:
        return [to_label(*p) for p in self.probs(items)]


class SentimentModel:
    """Target sentiment model.

    key: one of MODELS (single classifier), "ens" (mean of ENSEMBLE), one of NLI_MODELS
    (zero-shot NLI, target-aware), or "a+b" combinations (probabilities averaged with
    `weights`). The label uses the margin P(pos) - P(neg) with `threshold`, if given.
    """

    def __init__(
        self,
        key: str = "ens",
        batch_size: int = 32,
        max_length: int = 256,
        threshold: float | None = None,
        weights: list[float] | None = None,
        window_mode: str = "clause",
    ):
        import torch

        t = time.time()
        self.torch = torch
        parts = key.split("+")
        self.keys = [k for p in parts for k in (ENSEMBLE if p == "ens" else (p,))]
        self.models = [
            _Nli(NLI_MODELS[k], max_length) if k in NLI_MODELS else _Single(k, max_length) for k in self.keys
        ]
        self.weights = weights or [1.0] * len(self.models)
        self.threshold = threshold if threshold is not None else (THRESHOLD if key == "ens" else None)
        self.batch_size = batch_size
        self.window_mode = window_mode
        self.version = f"sent-{key}-{window_mode}@3"
        log.info("loaded sentiment %s in %.1fs", "+".join(self.keys), time.time() - t)

    def probs(self, items: list) -> list[tuple[float, float, float]]:
        """Weighted mean (p_pos, p_neu, p_neg) per (text, target name) item."""
        items = [x if isinstance(x, tuple) else (x, "") for x in items]
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        out: dict[int, tuple[float, float, float]] = {}
        wsum = sum(self.weights)
        with self.torch.inference_mode():
            for b in range(0, len(order), self.batch_size):
                idxs = order[b : b + self.batch_size]
                batch = [items[i] for i in idxs]
                per_model = [
                    m.probs(self.torch, batch) if isinstance(m, _Nli) else m.probs(self.torch, [x[0] for x in batch])
                    for m in self.models
                ]
                for j, i in enumerate(idxs):
                    out[i] = tuple(
                        sum(w * pm[j][c] for w, pm in zip(self.weights, per_model, strict=True)) / wsum
                        for c in range(3)
                    )
        return [out[i] for i in range(len(items))]

    def predict(self, items: list) -> list[tuple[str, float, float]]:
        """Return (label, score, confidence) per window; items are texts or (text, target name)."""
        return [to_label(*p, self.threshold) for p in self.probs(items)]


def to_label(p_pos: float, p_neu: float, p_neg: float, threshold: float | None = None) -> tuple[str, float, float]:
    """Map class probabilities to (label, score in [-1, 1], confidence in [0, 1]).

    Without threshold: argmax. With threshold t: pos if score > t, neg if score < -t, else neu.
    """
    probs = {"pos": p_pos, "neu": p_neu, "neg": p_neg}
    score = p_pos - p_neg
    if threshold is None:
        label = max(probs, key=probs.get)
    else:
        label = "pos" if score > threshold else "neg" if score < -threshold else "neu"
    return label, round(score, 4), round(probs[label], 4)


def combine(results: list[tuple[str, float, float]]) -> tuple[str, float, float]:
    """Merge several windows of the same (comment, entity): mean score, majority-by-mass label."""
    if len(results) == 1:
        return results[0]
    score = sum(r[1] for r in results) / len(results)
    mass = {"pos": 0.0, "neu": 0.0, "neg": 0.0}
    for lab, _s, conf in results:
        mass[lab] += conf
    label = max(mass, key=mass.get)
    conf = mass[label] / len(results)
    return label, round(score, 4), round(conf, 4)


def load_sentiment(key: str = DEFAULT_SENTIMENT, **kw):
    """Factory used by the pipeline: "cal" (default, calibrated NLI + classifier), "ft" (fine-tuned
    NLI model; needs data/sentiment_ft.json and its release) or any SentimentModel key."""
    if key == "cal":
        return CalibratedSentiment()
    if key == "ft":
        return FineTunedSentiment()
    return SentimentModel(key, **kw)
