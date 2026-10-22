"""Sentiment scoring of the text window around an entity mention.

The window is the sentence containing the mention; if the sentence mentions several
entities and contains a contrastive clause boundary ("aber", "während", ";" ...), the
window shrinks to the clause containing the mention (cheap target dependence). The
window is classified by a German 3-class sentiment BERT; the score is
P(pos) - P(neg) in [-1, 1], the label the argmax class, the confidence its probability.
"""

from __future__ import annotations

import logging
import re
import time

log = logging.getLogger(__name__)

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


class _Single:
    def __init__(self, key: str, max_length: int):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name, self.clean = MODELS[key]
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.max_length = max_length
        labels = {i: lab.lower() for i, lab in self.model.config.id2label.items()}
        self.idx = {k: next(i for i, lab in labels.items() if lab.startswith(k)) for k in ("pos", "neu", "neg")}

    def probs(self, torch, texts: list[str]) -> list[tuple[float, float, float]]:
        prepared = [(guhr_clean(t) if self.clean else t) or "." for t in texts]
        enc = self.tok(prepared, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        out = torch.softmax(self.model(**enc).logits, dim=-1).tolist()
        return [(p[self.idx["pos"]], p[self.idx["neu"]], p[self.idx["neg"]]) for p in out]


# Ensemble default, chosen on the hand-labelled sample (see eval/README.md).
ENSEMBLE = ("news", "guhr-raw")
THRESHOLD = 0.2  # |P(pos) - P(neg)| above this -> pos/neg, else neu (models are over-neutral on comments)


class SentimentModel:
    """One model key from MODELS, or "ens" = mean of ENSEMBLE models with a score threshold."""

    def __init__(self, key: str = "ens", batch_size: int = 32, max_length: int = 256):
        import torch

        t = time.time()
        self.torch = torch
        self.keys = list(ENSEMBLE) if key == "ens" else [key]
        self.models = [_Single(k, max_length) for k in self.keys]
        self.threshold = THRESHOLD if key == "ens" else None
        self.batch_size = batch_size
        self.version = f"sent-{key}@2"
        log.info("loaded sentiment %s in %.1fs", "+".join(self.keys), time.time() - t)

    def predict(self, texts: list[str]) -> list[tuple[str, float, float]]:
        """Return (label, score, confidence) per text."""
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        results: dict[int, tuple[str, float, float]] = {}
        with self.torch.inference_mode():
            for b in range(0, len(order), self.batch_size):
                idxs = order[b : b + self.batch_size]
                batch = [texts[i] for i in idxs]
                per_model = [m.probs(self.torch, batch) for m in self.models]
                for j, i in enumerate(idxs):
                    ps = [pm[j] for pm in per_model]
                    p_pos = sum(p[0] for p in ps) / len(ps)
                    p_neu = sum(p[1] for p in ps) / len(ps)
                    p_neg = sum(p[2] for p in ps) / len(ps)
                    results[i] = to_label(p_pos, p_neu, p_neg, self.threshold)
        return [results[i] for i in range(len(texts))]


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
