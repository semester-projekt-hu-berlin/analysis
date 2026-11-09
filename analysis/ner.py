"""Named entity recognition backends.

Default: Flair `flair/ner-german` (course recommendation, BiLSTM-CRF over Flair
contextual string embeddings, CoNLL-03 German, F1 ~87). A faster BERT alternative and
a no-op backend (gazetteer-only) are available via `--ner`.

All backends take a list of sentence strings and return, per sentence, a list of
(start, end, tag, score) character spans relative to that sentence.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

Span = tuple[int, int, str, float]


class NoNer:
    name = "none"
    version = "none@0"

    def predict(self, sentences: list[str]) -> list[list[Span]]:
        return [[] for _ in sentences]


class FlairNer:
    name = "flair"
    version = "flair-ner-german@1"

    def __init__(self, model: str = "flair/ner-german", batch_size: int = 32):
        from flair.models import SequenceTagger

        t = time.time()
        self.tagger = SequenceTagger.load(model)
        self.batch_size = batch_size
        log.info("loaded NER %s in %.1fs", model, time.time() - t)

    def predict(self, sentences: list[str]) -> list[list[Span]]:
        from flair.data import Sentence

        objs = [Sentence(s) if s.strip() else None for s in sentences]
        todo = [o for o in objs if o is not None and len(o) > 0]
        if todo:
            self.tagger.predict(todo, mini_batch_size=self.batch_size, verbose=False)
        out: list[list[Span]] = []
        for o in objs:
            if o is None or len(o) == 0:
                out.append([])
                continue
            out.append([(sp.start_position, sp.end_position, sp.tag, float(sp.score)) for sp in o.get_spans("ner")])
        return out


class BertNer:
    """Transformer token classifier fine-tuned on GermEval14 (~3x faster than Flair on CPU)."""

    name = "bert"
    version = "bert-germeval-ner@1"

    def __init__(self, model: str = "domischwimmbeck/bert-base-german-cased-fine-tuned-ner", batch_size: int = 32):
        from transformers import pipeline

        self.pipe = pipeline("token-classification", model=model, aggregation_strategy="simple", device=-1)
        self.batch_size = batch_size

    def predict(self, sentences: list[str]) -> list[list[Span]]:
        if not sentences:
            return []
        # Each distinct sentence once, shortest first: a batch is padded to its longest sentence,
        # so batches of similar length waste no compute on padding (the spans are the same).
        unique = sorted(dict.fromkeys(sentences), key=len)
        res = self.pipe(unique, batch_size=self.batch_size)
        by_sentence: dict[str, list[Span]] = {}
        for s, r in zip(unique, res, strict=True):
            spans = []
            for x in r:
                tag = x["entity_group"].split("-")[-1].upper()
                tag = {"PERSON": "PER", "LOCATION": "LOC", "OTH": "MISC"}.get(tag, tag)
                spans.append((int(x["start"]), int(x["end"]), tag, float(x["score"])))
            by_sentence[s] = spans
        return [list(by_sentence[s]) for s in sentences]


def load_ner(name: str):
    if name == "flair":
        return FlairNer()
    if name == "bert":
        return BertNer()
    if name == "none":
        return NoNer()
    raise ValueError(f"unknown NER backend {name}")
