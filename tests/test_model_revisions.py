"""Every Hugging Face model is loaded at a pinned commit, never at whatever `main` is today."""

from __future__ import annotations

import json
import re
import sys
import types

import pytest

from analysis import ner, sentiment

SHA = re.compile(r"[0-9a-f]{40}")


def test_every_model_has_a_pinned_commit():
    ids = {name for name, _ in sentiment.MODELS.values()} | set(sentiment.NLI_MODELS.values())
    assert ids <= set(sentiment.REVISIONS)
    assert ner.BERT_NER_MODEL in ner.REVISIONS
    for rev in [*sentiment.REVISIONS.values(), *ner.REVISIONS.values()]:
        assert SHA.fullmatch(rev)
    # the fine-tuned delta fits the base revision it was trained on, which is the pinned one
    cfg = json.loads(sentiment.FT_META.read_text(encoding="utf-8"))
    assert sentiment.REVISIONS[cfg["baseModel"]] == cfg["baseRevision"]


@pytest.fixture
def fake_transformers(monkeypatch):
    """A stand-in `transformers` module that records the revision of every load."""
    calls = []

    class _Model:
        def __init__(self, name=""):
            labels = ("entailment", "neutral", "contradiction") if "nli" in name else ("pos", "neu", "neg")
            self.config = types.SimpleNamespace(id2label=dict(enumerate(labels)))

        def eval(self):
            return self

    class _Auto:
        def __init__(self, kind):
            self.kind = kind

        def from_pretrained(self, name, **kw):
            calls.append((self.kind, name, kw.get("revision")))
            return _Model(name)

    def pipeline(task, model, **kw):
        calls.append(("pipeline", model, kw.get("revision")))
        return types.SimpleNamespace(model=_Model(), tokenizer=object())

    module = types.ModuleType("transformers")
    module.AutoTokenizer = _Auto("tokenizer")
    module.AutoModelForSequenceClassification = _Auto("model")
    module.pipeline = pipeline
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setenv("ANALYSIS_ONNX", "0")
    return calls


def test_loaders_pass_the_pinned_revision(fake_transformers):
    ner.BertNer()
    sentiment._Single("news", 256)
    sentiment._Nli(sentiment.NLI_MODELS["nli"], 256)
    news = "mdraw/german-news-sentiment-bert"
    nli = sentiment.NLI_MODELS["nli"]
    assert fake_transformers == [
        ("pipeline", ner.BERT_NER_MODEL, ner.REVISIONS[ner.BERT_NER_MODEL]),
        ("tokenizer", news, sentiment.REVISIONS[news]),
        ("model", news, sentiment.REVISIONS[news]),
        ("tokenizer", nli, sentiment.REVISIONS[nli]),
        ("model", nli, sentiment.REVISIONS[nli]),
    ]
