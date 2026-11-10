"""ONNX Runtime backend: used only when it reproduces the PyTorch model, never required."""

import pytest

from analysis import onnx_backend

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

WORDS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "das",
    "ist",
    "ein",
    "test",
    ".",
    "friedrich",
    "merz",
    "und",
    "die",
    "spd",
    "gut",
    "ja",
    "der",
    "kommentar",
]


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """A small random BERT classifier with its tokenizer (nothing is downloaded)."""
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(WORDS), encoding="utf-8")
    tok = transformers.BertTokenizerFast(vocab_file=str(vocab))
    cfg = transformers.BertConfig(
        vocab_size=len(WORDS), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64
    )
    torch.manual_seed(0)
    model = transformers.BertForSequenceClassification(cfg).eval()
    monkeypatch.setenv("ANALYSIS_ONNX_DIR", str(tmp_path / "onnx"))
    monkeypatch.setenv("ANALYSIS_ONNX", "1")
    return model, tok, tmp_path / "onnx"


def test_off_by_default(tiny, monkeypatch):
    model, tok, cache = tiny
    monkeypatch.delenv("ANALYSIS_ONNX")
    assert onnx_backend.accelerate(model, tok, "tiny/bert") is None
    assert not cache.exists()


def test_session_reproduces_the_model_and_is_cached(tiny):
    model, tok, cache = tiny
    session = onnx_backend.accelerate(model, tok, "tiny/bert")
    assert session is not None
    files = list(cache.glob("*.onnx"))
    assert len(files) == 1 and files[0].name.startswith("tiny-bert-")
    # other batch shapes than the probes, with padding
    enc = tok(["das ist gut", "friedrich merz und die spd . ja ja ja", "test"], padding=True, return_tensors="pt")
    with torch.inference_mode():
        want = model(**enc).logits
    assert float((session.logits(enc) - want).abs().max()) < 1e-4
    # pairs, and the second load uses the exported file
    pair = tok(["das ist gut", "ja"], ["der kommentar", "merz ist gut"], padding=True, return_tensors="pt")
    stamp = files[0].stat().st_mtime_ns
    again = onnx_backend.accelerate(model, tok, "tiny/bert", pair=True)
    assert files[0].stat().st_mtime_ns == stamp
    with torch.inference_mode():
        assert float((again.logits(pair) - model(**pair).logits).abs().max()) < 1e-4


def test_session_that_differs_is_not_used(tiny, monkeypatch):
    model, tok, cache = tiny
    monkeypatch.setattr(onnx_backend, "TOLERANCE", -1.0)
    assert onnx_backend.accelerate(model, tok, "tiny/bert") is None
    assert not list(cache.glob("*.onnx"))  # removed, exported again next time


def test_failing_export_falls_back(tiny, monkeypatch):
    model, tok, _cache = tiny

    def boom(*a, **kw):
        raise RuntimeError("exporter removed")

    monkeypatch.setattr(torch.onnx, "export", boom)
    assert onnx_backend.accelerate(model, tok, "tiny/bert") is None
