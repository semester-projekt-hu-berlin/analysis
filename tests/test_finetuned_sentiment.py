"""Fine-tuned sentiment model: target window, verified model files, label mapping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis import finetune
from analysis.analyze import mention_window
from analysis.linker import Match
from analysis.sentiment import FT_META
from analysis.textutil import split_sentences

TEXT = "Das war absehbar. Merz hat es versprochen, und die SPD hat Merz geglaubt. Jetzt zahlen alle. Schade."


def _matches():
    out = []
    for surface, eid in (("Merz", "friedrich-merz"), ("SPD", "spd")):
        start = 0
        while (i := TEXT.find(surface, start)) >= 0:
            out.append(Match(i, i + len(surface), surface, eid, "alias"))
            start = i + 1
    return sorted(out, key=lambda m: m.start)


def test_target_window_marks_the_target_in_its_sentence_and_neighbours():
    sents = split_sentences(TEXT)
    ms = _matches()
    merz = next(m for m in ms if m.entity_id == "friedrich-merz")
    w = mention_window(TEXT, sents, merz, ms, "target")
    # sentence +- 1, every mention of the target in brackets, other entities untouched
    assert w == "Das war absehbar. [Merz] hat es versprochen, und die SPD hat [Merz] geglaubt. Jetzt zahlen alle."
    spd = next(m for m in ms if m.entity_id == "spd")
    assert "die [SPD] hat Merz geglaubt" in mention_window(TEXT, sents, spd, ms, "target")
    assert "Schade" not in w


def test_target_window_of_a_single_sentence():
    text = "Die FPÖ macht das nicht besser."
    m = Match(4, 7, "FPÖ", "fpoe", "alias")
    assert mention_window(text, split_sentences(text), m, [m], "target") == "Die [FPÖ] macht das nicht besser."


def test_other_window_modes_are_unchanged():
    sents = split_sentences(TEXT)
    ms = _matches()
    merz = next(m for m in ms if m.entity_id == "friedrich-merz")
    assert "[" not in mention_window(TEXT, sents, merz, ms, "clause")
    assert "X" in mention_window(TEXT, sents, merz, ms, "entity")


# ---------------------------------------------------------------- model files


def _meta(tmp_path: Path, files: dict[str, bytes]) -> dict:
    src = tmp_path / "release"
    src.mkdir()
    for name, data in files.items():
        (src / name).write_bytes(data)
    return {
        "release": "sentiment-test",
        "url": "https://example.invalid/releases/download/sentiment-test",
        "files": {name: finetune.sha256(src / name) for name in files},
    }


def test_ensure_model_downloads_verifies_and_reuses(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYSIS_MODEL_DIR", str(tmp_path / "models"))
    meta = _meta(tmp_path, {"config.json": b"{}", "model.safetensors": b"weights"})
    calls = []

    def fetch(url, dest):
        calls.append(url)
        dest.write_bytes((tmp_path / "release" / dest.name).read_bytes())

    d = finetune.ensure_model(meta, fetch)
    assert d == tmp_path / "models" / "sentiment-test"
    assert sorted(calls) == [f"{meta['url']}/config.json", f"{meta['url']}/model.safetensors"]
    assert (d / "model.safetensors").read_bytes() == b"weights"
    finetune.ensure_model(meta, fetch)  # verified files are not fetched again
    assert len(calls) == 2
    (d / "model.safetensors").write_bytes(b"damaged")  # a damaged file is replaced
    finetune.ensure_model(meta, fetch)
    assert len(calls) == 3 and (d / "model.safetensors").read_bytes() == b"weights"


def test_ensure_model_rejects_a_wrong_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYSIS_MODEL_DIR", str(tmp_path / "models"))
    meta = _meta(tmp_path, {"model.safetensors": b"weights"})
    with pytest.raises(RuntimeError, match="SHA-256"):
        finetune.ensure_model(meta, lambda url, dest: dest.write_bytes(b"something else"))
    assert not (tmp_path / "models" / "sentiment-test" / "model.safetensors").exists()


def test_model_dir_is_inside_the_model_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("ANALYSIS_MODEL_DIR", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert finetune.model_dir("sentiment-ft-v3") == tmp_path / "hf" / "meinungspuls" / "sentiment-ft-v3"


def test_meta_file_names_a_release_with_hashes():
    if not FT_META.exists():
        pytest.skip("no fine-tuned model in this checkout")
    meta = json.loads(FT_META.read_text(encoding="utf-8"))
    assert meta["release"] == f"sentiment-ft-v{meta['version']}"
    assert meta["url"].endswith("/releases/download/" + meta["release"])
    assert {"config.json", "model.safetensors"} <= meta["files"].keys()
    assert all(len(h) == 64 and set(h) <= set("0123456789abcdef") for h in meta["files"].values())
    assert "{name}" in meta["hypothesis"] and meta["window"] == "target"
    assert sorted(meta["logitOrder"]) == [0, 1, 2] and meta["classes"] == ["pos", "neu", "neg"]


# ---------------------------------------------------------------- inference with a tiny random model

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "der", "kommentar", "äußert", "sich", "positiv", "über"]
WORDS += ["merz", "spd", "gut", "schlecht", "ist", ".", "[", "]", "die"]


@pytest.fixture
def tiny_ft(tmp_path, monkeypatch):
    """A small random BERT classifier published like the real model (nothing is downloaded)."""
    monkeypatch.setenv("ANALYSIS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("ANALYSIS_ONNX", "0")
    d = finetune.model_dir("sentiment-ft-v9")
    d.mkdir(parents=True)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(WORDS), encoding="utf-8")
    transformers.BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(d)
    cfg = transformers.BertConfig(
        vocab_size=len(WORDS), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64
    )
    cfg.num_labels = 3
    torch.manual_seed(0)
    model = transformers.BertForSequenceClassification(cfg).eval()
    finetune.round_half(model)
    reference = {k: v.clone() for k, v in model.state_dict().items()}
    model.half().save_pretrained(d)
    meta = {
        "version": 9,
        "release": "sentiment-ft-v9",
        "url": "https://example.invalid/none",
        "files": {f.name: finetune.sha256(f) for f in d.iterdir()},
        "hypothesis": "Der Kommentar äußert sich positiv über {name}.",
        "window": "target",
        "maxLength": 32,
        "classes": ["pos", "neu", "neg"],
        "logitOrder": [2, 0, 1],
    }
    path = tmp_path / "sentiment_ft.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path, reference


def test_fine_tuned_model_loads_verified_files_and_maps_the_logits(tiny_ft):
    from analysis.sentiment import FineTunedSentiment, to_label

    path, reference = tiny_ft
    model = FineTunedSentiment(batch_size=2, meta=path)
    assert model.version == "sent-ft@9" and model.window_mode == "target"
    # half precision on disk, single precision in memory, and exactly the rounded weights
    state = model.model.state_dict()
    assert all(v.dtype == torch.float32 for v in state.values() if v.is_floating_point())
    assert all(torch.equal(state[k], v) for k, v in reference.items())
    items = [("[Merz] ist gut.", "Merz"), ("Die [SPD] ist schlecht. Merz ist gut.", "SPD"), ("", "SPD")]
    probs = model.probs(items)
    assert len(probs) == 3 and all(abs(sum(p) - 1) < 1e-5 for p in probs)
    enc = model.tok(
        ["[Merz] ist gut."], ["Der Kommentar äußert sich positiv über Merz."], return_tensors="pt", truncation=True
    )
    with torch.inference_mode():
        want = torch.softmax(model.model(**enc).logits[0, [2, 0, 1]], dim=-1).tolist()
    assert probs[0] == pytest.approx(want, abs=1e-5)
    # deterministic, independent of batch composition and order
    assert model.probs(list(reversed(items))) == pytest.approx(list(reversed(probs)), abs=1e-5)
    assert model.predict(items) == [to_label(*p) for p in probs]
    assert model.probs([]) == []


def test_fine_tuned_model_refuses_changed_weights(tiny_ft, monkeypatch):
    from analysis.sentiment import FineTunedSentiment

    def no_network(url, dest):
        raise RuntimeError(f"download of {url} failed")

    monkeypatch.setattr(finetune, "_download", no_network)
    path, _ = tiny_ft
    weights = finetune.model_dir("sentiment-ft-v9") / "model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-1] + b"\x00")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if finetune.sha256(weights) == meta["files"]["model.safetensors"]:
        weights.write_bytes(weights.read_bytes()[:-1] + b"\x01")
    with pytest.raises(RuntimeError):  # the changed file is re-fetched, and the fetch fails here
        FineTunedSentiment(meta=path)


def test_load_sentiment_knows_the_fine_tuned_model(tiny_ft, monkeypatch):
    from analysis import sentiment

    path, _ = tiny_ft
    monkeypatch.setattr(sentiment, "FT_META", path)
    monkeypatch.setattr(sentiment.FineTunedSentiment.__init__, "__defaults__", (16, path))
    assert sentiment.load_sentiment("ft").version == "sent-ft@9"
