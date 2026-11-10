"""Optional ONNX Runtime inference for the transformer models (NER, NLI, news sentiment).

The same weights run through ONNX Runtime instead of PyTorch. The graph is exported once per
model revision into a cache directory, and every load checks the session against the PyTorch
model on a few probe sentences: if the logits differ by more than TOLERANCE, or anything in
export or loading fails, the caller keeps using PyTorch. So the backend can only change the
speed, not the results (measured: logits equal to ~1e-4, identical labels on the evaluation
sets; see eval/README.md).

    ANALYSIS_ONNX=0        use PyTorch only
    ANALYSIS_ONNX_DIR      where the exported graphs are kept (default: $HF_HOME/onnx, so the
                           CI cache of the models keeps them too)
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import time
from pathlib import Path

log = logging.getLogger(__name__)

TOLERANCE = 2e-3  # largest accepted difference of a logit between ONNX Runtime and PyTorch
OPSET = 17
FORMAT = 1  # bump to re-export after a change of the export below
PROBES = [
    "Das ist ein Test.",
    "Friedrich Merz und die SPD streiten in Berlin über den Haushalt, sagt der Kommentar zum Artikel.",
    "Gut.",
]
PROBE_PAIRS = ["Der Kommentar äußert sich positiv über Merz.", "Der Kommentar bewertet die SPD.", "Ja."]


def enabled() -> bool:
    return os.environ.get("ANALYSIS_ONNX", "0").strip().lower() in ("1", "true", "on", "yes")


def cache_dir() -> Path:
    d = os.environ.get("ANALYSIS_ONNX_DIR")
    if d:
        return Path(d)
    hf = os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface")
    return Path(hf) / "onnx"


class OrtModel:
    """An ONNX Runtime session that maps a tokenizer encoding to a logits tensor."""

    def __init__(self, session, torch):
        self.session = session
        self.torch = torch
        self.inputs = [i.name for i in session.get_inputs()]

    def logits(self, enc):
        feed = {k: enc[k].numpy() for k in self.inputs}
        return self.torch.from_numpy(self.session.run(["logits"], feed)[0])


def _path(name: str, model) -> Path:
    rev = str(getattr(model.config, "_commit_hash", None) or "local")[:12]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")
    return cache_dir() / f"{slug}-{rev}-v{FORMAT}.onnx"


def _export(model, enc: dict, path: Path, torch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    names = list(enc)
    old = model.config.return_dict
    model.config.return_dict = False  # the exporter traces tuples
    try:
        with torch.inference_mode(False), torch.no_grad():
            torch.onnx.export(
                model,
                tuple(enc[k] for k in names),
                str(tmp),
                input_names=names,
                output_names=["logits"],
                dynamic_axes={**{k: {0: "batch", 1: "tokens"} for k in names}, "logits": {0: "batch"}},
                opset_version=OPSET,
                dynamo=False,
            )
        tmp.replace(path)
    finally:
        model.config.return_dict = old
        tmp.unlink(missing_ok=True)


def accelerate(model, tokenizer, name: str, pair: bool = False, threads: int | None = None) -> OrtModel | None:
    """ONNX Runtime session for `model`, or None (keep PyTorch): disabled, onnxruntime missing,
    export or load failed, or the session does not reproduce the PyTorch logits."""
    if not enabled():
        return None
    try:
        import onnxruntime as ort
        import torch
    except ImportError:
        return None
    t = time.time()
    try:
        texts = (PROBES, PROBE_PAIRS) if pair else (PROBES,)
        raw = tokenizer(*texts, padding=True, return_tensors="pt")
        # the exporter passes the inputs by position: keep the order of the model's forward()
        enc = {k: raw[k] for k in inspect.signature(model.forward).parameters if k in raw}
        path = _path(name, model)
        exported = not path.exists()
        if exported:
            _export(model, enc, path, torch)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or torch.get_num_threads()
        opts.log_severity_level = 3
        session = OrtModel(ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"]), torch)
        with torch.inference_mode():
            want = model(**enc).logits
        diff = float((session.logits(enc) - want).abs().max())
        if not diff <= TOLERANCE:
            log.warning("ONNX Runtime differs from PyTorch for %s (max |diff| %.4f): using PyTorch", name, diff)
            path.unlink(missing_ok=True)  # a stale or broken export: build it again next time
            return None
        log.info(
            "ONNX Runtime for %s (%s in %.1fs, max |diff| to PyTorch %.1e)",
            name,
            "exported" if exported else "cached",
            time.time() - t,
            diff,
        )
        return session
    except Exception as e:  # any failure of the optional backend must not stop the run
        log.warning("ONNX Runtime not used for %s: %s: %s", name, type(e).__name__, str(e)[:300])
        return None
