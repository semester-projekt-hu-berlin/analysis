"""Storage of the fine-tuned sentiment model (see analysis/sentiment.py, FineTunedSentiment).

The weights (the NLI model fine-tuned on the hand-labelled pairs, half precision, about 560 MB)
are too large for git. They are assets of a GitHub release of this repository; data/sentiment_ft.json
names the release and the SHA-256 of every file. `ensure_model` downloads what is missing into
the model cache and verifies it, so a wrong or damaged file is never loaded.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)


def round_half(model) -> None:
    """Round all weights to half precision in place (they stay float32 tensors): the model that
    is evaluated after training is exactly the one that is stored."""
    import torch

    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.half().float())


def model_dir(release: str) -> Path:
    d = os.environ.get("ANALYSIS_MODEL_DIR")
    if d:
        return Path(d) / release
    hf = os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface")
    return Path(hf) / "meinungspuls" / release  # inside the CI model cache


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path, attempts: int = 3) -> None:
    tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.part")
    for k in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
            tmp.replace(dest)
            return
        except OSError as e:
            tmp.unlink(missing_ok=True)
            if k == attempts - 1:
                raise RuntimeError(f"download of {url} failed: {e}") from e
            time.sleep(5 * (k + 1))


def ensure_model(meta: dict, fetch=None) -> Path:
    """Directory with the verified files of the release named in `meta` (downloaded if missing)."""
    d = model_dir(meta["release"])
    d.mkdir(parents=True, exist_ok=True)
    for name, digest in meta["files"].items():
        path = d / name
        if path.exists() and sha256(path) == digest:
            continue
        t = time.time()
        (fetch or _download)(f"{meta['url'].rstrip('/')}/{name}", path)
        got = sha256(path)
        if got != digest:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"{name} of {meta['release']} has SHA-256 {got[:12]}, expected {digest[:12]}")
        log.info("downloaded %s (%.0f MB) in %.0fs", name, path.stat().st_size / 1e6, time.time() - t)
    return d
