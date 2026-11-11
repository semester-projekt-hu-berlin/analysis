"""Storage of the fine-tuned sentiment model (see analysis/sentiment.py, FineTunedSentiment).

The fine-tuned model is the public NLI model (pinned revision, from the Hugging Face cache) plus
what training changed, in half precision: every encoder and head tensor, and of the embedding
matrix (250,000 tokens) only the rows that changed (a few thousand). That delta is about 180 MB,
split into parts below GitHub's 100 MB file limit, and kept in the git branch `models` of this
repository, outside the history of `main`. data/sentiment_ft.json names the commit and the
SHA-256 of every part; `ensure_model` downloads what is missing into the model cache and
verifies it, so a wrong or damaged file is never loaded.

Loading rebuilds exactly the model that was evaluated: the base weights rounded to half
precision, then the stored tensors and rows put over them.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

ROWS = ".rows"  # suffix of the index tensor of a row-sparse delta ("<name>.rows" + "<name>")
PART_BYTES = 90_000_000  # below GitHub's 100 MB limit per file


def round_half(model) -> None:
    """Round all weights to half precision in place (they stay float32 tensors): the model that
    is evaluated after training is exactly the one that is stored."""
    import torch

    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.half().float())


def delta(trained: dict, base: dict) -> dict:
    """Tensors of `trained` that differ from `base` in half precision. A 2-d tensor of which only
    some rows changed (the embedding matrix) is stored as "<name>.rows" (indices) + "<name>"
    (those rows); other changed tensors whole."""
    import torch

    out = {}
    for name, t in trained.items():
        t16, b16 = t.half(), base[name].half()
        diff = t16 != b16
        if not diff.any():
            continue
        if t16.dim() == 2:
            rows = diff.any(dim=1).nonzero().flatten()
            if len(rows) < t16.shape[0] // 2:
                out[name + ROWS] = rows.to(torch.int64).contiguous()
                out[name] = t16[rows].contiguous()
                continue
        out[name] = t16.contiguous()
    return out


def apply_delta(model, tensors: dict) -> int:
    """Rebuild the fine-tuned model from a freshly loaded base model, in place: round to half
    precision, then put the stored tensors and rows over it. Returns the number of tensors.
    Raises ValueError if a tensor does not fit the model."""
    import torch

    round_half(model)
    params = dict(model.named_parameters())
    n = 0
    with torch.no_grad():
        for name, t in tensors.items():
            if name.endswith(ROWS):
                continue
            if name not in params:
                raise ValueError(f"fine-tuned tensor {name} is not in the base model")
            p = params[name]
            rows = tensors.get(name + ROWS)
            if rows is not None:
                if t.shape != (len(rows), p.shape[1]):
                    raise ValueError(f"rows of {name} do not fit the base model")
                p[rows] = t.to(p.dtype)
            elif t.shape != p.shape:
                raise ValueError(f"fine-tuned tensor {name} does not fit the base model")
            else:
                p.copy_(t.to(p.dtype))
            n += 1
    return n


def split(tensors: dict, limit: int = PART_BYTES) -> list[dict]:
    """Group tensors into parts of at most `limit` bytes (a tensor and its row index stay together)."""
    parts: list[dict] = [{}]
    size = 0
    for name in sorted(k for k in tensors if not k.endswith(ROWS)):
        group = {name: tensors[name]}
        if name + ROWS in tensors:
            group[name + ROWS] = tensors[name + ROWS]
        n = sum(t.numel() * t.element_size() for t in group.values())
        if parts[-1] and size + n > limit:
            parts.append({})
            size = 0
        parts[-1].update(group)
        size += n
    return parts


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
    """Directory with the verified files named in `meta` (downloaded if missing)."""
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
