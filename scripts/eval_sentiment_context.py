"""Does the article headline help the fine-tuned target-sentiment model? (GitHub Actions only)

Same protocol as sent-ft@3 (scripts/train_sentiment_finetune.py cv, scripts/eval_protocol.py):
all labelled pairs, 5 folds stratified over comments with fold seed 0, the same training settings
and seeds per fold, every pair scored once by a model that did not see its comment. Variants of
the premise (the hypothesis and everything else stay the same):

* base:        the mention window (sent-ft@3, sanity check against its 0.557)
* title:       "Artikel: {headline} | Kommentar: {window}" whenever the article has a headline
* title-short: the same, but only for windows of fewer than SHORT_WORDS words

Privacy: this repository is public, and so are its workflow logs and artifacts. Comment texts and
headlines are read from Firestore into memory only and never written to disk, printed or
uploaded; nothing per pair is printed. The only thing that leaves a fold job is its out-of-fold
labels and scores (no text), encrypted with a key derived from the Firestore secret, so the
report job can pool the folds; the artifact cannot be read without the secret and expires
after a day. The report prints aggregates only.

    python scripts/eval_sentiment_context.py fold --fold 0 --out fold-0.bin   # one fold, all variants
    python scripts/eval_sentiment_context.py report --dir folds/                 # pooled metrics

Workflow: .github/workflows/eval-context.yml (manual dispatch). Results: eval/README.md.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_protocol import (  # noqa: E402
    CLASSES,
    bootstrap_ci,
    bootstrap_diff,
    confusion,
    grouped_folds,
    macro_f1,
    pearson,
    per_class,
)

LABEL_FILES = [ROOT / "eval" / f for f in ("labels.csv", "labels_extra.csv", "labels_extra2.csv")]
VARIANTS = ("base", "title", "title-short")
SHORT_WORDS = 12  # a window of fewer words is "short"
TITLE_CHARS = 200
POLARITY = {"pos": 1.0, "neu": 0.0, "neg": -1.0}
KEY_INFO = b"meinungspuls eval-context v1"
# the labelled comments carry no sourceType: social posts are told apart by their source
SOCIAL = {"bluesky", "bluesky_jetstream", "mastodon"}


def premise(window: str, title: str, variant: str) -> str:
    """The text the model reads for one window under a variant."""
    title = " ".join((title or "").split())[:TITLE_CHARS]
    if not title or variant == "base":
        return window
    if variant == "title-short" and len(window.split()) >= SHORT_WORDS:
        return window
    return f"Artikel: {title} | Kommentar: {window}"


# --------------------------------------------------------------------------- encryption


def _fernet():
    from cryptography.fernet import Fernet

    secret = os.environ.get("EVAL_KEY_MATERIAL", "")
    if len(secret) < 32:
        raise SystemExit("EVAL_KEY_MATERIAL (the Firestore secret) is needed to encrypt the fold results")
    key = hashlib.sha256(KEY_INFO + b"\0" + secret.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(key))


def seal(data: dict) -> bytes:
    return _fernet().encrypt(json.dumps(data).encode())


def unseal(blob: bytes) -> dict:
    return json.loads(_fernet().decrypt(blob))


# --------------------------------------------------------------------------- data (in memory)


def fetch(fs, comment_ids: set[str]) -> tuple[dict, dict]:
    """Comment texts and article headlines, in memory only: ({comment id: {...}}, {article id: title})."""
    docs = fs.batch_get(
        [f"comments/{c}" for c in sorted(comment_ids)], mask=["text", "sourceId", "sourceType", "articleId"]
    )
    texts = {}
    for path, d in docs.items():
        if d:
            texts[path.split("/", 1)[1]] = {
                "text": d.data.get("text") or "",
                "source": d.data.get("sourceId") or "",
                "sourceType": d.data.get("sourceType") or "",
                "article": d.data.get("articleId") or "",
            }
    arts = sorted({t["article"] for t in texts.values() if t["article"]})
    titles = {}
    for path, d in fs.batch_get([f"articles/{a}" for a in arts], mask=["title"]).items():
        titles[path.split("/", 1)[1]] = ((d.data.get("title") if d else "") or "").strip()
    return texts, titles


def gold_pairs(texts: dict) -> list[dict]:
    from train_sentiment_finetune import load_pairs

    from analysis.linker import Registry

    reg = Registry.from_yaml()
    return [p for f in LABEL_FILES for p in load_pairs(f, reg, texts=texts)]


def comment_ids() -> set[str]:
    import csv

    out = set()
    for f in LABEL_FILES:
        for r in csv.DictReader(f.open(encoding="utf-8")):
            if r.get("linkOk", "1") != "0":
                out.add(r["mentionId"].rsplit("_", 1)[0])
    return out


# --------------------------------------------------------------------------- one fold


def cmd_fold(args) -> None:
    import torch
    from train_sentiment_finetune import CONFIG, FOLD_SEED, FOLDS, Trainer, examples

    from analysis.budget import Budget
    from analysis.firestore import Firestore

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise SystemExit(f"unknown variants {sorted(unknown)}")
    _fernet()  # fail before hours of training, not after

    fs = Firestore()
    budget = Budget.load(fs)
    why = budget.exhausted()
    if why:
        raise SystemExit(f"daily Firestore budget: {why}")
    try:
        texts, titles = fetch(fs, comment_ids())
    finally:
        budget.record()
    print(f"read {fs.reads} Firestore documents (recorded in meta/budget)", flush=True)

    gold = gold_pairs({k: {"text": v["text"], "source": v["source"]} for k, v in texts.items()})
    y = [CLASSES.index(p["gold"]) for p in gold]
    folds = grouped_folds(y, [p["comment"] for p in gold], FOLDS, FOLD_SEED)
    for p in gold:
        t = texts[p["comment"]]
        p["title"] = titles.get(t["article"], "")
        p["sourceType"] = t["sourceType"] or ("social" if p["source"] in SOCIAL else "news")
    with_title = sum(1 for p in gold if p["title"])
    short = sum(1 for p in gold if len(p["windows"][0].split()) < SHORT_WORDS)
    print(
        f"{len(gold)} gold pairs {dict(Counter(p['gold'] for p in gold))}; with a headline {with_title}; "
        f"short windows (< {SHORT_WORDS} words) {short}; fold sizes {[len(f) for f in folds]}",
        flush=True,
    )

    k = args.fold
    test = [gold[i] for i in folds[k]]
    held_out = {p["comment"] for p in test}
    train = [p for p in gold if p["comment"] not in held_out]
    torch.set_num_threads(os.cpu_count() or 4)
    trainer = Trainer()
    out = {
        "fold": k,
        "pairs": {
            p["id"]: {
                "sourceType": p["sourceType"],
                "source": p["source"],
                "short": len(p["windows"][0].split()) < SHORT_WORDS,
                "title": bool(p["title"]),
            }
            for p in test
        },
        "pred": {},
    }
    for v in variants:
        t = time.time()

        def view(pairs: list[dict], v=v) -> list[dict]:
            return [{**p, "windows": [premise(w, p["title"], v) for w in p["windows"]]} for p in pairs]

        tr, te = view(train), view(test)
        changed = sum(1 for a, b in zip(te, test, strict=True) if a["windows"] != b["windows"])
        over = sum(
            1
            for p in tr + te
            for w in p["windows"]
            if len(trainer.encode([(w, p["name"])])["input_ids"][0]) >= CONFIG["maxLength"]
        )
        model, order = trainer.fit(
            examples(tr),
            CONFIG["seed"] + k,
            lambda ep, loss, *_, v=v: print(f"  {v} epoch {ep}: loss {loss:.3f}", flush=True),
        )
        scored = trainer.score(model, order, te)
        out["pred"][v] = {p["id"]: [r["label"], r["score"]] for p, r in zip(te, scored, strict=True)}
        f1 = macro_f1([CLASSES.index(p["gold"]) for p in te], [CLASSES.index(r["label"]) for r in scored])
        print(
            f"fold {k} {v}: {len(tr)} train / {len(te)} test pairs ({changed} test pairs with the headline, "
            f"{over} windows at the {CONFIG['maxLength']}-token limit), macro-F1 {f1:.3f}, {time.time() - t:.0f}s",
            flush=True,
        )
        del model
    Path(args.out).write_bytes(seal(out))
    print(f"fold {k}: encrypted results written", flush=True)


# --------------------------------------------------------------------------- report


def _labels() -> dict[str, str]:
    import csv

    gold = {}
    for f in LABEL_FILES:
        for r in csv.DictReader(f.open(encoding="utf-8")):
            if r.get("linkOk", "1") != "0":
                gold[r["mentionId"]] = r["gold"]
    return gold


def _group_r(ids: list[str], key: dict[str, str], gold: dict[str, str], score: dict[str, float], min_n: int):
    by = defaultdict(list)
    for i in ids:
        by[key[i]].append(i)
    groups = [v for v in by.values() if len(v) >= min_n]
    if len(groups) < 3:
        return float("nan"), float("nan"), len(groups)
    g = [sum(POLARITY[gold[i]] for i in v) / len(v) for v in groups]
    s = [sum(score[i] for i in v) / len(v) for v in groups]
    return pearson(s, g), sum(abs(a - b) for a, b in zip(s, g, strict=True)) / len(groups), len(groups)


def _entity_r_diff(ids, ent, gold, sa, sb, n_boot=1000, seed=0):
    """Paired cluster bootstrap over comments of r(entity means) a - b (entities with >= 5 pairs in the draw)."""
    rng = random.Random(seed)
    by_c = defaultdict(list)
    for i in ids:
        by_c[i.rsplit("_", 1)[0]].append(i)
    cs = sorted(by_c)
    diffs = []
    for _ in range(n_boot):
        draw = []
        for j, c in enumerate(rng.choices(cs, k=len(cs))):
            draw += [(f"{i}#{j}", i) for i in by_c[c]]
        key = {d: ent[i] for d, i in draw}
        g = {d: gold[i] for d, i in draw}
        ra = _group_r([d for d, _ in draw], key, g, {d: sa[i] for d, i in draw}, 5)[0]
        rb = _group_r([d for d, _ in draw], key, g, {d: sb[i] for d, i in draw}, 5)[0]
        if ra == ra and rb == rb:
            diffs.append(ra - rb)
    diffs.sort()
    n = len(diffs)
    return diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


def cmd_report(args) -> None:
    from analysis.linker import Registry

    folds = [unseal(p.read_bytes()) for p in sorted(Path(args.dir).rglob("*.bin"))]
    got = sorted(f["fold"] for f in folds)
    print(f"folds: {got}")
    if got != list(range(5)):
        raise SystemExit("not all five folds are there")
    attrs = {i: a for f in folds for i, a in f["pairs"].items()}
    variants = [v for v in VARIANTS if all(v in f["pred"] for f in folds)]
    pred = {v: {i: r for f in folds for i, r in f["pred"][v].items()} for v in variants}
    gold_all = _labels()
    ids = sorted(attrs)
    reg = Registry.from_yaml()
    ent = {i: i.rsplit("_", 1)[1] for i in ids}
    etype = {i: reg.entities[ent[i]].type for i in ids}
    src = {i: attrs[i]["source"] for i in ids}
    y = [CLASSES.index(gold_all[i]) for i in ids]
    groups = [i.rsplit("_", 1)[0] for i in ids]
    lab = {v: [CLASSES.index(pred[v][i][0]) for i in ids] for v in variants}
    sc = {v: {i: pred[v][i][1] for i in ids} for v in variants}
    lines = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    n_title = sum(attrs[i]["title"] for i in ids)
    n_short = sum(attrs[i]["short"] for i in ids)
    out(f"{len(ids)} pairs from {len(set(groups))} comments; {n_title} with a headline; {n_short} short windows")
    out()
    out(
        "| variant | macro-F1 (95 % CI) | difference to base (paired 95 % CI; draws <= 0) "
        "| F1 pos / neu / neg | accuracy |"
    )
    out("|---|---|---|---|---|")
    for v in variants:
        f1, lo, hi = bootstrap_ci(y, lab[v], groups)
        pc = per_class(confusion(y, lab[v]))
        acc = sum(a == b for a, b in zip(y, lab[v], strict=True)) / len(y)
        if v == "base":
            diff = "–"
        else:
            d, dlo, dhi, p0 = bootstrap_diff(y, lab[v], lab["base"], groups)
            diff = f"{d:+.3f} ({dlo:+.3f} to {dhi:+.3f}; {p0:.3f})"
        cls = " / ".join(f"{c[2]:.3f}" for c in pc)
        out(f"| {v} | {f1:.3f} ({lo:.3f} to {hi:.3f}) | {diff} | {cls} | {acc:.3f} |")
    out()
    out("per class precision / recall:")
    for v in variants:
        pc = per_class(confusion(y, lab[v]))
        out(f"  {v:12s} " + "  ".join(f"{c} {p:.3f}/{r:.3f}" for c, (p, r, _f) in zip(CLASSES, pc, strict=True)))
        out(f"  {'':12s} predicted {dict(Counter(CLASSES[k] for k in lab[v]))}")
    out()
    out("per fold macro-F1:")
    by_fold = {f["fold"]: sorted(f["pairs"]) for f in folds}
    for v in variants:
        vals = []
        for k in range(5):
            idx = [ids.index(i) for i in by_fold[k]]
            vals.append(macro_f1([y[j] for j in idx], [lab[v][j] for j in idx]))
        out(f"  {v:12s} " + ", ".join(f"{x:.3f}" for x in vals))
    for v in variants:
        if v == "base":
            continue
        changed = [j for j in range(len(ids)) if lab[v][j] != lab["base"][j]]
        better = sum(lab[v][j] == y[j] for j in changed)
        worse = sum(lab["base"][j] == y[j] for j in changed)
        out(
            f"{v}: {len(changed)} labels differ from base; {better} now right, {worse} now wrong, "
            f"{len(changed) - better - worse} wrong both"
        )
    out()
    for name, key in (
        ("source type", {i: attrs[i]["sourceType"] for i in ids}),
        ("window", {i: ("short (< 12 words)" if attrs[i]["short"] else "long") for i in ids}),
        ("headline", {i: ("with headline" if attrs[i]["title"] else "without") for i in ids}),
        ("source", src),
    ):
        out(f"macro-F1 by {name} (n; " + " / ".join(variants) + "):")
        by = defaultdict(list)
        for j, i in enumerate(ids):
            by[key[i]].append(j)
        for g, idx in sorted(by.items(), key=lambda kv: -len(kv[1]))[:10]:
            if len(idx) < 20:
                continue
            vals = " / ".join(f"{macro_f1([y[j] for j in idx], [lab[v][j] for j in idx]):.3f}" for v in variants)
            out(f"  {g:20s} n={len(idx):4d}  {vals}")
    out()
    out("score checks (mean score per group against the mean gold polarity):")
    gold_pol = [POLARITY[gold_all[i]] for i in ids]
    for v in variants:
        s = [sc[v][i] for i in ids]
        bias = sum(s) / len(s) - sum(gold_pol) / len(gold_pol)
        parts = []
        for name, key, min_n in (("entity", ent, 5), ("source", src, 10), ("type", etype, 10)):
            r, mae, k = _group_r(ids, key, gold_all, sc[v], min_n)
            parts.append(f"{name} (k={k}) r {r:.3f} MAE {mae:.3f}")
        out(f"  {v:12s} bias {bias:+.3f}, r per mention {pearson(s, gold_pol):.3f}; " + "; ".join(parts))
    for v in variants:
        if v == "base":
            continue
        lo, hi = _entity_r_diff(ids, ent, gold_all, sc[v], sc["base"])
        out(f"  entity-level r, {v} - base: paired 95 % CI {lo:+.3f} to {hi:+.3f}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("## Headline as context: cross-validated\n\n```\n" + "\n".join(lines) + "\n```\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fold")
    f.add_argument("--fold", type=int, required=True)
    f.add_argument("--variants", default=",".join(VARIANTS))
    f.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True)
    args = ap.parse_args()
    {"fold": cmd_fold, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
