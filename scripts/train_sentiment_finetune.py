"""Train the fine-tuned target-sentiment model ("ft").

The public NLI model already answers "Der Kommentar äußert sich positiv über {Name}." with
entailment / neutral / contradiction. Here it is fine-tuned on the hand-labelled pairs to read
that as positive / neutral / negative toward the entity. One model pass per window; the window
is the mention's sentence +- one sentence with the target in [brackets].

Training pairs: every labelled pair (eval/labels.csv, labels_extra.csv, labels_extra2.csv; see
scripts/eval_sentiment_cv.py). Comment texts are read from the git-ignored cache
eval/texts.local.json (fetched from Firestore if missing), never written to git.

    python scripts/train_sentiment_finetune.py cv    # grouped 5-fold CV -> eval/ft_cv.local.json
    python scripts/train_sentiment_finetune.py fit   # train on all pairs -> delta parts + data/sentiment_ft.json

`fit` stores what training changed against the base model (analysis/finetune.py: about 180 MB
in parts below 100 MB) in the model cache; the parts are committed to the branch `models`, and
the commit goes into the url of data/sentiment_ft.json, from where the pipeline downloads and
verifies them.

Evaluate with scripts/eval_sentiment_cv.py (same folds for the baseline, paired bootstrap).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_protocol import CLASSES, grouped_folds, macro_f1  # noqa: E402

from analysis.analyze import MAX_WINDOWS_PER_MENTION, mention_window  # noqa: E402
from analysis.finetune import delta, model_dir, round_half, sha256, split  # noqa: E402
from analysis.linker import Match, Registry  # noqa: E402
from analysis.sentiment import FT_META, NLI_MODELS, combine, to_label  # noqa: E402
from analysis.textutil import clean_text, split_sentences  # noqa: E402

LABELS = ROOT / "eval" / "labels.csv"
EXTRA = ROOT / "eval" / "labels_extra.csv"
TEXTS = ROOT / "eval" / "texts.local.json"
CV_FILE = ROOT / "eval" / "ft_cv.local.json"
COUNTRY = {"derstandard": "AT", "blick": "CH", "taz": "DE"}
MODELS_URL = "https://raw.githubusercontent.com/semester-projekt-hu-berlin/analysis"

# Fixed before the cross-validation was run; nothing here was tuned on the test folds.
CONFIG = {
    "version": 3,
    "baseModel": NLI_MODELS["nli"],
    "hypothesis": "Der Kommentar äußert sich positiv über {name}.",
    "window": "target",
    "maxLength": 192,
    "lr": 3e-5,
    "epochs": 4,
    "batchSize": 16,
    "classWeightPower": 0.5,
    "seed": 0,
}
FOLDS, FOLD_SEED = 5, 0


def load_texts(comment_ids: set[str]) -> dict:
    try:
        texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    except OSError:
        texts = {}
    missing = sorted(comment_ids - texts.keys())
    if missing:
        from analysis.firestore import Firestore

        docs = Firestore().batch_get([f"comments/{c}" for c in missing], mask=["text", "sourceId"])
        for p, d in docs.items():
            if d:
                texts[p.split("/", 1)[1]] = {"text": d.data.get("text", ""), "source": d.data.get("sourceId", "")}
        TEXTS.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    return texts


def load_pairs(path: Path = LABELS, reg: Registry | None = None) -> list[dict]:
    """Labelled (comment, entity) pairs with the windows the pipeline would build for them:
    `clause` (calibrated model) and `windows` (fine-tuned model, up to 3 per pair)."""
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r.get("linkOk", "1") != "0"]
    texts = load_texts({r["mentionId"].rsplit("_", 1)[0] for r in rows})
    reg = reg or Registry.from_yaml()
    out = []
    for r in rows:
        cid, eid = r["mentionId"].rsplit("_", 1)
        t = texts.get(cid)
        if not t or eid not in reg.entities:
            continue
        text = clean_text(t["text"])
        sents = split_sentences(text)
        matches = reg.gazetteer(text, [], "", COUNTRY.get(t["source"]))
        mine = [m for m in matches if m.entity_id == eid]
        if not mine:  # weak alias confirmed by NER at analysis time: locate the surface
            surf = next((a for a in reg.entities[eid].aliases if a.casefold() in text.casefold()), None)
            if surf is None:
                continue
            i = text.casefold().index(surf.casefold())
            mine = [Match(i, i + len(surf), text[i : i + len(surf)], eid, "surface")]
            matches = [*matches, mine[0]]
        windows: list[str] = []
        for m in mine[:MAX_WINDOWS_PER_MENTION]:
            w = mention_window(text, sents, m, matches, CONFIG["window"])
            if w not in windows:
                windows.append(w)
        out.append(
            {
                "id": r["mentionId"],
                "comment": cid,
                "entity": eid,
                "gold": r["gold"],
                "source": t["source"],
                "type": reg.entities[eid].type,
                "name": reg.entities[eid].name,
                "clause": mention_window(text, sents, mine[0], matches, "clause"),
                "windows": windows,
            }
        )
    return out


class Trainer:
    def __init__(self, cfg: dict = CONFIG):
        import torch
        from transformers import AutoTokenizer

        self.torch, self.cfg = torch, cfg
        self.tok = AutoTokenizer.from_pretrained(cfg["baseModel"])

    def encode(self, batch: list[tuple[str, str]]):
        return self.tok(
            [w or "." for w, _ in batch],
            [self.cfg["hypothesis"].format(name=n) for _, n in batch],
            padding=True,
            truncation="only_first",
            max_length=self.cfg["maxLength"],
            return_tensors="pt",
        )

    def new_model(self):
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(self.cfg["baseModel"])
        self.revision = getattr(model.config, "_commit_hash", None)
        labels = {i: lab.lower() for i, lab in model.config.id2label.items()}
        ent = next(i for i, lab in labels.items() if lab.startswith("entail"))
        con = next(i for i, lab in labels.items() if lab.startswith("contra"))
        neu = next(i for i in labels if i not in (ent, con))
        return model, [ent, neu, con]

    def fit(self, train: list[tuple[str, str, int]], seed: int, progress=None):
        """Train on (window, entity name, class index) examples; returns (model, logit order)."""
        from transformers import get_linear_schedule_with_warmup

        torch, cfg = self.torch, self.cfg
        torch.manual_seed(seed)
        rng = random.Random(seed)
        train = list(train)
        model, order = self.new_model()
        counts = Counter(y for _, _, y in train)
        weight = torch.tensor([(len(train) / (3 * counts[k])) ** cfg["classWeightPower"] for k in range(3)])
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01)
        bs = cfg["batchSize"]
        steps = (len(train) + bs - 1) // bs * cfg["epochs"]
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        for ep in range(cfg["epochs"]):
            model.train()
            rng.shuffle(train)
            # batches of similar length (little padding), in random order
            chunks = [sorted(train[i : i + bs * 8], key=lambda x: len(x[0])) for i in range(0, len(train), bs * 8)]
            batches = [c[i : i + bs] for c in chunks for i in range(0, len(c), bs)]
            rng.shuffle(batches)
            total = 0.0
            for b in batches:
                # in slices of at most 8 windows (memory); the step is taken over the whole batch
                loss = 0.0
                for i in range(0, len(b), 8):
                    part = b[i : i + 8]
                    logits = model(**self.encode([(w, n) for w, n, _ in part])).logits[:, order]
                    target = torch.tensor([y for _, _, y in part])
                    # weighted mean over the batch, as one pass over the whole batch would give
                    part_loss = torch.nn.functional.cross_entropy(logits, target, weight=weight, reduction="sum")
                    part_loss = part_loss / float(sum(weight[y] for _, _, y in b))
                    part_loss.backward()
                    loss = loss + part_loss.detach()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                total += loss.item()
            if progress:
                progress(ep, total / len(batches), model, order)
        round_half(model)  # what is evaluated is what is stored (half precision)
        model.eval()
        return model, order

    def probs(self, model, order: list[int], items: list[tuple[str, str]]) -> list[tuple[float, float, float]]:
        torch = self.torch
        model.eval()
        out: dict[int, tuple] = {}
        idx = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        with torch.inference_mode():
            for s in range(0, len(idx), 32):
                chunk = idx[s : s + 32]
                p = torch.softmax(model(**self.encode([items[i] for i in chunk])).logits[:, order], dim=-1).tolist()
                for i, row in zip(chunk, p, strict=True):
                    out[i] = tuple(row)
        return [out[i] for i in range(len(items))]

    def score(self, model, order: list[int], pairs: list[dict]) -> list[dict]:
        """Score pairs as the pipeline does: every window, merged with `combine`."""
        flat = [(w, p["name"]) for p in pairs for w in p["windows"]]
        pr = self.probs(model, order, flat)
        out, k = [], 0
        for p in pairs:
            mine = pr[k : k + len(p["windows"])]
            k += len(p["windows"])
            label, score, conf = combine([to_label(*q) for q in mine])
            out.append({"label": label, "score": score, "confidence": conf, "probs": [round(v, 5) for v in mine[0]]})
        return out


def examples(pairs: list[dict]) -> list[tuple[str, str, int]]:
    return [(p["windows"][0], p["name"], CLASSES.index(p["gold"])) for p in pairs]


def cmd_cv(args) -> None:
    from eval_sentiment_cv import load_gold

    gold, extra = load_gold(), []  # every labelled pair is tested once (grouped folds)
    y = [CLASSES.index(p["gold"]) for p in gold]
    print(f"{len(gold)} gold pairs {dict(Counter(p['gold'] for p in gold))}")
    trainer = Trainer()
    folds = grouped_folds(y, [p["comment"] for p in gold], FOLDS, FOLD_SEED)
    try:
        result = json.loads(CV_FILE.read_text(encoding="utf-8")) if args.resume else {}
    except OSError:
        result = {}
    for k, test in enumerate(folds):
        test_pairs = [gold[i] for i in test]
        if all(p["id"] in result for p in test_pairs):
            continue
        held_out = {p["comment"] for p in test_pairs}
        train = [p for p in gold + extra if p["comment"] not in held_out]
        t = time.time()

        def progress(ep, loss, model, order, k=k, test_pairs=test_pairs, t=t):
            if args.verbose:  # information only: the number of epochs is fixed in CONFIG
                pred = [CLASSES.index(r["label"]) for r in trainer.score(model, order, test_pairs)]
                f1 = macro_f1([CLASSES.index(p["gold"]) for p in test_pairs], pred)
                print(
                    f"  fold {k} epoch {ep}: loss {loss:.3f}, test macro-F1 {f1:.3f} ({time.time() - t:.0f}s)",
                    flush=True,
                )

        model, order = trainer.fit(examples(train), CONFIG["seed"] + k, progress)
        scored = trainer.score(model, order, test_pairs)
        for p, r in zip(test_pairs, scored, strict=True):
            result[p["id"]] = {**r, "fold": k}
        pred = [CLASSES.index(r["label"]) for r in scored]
        f1 = macro_f1([CLASSES.index(p["gold"]) for p in test_pairs], pred)
        print(
            f"fold {k}: {len(train)} train / {len(test)} test pairs, macro-F1 {f1:.3f}, {time.time() - t:.0f}s",
            flush=True,
        )
        CV_FILE.write_text(json.dumps(result), encoding="utf-8")
    pred = [CLASSES.index(result[p["id"]]["label"]) for p in gold]
    print(f"out-of-fold macro-F1 {macro_f1(y, pred):.3f} on {len(gold)} pairs -> {CV_FILE.name}")


def cmd_fit(args) -> None:
    from eval_sentiment_cv import load_gold
    from safetensors.torch import save_file

    gold = load_gold()
    pairs = gold
    trainer = Trainer()
    t = time.time()
    model, order = trainer.fit(
        examples(pairs), CONFIG["seed"], lambda ep, loss, *_: print(f"epoch {ep}: loss {loss:.3f}", flush=True)
    )
    scored = trainer.score(model, order, gold)
    fit_f1 = macro_f1([CLASSES.index(p["gold"]) for p in gold], [CLASSES.index(r["label"]) for r in scored])
    release = f"sentiment-ft-v{CONFIG['version']}"
    out = model_dir(release)
    out.mkdir(parents=True, exist_ok=True)
    base_model, _order = trainer.new_model()  # the untouched base, same revision
    parts = split(delta(model.state_dict(), base_model.state_dict()))
    files = {}
    for k, part in enumerate(parts):
        name = f"delta-{k}.safetensors"
        save_file(part, str(out / name))
        files[name] = sha256(out / name)
    meta = {
        "version": CONFIG["version"],
        "release": release,
        "url": f"{MODELS_URL}/COMMIT/{release}",
        "files": files,
        **{k: CONFIG[k] for k in ("baseModel", "hypothesis", "window", "maxLength")},
        "baseRevision": trainer.revision,
        "classes": list(CLASSES),
        "logitOrder": order,
        "training": {k: CONFIG[k] for k in ("lr", "epochs", "batchSize", "classWeightPower", "seed")},
        "trainedOn": len(gold),
        "fitMacroF1": round(fit_f1, 3),
    }
    FT_META.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    size = sum((out / f).stat().st_size for f in files) / 1e6
    print(f"trained on {len(pairs)} pairs in {time.time() - t:.0f}s; in-sample macro-F1 {fit_f1:.3f}")
    print(f"saved {len(files)} parts ({size:.0f} MB) to {out} and {FT_META.name}")
    print(f"commit them to the branch `models` under {release}/ and put that commit into the url of {FT_META.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    cv = sub.add_parser("cv")
    cv.add_argument("--resume", action="store_true", help="keep the folds already in eval/ft_cv.local.json")
    cv.add_argument("--verbose", action="store_true", help="print the test score after every epoch (information only)")
    sub.add_parser("fit")
    args = ap.parse_args()
    {"cv": cmd_cv, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
