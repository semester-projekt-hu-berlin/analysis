"""The headline-context evaluation: input variants, encrypted fold results, a report of aggregates only."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_sentiment_context as ec

KEY = "x" * 64


def test_premise_variants():
    short, long = "Merz kann es nicht.", " ".join(["Wort"] * 20)
    assert ec.premise(short, "Titel", "base") == short
    assert ec.premise(short, "", "title") == short
    assert ec.premise(short, " Ein\n Titel ", "title") == "Artikel: Ein Titel | Kommentar: Merz kann es nicht."
    assert ec.premise(long, "Titel", "title").startswith("Artikel: Titel | Kommentar: ")
    assert ec.premise(long, "Titel", "title-short") == long
    assert ec.premise(short, "Titel", "title-short").startswith("Artikel: Titel")
    assert ec.premise(short, "T" * 500, "title").count("T") == 200


def test_sealed_results_need_the_key(monkeypatch):
    monkeypatch.setenv("EVAL_KEY_MATERIAL", KEY)
    blob = ec.seal({"fold": 1, "pred": {"base": {"a_b": ["neg", -0.5]}}})
    assert b"neg" not in blob
    assert ec.unseal(blob)["pred"]["base"]["a_b"] == ["neg", -0.5]
    monkeypatch.setenv("EVAL_KEY_MATERIAL", "y" * 64)
    with pytest.raises(Exception):  # noqa: B017 - any decryption failure
        ec.unseal(blob)
    monkeypatch.setenv("EVAL_KEY_MATERIAL", "")
    with pytest.raises(SystemExit):
        ec.seal({})


def test_report_prints_aggregates_only(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("EVAL_KEY_MATERIAL", KEY)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    from analysis.linker import Registry

    reg = Registry.from_yaml()
    gold = {i: g for i, g in ec._labels().items() if i.rsplit("_", 1)[1] in reg.entities}
    rng = random.Random(0)
    ids = sorted(gold)
    for k in range(5):
        mine = ids[k::5]
        data = {
            "fold": k,
            "pairs": {
                i: {"sourceType": "news", "source": "taz", "short": rng.random() < 0.2, "title": True} for i in mine
            },
            "pred": {},
        }
        for v in ec.VARIANTS:
            data["pred"][v] = {
                i: [gold[i] if rng.random() < 0.6 else rng.choice(["pos", "neu", "neg"]), rng.uniform(-1, 1)]
                for i in mine
            }
        (tmp_path / f"fold-{k}.bin").write_bytes(ec.seal(data))

    class Args:
        dir = str(tmp_path)

    ec.cmd_report(Args)
    out = capsys.readouterr().out
    assert "| base |" in out and "| title |" in out and "difference to base" in out
    assert not any(i in out for i in ids)  # no mention id, so nothing per pair
