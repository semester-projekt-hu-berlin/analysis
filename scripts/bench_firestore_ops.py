"""Count Firestore reads/writes per 1,000 comments on an in-memory Firestore (fake NLP models).

python scripts/bench_firestore_ops.py [N]
"""

from __future__ import annotations

import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis.ner
import analysis.sentiment
from analysis import pipeline
from analysis.linker import Registry
from analysis.textutil import day_key
from tests.fakes import FakeFirestore, FakeNer, FakeSentiment

NAMES = ["Merz", "Trump", "die SPD", "Kickl", "die FPÖ", "Putin", "Söder", "die AfD", "Tesla", "Musk", "Babler"]
ADJ = ["ist gut", "hat versagt", "redet viel", "ist eine Katastrophe", "macht Politik"]


def main(n: int = 4000) -> None:
    analysis.ner.load_ner = lambda name: FakeNer({})
    analysis.sentiment.load_sentiment = lambda key, **kw: FakeSentiment()
    rnd = random.Random(1)
    reg = Registry.from_yaml()
    docs = {f"entities/{e.id}": {**e.metadata(), **pipeline.ZERO_STATS} for e in reg.entities.values()}
    docs["sources/s"] = {"country": "DE"}
    now = datetime.now(UTC)
    for i in range(n):
        ts = now - timedelta(hours=rnd.random() * 30)
        k = rnd.choice([0, 1, 1, 2])
        text = " ".join([f"{rnd.choice(NAMES)} {rnd.choice(ADJ)}." for _ in range(k)] + ["Das sehe ich anders."])
        docs[f"comments/c{i}"] = {
            "articleId": "a",
            "sourceId": "s",
            "text": text,
            "publishedAt": ts,
            "crawledAt": ts + timedelta(seconds=i),
            "day": day_key(ts),
            "analyzed": False,
            "analyzedAt": None,
            "analysisVersion": None,
        }
    fs = FakeFirestore(docs)
    out = pipeline.run(fs, limit=n, use_wikidata=False)
    print(f"per 1000 comments: reads={fs.reads * 1000 / n:.0f} writes={fs.writes * 1000 / n:.0f}")
    for kind, ops in out["firestoreOps"].items():  # the same breakdown every real run logs
        print(f"  {kind}: " + ", ".join(f"{k} {v * 1000 / n:.0f}" for k, v in ops.items()))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
