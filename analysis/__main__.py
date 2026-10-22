"""CLI: python -m analysis {run,reaggregate,seed}."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m analysis", description="Meinungspuls analysis pipeline")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="analyze unanalyzed comments and update aggregates")
    r.add_argument("--limit", type=int, default=int(os.environ.get("ANALYSIS_LIMIT", 3000)))
    r.add_argument("--batch", type=int, default=250, help="comments per NLP/commit batch")
    r.add_argument("--ner", choices=["flair", "bert", "none"], default=os.environ.get("ANALYSIS_NER", "flair"))
    r.add_argument("--sentiment", choices=["guhr", "news"], default=os.environ.get("ANALYSIS_SENTIMENT", "guhr"))
    r.add_argument("--no-wikidata", action="store_true", help="disable the Wikidata fallback linker")
    r.add_argument("--max-wikidata-lookups", type=int, default=150)

    sub.add_parser("reaggregate", help="rebuild all day docs, entity counters and trending from mentions")
    s = sub.add_parser("seed", help="write curated entities and events")
    s.add_argument("--no-images", action="store_true")

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("urllib3", "google", "flair", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from .firestore import Firestore
    from .pipeline import reaggregate, run, seed

    fs = Firestore()
    if args.cmd == "run":
        try:
            import torch

            torch.set_num_threads(max(1, os.cpu_count() or 1))
        except ImportError:
            pass
        out = run(
            fs,
            limit=args.limit,
            batch=args.batch,
            ner_backend=args.ner,
            sentiment_key=args.sentiment,
            use_wikidata=not args.no_wikidata,
            max_wikidata_lookups=args.max_wikidata_lookups,
        )
    elif args.cmd == "reaggregate":
        out = reaggregate(fs)
    else:
        out = seed(fs, with_images=not args.no_images)
    out["firestoreReads"] = fs.reads
    out["firestoreWrites"] = fs.writes
    json.dump(out, sys.stdout, ensure_ascii=False, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
