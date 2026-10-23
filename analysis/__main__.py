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
    r.add_argument("--limit", type=int, default=int(os.environ.get("ANALYSIS_LIMIT", 20000)))
    r.add_argument("--batch", type=int, default=250, help="comments per NLP/commit batch")
    r.add_argument("--ner", choices=["flair", "bert", "none"], default=os.environ.get("ANALYSIS_NER", "bert"))
    r.add_argument(
        "--sentiment",
        choices=["cal", "ens", "guhr", "guhr-raw", "news", "nli"],
        default=os.environ.get("ANALYSIS_SENTIMENT", "cal"),
    )
    r.add_argument("--no-wikidata", action="store_true", help="disable the Wikidata fallback linker")
    r.add_argument("--max-wikidata-lookups", type=int, default=150)
    r.add_argument("--agg-every", type=int, default=2000, help="aggregate after this many comments (and at the end)")

    ra = sub.add_parser("reaggregate", help="rebuild day docs, entity counters and trending from mentions")
    ra.add_argument("--since", help="only recompute (entity, day) pairs from this day (YYYY-MM-DD) on; default: all")
    rs = sub.add_parser("rescore", help="re-run only the sentiment step for all mentions, then re-aggregate")
    rs.add_argument("--sentiment", default=os.environ.get("ANALYSIS_SENTIMENT", "cal"))
    rl = sub.add_parser("relink", help="add mentions of new entities to analysed comments of the last N days")
    rl.add_argument("--days", type=int, default=14)
    rl.add_argument("--queue", action="store_true", help="only register the job; `run` continues it")
    rl.add_argument(
        "--ids-file", help="file with the entity ids to relink (default: generated entities without mentions)"
    )
    pr = sub.add_parser("prune", help="re-check mentions of given entities, delete those no longer linked")
    pr.add_argument("--ids", default="", help="comma-separated entity ids")
    pr.add_argument("--ids-file", help="file with one entity id per line")
    pr.add_argument("--add", default="", help="comma-separated entity ids whose mentions may be added")
    pr.add_argument("--queue", action="store_true", help="only register the job; `run` continues it")
    sub.add_parser("seed-requests", help="validate accepted entity_requests via Wikidata and seed them")
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
    from .pipeline import prune, reaggregate, reaggregate_since, relink, rescore, run, seed

    fs = Firestore()
    if args.cmd == "run":
        try:
            import torch

            torch.set_num_threads(int(os.environ.get("ANALYSIS_THREADS") or max(1, os.cpu_count() or 1)))
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
            agg_every=args.agg_every,
        )
    elif args.cmd == "relink":
        ids = None
        if args.ids_file:
            with open(args.ids_file, encoding="utf-8") as f:
                ids = {line.strip() for line in f if line.strip()}
        if args.queue:
            from datetime import date, timedelta

            from .pipeline import queue_job
            from .textutil import today_berlin

            cutoff = (date.fromisoformat(today_berlin()) - timedelta(days=args.days - 1)).isoformat()
            out = queue_job(fs, "relink", {"cutoff": cutoff, "entityIds": sorted(ids or [])})
        else:
            out = relink(fs, days=args.days, entity_ids=ids)
    elif args.cmd == "prune":
        ids = {x.strip() for x in args.ids.split(",") if x.strip()}
        if args.ids_file:
            with open(args.ids_file, encoding="utf-8") as f:
                ids |= {line.strip() for line in f if line.strip()}
        add = {x.strip() for x in args.add.split(",") if x.strip()}
        if args.queue:
            from .pipeline import queue_job

            out = queue_job(fs, "prune", {"ids": sorted(ids), "add": sorted(add)})
        else:
            out = prune(fs, ids, add_ids=add)
    elif args.cmd == "rescore":
        out = rescore(fs, sentiment_key=args.sentiment)
    elif args.cmd == "reaggregate":
        out = reaggregate_since(fs, args.since) if args.since else reaggregate(fs)
    else:
        # seeding only writes changed documents; it is counted in the daily budget too
        from .budget import Budget

        budget = Budget.load(fs)
        why = budget.exhausted()
        if why:
            logging.getLogger("analysis").warning("daily Firestore budget: %s - skipping %s", why, args.cmd)
            out = {"skipped": why}
            if args.cmd == "seed":  # the regular run seeds once the budget allows
                from .pipeline import queue_job

                out["queued"] = queue_job(fs, "seed", {})
        else:
            try:
                if args.cmd == "seed-requests":
                    from .entity_requests import process_entity_requests

                    out = process_entity_requests(fs)
                else:
                    out = seed(fs, with_images=not args.no_images, budget=budget)
                    if out.get("stopped"):
                        from .pipeline import queue_job

                        out["queued"] = queue_job(fs, "seed", {})
            finally:
                budget.record()
            out["budget"] = budget.summary()
    out["firestoreReads"] = fs.reads
    out["firestoreWrites"] = fs.writes
    json.dump(out, sys.stdout, ensure_ascii=False, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
