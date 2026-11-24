"""Export a labelling sample of implicit-mention candidates (analysis/implicit.py). Private use only.

Runs in the PRIVATE infrastructure repository (workflow `export-implicit-sample.yml`, manual), never
in this public one: the output file holds comment texts and headlines, so it may only go to a
short-lived artifact of a private repository. Nothing per comment is printed; the log and the
summary hold counts only.

What it does (read-only, about 1 read per article of the window + 1 per sampled comment, recorded
in meta/budget):
1. the non-social articles published in the last `--days` Berlin days (social sources have one
   pseudo-article per day and are skipped), a seeded sample of `--articles` of them, all their comments;
2. the production linker (registry incl. Firestore-only entities, BERT NER) on every comment, and
   every implicit candidate of every rule and cue (`ImplicitInput.collect`), accepted or not;
3. a stratified sample of candidates per (rule, cue) and a random sample of comments with neither
   an explicit entity nor a candidate (recall), with their texts, headline and parent comment;
4. the sentiment of each sampled candidate under three premises (comment window only; with the
   headline; with headline and parent comment) with the production sentiment model.

    python scripts/export_implicit_sample.py --days 3 --articles 500 --out sample/implicit.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import implicit as imp  # noqa: E402
from analysis.analyze import ImplicitInput, Stats, analyze_comments  # noqa: E402
from analysis.budget import Budget  # noqa: E402
from analysis.firestore import Firestore  # noqa: E402
from analysis.registry_store import load_registry  # noqa: E402
from analysis.textutil import BERLIN, clean_text, today_berlin  # noqa: E402

log = logging.getLogger("export")
COMMENT_MASK = ["articleId", "sourceId", "text", "parentId", "publishedAt", "day"]


class _NoScore:
    """Explicit mentions need no sentiment here."""

    window_mode = "target"
    version = "none"

    def predict(self, items):
        return [("neu", 0.0, 1.0) for _ in items]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--articles", type=int, default=500)
    ap.add_argument("--max-comments", type=int, default=12000)
    ap.add_argument("--per-stratum", type=int, default=45)
    ap.add_argument("--rejected", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-sentiment", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rng = random.Random(a.seed)

    fs = Firestore()
    budget = Budget.load(fs)
    est = 4000 + a.max_comments
    if not budget.allows(reads=est):
        raise SystemExit(f"daily read budget too small for the export (needs about {est} reads)")

    sources = {d.id: d.data for d in fs.query("sources", select=["country", "sourceType"])}
    countries = {k: v.get("country") or "" for k, v in sources.items()}
    social_sources = {k for k, v in sources.items() if v.get("sourceType") == "social"}

    end = datetime.fromisoformat(today_berlin()).replace(tzinfo=BERLIN)
    start = end - timedelta(days=a.days)
    arts = list(
        fs.query(
            "articles",
            where=[("publishedAt", "GREATER_THAN_OR_EQUAL", start), ("publishedAt", "LESS_THAN", end)],
            order_by=[("publishedAt", "ASCENDING")],
            select=["title", "sourceId", "sourceType", "commentCount"],
        )
    )
    real = [
        d
        for d in arts
        if d.data.get("sourceType") != "social"
        and (d.data.get("sourceId") or "") not in social_sources
        and (d.data.get("title") or "").strip()
        and int(d.data.get("commentCount") or 0) > 0
    ]
    by_source: dict[str, list] = defaultdict(list)
    for d in real:
        by_source[d.data.get("sourceId") or ""].append(d)
    # stratified by source: an equal share each, the rest filled at random
    share = max(5, a.articles // max(1, len(by_source)))
    picked = []
    for _sid, ds in sorted(by_source.items()):
        rng.shuffle(ds)
        picked += ds[:share]
    rest = [d for d in real if d not in picked]
    rng.shuffle(rest)
    picked += rest[: max(0, a.articles - len(picked))]
    titles = {d.id: d.data.get("title") or "" for d in picked}
    log.info(
        "%d articles in %d days (%d real, %d sources), %d sampled",
        len(arts),
        a.days,
        len(real),
        len(by_source),
        len(picked),
    )

    comments: list[dict] = []
    ids = sorted(titles)
    for i in range(0, len(ids), 30):
        if len(comments) >= a.max_comments:
            break
        for d in fs.query("comments", where=[("articleId", "IN", ids[i : i + 30])], select=COMMENT_MASK):
            comments.append({"id": d.id, **d.data})
    comments = comments[: a.max_comments]
    by_id = {c["id"]: c for c in comments}
    log.info("%d comments", len(comments))

    from analysis.ner import load_ner

    registry = load_registry(fs)
    ner = load_ner("bert")
    inp = ImplicitInput(
        titles=titles,
        parents={c["parentId"]: by_id[c["parentId"]]["text"] for c in comments if c.get("parentId") in by_id},
        collect=[],
    )
    stats = Stats()
    explicit_comments = 0
    no_entity: list[dict] = []
    comments.sort(key=lambda c: (c.get("articleId") or "", str(c.get("publishedAt") or "")))
    for i in range(0, len(comments), 250):
        chunk = comments[i : i + 250]
        res = analyze_comments(
            chunk,
            registry,
            ner,
            _NoScore(),
            source_country=countries,
            article_context={k: v for k, v in titles.items()},
            stats=stats,
            implicit=inp,
        )
        for c in chunk:
            explicit = [m for m in res[c["id"]] if not m.get("implicit")]
            if explicit:
                explicit_comments += 1
            else:
                no_entity.append(c)
        log.info("linked %d / %d comments", i + len(chunk), len(comments))

    cands = inp.collect
    with_cand = {c["id"] for c, *_ in cands}
    strata: dict[tuple[str, str], list] = defaultdict(list)
    for item in cands:
        x = item[3]
        strata[(x.rule, x.via if x.rule == "A" else "", x.cue)].append(item)
    sample = []
    for _key, items in sorted(strata.items()):
        rng.shuffle(items)
        sample += items[: a.per_stratum]
    rejected = [c for c in no_entity if c["id"] not in with_cand and c.get("text")]
    rng.shuffle(rejected)
    rejected = rejected[: a.rejected]

    rows = []
    for c, text, ctx, x in sample:
        title = titles.get(c.get("articleId") or "", "")
        parent = clean_text(inp.parents.get(c.get("parentId") or "", "")) if c.get("parentId") else ""
        w = imp.window(text, x.start, x.end)
        rows.append(
            {
                "kind": "candidate",
                "commentId": c["id"],
                "sourceId": c.get("sourceId"),
                "day": c.get("day"),
                "rule": x.rule,
                "via": x.via,
                "cue": x.cue,
                "nTargets": x.n_targets,
                "accepted": imp.accepted(x),
                "entityId": x.entity_id,
                "entityName": registry.entities[x.entity_id].name,
                "entityDescription": registry.entities[x.entity_id].description,
                "cueText": text[x.start : x.end],
                "isReply": ctx.is_reply,
                "headlineEntities": ctx.title_entities,
                "parentEntities": ctx.parent_entities,
                "title": title,
                "parent": parent,
                "text": text,
                "premises": {
                    "window": w,
                    "title": imp.premise(w, title),
                    "title+parent": imp.premise(w, title, parent),
                },
            }
        )
    for c in rejected:
        pid = c.get("parentId") or ""
        rows.append(
            {
                "kind": "rejected",
                "commentId": c["id"],
                "sourceId": c.get("sourceId"),
                "isReply": bool(pid),
                "title": titles.get(c.get("articleId") or "", ""),
                "headlineEntities": inp.title_entities.get(c.get("articleId") or "", []),
                "parent": clean_text(inp.parents.get(pid, "")) if pid else "",
                "parentEntities": inp.parent_entities.get(pid) if pid else None,
                "text": clean_text(c.get("text") or ""),
            }
        )
    if not a.no_sentiment:
        from analysis.sentiment import load_sentiment

        model = load_sentiment("ft")
        items, where = [], []
        for r in rows:
            for k, p in (r.get("premises") or {}).items():
                items.append((p, r["entityName"]))
                where.append((r, k))
        for (r, k), (label, score, conf) in zip(where, model.predict(items), strict=True):
            r.setdefault("sentiment", {})[k] = {"label": label, "score": score, "confidence": conf}
        sentiment_version = model.version
    else:
        sentiment_version = None

    summary = {
        "days": a.days,
        "articlesInWindow": len(arts),
        "realArticlesInWindow": len(real),
        "realCommentCountInWindow": sum(int(d.data.get("commentCount") or 0) for d in real),
        "sampledArticles": len(picked),
        "comments": len(comments),
        "commentsBySource": dict(Counter(c.get("sourceId") or "" for c in comments)),
        "replies": sum(1 for c in comments if c.get("parentId")),
        "commentsWithExplicit": explicit_comments,
        "commentsWithoutEntity": len(no_entity),
        "candidates": len(cands),
        "candidatesByStratum": {"/".join(k): len(v) for k, v in sorted(strata.items())},
        "candidatesBySourceRule": dict(Counter(f"{c.get('sourceId')}/{x.rule}" for c, _t, _x, x in cands)),
        "sampled": len(sample),
        "rejectedSampled": len(rejected),
        "sentimentVersion": sentiment_version,
        "socialSources": sorted(social_sources),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "summary", **summary}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    budget.record()
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
