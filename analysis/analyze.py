"""Turn comments into (comment, entity) mentions: NER -> linking -> target sentiment.

Pure with respect to Firestore: takes comment dicts, returns mention dicts. Comment
text is only held in memory; mentions keep nothing but the matched surface span.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field

from . import implicit as imp
from .linker import Match, Registry, strip_titles
from .sentiment import combine, target_window
from .textutil import clean_text, day_key, split_sentences

log = logging.getLogger(__name__)

NER_MIN_SCORE = 0.75
MAX_WINDOWS_PER_MENTION = 3
MAX_WINDOW_CHARS = 1000
MIN_NER_CHARS = 4
_HAS_UPPER = re.compile(r"[A-ZÄÖÜ]")


@dataclass
class Stats:
    comments: int = 0
    sentences: int = 0
    ner_sentences: int = 0
    mentions: int = 0
    ner_seconds: float = 0.0
    sentiment_seconds: float = 0.0
    link_seconds: float = 0.0
    wikidata_lookups: int = 0
    discovered: list[str] = field(default_factory=list)
    implicit: Counter = field(default_factory=Counter)  # implicit mentions by "rule/cue"
    implicit_candidates: int = 0


@dataclass
class ImplicitInput:
    """Context for implicit mentions (analysis/implicit.py), kept by the caller for a whole run.

    titles: article id -> headline, only for real articles (social pseudo-articles are left out,
    their comments get no implicit mentions); parents: comment id -> text of parent comments of
    this batch's replies (in memory only). The other fields are filled here and reused by later
    batches: the linked entities of each headline and parent, and the explicit mentions per article.
    `collect` (evaluation only) receives every candidate, accepted or not."""

    titles: dict[str, str] = field(default_factory=dict)
    parents: dict[str, str] = field(default_factory=dict)
    title_entities: dict[str, list[str]] = field(default_factory=dict)
    parent_entities: dict[str, list[str]] = field(default_factory=dict)
    article_counts: dict[str, Counter] = field(default_factory=dict)
    collect: list | None = None


def link_comment(
    text: str,
    sentences: list[tuple[int, int]],
    ner_per_sentence: list[list[tuple[int, int, str, float]]],
    registry: Registry,
    context_text: str = "",
    country: str | None = None,
    wikidata=None,
    stats: Stats | None = None,
) -> list[Match]:
    """Link all entity mentions in one (cleaned) comment text."""
    spans: list[tuple[int, int, str, float]] = []
    for (s0, _s1), ner in zip(sentences, ner_per_sentence, strict=True):
        spans.extend((s0 + a, s0 + b, tag, score) for a, b, tag, score in ner)
    matches = registry.gazetteer(text, spans, context_text=context_text, country=country)
    # NER spans not covered by the gazetteer: exact alias lookup, then Wikidata fallback.
    for a, b, tag, score in spans:
        if tag not in ("PER", "ORG") or score < NER_MIN_SCORE:
            continue
        if any(m.start < b and a < m.end for m in matches):
            continue
        surface = text[a:b]
        eid = registry.lookup_surface(surface, tag, country)
        via = "ner"
        if eid is None and wikidata is not None:
            info = wikidata.lookup(strip_titles(surface), tag)
            if info and registry.rejected_alias(strip_titles(surface), info["qid"]):
                info = None  # "Signal" without messenger context: Wikidata must not bypass the alias rules
            if info:
                before = len(registry.entities)
                ent = registry.add_discovered(info)
                if len(registry.entities) > before and stats is not None:
                    stats.discovered.append(ent.id)
                eid, via = ent.id, "wikidata"
        if eid:
            matches.append(Match(a, b, strip_titles(surface), eid, via))
    matches.sort(key=lambda m: m.start)
    return matches


def mention_window(
    text: str, sents: list[tuple[int, int]], m: Match, matches: list[Match], mode: str = "clause"
) -> str:
    """Text window used to judge the sentiment toward the entity of match `m`.

    clause: the mention's sentence plus neighbouring sentences without other entities; narrowed
            to the clause when the sentence mentions other entities.
    entity: the mention's sentence +- one sentence, other entities' surfaces masked as "X".
    target: the mention's sentence +- one sentence, the target's own mentions in [brackets]
            (for the fine-tuned model, which reads the context and has to know which name is meant).
    """
    k = next((j for j, (a, b) in enumerate(sents) if a <= m.start < b), None)
    if k is None:
        return text[:MAX_WINDOW_CHARS]
    s0, s1 = sents[k]
    if mode == "target":
        w0, w1 = sents[max(0, k - 1)][0], sents[min(len(sents) - 1, k + 1)][1]
        parts, pos = [], w0
        for o in sorted(matches, key=lambda o: o.start):
            if o.entity_id == m.entity_id and pos <= o.start and o.end <= w1:
                parts += [text[pos : o.start], "[", text[o.start : o.end], "]"]
                pos = o.end
        parts.append(text[pos:w1])
        return "".join(parts)[:MAX_WINDOW_CHARS]
    if mode == "entity":
        w0, w1 = sents[max(0, k - 1)][0], sents[min(len(sents) - 1, k + 1)][1]
        chunk = list(text[w0:w1])
        for o in sorted(matches, key=lambda o: -o.start):
            if o.entity_id != m.entity_id and w0 <= o.start and o.end <= w1:
                chunk[o.start - w0 : o.end - w0] = list("X")
        return "".join(chunk)[:MAX_WINDOW_CHARS]
    others = [(o.start - s0, o.end - s0) for o in matches if o.entity_id != m.entity_id and s0 <= o.start < s1]
    if others:
        return target_window(text[s0:s1], m.start - s0, m.end - s0, others)

    def free(j: int) -> bool:
        a, b = sents[j]
        return not any(o.entity_id != m.entity_id and a <= o.start < b for o in matches)

    lo = k - 1 if k > 0 and free(k - 1) else k
    hi = k + 1 if k + 1 < len(sents) and free(k + 1) else k
    return text[sents[lo][0] : sents[hi][1]][:MAX_WINDOW_CHARS]


def analyze_comments(
    comments: list[dict],
    registry: Registry,
    ner,
    sentiment,
    *,
    source_country: dict[str, str] | None = None,
    article_context: dict[str, str] | None = None,
    wikidata=None,
    stats: Stats | None = None,
    implicit: ImplicitInput | None = None,
) -> dict[str, list[dict]]:
    """Return {commentId: [mention dicts]} for the given comments.

    Comment dicts need: id, text, sourceId, articleId, publishedAt (datetime), day (optional);
    parentId for replies. With `implicit`, comments under real articles that name no entity can
    get implicit mentions (analysis/implicit.py).
    """
    stats = stats or Stats()
    source_country = source_country or {}
    article_context = article_context or {}
    prepared = []
    all_sentences: list[str] = []
    for c in comments:
        text = clean_text(c.get("text") or "")
        sents = split_sentences(text) if text else []
        prepared.append((c, text, sents, len(all_sentences)))
        all_sentences.extend(text[a:b] for a, b in sents)
    stats.comments += len(comments)
    stats.sentences += len(all_sentences)
    n_comment_sentences = len(all_sentences)
    # headlines and parent comments that the implicit rules need, linked once per run
    extra: list[tuple[dict, str, str, list[tuple[int, int]], int, str | None]] = []
    if implicit is not None:
        seen: set[tuple[int, str]] = set()
        for c, _text, _sents, _o in prepared:
            country = source_country.get(c.get("sourceId") or "")
            aid, pid = c.get("articleId") or "", c.get("parentId") or ""
            todo = []
            if aid in implicit.titles and aid not in implicit.title_entities:
                todo.append((implicit.title_entities, aid, implicit.titles[aid]))
            if pid in implicit.parents and pid not in implicit.parent_entities:
                todo.append((implicit.parent_entities, pid, implicit.parents[pid]))
            for store, key, raw in todo:
                if (id(store), key) in seen:
                    continue
                seen.add((id(store), key))
                t = clean_text(raw)
                ss = split_sentences(t) if t else []
                extra.append((store, key, t, ss, len(all_sentences), country))
                all_sentences.extend(t[a:b] for a, b in ss)

    # NER only where it can matter: sentences with an upper-case letter and some length. Weak
    # aliases are case-sensitive and the NER model finds nothing in all-lowercase text anyway.
    t = time.time()
    todo = [i for i, x in enumerate(all_sentences) if len(x) >= MIN_NER_CHARS and _HAS_UPPER.search(x)]
    ner_out: list[list] = [[] for _ in all_sentences]
    if todo:
        for i, spans in zip(todo, ner.predict([all_sentences[i] for i in todo]), strict=True):
            ner_out[i] = spans
    stats.ner_sentences += len([i for i in todo if i < n_comment_sentences])
    stats.ner_seconds += time.time() - t

    t = time.time()
    for store, key, text, sents, offset, country in extra:
        ner_x = ner_out[offset : offset + len(sents)]
        ms = link_comment(text, sents, ner_x, registry, country=country) if sents else []
        store[key] = list(dict.fromkeys(m.entity_id for m in ms))
    linked: list[list[Match]] = []
    for c, text, sents, offset in prepared:
        if not sents:
            linked.append([])
            continue
        ner_c = ner_out[offset : offset + len(sents)]
        linked.append(
            link_comment(
                text,
                sents,
                ner_c,
                registry,
                context_text=article_context.get(c.get("articleId") or "", ""),
                country=source_country.get(c.get("sourceId") or ""),
                wikidata=wikidata,
                stats=stats,
            )
        )
    implicit_of: dict[str, list[imp.Candidate]] = {}
    if implicit is not None:
        for (c, _t, _s, _o), matches in zip(prepared, linked, strict=True):
            aid = c.get("articleId") or ""
            if matches and aid in implicit.titles:
                implicit.article_counts.setdefault(aid, Counter()).update({m.entity_id for m in matches})
        for (c, text, sents, _o), matches in zip(prepared, linked, strict=True):
            aid = c.get("articleId") or ""
            if not sents or aid not in implicit.titles:
                continue
            published = c.get("publishedAt")
            own = {m.entity_id for m in matches}
            counts = Counter(implicit.article_counts.get(aid) or {})
            counts.subtract(own)  # the article's *other* comments
            pid = c.get("parentId") or ""
            ctx = imp.CommentContext(
                title=implicit.titles.get(aid, ""),
                title_entities=implicit.title_entities.get(aid, []),
                article_counts=+counts,
                is_reply=bool(pid),
                parent_entities=implicit.parent_entities.get(pid) if pid else None,
                country=source_country.get(c.get("sourceId") or ""),
                day=c.get("day") or (day_key(published) if published else ""),
            )
            cands = imp.find(text, own, ctx, registry)
            stats.implicit_candidates += len(cands)
            if implicit.collect is not None:
                implicit.collect += [(c, text, ctx, x) for x in cands]
            implicit_of[c["id"]] = [x for x in cands if imp.accepted(x)]

    windows: list[tuple[str, str]] = []  # (window text, target entity name)
    window_index: dict[tuple[str, str], int] = {}
    drafts: list[tuple[dict, str, str, list[int], str | None]] = []  # comment, entity, surface, windows, via

    def add_window(w: tuple[str, str], idxs: list[int]) -> None:
        if w not in window_index:
            window_index[w] = len(windows)
            windows.append(w)
        if window_index[w] not in idxs:
            idxs.append(window_index[w])

    for (c, text, sents, _o), matches in zip(prepared, linked, strict=True):
        by_entity: dict[str, list[Match]] = {}
        for m in matches:
            by_entity.setdefault(m.entity_id, []).append(m)
        for eid, ms in by_entity.items():
            idxs: list[int] = []
            for m in ms[:MAX_WINDOWS_PER_MENTION]:
                w = mention_window(text, sents, m, matches, mode=getattr(sentiment, "window_mode", "clause"))
                add_window((w, registry.entities[eid].name), idxs)
            drafts.append((c, eid, ms[0].surface, idxs, None))
        for x in implicit_of.get(c["id"], []):
            idxs = []
            parent = (implicit.parents.get(c.get("parentId") or "", "") if x.rule == "B" else "") if implicit else ""
            title = implicit.titles.get(c.get("articleId") or "", "") if implicit else ""
            premise = imp.premise(imp.window(text, x.start, x.end), title, clean_text(parent))
            add_window((premise, registry.entities[x.entity_id].name), idxs)
            drafts.append((c, x.entity_id, text[x.start : x.end], idxs, x.via))
            stats.implicit[f"{x.rule}/{x.cue}"] += 1
    stats.link_seconds += time.time() - t

    t = time.time()
    scored = sentiment.predict(windows) if windows else []
    stats.sentiment_seconds += time.time() - t

    ents_of: dict[str, set[str]] = {}
    for c, eid, *_ in drafts:
        ents_of.setdefault(c["id"], set()).add(eid)
    out: dict[str, list[dict]] = {c["id"]: [] for c in comments}
    for c, eid, surface, idxs, via in drafts:
        label, score, conf = combine([scored[i] for i in idxs])
        published = c.get("publishedAt")
        day = c.get("day") or (day_key(published) if published else None)
        if not day:
            continue
        out[c["id"]].append(
            {
                "commentId": c["id"],
                "articleId": c.get("articleId") or "",
                "sourceId": c.get("sourceId") or "",
                "entityId": eid,
                "day": day,
                "publishedAt": published,
                "surface": surface[:80],
                "label": label,
                "score": max(-1.0, min(1.0, float(score))),
                "confidence": max(0.0, min(1.0, float(conf))),
                "coEntityIds": sorted(ents_of[c["id"]] - {eid}),
            }
        )
        if via:
            out[c["id"]][-1].update({"implicit": True, "via": via})
        if c.get("sourceType"):
            out[c["id"]][-1]["sourceType"] = c["sourceType"]
        stats.mentions += 1
    if wikidata is not None:
        stats.wikidata_lookups = wikidata.lookups
    return out
