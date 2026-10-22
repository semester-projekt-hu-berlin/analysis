"""Turn comments into (comment, entity) mentions: NER -> linking -> target sentiment.

Pure with respect to Firestore: takes comment dicts, returns mention dicts. Comment
text is only held in memory; mentions keep nothing but the matched surface span.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

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
    """
    k = next((j for j, (a, b) in enumerate(sents) if a <= m.start < b), None)
    if k is None:
        return text[:MAX_WINDOW_CHARS]
    s0, s1 = sents[k]
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
) -> dict[str, list[dict]]:
    """Return {commentId: [mention dicts]} for the given comments.

    Comment dicts need: id, text, sourceId, articleId, publishedAt (datetime), day (optional).
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

    # NER only where it can matter: sentences with an upper-case letter and some length. Weak
    # aliases are case-sensitive and the NER model finds nothing in all-lowercase text anyway.
    t = time.time()
    todo = [i for i, x in enumerate(all_sentences) if len(x) >= MIN_NER_CHARS and _HAS_UPPER.search(x)]
    ner_out: list[list] = [[] for _ in all_sentences]
    if todo:
        for i, spans in zip(todo, ner.predict([all_sentences[i] for i in todo]), strict=True):
            ner_out[i] = spans
    stats.ner_sentences += len(todo)
    stats.ner_seconds += time.time() - t

    t = time.time()
    windows: list[tuple[str, str]] = []  # (window text, target entity name)
    window_index: dict[tuple[str, str], int] = {}
    drafts: list[tuple[dict, str, Match, list[int]]] = []
    for c, text, sents, offset in prepared:
        if not sents:
            continue
        ner_c = ner_out[offset : offset + len(sents)]
        matches = link_comment(
            text,
            sents,
            ner_c,
            registry,
            context_text=article_context.get(c.get("articleId") or "", ""),
            country=source_country.get(c.get("sourceId") or ""),
            wikidata=wikidata,
            stats=stats,
        )
        by_entity: dict[str, list[Match]] = {}
        for m in matches:
            by_entity.setdefault(m.entity_id, []).append(m)
        for eid, ms in by_entity.items():
            idxs: list[int] = []
            for m in ms[:MAX_WINDOWS_PER_MENTION]:
                w = mention_window(text, sents, m, matches, mode=getattr(sentiment, "window_mode", "clause"))
                w = (w, registry.entities[eid].name)
                if w not in window_index:
                    window_index[w] = len(windows)
                    windows.append(w)
                if window_index[w] not in idxs:
                    idxs.append(window_index[w])
            drafts.append((c, eid, ms[0], idxs))
    stats.link_seconds += time.time() - t

    t = time.time()
    scored = sentiment.predict(windows) if windows else []
    stats.sentiment_seconds += time.time() - t

    ents_of: dict[str, set[str]] = {}
    for c, eid, _first, _idxs in drafts:
        ents_of.setdefault(c["id"], set()).add(eid)
    out: dict[str, list[dict]] = {c["id"]: [] for c in comments}
    for c, eid, first, idxs in drafts:
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
                "surface": first.surface[:80],
                "label": label,
                "score": max(-1.0, min(1.0, float(score))),
                "confidence": max(0.0, min(1.0, float(conf))),
                "coEntityIds": sorted(ents_of[c["id"]] - {eid}),
            }
        )
        if c.get("sourceType"):
            out[c["id"]][-1]["sourceType"] = c["sourceType"]
        stats.mentions += 1
    if wikidata is not None:
        stats.wikidata_lookups = wikidata.lookups
    return out
