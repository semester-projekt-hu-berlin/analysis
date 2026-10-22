"""Turn comments into (comment, entity) mentions: NER -> linking -> target sentiment.

Pure with respect to Firestore: takes comment dicts, returns mention dicts. Comment
text is only held in memory; mentions keep nothing but the matched surface span.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .linker import Match, Registry, strip_titles
from .sentiment import combine, target_window
from .textutil import clean_text, day_key, split_sentences

log = logging.getLogger(__name__)

NER_MIN_SCORE = 0.75
MAX_WINDOWS_PER_MENTION = 3


@dataclass
class Stats:
    comments: int = 0
    sentences: int = 0
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

    t = time.time()
    ner_out = ner.predict(all_sentences) if all_sentences else []
    stats.ner_seconds += time.time() - t

    t = time.time()
    windows: list[str] = []
    window_index: dict[str, int] = {}
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
                s0, s1 = next(((a, b) for a, b in sents if a <= m.start < b), (0, len(text)))
                others = [(o.start - s0, o.end - s0) for o in matches if o.entity_id != eid and s0 <= o.start < s1]
                w = target_window(text[s0:s1], m.start - s0, m.end - s0, others)
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
            }
        )
        stats.mentions += 1
    if wikidata is not None:
        stats.wikidata_lookups = wikidata.lookups
    return out
