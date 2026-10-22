# Analysis · Meinungspuls

Component 2 of the HU Berlin semester project *Stimmungsanalyse im Web*: finds which **public
persons and organisations** German-language news comments talk about, determines the
**sentiment toward each of them**, and writes the aggregates that the API and the dashboard
serve ("how often is the SPD mentioned positively / negatively per day").

The data contract is [`infrastructure/SCHEMA.md`](https://github.com/semester-projekt-hu-berlin/infrastructure/blob/main/SCHEMA.md).

## Pipeline

```
 Firestore comments (analyzed == false, oldest crawledAt first, ≤ 20000 per run, batches of 250)
        │   only articleId, sourceId, text, publishedAt, day, analysis fields are read (field mask)
        ▼
 clean (URLs, whitespace, 2000-char cap) ─► sentence split (segtok, ≤ 400 chars)
        │
        ▼
 NER  ── German BERT NER (default) or Flair flair/ner-german (PER / ORG / LOC / MISC), batched;
        │   skipped for sentences without an upper-case letter or shorter than 4 characters
        │
        ▼
 Entity linking
   1. gazetteer over data/entities.yaml (384 curated entities, ~1100 aliases)
        plain alias  ─ always             "Merz", "Sozis", "AfD-Wähler"
        weak alias   ─ needs NER span     "Reiche" (PER) vs. "die Reiche Elite"
        context alias─ needs keywords     "Leverkusen" + Spiel/Tor/Bundesliga… (comment or article title)
        tie-breaks                        other name words in the text ("Eintracht … Braunschweig"),
                                          context-confirmed reading beats plain ("RWE" + Spiel → Rot-Weiss Essen),
                                          source country ("Grüne"/"FCB" on derstandard.at / blick.ch)
   2. NER span not covered → exact alias lookup (titles stripped: "Kanzler Merz" → "Merz")
   3. still unknown PER/ORG → Wikidata fallback with strict public-figure filter (cached, capped)
        │
        ▼
 Target sentiment ─ window = sentence of the mention plus neighbouring sentences without other
        │           entities; narrowed to the clause if the sentence mentions several entities
        │           ("Merz macht das gut, aber die SPD hat versagt")
        │           → calibrated target sentiment: NLI (mDeBERTa XNLI) "Der Kommentar äußert sich
        │             positiv/negativ über {Entity}", "Der Kommentar bewertet {Entity}"
        │             + news-sentiment BERT → logistic regression → pos/neu/neg
        ▼
 mentions/{commentId}_{entityId}  (surface span only, never the comment text)
 comments/{id}: analyzed=true, analyzedAt, analysisVersion
        │
        ▼
 Aggregation (idempotent, every 2000 comments and at the end of a run)
   for every affected (entity, day) pair: recompute entities/{id}/days/{day} from the mentions
     of that pair (query entityId == X AND day == D; coMentions from mentions.coEntityIds)
   entity totals += (new day doc − old day doc), committed atomically with the day docs
   rolling mentions7d / avgSentiment7d / mentionsPrev7d / avgSentimentPrev7d from recentDays
   meta/trending (top 50 with 30-day sparkline, risers / fallers ≥ 20 mentions in both windows),
   meta/stats
```

All Firestore access goes through the REST API (`runQuery`, `batchGet`, `commit` in batches
of ≤ 100 writes, `runAggregationQuery` for counts; every request has a 120 s timeout and
exponential backoff, and batches that hit 429 are split in halves) using a service account and
`AuthorizedSession` — no gRPC (the development sandbox cannot use it; REST also works in CI).

## Models and why

| Step | Model | Why |
|---|---|---|
| NER (default) | `domischwimmbeck/bert-base-german-cased-fine-tuned-ner` (`--ner bert`, GermEval 2014) | ≈ 2.9× faster than Flair on CPU (52 vs 18 sentences/s), same spans on our smoke tests. The runs are twice a day with a half-day backlog, so CPU time matters. |
| NER (alt.) | [`flair/ner-german`](https://huggingface.co/flair/ner-german) (`--ner flair`; Flair 0.15, BiLSTM-CRF, CoNLL-03 German F1 ≈ 87) | Course recommendation; kept as an option. |
| Sentiment (default `cal`, v2) | zero-shot NLI [`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) with the hypotheses "Der Kommentar äußert sich positiv / negativ über {Entity}." and "Der Kommentar bewertet {Entity}." (target vs. context; 3 passes) + [`mdraw/german-news-sentiment-bert`](https://huggingface.co/mdraw/german-news-sentiment-bert) (1 pass), combined by a logistic regression trained on 456 hand-labelled pairs (`data/sentiment_calibration.json`) | Target-aware. Test macro-F1 0.483 (5-fold CV 0.485), vs 0.383 for the original ensemble ([eval](eval/README.md)). Free, CPU, 1.7 mentions/s with 2 threads. |

**Throughput** (production data, CPU torch, BERT NER + calibrated sentiment): about 280
comments/min with **2 threads** (500 newest comments: NER 67 s, sentiment 41 s, 110 mentions;
measured with `ANALYSIS_THREADS=2`), so a half-day of 6k comments takes about 25 min on a 2-vCPU
runner. The workflow uses `--limit 9000` and a 150 min timeout; threads default to the number
of CPUs. Length caps (2000 chars per comment, 400 per sentence, 1000 per sentiment window, 256
tokens) bound the worst case. Social posts (Mastodon, Bluesky) get the same pipeline after
cleaning: URLs and @handles are removed, and hashtags are split into words ("#MerzMussWeg" →
"Merz Muss Weg").

**Firestore cost** (`python scripts/bench_firestore_ops.py`, in-memory Firestore with counters,
4000 synthetic comments over two days):

| | reads / 1000 comments | writes / 1000 comments |
|---|---|---|
| before (day-wide recompute after every 250 comments) | 6338 | 2148 |
| after (pair-wise recompute every 2000 comments, unchanged docs not written, no mention lookup for new comments) | 2581 | 2004 |

Of the remaining reads, 1000 are the comments themselves. The writes are about one per comment
(the `analyzed` flag) plus one per mention.

The default pipeline is entirely open-source and free. The HU LLM offer is not used.

## Entity linking

* **Seed list** [`data/entities.yaml`](data/entities.yaml): all Bundestag parties and major
  Landtag parties (incl. Freie Wähler, SSW), Austrian parties, US parties; the federal cabinet,
  party and parliamentary group leaders, Ministerpräsident:innen, the Bundespräsident and
  names discussed for the Bundesversammlung on 30 Jan 2027 (Ilse Aigner, Udo Di Fabio, Karin
  Prien, Julia Klöckner, Boris Rhein); major Austrian politicians; the US president,
  vice-president, cabinet and leading figures of the 2026 midterms; key international leaders;
  institutions (Bundestag, Bundesrat, BVerfG, EU, Commission, EP, ECB, NATO, UN, …), DAX
  companies plus large tech companies. Football: all clubs of the Bundesliga and 2. Bundesliga
  2026/27, popular 3. Liga and traditional clubs, the Austrian Bundesliga and the Swiss Super
  League, big international clubs, the German/Austrian/Swiss national teams, DFB, DFL, ÖFB, SFV,
  UEFA, FIFA (91 football entities, with nicknames like "Werkself", "Fohlen", "Knappen",
  "Effzeh"). Because the crawler also covers derstandard.at (AT) and blick.ch (CH), Austrian and
  Swiss parties, Bundesrät:innen, institutions and companies are included as well.
  Every entry has a Wikidata id, resolved with `scripts/resolve_wikidata.py` (SPARQL, label
  match, type/description plausibility check, largest sitelink count) and checked by hand
  (44 manual overrides for homonyms such as "Daniel Günther" the Swedish singer; clubs via a
  football-specific search that excludes women's/youth teams).
* **Images** (`seed`): direct `upload.wikimedia.org` thumbnail URLs (250 px, the smallest
  standard width Wikimedia serves near the requested 160 px; 160 px returns HTTP 400). Persons:
  portrait (P18). Organisations, parties, clubs, media: logo only (P154, P8972, P158), never
  P18 (often a photo of the headquarters). No logo → `imageUrl: null`. `imageKind` is
  `"logo" | "portrait" | null`.
* **Alias modes** (see the header of the YAML): plain (case-insensitive), `=` case-sensitive
  (`=UN`, `=SAP`), `~` weak (must overlap an NER span of a compatible type, or a context
  keyword is present: `~Reiche`, `~Union`, `~Bas`), `@` contextual (needs a keyword from a
  context set – football, US politics, business, politics – in the comment or the article
  title: `@Leverkusen`, `@Bayern`, `@Demokraten`), `^` local (only on sources of the
  entity's own country: `^Linke` is *Die Linke* on taz.de but just "the left" on blick.ch). Longest match wins ("Siemens Energy" vs
  "Siemens", "CDU/CSU" vs "CDU"); German inflection suffixes are accepted ("Trumps", "Merz'",
  "Grünen").
* **Ambiguity**: the same alias may belong to several entities; the source country (from
  `sources/{id}.country`) decides ("Grüne" → `gruene` on DE sites, `gruene-at` on AT sites),
  context keywords decide club vs. company vs. party ("Bayer", "Union").
* **Wikidata fallback** (optional, on by default, ≤ 150 lookups per run): for PER/ORG spans
  that match nothing. A candidate is accepted only if the Wikidata label or an alias equals the
  surface form **and**
  * persons: instance of human (Q5), the comment contains the full name (≥ 2 tokens), the item
    has an occupation (P106) or position held (P39), and ≥ 15 sitelinks;
  * organisations: instance of a whitelisted organisation class (party, company, club,
    institution, union, broadcaster, …) and ≥ 10 sitelinks.

  If the item's QID belongs to a seed entity, the seed entity is used. New entities get
  `allowlisted: true` and only weak aliases (later matches need NER confirmation). Lookup
  results are cached in `.cache/wikidata.json` (restored between CI runs via `actions/cache`)
  keyed by **SHA-1 of the surface form**, so names typed by commenters are never stored in
  clear text. On Wikidata rate limiting the fallback is switched off for the rest of the run.

## Privacy rules

* Only entities from the curated seed list or Wikidata items that pass the public-figure
  filter above are ever written to `entities` (all with `allowlisted: true`). A private person
  named in a comment ("mein Nachbar Klaus Müller") produces no entity, no mention, no cache
  entry in clear text (tested in `tests/test_linker.py` and `tests/test_pipeline.py`).
* Commenter user names are never read: comments are fetched with a field mask
  (`articleId, sourceId, text, publishedAt, crawledAt, day, analysisVersion`); the crawler does
  not store author fields at all.
* Comment text stays inside the pipeline process. `mentions` store only the matched surface
  span (≤ 80 chars, e.g. "Kanzler Merz"); logs contain counts only. The evaluation gold file
  stores mention ids + labels, never text (`eval/sample.local.jsonl` with text is git-ignored).
* Service account keys are never committed (`.gitignore`); CI uses the organisation secret.

## Aggregation and idempotency

* Mention ids are deterministic (`commentId_entityId`): re-analysing a comment overwrites its
  mentions; mentions of entities that no longer match are deleted.
* For each affected (entity, day) pair, the day document is **recomputed** from the mentions of
  that pair (counts, `scoreSum`, `bySource`, top-20 `coMentions` via `mentions.coEntityIds`),
  never incremented. Unchanged day docs are not written.
* The pairs still to aggregate are stored in `meta/analysis.pendingPairs` *before* the mentions
  are written; a run that stops in between is completed by the next run. Each comment's
  mentions and its `analyzed` flag are committed atomically together.
* Entity totals change by `new − old` of each recomputed day doc, written in the **same atomic
  commit** as the day docs. A crash before the commit changes nothing; re-running yields a zero
  delta. `python -m analysis reaggregate` rebuilds everything from scratch (and deletes orphan
  day docs) if anything ever drifts.
* **Baseline**: comment sections are negative on average, so `meta/stats` and `meta/trending`
  carry `baseline7d` (mention-weighted mean sentiment of all entities over 7 days) and
  `baselineBySource7d`; every entity with ≥ 10 mentions in 7 days gets
  `relative7d = avgSentiment7d − baseline7d`. `top` lists only entities with ≥ 10 mentions in
  7 days; risers/fallers need ≥ 20 in both windows.
* **Relative context**: every entity with ≥ 10 mentions in 7 days gets `rankInType` /
  `peersInType` / `percentileInType` (by `avgSentiment7d` within its type; 100 = most
  positive) and the same across all types (`rankAll`, `peersAll`, `percentileAll`);
  `meta/peers` holds the peer strip per type (one read).
* **Evidence**: each day doc lists `topArticles` (the 20 articles with the most mentions that
  day, with label counts).
* **Community texts** (`sourceType: "community"`, submitted through the API) are analysed like
  comments but never enter `days`, entity totals, windows, baselines, trending or peers. They
  are aggregated into `entities/{id}/community_text/{day}` and the entity field
  `communityText7d`.
* **Entity requests**: at the start of every run (and with `python -m analysis seed-requests`),
  accepted `entity_requests/{qid}` are re-validated with Wikidata (≥ 10 sitelinks; persons
  need an occupation or position, organisations an organisation class) and seeded as allowlisted
  entities, or marked `failed` with a reason. If Wikidata rate-limits, they are retried in the
  next run.
* Rolling windows are recomputed every run for all entities (they slide even without new
  mentions) from `recentDays`; only entity docs whose values change are written. `meta/trending` and `meta/stats` (`mentions`, `entities` = allowlisted entities
  with ≥ 1 mention, `lastAnalysisAt`) are rewritten after each run.
* Tests: `tests/test_aggregate.py` (math), `tests/test_pipeline.py` (end-to-end on an in-memory
  Firestore: re-run gives an identical snapshot, reaggregate matches incremental, stale mention
  removal, privacy).

### Optional fields added to the schema

All optional, no existing field changed:

| where | field | why |
|---|---|---|
| `entities` | `scoreSum` | exact running sum, so `avgSentiment` can be updated by deltas without rounding drift |
| `entities` | `recentDays` | `{day: {mentions,pos,neu,neg,scoreSum}}` for the last 30 days, so rolling windows and the sparkline need no reads of the `days` subcollection |
| `entities` | `imageKind` | `"logo" \| "portrait" \| null` |
| `entities/{id}/days/{day}` | `lastMentionAt` | latest mention timestamp of the day, used for `entities.lastMentionAt` |
| `mentions` | `coEntityIds` | other entities of the same comment, so one (entity, day) can be recomputed from its own mentions |
| `meta/trending` items | `spark` | 30 daily `avgSentiment` values (oldest first, ending on the Berlin day of `updatedAt`, `null` without mentions) |
| `meta/analysis` | `pendingPairs` | internal crash-recovery state |

These are documented in `infrastructure/SCHEMA.md`, which also lists the composite index
`mentions (entityId, day)`. Equality-only queries also work without it.

## Evaluation and limitations

[`eval/README.md`](eval/README.md) has the protocol and all numbers. In short: 480
hand-labelled (comment, entity) pairs from all major sources. Linking precision is 93.6 %.
Sentiment toward the entity reaches test macro-F1 0.483 (5-fold CV 0.485); per class F1 is
pos 0.41, neu 0.44, neg 0.60.

**Limitations.** A single label is often wrong; only aggregates over many mentions are
meaningful. Neutral mentions are under-detected and often scored as negative, and sarcasm is
the main error source, along with entities that are only context (victims, comparisons). The
gold labels come from one annotator. Comment sections are negative overall, so compare entities via `relative7d`
(entity average minus `baseline7d`) rather than by absolute averages.

## Running

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt

export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json   # or FIREBASE_SERVICE_ACCOUNT='{"type": ...}'

python -m analysis seed                 # curated entities (+ Wikimedia images) and events
python -m analysis run                  # analyze unanalyzed comments (≤ 20000), update aggregates
python -m analysis run --ner flair      # Flair NER instead of BERT; --no-wikidata disables the fallback
python -m analysis rescore              # re-run only the sentiment step for all mentions, re-aggregate
python -m analysis seed-requests        # seed accepted entity_requests (also done at the start of `run`)
python -m analysis reaggregate --since 2026-10-20   # recompute only pairs from that day on (cheap)
python -m analysis reaggregate          # full rebuild from all mentions (full scan, use rarely)
python scripts/eval_sentiment.py score  # sentiment evaluation (see eval/README.md)
python scripts/bench_firestore_ops.py   # reads/writes per 1000 comments

pytest -q && ruff check . && ruff format --check .
```

Re-analysing everything with a new model version: set `analyzed = false` on the comments
(their old `analysisVersion` is kept, so stale mentions are cleaned up) and run again.

## Schedule (GitHub Actions)

| workflow | trigger | what |
|---|---|---|
| `analyze.yml` | twice a day at 05:30 and 17:30 UTC (`30 5,17 * * *`), about an hour after the crawler (`17 4,16 * * *`), + manual | `python -m analysis run` over the half-day backlog, newest first (about 6k comments; `--limit 9000`, timeout 150 min, aggregation every 2000 comments); CPU-only torch, pip cache, Hugging Face + Flair model cache, Wikidata lookup cache; `concurrency: analysis` prevents overlapping runs |
| `seed.yml` | push to `data/*.yaml` + manual | `python -m analysis seed` |
| `ci.yml` | push / PR | ruff + pytest (fake models, no torch needed) |

The workflows need the organisation secret `FIREBASE_SERVICE_ACCOUNT`.

## Events

`data/events.yaml` (written by `seed`): 3 Nov 2026 US midterms; 30 Jan 2027 Bundesversammlung
(date set by Bundestagspräsidentin Klöckner); 18 Apr 2027 Landtagswahlen Saarland and
Schleswig-Holstein; 25 Apr 2027 Landtagswahl Nordrhein-Westfalen (dates from
[wahlrecht.de](https://www.wahlrecht.de/termine.htm), confirmed by the Bundeswahlleiterin's list).

## Layout

```
analysis/
  firestore.py   REST client (typed values, runQuery with cursor paging, batchGet, commit)
  textutil.py    cleaning, sentence split, slugs, searchTokens, Berlin day keys
  linker.py      registry, alias modes, gazetteer, disambiguation
  wikidata.py    Wikidata client, public-figure filter, hashed lookup cache
  ner.py         Flair / BERT / none backends
  sentiment.py   target window, sentiment models, label/score mapping
  analyze.py     comments → mentions (pure, no I/O)
  aggregate.py   day docs, deltas, rolling windows, trending (pure)
  pipeline.py    Firestore orchestration: run, reaggregate, seed
data/            entities.yaml, events.yaml
scripts/         resolve_wikidata.py, eval_sentiment.py
tests/           pytest (fakes for Firestore and models)
```
