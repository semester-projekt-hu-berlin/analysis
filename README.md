# Analysis · Meinungspuls

Component 2 of the HU Berlin semester project *Stimmungsanalyse im Web*: finds which **public
persons and organisations** German-language news comments talk about, determines the
**sentiment toward each of them**, and writes the aggregates that the API and the dashboard
serve ("how often is the SPD mentioned positively / negatively per day").

The data contract is [`infrastructure/SCHEMA.md`](https://github.com/semester-projekt-hu-berlin/infrastructure/blob/main/SCHEMA.md).

## Pipeline

```
 Firestore comments (analyzed == false, oldest crawledAt first, ≤ 3000 per run)
        │   only articleId, sourceId, text, publishedAt, day are read (field mask)
        ▼
 clean (URLs, whitespace, 2000-char cap) ─► sentence split (segtok, ≤ 400 chars)
        │
        ▼
 NER  ── Flair flair/ner-german (PER / ORG / LOC / MISC), batched over all sentences of a run
        │
        ▼
 Entity linking
   1. gazetteer over data/entities.yaml (331 curated entities, ~780 aliases)
        plain alias  ─ always             "Merz", "Sozis", "AfD-Wähler"
        weak alias   ─ needs NER span     "Reiche" (PER) vs. "die Reiche Elite"
        context alias─ needs keywords     "Leverkusen" + Spiel/Tor/Bundesliga… (comment or article title)
        country tie-break                 "Grüne" on derstandard.at → Austrian Greens
   2. NER span not covered → exact alias lookup (titles stripped: "Kanzler Merz" → "Merz")
   3. still unknown PER/ORG → Wikidata fallback with strict public-figure filter (cached, capped)
        │
        ▼
 Target sentiment ─ window = sentence of the mention, narrowed to the clause if the sentence
        │           mentions several entities ("Merz macht das gut, aber die SPD hat versagt")
        │           → German sentiment BERT → label pos/neu/neg, score P(pos)−P(neg), confidence
        ▼
 mentions/{commentId}_{entityId}  (surface span only, never the comment text)
 comments/{id}: analyzed=true, analyzedAt, analysisVersion
        │
        ▼
 Aggregation (idempotent)
   for every affected day: recompute entities/{id}/days/{day} from ALL mentions of that day
   entity totals += (new day doc − old day doc), committed atomically with the day docs
   rolling mentions7d / avgSentiment7d / mentionsPrev7d / avgSentimentPrev7d
   meta/trending (top 50, risers / fallers ≥ 20 mentions in both windows), meta/stats
```

All Firestore access goes through the REST API (`runQuery`, `batchGet`, `commit` with ≤ 500
writes per batch, `runAggregationQuery` for counts) using a service account and
`AuthorizedSession` — no gRPC (the development sandbox cannot use it; REST also works in CI).

## Models and why

| Step | Model | Why |
|---|---|---|
| NER | [`flair/ner-german`](https://huggingface.co/flair/ner-german) (Flair 0.15, BiLSTM-CRF over contextual string embeddings, CoNLL-03 German F1 ≈ 87) | Course recommendation; robust on noisy, lowercased comment text; CPU-only. |
| NER (alt.) | `domischwimmbeck/bert-base-german-cased-fine-tuned-ner` (`--ner bert`) | ≈ 2.9× faster; same spans on our smoke tests. Switch if runs get too slow. |
| Sentiment | [`oliverguhr/german-sentiment-bert`](https://huggingface.co/oliverguhr/german-sentiment-bert) (`--sentiment guhr`) | Trained on 1.8 M German texts incl. social-media comments; free, CPU-friendly (≈ 57 windows/s). See evaluation for the comparison with alternatives. |

**Benchmark** (this sandbox, 4 CPU threads, CPU torch 2.x): Flair NER 18 sentences/s,
BERT NER 52 sentences/s, sentiment 55–57 windows/s. A typical comment has 2–4 sentences, so an
hourly run with a few hundred new comments takes 1–3 minutes plus ≈ 1 minute of model loading;
the default cap of 3000 comments per run (~9000 sentences) stays below ~12 minutes of NER on a
4-vCPU GitHub runner. Length caps (2000 chars per comment, 400 per sentence, 256 tokens per
sentiment window) bound the worst case.

The default pipeline is entirely open-source and free. The HU LLM offer is not used.

## Entity linking

* **Seed list** [`data/entities.yaml`](data/entities.yaml): all Bundestag parties and major
  Landtag parties (incl. Freie Wähler, SSW), Austrian parties, US parties; the federal cabinet,
  party and parliamentary group leaders, Ministerpräsident:innen, the Bundespräsident and
  names discussed for the Bundesversammlung on 30 Jan 2027 (Ilse Aigner, Udo Di Fabio, Karin
  Prien, Julia Klöckner, Boris Rhein); major Austrian politicians; the US president,
  vice-president, cabinet and leading figures of the 2026 midterms; key international leaders;
  institutions (Bundestag, Bundesrat, BVerfG, EU, Commission, EP, ECB, NATO, UN, …), DAX
  companies plus large tech companies, Bundesliga / 2. Bundesliga / Austrian clubs. Because the
  crawler also covers derstandard.at (AT) and blick.ch (CH), Austrian and Swiss parties,
  Bundesrät:innen, institutions, companies and clubs are included as well.
  Every entry has a Wikidata id, resolved with `scripts/resolve_wikidata.py` (SPARQL, label
  match, type/description plausibility check, largest sitelink count) and checked by hand
  (43 manual overrides for homonyms such as "Daniel Günther" the Swedish singer).
* **Alias modes** (see the header of the YAML): plain (case-insensitive), `=` case-sensitive
  (`=UN`, `=SAP`), `~` weak (must overlap an NER span of a compatible type, or a context
  keyword is present: `~Reiche`, `~Union`, `~Bas`), `@` contextual (needs a keyword from a
  context set – football, US politics, business, politics – in the comment or the article
  title: `@Leverkusen`, `@Bayern`, `@Demokraten`). Longest match wins ("Siemens Energy" vs
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
* For each affected day, the day documents of all entities are **recomputed** from all mentions
  of that day (counts, `scoreSum`, `bySource`, top-20 `coMentions`) – never incremented.
* Entity totals change by `new − old` of each recomputed day doc, written in the **same atomic
  commit** as the day docs. A crash before the commit changes nothing; re-running yields a zero
  delta. `python -m analysis reaggregate` rebuilds everything from scratch (and deletes orphan
  day docs) if anything ever drifts.
* Rolling windows are recomputed every run for all entities (they slide even without new
  mentions). `meta/trending` and `meta/stats` (`mentions`, `entities` = allowlisted entities
  with ≥ 1 mention, `lastAnalysisAt`) are rewritten after each run.
* Tests: `tests/test_aggregate.py` (math), `tests/test_pipeline.py` (end-to-end on an in-memory
  Firestore: re-run gives an identical snapshot, reaggregate matches incremental, stale mention
  removal, privacy).

### Optional fields added to the schema

All optional, no existing field changed:

| where | field | why |
|---|---|---|
| `entities` | `scoreSum` | exact running sum, so `avgSentiment` can be updated by deltas without rounding drift |
| `entities` | `recentDays` | `{day: {mentions,pos,neu,neg,scoreSum}}` for the last 14 days, so rolling 7-day windows can slide every run without reading the `days` subcollection |
| `entities/{id}/days/{day}` | `lastMentionAt` | latest mention timestamp of the day, used for `entities.lastMentionAt` |

## Evaluation

See [`eval/README.md`](eval/README.md) for the protocol and results.

## Running

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt

export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json   # or FIREBASE_SERVICE_ACCOUNT='{"type": ...}'

python -m analysis seed                 # curated entities (+ Wikimedia images) and events
python -m analysis run --limit 3000     # analyze unanalyzed comments, update aggregates
python -m analysis reaggregate          # rebuild all aggregates from mentions
python -m analysis run --ner bert       # faster NER backend; --no-wikidata disables the fallback

pytest -q && ruff check . && ruff format --check .
```

Re-analysing everything with a new model version: set `analyzed = false` on the comments
(their old `analysisVersion` is kept, so stale mentions are cleaned up) and run again.

## Schedule (GitHub Actions)

| workflow | trigger | what |
|---|---|---|
| `analyze.yml` | hourly at minute 30 (crawler runs on the hour) + manual | `python -m analysis run`; CPU-only torch, pip cache, Hugging Face + Flair model cache, Wikidata lookup cache; `concurrency: analysis` prevents overlapping runs |
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
