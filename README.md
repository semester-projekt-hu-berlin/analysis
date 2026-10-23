# Analysis · Meinungspuls

Component 2 of the HU Berlin semester project *Stimmungsanalyse im Web*: finds which **public
persons and organisations**, and which **apps and digital services**, German-language news
comments talk about, determines the
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
   1. gazetteer over data/entities.yaml (591 curated entities incl. 207 apps & services, ~1560
        aliases) and data/entities_generated.yaml (6,343 entities)
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
* **Apps & Dienste** (type `PRODUCT`, 207 entities in the seed list): apps, platforms, digital
  services, operating systems and games that people in Germany discuss. See
  [Apps & Dienste](#apps--dienste-type-product) below.
* **Generated list** [`data/entities_generated.yaml`](data/entities_generated.yaml), built by
  `scripts/generate_entities.py` from a few large Wikidata SPARQL queries (descriptive
  User-Agent, backoff on 429, results cached in `data/wikidata_cache/`; `--refresh` re-queries)
  and loaded after the seed list (curated QIDs and aliases always win). It has 6,343 entities
  (6,934 with the curated list); groups overlap:
  * **1,423 politicians**:

    | group | count |
    |---|---|
    | Bundestag members, 21st term | 630 |
    | German MEPs | 95 |
    | current federal and state office holders | 130 |
    | federal and state party leaders | 168 |
    | parliamentary group leaders | 10 |
    | Landtag members with ≥ 5 sitelinks | 134 |
    | living former chancellors, federal ministers and heads of state governments | 138 |
    | German politicians with ≥ 10 sitelinks | 598 |

    Descriptions come from the role, e.g. "Bundestagsabgeordnete (SPD)" or
    "Landtagsabgeordneter (CDU)"; the party comes from P102.
  * **3,772 public persons** (German citizens, alive):

    | group | count |
    |---|---|
    | athletes with ≥ 20 sitelinks (sports bots inflate sitelinks) | 1,195 |
    | current Bundesliga players and coaches, any nationality | 116 |
    | TV presenters and journalists with ≥ 5 sitelinks | 696 |
    | everyone else with ≥ 10 sitelinks (actors, musicians, entrepreneurs, scientists, YouTubers…) | 1,760 |
    | DAX/MDAX CEOs | 67 |
    | BVerfG judges, EKD and DBK heads, Bundesbank and RKI presidents | 6 |

    The description comes from Wikidata ("deutscher Fernsehmoderator"), or from the role
    ("Vorstandsvorsitz SAP").
  * **781 companies**: DAX/MDAX/SDAX/TecDAX members, German companies with > 10,000 employees
    (P1128), companies and brands with ≥ 8 sitelinks, and a brand list (Aldi, Kaufland, Penny,
    Rossmann, MediaMarkt, Bertelsmann, Otto, Flixbus…). Public bodies and non-profits
    (ministries, Bundeswehr, police, churches, clinics…) are filtered out by description. Legal
    forms are stripped for short names ("Rational AG" → "Rational").
  * **67 tech and media entities**, from a hand-checked QID list (`MEDIA_TECH`):
    * Tech: Alphabet, X Corp., Twitter Inc., ByteDance, Samsung, Huawei, Delivery Hero, Celonis,
      United Internet/1&1, TeamViewer, Helsing, Aleph Alpha, Anthropic, Mistral…
    * Broadcasters: WDR, NDR, BR, SWR, MDR, RBB, HR, SR, Radio Bremen, DW, Deutschlandfunk,
      RTL, Sat.1, VOX, n-tv, WELT TV, 3sat, KiKA…
    * Newspapers and agencies: SZ, FAZ, ZEIT, WELT, Handelsblatt, Stern, Tagesspiegel, NZZ,
      dpa, Correctiv…

    Apple, Google, Meta, Microsoft, Amazon, OpenAI, Nvidia, Tesla, SAP, Telekom, ARD and ZDF
    were already curated. Brands that are mainly discussed as a service (Netflix, Spotify,
    PayPal, N26, Trade Republic, Temu, Shein, Uber, Wolt, Airbnb, DeepL, DAZN, idealo, Check24,
    trivago, GMX) are curated `PRODUCT` entities with their old ids, so the generator skips them.
  * **28 TV shows** (`SHOWS`, type ORG, description "TV-Sendung (ZDF)"): Tagesschau,
    Tagesthemen, heute journal, Markus Lanz, maischberger, hart aber fair, Caren Miosga, Maybrit
    Illner, Tatort, GNTM, Wer wird Millionär?, Dschungelcamp…
    * Shows named after their presenter link only with a TV context word (Sendung, Talkshow,
      Gäste, Moderator, Mediathek…) or as "bei Lanz". Otherwise "Markus Lanz" is the person.
    * Show names that are ordinary words ("Tatort", "Monitor", "Panorama", "hart aber fair")
      also need TV context.
  * **296 universities** (P31/P279* Q3918 in Germany, with abbreviations); universities of
    applied sciences only with ≥ 10 sitelinks.

  Alias policy, for linking precision: full names are plain aliases. A person's surname alone
  is an **NER-only** alias (`!Reddig`), which needs an overlapping NER PER span (context
  keywords are not enough) and works only on sources of the person's own country. A surname
  gets one only if it is unique among all entities and is not a curated alias, a common
  surname (Müller, Schmidt, Wagner, Klein, Fischer…), a first name (Clemens, Paulus), a
  frequent German word (wordfreq zipf ≥ 4.0, e.g. "Wiese"), or a name that is frequent in
  English text (zipf ≥ 2.5: Thiel, Kant, Moser, Maas are mostly other people). This gives 774
  surname aliases for the 1,141 politicians. Company names:
  * Single-word names that are common words (Post, Bahn, Bild) are weak.
  * Names that are ordinary words in comments ("3000 Netto", "beim Globus") are contextual and
    need a business keyword.
  * Ordinary words like "Rational" are case-sensitive.
  * Acronyms are kept only for well-known items (≥ 30 sitelinks; no "EUV" for the Viadrina, no
    "MBS").
  * One- and two-letter aliases and very frequent words ("CL", "Deutsche") are dropped.

  Aliases shared by several generated entities stay with the most notable one, and a less
  notable namesake is skipped. A bare surname directly after a *different* known first name
  is not linked ("Hermann Merz" is not Friedrich Merz; "Olav Scholz" is still Olaf Scholz). NER
  spans resolved by exact alias lookup follow the same contextual, local and NER-only rules.
  More rules for the larger list:
  * One-word artist names link only when they are not first names or words ("Nena", "Sido";
    not "Sandra", "Haftbefehl").
  * Generated organisation names take no inflection endings, and no bare genitive "s"
    ("Hama" never matches "Hamas").
  * An NER span of the other type vetoes a generated alias ("Lanz" tagged as a person is not
    the Heinrich Lanz AG).
  * Names that are everyday phrases ("Die Welt", "Die Zeit") need media context; "WELT",
    "ZEIT" and "Zeit Online" link directly.
  * The alias regex is built as a trie, so tens of thousands of aliases stay fast.

  Collision tests: `tests/test_collisions.py`. Precision check: `scripts/eval_linking.py`.
* **Images** (`seed`): direct `upload.wikimedia.org` thumbnail URLs (250 px, the smallest
  standard width Wikimedia serves near the requested 160 px; 160 px returns HTTP 400). Persons:
  portrait (P18). Organisations, parties, clubs, media, products: logo only (P154, P8972, P2910
  app icon, P158), never P18 (often a photo of the headquarters). No logo → `imageUrl: null`. `imageKind` is
  `"logo" | "portrait" | null`.
* **Alias modes** (see the header of the YAML): plain (case-insensitive), `=` case-sensitive
  (`=UN`, `=SAP`), `~` weak (must overlap an NER span of a compatible type, or a context
  keyword is present: `~Reiche`, `~Union`, `~Bas`), `@` contextual (needs a keyword from a
  context set – football, US politics, business, politics, apps, AI, messenger, games … – in the comment or the article
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
    institution, union, broadcaster, …) and ≥ 10 sitelinks;
  * apps & services (ORG spans): instance of a product class (software, mobile app, website,
    online service, social network, streaming service, chatbot, video game, …; `PRODUCT_CLASSES`
    in `analysis/wikidata.py`) and ≥ 20 sitelinks; type `PRODUCT`.

  If the item's QID belongs to a seed entity, the seed entity is used, unless the surface is an
  alias of that entity which the gazetteer rejected ("Signal" without messenger context is not
  linked through Wikidata either). New entities get
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
  need an occupation or position, organisations an organisation class; apps and services a
  product class → type `PRODUCT`) and seeded as allowlisted entities, or marked `failed` with a
  reason. Type inference (`wikidata.entity_type`): human → `PER`, party → `PARTY`, newspaper /
  broadcaster / news agency → `ORG` (even with a website), software / mobile app / website /
  online service / video game … → `PRODUCT` (also when the item is a company too, e.g. Netflix),
  other organisation classes → `ORG`. If Wikidata rate-limits, they are retried in the
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
python -m analysis relink --days 14     # add mentions of new (generated) entities to the last 14 days
python -m analysis prune --ids-file ids.txt --add rewe   # re-check mentions of these entities after an alias change
python -m analysis prune --queue --ids meta-platforms --add facebook   # same, continued by `run` within the budget
python -m analysis reaggregate --since 2026-10-20   # recompute only pairs from that day on (cheap)
python -m analysis reaggregate          # full rebuild from all mentions (full scan, use rarely)
python scripts/eval_sentiment.py score  # sentiment evaluation (see eval/README.md)
python scripts/bench_firestore_ops.py   # reads/writes per 1000 comments

pytest -q && ruff check . && ruff format --check .
```

Re-analysing everything with a new model version: set `analyzed = false` on the comments
(their old `analysisVersion` is kept, so stale mentions are cleaned up) and run again.

## Rollout of the generated entities (23 Oct 2026)

1. `seed` wrote the 1,750 new entities (1,650 of all 2,130 have a portrait or logo).
2. `relink --days 14` scanned 52,126 comments and re-ran NER on the 8,981 that contain a new
   alias. It added 537 mentions in one pass (69,977 reads, 1,495 writes, 27 min).
3. Spot check 1 (50 random new mentions): 60 % correct. The aliases were tightened and
   `prune` re-checked the 537 mentions: it deleted 321, moved 97 "Rewe" mentions to the new
   entity `rewe`, and removed 5 entities (incl. the Bundesministerium des Innern).
4. Spot check 2: 88 % correct. A second tightening and `prune` deleted 12 more.
5. Spot check 3 (final state): **50 / 50 correct**. 301 mentions on 83 new entities. Top 10:
   Rewe 97, Lidl 33, Mercedes-AMG 29, Audi 10, Ionos 6, RTL Group 5, ADAC 4, Kaufland 4,
   Mechthilde Wittmann 4, MediaMarkt 4.

The relink and prune steps aggregated every affected (entity, day) pair. A check found
that the totals of all 91 touched entities match their mention docs, so no full
`reaggregate` was run (it would have used most of the day's remaining read budget). The
whole rollout used 4,526 writes and 96,655 reads of the analysis share on 23 Oct.

## Apps & Dienste (type `PRODUCT`)

207 curated entities in [`data/entities.yaml`](data/entities.yaml), each with a hand-checked
Wikidata id, a German description, aliases and a context set. The type string is exactly
`PRODUCT` (the dashboard label is "Apps & Dienste"); ids are stable slugs (`whatsapp`,
`x-twitter`, `ard-mediathek`), so a source that already knows its target (e.g. app-store
reviews with a `targetEntityId`) can refer to them directly.

| group | examples |
|---|---|
| messenger, social media | WhatsApp, Instagram, Facebook, TikTok, YouTube, X (Twitter), Threads, Telegram, Signal, Threema, Bluesky, Mastodon, Snapchat, Reddit, LinkedIn, Discord, Twitch, Truth Social, OnlyFans, Teams, Zoom |
| AI | ChatGPT, Gemini, Claude, Microsoft Copilot, Perplexity, DeepL, Grok, DeepSeek, Midjourney, Sora, Siri, Alexa |
| search, browsers, platforms | Google Suche, Google Maps, Gmail, Chrome, Firefox, Wikipedia, App Store, Google Play, Microsoft Office/365, Excel, Outlook, iCloud, Starlink |
| operating systems | iOS, Android, Windows, macOS, Linux, HarmonyOS |
| streaming | Netflix, Disney+, Prime Video, Amazon Prime, Joyn, RTL+, ARD Mediathek, ZDFmediathek, DAZN, Spotify, Apple Music, Audible |
| shopping, travel | eBay, Kleinanzeigen, Temu, Shein, AliExpress, Vinted, idealo, Check24, Booking.com, Airbnb, Too Good To Go, Payback, Doctolib |
| delivery, mobility | Uber, Uber Eats, Bolt, Free Now, Lieferando, Wolt, Flink, BlaBlaCar, DB Navigator |
| finance | PayPal, Klarna, N26, Trade Republic, Scalable Capital, Revolut, Bitpanda, Coinbase |
| public services | ELSTER, BundID, elektronische Patientenakte (ePA), E-Rezept, Corona-Warn-App, NINA, KATWARN, AusweisApp, EUDI-Wallet |
| dating, learning, games | Tinder, Bumble, Parship, Duolingo, Moodle, Pokémon Go, Fortnite, Minecraft, Roblox, Steam, GTA VI, EA Sports FC |

Decisions:

* The platform is the product, the company stays an organisation: `facebook` (PRODUCT) vs.
  `meta-platforms` (ORG, keeps "Meta", "Meta-Konzern", "Facebook-Konzern"), `google-suche` vs.
  `google`, `amazon-prime-video` vs. `amazon`. Zalando, Otto and Delivery Hero stay companies.
* `tiktok` moved from ORG to PRODUCT and kept its id; 16 generated companies that are mainly
  the service (Netflix, PayPal, Uber …) became curated products with their ids and QIDs.
* Hardware (iPhone, PlayStation) and tariffs (Deutschlandticket) are not products here.

Precision rules for names that are words or first names:

* **X** never links alone (aliases need ≥ 2 characters anyway). "X (Twitter)", "X (ehemals
  Twitter)", "Twitter/X", "x.com", "Plattform X", "Twitter" link directly; "auf X" / "bei X"
  only with a social-media keyword (Musk, gepostet, Follower, Plattform …).
* **Contextual** (`@`, need a keyword of the entity's context): Signal (messenger words:
  WhatsApp, verschlüsselt, Gruppenchat …), Threads (Meta, Instagram, Bluesky …), Claude and
  Gemini (KI, Chatbot, ChatGPT …), Copilot (Microsoft, KI …; "der Copilot im Flugzeug" is not
  linked), Teams, Bolt, Flink, Sora, Alexa, Safari, Edge, Kick, Hinge (Austrian dialect
  "hinge"), Elster (the bird), NINA (Nina Warken), Payback. Compounds link directly
  ("Signal-App", "Threads-App", "Google Gemini", "Anthropic Claude", "Bolt-App").
* For products the alias itself is never its own context ("Claude ist mein Nachbar" has no AI
  keyword even though "Claude" is in the AI set).
* **Weak** (`~`, NER ORG or MISC span, or context): Steam, Wolt ("Wolt ihr …" is a typo),
  Zoom, Siri, Bing, Chrome, Outlook, Kleinanzeigen.
* **Case-sensitive** (`=`): Uber (never "über"/"Über"), iOS, Windows, Excel, GTA, DAZN, Grok,
  Slack, Perplexity, AWS, Azure, WoW, PSN.
* An NER **person** span (score ≥ 0.9) over a product alias vetoes it ("Jean-Claude Juncker",
  "Usain Bolt", "Claude" as a person).
* Product names inside links or handles are not mentions ("de.wikipedia.org/wiki/…",
  "krugman.substack.com/p/the-starlink", "@spotify"); domain aliases ("x.com", "booking.com",
  "web.de") still link.
* The case-insensitive alias trie is built over lower-cased aliases, so "twitter.com" beats
  "Twitter" (before, the first case variant of a branch won).

Tests: `tests/test_products.py` (X, Signal, Threads, Claude, Gemini, Uber/über, Bolt, Steam,
Wolt, Temu, Alexa, Copilot, Elster, NINA, Teams, Kleinanzeigen, links, type inference, the
rollout jobs).

**Evaluation** (`scripts/eval_linking.py`, 480 gold pairs): precision on gold 0.975 before and
after (436 → 433 gold pairs linked; the 3 missing are re-assignments to the more specific
product: 2× "Facebook" → `facebook` instead of `meta-platforms`, 1× "Google Drive" →
`google-drive` instead of `google`); no new wrong link. All 11 product links in the gold
comments are correct. Read-only spot check on 2 × 1,000 recent production comments: 36 of 36
product links correct (two links inside URLs in the first sample led to the URL rule).

## Reads per run do not grow with the number of entities

* The registry comes from the two entity files plus one compact doc, `meta/registry`, which
  holds the Firestore-only entities (Wikidata fallback, entity requests). `seed` rebuilds it.
* Aggregation reads only the entity docs listed in `meta/active` (entities whose windows,
  relative sentiment or peers can still change) plus the docs the batch touches.
* The number of entities with mentions is a count query.
* An entity that is never mentioned costs its one seed write and a few KB of storage.

Before this change, every run read all entity docs (2,130 reads at the time, and it would have
been about 6,750 with the larger list). Now it reads 1 doc for the registry, plus the active
set and the touched entities. That is a few hundred reads, whatever the total entity count
(`tests/test_scaling.py` adds 5,000 never-mentioned entities and checks that the reads of a
run stay the same).

## Daily Firestore budget

To keep the project under about €3.70 a month, every command follows the shared
`meta/budget` contract (SCHEMA.md, [`analysis/budget.py`](analysis/budget.py)):

* **Limits** (env): `DAILY_WRITE_LIMIT=60000` and `DAILY_READ_LIMIT=200000` for all
  components combined; the analysis share is `ANALYSIS_DAILY_WRITE_LIMIT=30000` and
  `ANALYSIS_DAILY_READ_LIMIT=120000`. A reserve (`ANALYSIS_AGG_RESERVE_WRITES=2000`,
  `ANALYSIS_AGG_RESERVE_READS=10000`) is kept back for the aggregation.
* **Start:** 1 read of `meta/budget` (counters of another Berlin day count as 0). If the share
  or the combined limit is used up, the command logs a warning and exits 0.
* **During a run:** the fetch is capped to what the budget can process (about 1.5 writes and 2
  reads per comment), and before every batch the run checks the estimate against what is left
  minus the reserve. When it stops, it finishes the current batch and always aggregates what it
  processed, so the data stays consistent; unanalysed comments wait for the next run.
* **End:** the run's reads and writes go into `meta/budget` in 1 write: an atomic increment on
  the same day, or a reset for a new day (guarded by an `updateTime` precondition and retried,
  so the crawler and the API never overwrite each other).
* **One-off jobs** (`relink`, `rescore`, `reaggregate [--since]`) check the budget before each
  step (a page of comments, a batch, a group of entities). When it runs out they aggregate what
  they did and keep a cursor in `meta/analysis.jobs.<name>`. The next call continues from
  there, and so does the regular `run` once its own backlog is done. A 14-day relink can
  therefore spread over several days without manual work. `seed` and `seed-requests` are
  counted too.
* **Rollout of new entities** needs no manual Firestore work: `seed` (run by `seed.yml` on
  every push to `data/*.yaml`, or queued by itself when the budget is used up) queues
  * a `relink` of the entities it created (last 14 days). A relink that is queued but has not
    started takes the new ids into the same scan; one that is already scanning keeps its set,
    and the new ids wait in `jobs.relink_next`, which becomes the relink when it is done;
  * a `prune` of existing entities that lost an alias to another entity (here: "Facebook" moved
    from `meta-platforms` to `facebook`): their mentions are re-checked and the new owner takes
    them over (`prune --ids meta-platforms --add facebook`, but within the budget).

  `run` continues the jobs in the order seed → prune → relink → rescore → reaggregate.

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
  firestore.py   REST client (typed values, runQuery with cursor paging, batchGet, commit,
                 updateTime preconditions, increment transforms)
  textutil.py    cleaning, sentence split, slugs, searchTokens, Berlin day keys
  linker.py      registry, alias modes, gazetteer, disambiguation
  wikidata.py    Wikidata client, public-figure filter, hashed lookup cache
  ner.py         Flair / BERT / none backends
  sentiment.py   target window, sentiment models, label/score mapping
  analyze.py     comments → mentions (pure, no I/O)
  aggregate.py   day docs, deltas, rolling windows, trending (pure)
  budget.py      daily Firestore budget (meta/budget)
  pipeline.py    Firestore orchestration: run, relink, rescore, reaggregate, seed (resumable jobs)
data/            entities.yaml (curated), entities_generated.yaml (Wikidata), wikidata_cache/, events.yaml
scripts/         generate_entities.py, eval_linking.py, resolve_wikidata.py, eval_sentiment.py
tests/           pytest (fakes for Firestore and models)
```
