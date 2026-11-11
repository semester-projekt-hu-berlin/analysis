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
        aliases) and data/entities_generated.yaml (6,452 entities)
        plain alias  ─ always             "Merz", "Sozis", "AfD-Wähler"
        weak alias   ─ needs NER span     "Reiche" (PER) vs. "die Reiche Elite"
        context alias─ needs keywords     "Leverkusen" + Spiel/Tor/Bundesliga… (comment or article title)
        cue-only     ─ needs a cue        "Bild": "laut Bild", "der Bild", "„Bild“", "Spiegel, FAZ und Bild"
        tie-breaks                        other name words in the text ("Eintracht … Braunschweig"),
                                          context-confirmed reading beats plain ("RWE" + Spiel → Rot-Weiss Essen),
                                          source country ("Grüne"/"FCB" on derstandard.at / blick.ch)
   2. NER span not covered → exact alias lookup (titles stripped: "Kanzler Merz" → "Merz")
   3. still unknown PER/ORG → Wikidata fallback with strict public-figure filter (cached, capped)
        │
        ▼
 Target sentiment ─ window = sentence of the mention ± one sentence, the entity in [brackets]
        │           → NLI model (mDeBERTa XNLI) fine-tuned on 1,021 labelled pairs reads
        │             "Der Kommentar äußert sich positiv über {Entity}." as pos / neu / neg
        │             (1 pass; entailment = pos, neutral = neu, contradiction = neg)
        ▼
 mentions/{commentId}_{entityId}  (surface span only, never the comment text)
 comments/{id}: analyzed=true, analyzedAt, analysisVersion
        │
        ▼
 Aggregation (idempotent, once at the end of a run; `--agg-every N` adds rounds in between)
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
| Sentiment (default `ft`, sent-ft@3) | [`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) fine-tuned on 1,021 labelled (comment, entity) pairs: the mention's sentence ± one sentence with the entity in [brackets] against "Der Kommentar äußert sich positiv über {Entity}.", entailment / neutral / contradiction = pos / neu / neg, score P(pos) − P(neg). Weights: release `sentiment-ft-v3`, verified by SHA-256 (`data/sentiment_ft.json`). The previous model `cal` (sent-cal@3: 3 NLI hypotheses + news-sentiment BERT + logistic regression) stays available. | Target-aware. Macro-F1 0.56 (grouped 5-fold CV, 95 % CI 0.52–0.60) vs 0.49 for `cal`; difference +0.07 (paired CI +0.03 to +0.11) ([eval](eval/README.md)). Free, CPU, 1 model pass per window. |

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
  and loaded after the seed list (curated QIDs and aliases always win). It has 6,452 entities
  (7,043 with the curated list); groups overlap:
  * **1,451 politicians**:

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
  * **3,844 public persons** (German citizens, alive; the press list also has a few others):

    | group | count |
    |---|---|
    | athletes with ≥ 20 sitelinks (sports bots inflate sitelinks) | 1,195 |
    | current Bundesliga players and coaches, any nationality | 116 |
    | TV presenters and journalists with ≥ 5 sitelinks | 696 |
    | everyone else with ≥ 10 sitelinks (actors, musicians, entrepreneurs, scientists, YouTubers…) | 1,760 |
    | DAX/MDAX CEOs | 67 |
    | BVerfG judges, EKD and DBK heads, Bundesbank and RKI presidents | 6 |
    | people in the press, hand-checked QIDs (`PRESS_PERSONS`, see below) | 59 |

    The description comes from Wikidata ("deutscher Fernsehmoderator"), or from the role
    ("Vorstandsvorsitz SAP").
  * **790 companies**: DAX/MDAX/SDAX/TecDAX members, German companies with > 10,000 employees
    (P1128), companies and brands with ≥ 8 sitelinks, and a brand list (Aldi, Kaufland, Penny,
    Rossmann, MediaMarkt, Bertelsmann, Otto, Flixbus…). Public bodies and non-profits
    (ministries, Bundeswehr, police, churches, clinics…) are filtered out by description. Legal
    forms are stripped for short names ("Rational AG" → "Rational").
  * **68 tech and media entities**, from a hand-checked QID list (`MEDIA_TECH`):
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
  * **People in the press** (`PRESS_PERSONS`, 59 hand-checked QIDs, each P31 human and alive):
    people whose popularity swings with the news but who miss the sitelink thresholds (or lack
    a citizenship statement); the cast of "Die Höhle der Löwen" is not on Wikidata at all. Each
    gets a context set of its own in `data/entities.yaml`:
    * `investors` (Löwen, DHDL, Investor, Deal, Startup, Gründer, Beteiligung, VOX…): Frank
      Thelen, Carsten Maschmeyer, Judith Williams, Dagmar Wöhrl, Georg Kofler, Ralf Dümmel, Nils
      Glagau, Tillman Schulz, Janna Ensthaler, Tijen Onaran, Lea-Sophie Cramer, Vural Öger, Jochen
      Schweizer, Nico Rosberg, Oliver Samwer, Philipp Westermeyer; with `business`/`transit`:
      Reinhold Würth, Martin Brudermüller, Richard Lutz, Evelyn Palla;
    * `entertainment` (Show, RTL, ProSieben, Jury, DSDS, Wetten dass, Konzert, Album, Film,
      Rapper…): Dieter Bohlen, Thomas Gottschalk, Stefan Raab, Heidi Klum, Günther Jauch, Jan
      Böhmermann, Joko Winterscheidt, Barbara Schöneberger, Elton, Sylvie Meis, Daniela
      Katzenberger, Luke Mockridge, Atze Schröder, Johannes B. Kerner, Frank Rosin, Christian
      Rach, Steffen Henssler, Harald Glööckler, Oliver Pocher, Amira Aly, Robert and Carmen Geiss,
      Bushido, Kollegah, Capital Bra, Apache 207, Luciano, Katja Krasavice;
    * `creators` (YouTube, Twitch, Stream, Influencer, Instagram, TikTok, Video, Kanal…): Rezo,
      MontanaBlack, Knossi, Papaplatte, Gronkh, Julien Bam, Bianca „Bibi“ Claßen, Pamela Reif,
      Fynn Kliemann, Unge, Dagi Bee.

    Hand-written aliases replace generated ones: distinctive surnames are weak instead of
    NER-only ("~Maschmeyer", "~Böhmermann", "~Dümmel": an NER person tag or a context word, also
    on Austrian and Swiss sources); nicknames, words and first names are contextual ("@Maschi",
    "@Poptitan", "@Gottschalk" (Kay Gottschalk, AfD, shares the surname), "@Elton", "@Luciano",
    "@Unge", "@Kerner", "@Rosin"). Never alone: first names ("Joko", "Bibi", "Atze"), "Capital"
    (the magazine), "Apache", common surnames (Müller, Becker, Fischer, Schweiger, Schulz,
    Williams), "Lichter" (Horst Lichter), "Reif" (also Marcel Reif). Claus Kleber keeps "!Kleber"
    and the company Würth "=Würth" against the new namesakes; Oliver Pocher keeps "!Pocher".
  * Labels: many persons have only a `mul` label on Wikidata since 2024 (no `de`/`en` one); they
    are filled in from it (Barbara Schöneberger, Elton, Norbert Lammert, Kurt Beck, Bernd Lucke,
    Alexander Zverev, Iris Berben, Juli Zeh, Armin Papperger, phoenix, Boehringer Ingelheim…).
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
  * Single-word names that are common words (Post, Bahn) are weak.
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
    not "Sandra", "Haftbefehl"). Until Nov 2026 every one-word name counted as a first name of
    itself, so none of them linked; a few that are first names or brands elsewhere stay off
    ("Nicki", "Nana", "Fancy", "Dillon", "Hauschka").
  * Wikidata nicknames made of words ("Die Katze", "der Boss", "Die Axt", "Der Gentleman", "Herr
    Sorge") are dropped: as plain aliases they linked every cat and every boss.
  * A one-word name of a generated person inside a longer NER person span is someone else
    ("Elton John", "Luciano Pavarotti", "Frank Raab"); hyphenated pairs ("Wegner-Giffey") and
    titles ("Herr Bohlen") are fine.
  * Generated organisation names take no inflection endings, and no bare genitive "s"
    ("Hama" never matches "Hamas").
  * An NER span of the other type vetoes a generated alias ("Lanz" tagged as a person is not
    the Heinrich Lanz AG).
  * Newspapers named with words get the `press` context and a cue set (`MEDIA_CUE`, see
    *Alias modes*): "Die Welt" / "Die Zeit" link mid-sentence ("stand in Die Welt"), with press
    words or with a cue; at the start of a sentence only with a cue ("Die Welt berichtet"), since
    "Die Welt steuert auf 2,8 Grad zu" is the world. "WELT", "ZEIT", "STERN" are contextual too
    and never link in all-caps text ("IN DIESER IRREN ZEIT"); "welt.de", "WELT TV", "WELT News",
    "Zeit Online" link directly.
  * Brand names that are words, first names or surnames are contextual with a context set of
    their own (`WORD_CONTEXT`): Sinn (watches; not "macht Sinn"), Nordsee (restaurants; not the
    sea), Focus (bikes; not the magazine or "The Focus"), Müller (drugstores), Taschen
    (publishing), Sky (pay TV; "Premiere" dropped), Saturn (electronics), Emma (feminism + press
    cues), One (ARD channels; only "ONE" and "ARD One", never "Apple One"). A short form that is
    a person's surname or first name is dropped ("Lanz" is not the Heinrich Lanz AG), unless the
    company is very well known ("Bosch", weak). "BR" and "SZ" are local to German sources
    ("BR" on blick.ch is the Bundesrat, "Tuggen SZ" the canton).
  * A one-word generated name right after "The" or another entity's name is part of that name
    ("Apple One", "The Focus"); an NER person or place tag vetoes one-word organisation names in
    every mode ("Fulda", "Salzgitter" as cities); "pt-BR" is not BR.
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
  entity's own country: `^Linke` is *Die Linke* on taz.de but just "the left" on blick.ch;
  `^~BR` / `^@X` combine local with weak / contextual), `%` cue-only (`%Bild`, `%Spiegel`: only
  with a cue of the entity's cue set). Cue sets (`cues:` in the YAML, `cue: press_f`) are local
  evidence that a word is the newspaper: a word before ("laut Bild", "der ZEIT", "Quelle:
  WELT"), a word after ("Bild berichtet", "Spiegel Interview"), quotes ("„Spiegel“"), a compound
  ("ZEIT-Redakteurin", "Bild-Chef") or another medium in a list ("Spiegel, FAZ und Bild",
  "ARD/ZDF/Spiegel"). Contextual aliases of an entity with a cue set link with a cue or a context
  word, cue-only ones never with a context word alone ("ein klares Bild" next to "Artikel"). The
  NER lookup never uses contextual or cue-only aliases, and weak or NER-only ones only in their
  spelling ("Zeit" is not "ZEIT", "die reiche Elite" not "Reiche"). Longest match wins ("Siemens Energy" vs
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
  that pair (counts, `scoreSum`, `scoreSqSum`, `bySource`, top-20 `coMentions` via `mentions.coEntityIds`),
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
  mentions) from `recentDays`; only entity docs whose values change are written. `meta/trending` and `meta/stats` (`entities` = allowlisted entities
  with ≥ 1 mention, `lastAnalysisAt`) are rewritten after each run. `meta/stats.mentions` changes
  by the difference of every day doc, in the commit that writes the day doc; a count query
  (1 read per 1000 mentions) corrects it once a week and after a full `reaggregate`.
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
| `entities/{id}/days/{day}` | `scoreSqSum` | sum of squared mention scores, so the API can give the sample SD of a day or week; missing on day docs written before it existed (filled in when the pair is re-aggregated) |
| `mentions` | `coEntityIds` | other entities of the same comment, so one (entity, day) can be recomputed from its own mentions |
| `meta/trending` items | `spark` | 30 daily `avgSentiment` values (oldest first, ending on the Berlin day of `updatedAt`, `null` without mentions) |
| `meta/analysis` | `pendingPairs` | internal crash-recovery state |
| `meta/analysis` | `indexSince` | first day whose analysed comments are all in `comment_index` (relinks scan earlier days) |
| `meta/stats` | `mentionsCountedAt`, `mentionsUpdatedAt` | internal: last full count of `mentions`, last increment |
| `comment_index/{day}_{hash}` | `day`, `n`, `v`, `blob`, `createdAt` | internal, new collection: word prefixes of the comments of one batch and day (no text), for relinks; see "Cost and speed" |

These are documented in `infrastructure/SCHEMA.md`, which also lists the composite index
`mentions (entityId, day)`. Equality-only queries also work without it.

## Evaluation and limitations

[`eval/README.md`](eval/README.md) has the protocol and all numbers. In short: 480
hand-labelled (comment, entity) pairs from all major sources. Linking precision is 93.6 %.
For sentiment, 1,021 labelled pairs from 686 comments: the 455 usable pairs of the original set
plus 566 pairs labelled in November 2026 by an automated annotator (see eval/README.md).
Sentiment toward the entity reaches macro-F1 0.56 with the fine-tuned model (grouped 5-fold
cross-validation over comments, 95 % CI 0.52–0.60; per class F1 pos 0.41, neu 0.57, neg 0.69),
against 0.49 for the calibrated model before it. The mean score matches the mean of the labels
(bias −0.03); the model before (sent-cal@2) was +0.25 too positive, so absolute averages and
`baseline7d` drop by about that much after the switch.

**Limitations.** A single label is often wrong; only aggregates over many mentions are
meaningful. Neutral mentions are under-detected and often scored as negative, and sarcasm is
the main error source, along with entities that are only context (victims, comparisons). The
gold labels come from a single annotator per pair; the newer 566 from an automated one. Comment sections are negative overall, so compare entities via `relative7d`
(entity average minus `baseline7d`) rather than by absolute averages.

## Running

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt

export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json   # or FIREBASE_SERVICE_ACCOUNT='{"type": ...}'

python -m analysis seed                 # curated entities (+ Wikimedia images) and events
python -m analysis run                  # analyze unanalyzed comments (≤ 20000), update aggregates
python -m analysis run --ner flair      # Flair NER instead of BERT (pip install -r requirements-flair.txt first); --no-wikidata disables the fallback
ANALYSIS_ONNX=1 python -m analysis run  # the same models through ONNX Runtime (as in CI), about 1.4-2x faster
python -m analysis rescore              # re-run only the sentiment step, day by day (newest first), re-aggregate
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

## Rollout of the larger list (23–28 Oct 2026)

The second generator run (public persons, media, tech, TV shows) added **4,807 entities**
(6,343 generated in total, 6,934 with the curated list).

1. `seed` wrote the new entities and queued a 14-day `relink`, which the regular runs continued
   within the budget. Three of those runs (27 Oct) were cancelled by the 150-minute job timeout,
   which led to the time budget below. The relink finished on 27 Oct 22:02 UTC: **110,643
   comments scanned**, 39,761 candidates re-run, **5,785 new mentions**. The last part used
   17,245 reads and 1,700 writes.
2. `prune` of 112 entities whose aliases had changed (28 Oct, morning): 6 mentions deleted,
   3 entities removed (`deutsche-pfandbriefbank`, `michael-arndt`, `renk`), 2,025 reads and
   49 writes.
3. Hand check 1 (50 random mentions of the new entities; 210 of them had 2,893 mentions):
   **44 / 50 (88 %)**. Errors: "Sinn" (macht Sinn), "Die Welt" (the world), "Nordsee" (the sea),
   "BR" on blick.ch (the Swiss Bundesrat), "One" in "Apple One", "Focus" in "The Focus" (linked to
   the bike brand). A scan of all 1,736 mentions of 64 word-like names (Firestore texts, local
   only) showed the same problem in bulk: Sinn 64 of 64 wrong, One 40/42, Nordsee 22/22, Taschen
   9/9, BR 58 of 91 (mostly the Bundesrat), the curated `bild` (≈ 130 of 209 "ein Bild"),
   `der-spiegel` (mirror) and `die-mitte` (88 of 102 outside Swiss sources). The rules in
   *Alias modes* and *Generated list* above (cue sets, `%` cue-only, local-weak, word contexts,
   all-caps and sentence-start rules, NER lookup spelling) fix them in the generator and the
   linker.
4. `prune` of the 55 entities whose aliases changed plus 4 companies named after places
   (Fulda, Salzgitter, Deutz, Eppendorf): 1,424 mentions re-checked, **602 deleted** (bild 133,
   die-mitte 88, sinn 64, bayerischer-rundfunk 58, one 40, der-spiegel 35, sky-deutschland 32,
   die-welt 27, nordsee 22, focus 18, die-zeit 11, heinrich-lanz-ag 11 …), no entity removed,
   206 entities and 442 day docs re-aggregated; **35,630 reads, 2,243 writes**.
5. Hand check 2 (new seed, 211 new entities with 2,689 mentions): **48 / 50 (96 %)**. The two
   errors ("Cherry-picking", "BVG" for the Bundesverwaltungsgericht) were fixed (lower-case
   compounds, BVG contextual with a feminine cue: "die BVG" vs. "das BVG") and pruned: 10 more
   mentions deleted, 2,241 reads, 31 writes.

Reads: a prune costs about 2,000 reads of fixed overhead (active entity docs, rolling windows)
plus about 3 reads per re-checked mention and the re-aggregation of the affected days; a hand
check about 3,000 reads. The regular run of 28 Oct 05:33 UTC analysed 3,869 comments with
19,673 reads and 7,887 writes (with 150 Wikidata lookups and the aggregation); the entity count
does not enter it (see below).

## Rollout of the people in the press (9 Nov 2026)

The generator run with `PRESS_PERSONS`, the `mul` labels and the alias fixes above added **110
generated entities** (47 of the 59 people in the press; 63 others: persons, companies and the
channel phoenix that only had a `mul` label, and one-word artist names such as Campino that had
no usable alias) and changed the aliases of 61 existing ones (6,452 generated, 7,043 with the
curated list; `volker-beck` is now the Green politician, Q57322, instead of a namesake). The
push to `data/entities.yaml` starts `seed.yml`, which writes the new entity docs and the changed
aliases (about 175 writes) and
queues two jobs that the regular runs continue within the daily budget:

1. a 14-day `relink` of the created entities: it scans every analysed comment of the last 14
   days once (about 1.3 reads per comment, i.e. roughly 140,000 reads for ~110,000 comments,
   spread over one or two days of the 120,000-read analysis share) and re-runs NER only on the
   few thousand comments that contain one of the new aliases; writes are the new mentions plus
   their day docs (expected in the hundreds);
2. a `prune` of the 23 entities that lost an alias ("Die Katze", "der Boss", "Die Axt",
   "!Lichter", "!Zverev", "!Kofler", "!Lucke" …): about 2,000 reads of fixed overhead plus about
   3 reads per re-checked mention. `seed` now queues this for every lost alias, not only for
   aliases that moved to another entity.

New aliases of existing entities ("@Gottschalk", "@Poptitan", "Kollegah", "!Sido") apply to new
comments only; they are not relinked.

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

## Cost and speed (10 Nov 2026)

Measured first (run logs of 5–9 Nov, a dry run of 1,500 production comments against the
in-memory Firestore, model benchmarks on 4 CPUs), then changed, then measured again.

**Where a run's Firestore operations went** (runs of 2,800–5,000 comments: 13–22k reads, 7–10k
writes, i.e. 3.5–5 reads and 2.0–2.3 writes per comment):

| reads | share | what |
|---|---|---|
| unanalysed comments | 1 per comment | the fetch (a field mask does not make it cheaper: Firestore bills per document) |
| entity docs of `meta/active` | ~2,220 per run, **and again for every job the run continued** | windows, baseline, peers, trending |
| mentions of the touched (entity, day) pairs | ~1.1 per mention **per aggregation round** | day docs are recomputed from all mentions of the pair |
| day docs + `community_text` docs | 2 per pair per round | the second one almost always missing |
| count of `mentions` | 111 per aggregation (1 per 1000 mentions, growing) | `meta/stats.mentions`; the budget counted it as 1 |
| article titles | ~30 per batch of 250 | context for contextual aliases |
| relink of new entities | 1 per comment of the last 14 days (~140,000) | 3 % contain an alias, 0.06 % get a mention |

| writes | share | what |
|---|---|---|
| `analyzed` flag | 1 per comment (45–50 %) | |
| mentions | ~0.6 per comment (27 %) | |
| day docs + entity docs | ~800 per aggregation round (25 % with 2–3 rounds) | entity docs also for `relative7d` and peer ranks, which move with every round |

**What changed**

| change | effect |
|---|---|
| one aggregation at the end of a run instead of every 2000 comments | mentions, day docs and entity docs of a pair are read and written once per run |
| `community_text` docs are read only where the entity's `communityRecentDays` (or a community mention) says one can exist | −1 read per pair |
| the jobs a run continues (prune, relink, …) use the entity docs the run holds | −2,220 reads per continued job |
| `meta/stats.mentions` by increments, counted weekly | −111 reads per aggregation today, and no growth with the number of mentions |
| article titles cached for the run | fewer article reads when an article spans batches |
| a run with nothing to do (no comments, no pending pairs, windows already moved today) | 5 reads instead of ~2,350 |
| comment index for relinks (below) | a 14-day relink: ~140,000 → about 5,000–8,000 reads once the index covers the days |
| NER batches sorted by sentence length | NER 1.6–2.1× faster, identical spans |
| models through ONNX Runtime (`ANALYSIS_ONNX=1`, set in `analyze.yml`) | NER another 1.1×, sentiment 1.4× faster, identical labels |
| Flair is installed only for `--ner flair` | ~30 s less setup per run, ~60 fewer packages |

Dry run, the same 1,500 comments (1,132 mentions) before and after: 5,954 → 5,216 reads,
3,846 → 3,413 writes (one intermediate round before, none after; the fixed 2,224 entity reads
weigh more in this small run than in a real one). Every run now logs its reads and writes per
collection (`firestoreOps` in the run summary), so the shares above can be read off each run.

**Comment index** ([`analysis/textindex.py`](analysis/textindex.py)). `run` writes one small
document per batch and day, `comment_index/{day}_{hash}`: for every comment of the batch the
first 8 characters of each of its words, normalised (about 80–120 bytes per comment, no text,
no order). It is written before the comments are marked as analysed. A relink reads these
documents, selects the comments whose words can contain an alias of a new entity, reads only
those and applies the exact alias pattern as before. Nothing is missed: wherever the pattern
matches, the index selects the comment (argument in the module, tested with random texts and
against all case-insensitive character pairs of Python's `re`). On production comments and
the aliases of 109 new entities the index selects 2.3 % / 4.4 % of the comments, the exact
pattern 2.1 % / 4.4 %. `meta/analysis.indexSince` is the first day the index covers
completely (two days after the first run that writes it); a relink scans earlier days as
before, so the index takes over within 14 days. `ANALYSIS_TEXT_INDEX=0` switches it off (the
next run removes `indexSince`, relinks scan again). A failed index write never fails a run.
Documents older than 45 days are deleted.

**Evaluated and rejected: int8 quantization.** Dynamic quantization makes the NLI model 1.5×
(PyTorch) to 2.1× (ONNX Runtime) and the news classifier 2.2× faster, but the outputs are no
longer the same model: the NLI logits move by 1.3 on average (argmax equal for 39 % of the
inputs), the classifier's label agrees for 47 %. Not used.

**Not changed: the `analyzed` flag** (1 write per comment, about 8,000 a day). A cursor on
`crawledAt` would save it, but the crawler sets `crawledAt` before it fetches a thread or a
stream and commits the comments afterwards, so a cursor can move past comments that are not
stored yet and would never be analysed. An exact design needs the crawler: with every batch
of comments it inserts it writes one small inbox document with their ids; the analysis reads
the inbox, fetches those comments and deletes each inbox document once its comments are
analysed. That is not built. At today's volume it would save about €0.10 a month.

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
* **Rescore after a model change.** `run` records the model that scores new mentions in
  `meta/analysis.sentiment`; when it differs from the recorded one (first time: `sent-cal@2`), it
  queues a `rescore`. The job works day by day, newest first: it reads the mentions of one day
  (not all mentions, as before), rewrites label / score / confidence where they change,
  re-aggregates that day and moves to the next older day. An unfinished rescore therefore leaves
  every day on one model except the day it stopped in, and the 7-day windows are converted
  first. Continued by a regular run it leaves 1.3 × that run's usage for each later run of the
  day (`ANALYSIS_RUN_INTERVAL_H`, default 12). The comments' `analysisVersion` is not rewritten
  (one write per comment saved). About 2 reads and 1.2 writes per mention; with the regular runs
  taking about 16k of the 30k daily writes, roughly 10k mentions per day (`tests/test_rescore.py`).

## Time budget per run

GitHub cancels the analysis job after `timeout-minutes: 150`. On 27 Oct three runs (05:33,
10:46, 17:58 UTC) were cancelled while they continued the 14-day relink; the cursor and
`pendingPairs` kept the progress, but each lost its last page and its budget record. Now every
run has a soft deadline ([`analysis/timebudget.py`](analysis/timebudget.py)):

* `ANALYSIS_TIME_BUDGET_MIN` (default 135) counts from `ANALYSIS_JOB_START`, which the first
  step of `analyze.yml` sets, so the setup steps (pip, model cache) count too; without it the
  clock starts with the process. `0` disables the limit.
* The main loop checks it before each batch, `relink` and `rescore` before each page or batch
  (keeping the duration of the last step in reserve), `reaggregate [--since]` before each group,
  `continue_jobs` before each job (`"waiting for time"`) and `prune` before it starts.
* When it is reached the run logs `time budget reached (… of 135 min used …)`, saves the job
  cursor, aggregates what it processed (no `pendingPairs` are left), records its reads and
  writes in `meta/budget` and exits 0 with `stopped: "time budget reached after N comments"`
  and `timeBudget` in its summary. The next run continues.

Tests: `tests/test_timebudget.py`.

## Schedule (GitHub Actions)

| workflow | trigger | what |
|---|---|---|
| `analyze.yml` | twice a day at 05:30 and 17:30 UTC (`30 5,17 * * *`), about an hour after the crawler (`17 4,16 * * *`), + manual | `python -m analysis run` over the half-day backlog, newest first (about 4k comments; `--limit 9000`, timeout 150 min with a soft time budget of 135 min, one aggregation at the end); CPU-only torch, ONNX Runtime (`ANALYSIS_ONNX=1`), pip cache, model cache (Hugging Face weights and their ONNX exports), Wikidata lookup cache; `concurrency: analysis` prevents overlapping runs |
| `seed.yml` | push to `data/*.yaml` + manual | `python -m analysis seed` |
| `ci.yml` | push / PR | ruff + pytest (fake models, no torch needed) |

The workflows need the organisation secret `FIREBASE_SERVICE_ACCOUNT`.

**Triggering.** GitHub's `schedule` started runs 4–6 h late (or not for hours), so `analyze.yml` is started by
Cloud Scheduler (job `meinungspuls-analyze` in the GCP project `meinungspuls-hu`, 05:30 / 17:30 UTC), which calls the
API (`GET /internal/cron/dispatch/analyze`, see `deploy/setup-gcp.sh` in the `api` repo); the API sends a
`workflow_dispatch` with the input `trigger=vercel-cron` (the value is historical, from when the cron ran on Vercel). The `schedule` stays as a fallback: the `gate` job
(`actions: read`) skips a scheduled run when a successful or still queued/running analysis was created since one
hour before its slot, and drops duplicate cron deliveries (another run within 3 h). Manual runs are never skipped.

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
  onnx_backend.py  optional ONNX Runtime inference, checked against PyTorch at every load
  textindex.py   comment index: word prefixes per comment, so relinks read only candidates
  analyze.py     comments → mentions (pure, no I/O)
  aggregate.py   day docs, deltas, rolling windows, trending (pure)
  budget.py      daily Firestore budget (meta/budget)
  pipeline.py    Firestore orchestration: run, relink, rescore, reaggregate, seed (resumable jobs)
data/            entities.yaml (curated), entities_generated.yaml (Wikidata), wikidata_cache/, events.yaml
scripts/         generate_entities.py, eval_linking.py, resolve_wikidata.py, eval_sentiment.py
tests/           pytest (fakes for Firestore and models)
```
