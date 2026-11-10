# Sentiment evaluation

## Data

* `labels.csv`: 480 real (comment, entity) pairs from production `mentions`, labelled by hand
  with the stance **toward that entity** (`pos` / `neu` / `neg`), not the overall mood of the
  comment. Sarcasm is labelled by its intended meaning ("Gute Kontakte zur ÖVP reichen
  vollauf" → neg). 60 pairs come from a first sample, and 330 from a second sample stratified by
  source (derstandard 131, blick 110, taz 89). The second sample has at most two pairs per
  comment, so it includes comments that mention several entities with different stances. A
  third sample of 90 pairs adds the new sources (tagesschau 15, Tichys Einblick 15, Bluesky 15,
  Mastodon 10, Freitag 5) and 30 "context" cases: mentions labelled negative in comments with
  several entities, where the entity is often only a victim, a comparison or a place.
* The file stores only `mentionId` (= `commentId_entityId`), source, gold label and `linkOk`
  (0 = the entity was linked wrongly, e.g. "Mitte" as a generic word or "BVB" as the Basel
  transport company). It never stores comment text. Texts are fetched from Firestore at
  evaluation time and cached in git-ignored `*.local.*` files.
* **Linking precision** on the 330 stratified pairs: 309 / 330 = 93.6 %. The errors were fixed
  in `data/entities.yaml` (local `^` aliases, removed generic aliases).
* For sentiment, the 456 correctly linked pairs are used: 233 neg (51 %), 164 neu (36 %),
  59 pos (13 %). They are split deterministically by hash into dev (241) and test (215). The
  candidate table below was computed on the first 369 pairs (dev 197 / test 172).
* **Consistency**: 50 random pairs were labelled a second time without looking at the first
  labels (`relabel.csv`). Agreement was 50/50 (κ = 1.0). Both passes were done by the same
  annotator in the same session, so this measures self-consistency, not inter-annotator
  agreement. A second annotator would be needed for that.

## Candidates (`scripts/eval_target_sentiment.py`)

Thresholds are tuned on dev only; the table shows test scores.

| candidate | passes / mention | dev macro-F1 | **test macro-F1** | test acc |
|---|---|---|---|---|
| a  previous default: ensemble news + guhr (raw), clause window, threshold 0.2 | 2 | 0.405 | 0.383 | 0.459 |
| b  zero-shot NLI (mDeBERTa-v3 XNLI), clause window | 3 | 0.466 | 0.465 | 0.494 |
| b  zero-shot NLI, entity window (±1 sentence, other entities masked) | 3 | 0.525 | 0.413 | 0.465 |
| b  zero-shot NLI MiniLM (small), entity window | 3 | 0.396 | 0.422 | 0.488 |
| c  ensemble, entity window, others masked | 2 | 0.443 | 0.394 | 0.494 |
| d  NLI + ensemble, averaged probabilities | 5 | 0.47–0.50 | 0.38–0.46 | ≤ 0.50 |
| **calibrated (default `cal`)**: NLI logits for "positiv über {name}" / "negativ über {name}" + news classifier, multinomial logistic regression | **3** | – | **0.476** | 0.494 |
| same features + Guhr classifier (over the 3-pass budget) | 5 | – | 0.517 | 0.552 |

`cal` was trained with `scripts/train_sentiment_calibration.py`: L2 = 0.1, class weights =
balanced^0.75 (chosen so that the predicted class rates match the gold rates: 52 / 29 / 19 %
predicted vs. 49 / 36 / 14 % gold). Its 5-fold cross-validated macro-F1 (3 repetitions, all
369 pairs) is **0.466**. Per class on the test split:

| | precision | recall | F1 |
|---|---|---|---|
| pos | 0.52 | 0.42 | 0.47 |
| neu | 0.45 | 0.37 | 0.40 |
| neg | 0.51 | 0.61 | 0.56 |

A higher-recall operating point (additive pos/neg logit biases, tuned on dev) did not help on
test (0.453), so no bias is applied.

Throughput of `cal` on this 4-CPU machine is about 4.8 mentions/s. At 2,500 mentions a day
that is about 9 minutes of CPU.

## Version 2: target vs. context (current default, `sent-cal@2`)

A review of live negative mentions found a consistent error: the comment is negative, but the
entity is only context ("russische Hacker greifen Parteien wie die CDU an", "Trump will regieren
wie Putin"). Version 2 adds a fourth model pass, the NLI hypothesis "Der Kommentar bewertet
{Name}." (is the entity judged at all?), and a small neutral logit bias (+0.1, chosen by 5-fold
CV inside the dev split). Both models were trained on the dev split and evaluated on the same
215-pair test split of the extended set:

| | v1 (3 passes) | **v2 (4 passes)** |
|---|---|---|
| 5-fold CV macro-F1 (456 pairs) | 0.460 | **0.485** |
| test macro-F1 | 0.469 | **0.483** |
| pos P / R / F1 | 0.46 / 0.40 / 0.43 | 0.46 / 0.37 / 0.41 |
| neu P / R / F1 | 0.48 / 0.31 / 0.38 | 0.47 / **0.42** / 0.44 |
| neg P / R / F1 | 0.53 / 0.69 / 0.60 | **0.57** / 0.64 / 0.60 |

Neutral recall and negative precision improve; positive recall drops slightly. Two further
variants did not hold up in cross-validation: a "wird nur nebenbei erwähnt" hypothesis
(5 passes, CV 0.474) and fully balanced class weights (CV 0.480 or lower). Of the five context
examples from the review, v2 now labels two as neutral (EU as background, the Greens in a list).
It still labels the CDU-as-victim and Putin-as-comparison cases negative.

Throughput with 2 threads: 1.7 mentions/s for sentiment and about 7.5 comments/s for NER. 9,000
comments (about 2,000 mentions) therefore take about 40–45 minutes, well within the 150-minute
CI window.

## Linking with the generated entities (Wikidata)

`python scripts/eval_linking.py` re-links the texts of all 480 gold pairs (BERT NER, without
article titles and without the Wikidata fallback) with the seed-only registry and with
seed + `data/entities_generated.yaml`:

| registry | gold pairs found | of them correct | precision on gold pairs | correct pairs lost |
|---|---|---|---|---|
| seed only | 436 / 480 | 425 | 97.5 % | 31 |
| seed + generated (1,745) | 436 / 480 | 425 | 97.5 % | 31 |

Both lose the same 31 pairs, because this offline check has no article titles (contextual
aliases such as club names) and no Wikidata fallback. The generated list loses no gold pair.
(Before the NER-lookup rule below, both found 439 pairs with 425 correct, 96.8 %.) On these
comments the generated list adds 10 new matches, and all 10 are correct (Reddig, Eralp,
Giffey, Curio, Scheuer, Wittmann, Lidl, Edeka, Kaufland, Rewe).

**Production spot checks** (random samples of 50 new mentions after the 14-day relink,
judged by hand against the comment):

| round | correct | typical errors | fix |
|---|---|---|---|
| 1: first list | 30 / 50 (60 %) | "Thiel" = Peter Thiel, "Kant", "Paulus", "Clemens", "Moser"(-Roth); "3000 Netto"; Austrian "Innenministerium"/"BMI" on derstandard.at; "EUV", "MBS" | NER-only surnames only on sources of the person's country, never first names or names frequent in English; ambiguous company words contextual; acronyms only for well-known items; public bodies removed |
| 2: after `prune` | 44 / 50 (88 %) | "Binz" (the town), "Basler Schulen", "Aida", "Netto vom Brutto", "Norma" in "Norman", "smart" in "smarten" | inflection suffixes only for curated names; Basler/Aida/smart/Norma contextual, "Netto" dropped |

The fixes of round 2 keep all 44 correct matches of that sample and remove 5 of the 6
errors (offline re-check). "Binz" is still wrong: the NER model tags the town as a person. NER
spans resolved by exact alias lookup now follow the contextual, local and NER-only rules too.
That also removed three wrong seed links among the gold pairs, which is why precision rose
from 96.8 % to 97.5 %. Round 3 (the final state in production) is in the rollout notes of
the analysis README.

### Names that are words (larger list, 28 Oct 2026)

Production hand checks of the 4,807 entities of the larger list (50 random new mentions each):

| round | correct | errors | fix |
|---|---|---|---|
| 1: after the 14-day relink (210 entities, 2,893 mentions) | 44 / 50 (88 %) | "Sinn" (macht Sinn), "Die Welt" (the world), "Nordsee" (the sea), "BR" on blick.ch (Bundesrat), "One" in "Apple One", "Focus" in "The Focus" | cue sets and cue-only aliases, local-weak "^~BR", word contexts for brand names, sentence-start rule for "Die Welt", all-caps rule, NER lookup keeps the spelling; `prune` deleted 602 mentions |
| 2: after that prune (211 entities, 2,689 mentions), new seed | 48 / 50 (96 %) | "Cherry-picking", "BVG" = Bundesverwaltungsgericht | lower-case compounds, "BVG" contextual with "die BVG" cue; 10 more deleted |

Before the prune, a local re-link of all 1,736 Firestore mentions of 64 word-like names (texts
are only cached locally) showed the size of the problem; the new rules keep 1,135 of them and
drop 601 (Sinn 64/64, One 40/42, Nordsee 22/22, Bild 133/209, Die Mitte 88/102 (every one
outside a Swiss source), BR 58/91, Spiegel 35/87, Die Welt 27/47, Die Zeit 11/166, Sky 32/39). A review of the
kept and dropped mentions per entity found few wrong drops (a few lower-case or quoted "Zeit",
"Emma war eines der wenigen Medien", "Der Stern hat nachgefragt"); the brand names in clear
use (Lidl, Rewe, Audi, Edeka, Kaufland, MediaMarkt, ADAC, Opel, WDR, NDR, dpa, t-online) lose
nothing.

`scripts/eval_linking.py` on the 480 gold pairs (seed + generated):

| state | gold pairs found | of them correct | precision on gold pairs | correct pairs lost |
|---|---|---|---|---|
| before (27 Oct) | 433 / 480 | 422 | 97.5 % | 34 |
| **after** | 428 / 480 | 422 | **98.6 %** | 34 |

The five links that disappeared were all wrong (four "die Mitte" on taz.de and tagesschau.de,
one "Bild" for a picture); no correct pair was lost. Outside the gold pairs the new registry
drops "Söder, schau in den Spiegel!" and "ein düsteres Bild" and adds "Quelle: Die Zeit vom
15.10.26". Tests: `tests/test_word_names.py`.

## Faster inference, same results (10 Nov 2026)

The NER model and the two sentiment models can run through ONNX Runtime
(`ANALYSIS_ONNX=1`, [`analysis/onnx_backend.py`](../analysis/onnx_backend.py)), and NER
batches are sorted by sentence length. Both change the speed only. Measured on 4 CPUs
(PyTorch 2.14 / 2.15, ONNX Runtime 1.31), each variant on the same inputs:

| | before | sorted batches | + ONNX Runtime |
|---|---|---|---|
| NER, 3,243 sentences of production comments (social-heavy) | 23.5 /s | 37.1 /s | 41.9 /s |
| NER, 1,436 sentences of the gold comments (news) | 21.3 /s | 44.9 /s | 47.5 /s |
| NER spans equal to before (4,679 sentences) | – | all | all (largest score difference 4e-6) |
| `scripts/eval_linking.py` (seed + generated) | 428 / 480 linked, 422 correct, 98.6 % | the same | the same |
| sentiment `cal`, 455 gold windows | 2.85 /s | – | 4.07 /s |
| sentiment features / labels equal to PyTorch | – | – | largest difference 2e-5 / 455 of 455 labels |
| macro-F1, fit on dev, scored on test | 0.474 | – | 0.474 |
| macro-F1 of the shipped calibration on the test split | 0.495 | – | 0.495 |

The sentiment rows use the production code path (`CalibratedSentiment.features`) and today's
entity lists: the windows of 455 of the 456 pairs can still be built (the lists have grown
since the calibration was trained, so one target is no longer located and some windows are
narrower). That is why the dev → test number is 0.474 here and not the 0.483 of the table
above; it is the same with PyTorch and with ONNX Runtime, before and after the change.

At every model load the ONNX session is compared with the PyTorch model on probe sentences;
if a logit differs by more than 0.002 (measured: 7e-6), or the export or the session fails,
PyTorch is used.

**int8 quantization was rejected.** Dynamic quantization (PyTorch, engines x86 / fbgemm /
onednn; ONNX Runtime, with and without per-channel weights) is 1.5–2.2× faster but changes the
models: NLI logits differ by 1.3 on average and the strongest class is the same for only
39 % of the inputs; the news classifier keeps its label for 47 % (88 % with qnnpack, which is
ten times slower). With features this far off, the calibration would have to be retrained
and re-validated on quantized features; that is a new model, not a faster one.

## Limitations

* The best free CPU method reaches only **≈ 0.48 macro-F1** on stance toward the entity. Single labels are often wrong. The aggregates over many mentions (daily
  counts, 7-day averages, the relative sentiment against the baseline) are much more reliable
  than any single label, but they are still biased. The model under-detects neutral mentions
  (recall 0.42) and confuses them with negative ones, especially when the entity is a victim or
  a comparison.
* Sarcasm, rhetorical questions and implicit criticism are common in comment sections and are
  the main source of errors.
* The gold labels come from one annotator. The newer sources are covered by only 60 pairs.
* Comment sections are negative on average, so absolute averages are mostly below zero. The
  dashboard should compare entities with `relative7d` (entity average minus the overall
  `baseline7d`).
