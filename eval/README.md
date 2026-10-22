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
