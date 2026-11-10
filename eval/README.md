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

## Clean protocol and candidates for a better model (10 Nov 2026)

**Protocol** (`scripts/eval_protocol.py`, `scripts/eval_sentiment_cv.py`). The 455 usable gold
pairs come from 363 comments; in the old hash split 45 comments have pairs on both sides (47 of
the 215 test pairs), and the settings of `cal` were chosen on the same pairs. All numbers below
therefore use: 5 folds stratified by class **over comments** (a comment's pairs are never
split), everything fitted inside the training folds, a 95 % cluster bootstrap over comments,
and a paired bootstrap on the same folds to compare two models. The pair that can no longer be
built is `d5d4a69e83249d2e1e3f_meta-platforms`: the comment says "Facebook", and that alias
moved to the product entity.

| baseline `cal` (sent-cal@2) | macro-F1 | 95 % CI | accuracy |
|---|---|---|---|
| old protocol: fit on dev (240), scored on test (215) | 0.474 | 0.394 to 0.550 | |
| shipped calibration scored on pairs it was trained on (all / test split) | 0.502 / 0.495 | (in-sample) | |
| **grouped 5-fold CV, mean of 10 fold seeds** | **0.482** (0.476 to 0.492 over seeds) | 0.43 to 0.53 | 0.505 |

Per class F1 0.42 (pos) / 0.46 (neu) / 0.56 (neg). The accuracy equals that of always answering
"negative" (0.512). The mean score per entity (29 entities with at least 5 pairs) correlates
with the gold mean at r = 0.65, but all scores are shifted by +0.23: the class weights of the
calibration favour pos and neu, so `score` is too positive (mean -0.15, gold -0.38). Fitting
without class weights removes the shift but costs labels (macro-F1 0.40 on the larger set
below); sent-cal@3 keeps the weighted label and takes the weights out of the score only.

**Labels are not the ceiling.** A second, blind annotation of 100 random pairs by the automated
annotator of the next section (`relabel2.csv`) agrees with the gold labels on 91 (kappa 0.85,
macro-F1 0.91).

**Where `cal` fails** (224 of 455 out-of-fold errors, 70 read by hand): criticism without
sentiment words that needs world knowledge or the rest of the comment (neg predicted neu: 88),
entities that are only context, victim or comparison in an angry comment (neu predicted neg:
62), support expressed by defending or by attacking the other side (pos missed), and windows
cut too narrow. Macro-F1 is 0.39 when another entity is in the same sentence (0.52 otherwise),
0.34 on taz.de (long comments, many entities), 0.43 with a question mark, 0.39 with quotes.

**Candidates** (same folds; difference to `cal` with its paired 95 % CI; CPU seconds per 1,000
mentions on 4 threads, PyTorch; * = 2 threads on a shared machine):

| candidate | macro-F1 | difference to `cal` | s / 1,000 |
|---|---|---|---|
| `cal` (3 NLI hypotheses + news classifier, logistic regression) | 0.482 | – | 339 |
| surface form instead of the entity name in the hypotheses | 0.487 | +0.004 (-0.035 to +0.044) | 339 |
| + hypothesis "neutral über" / "Der Autor kritisiert" | 0.498 / 0.499 | +0.016 / +0.017 (n.s.) | 410 |
| "Der Autor äußert sich …" templates | 0.451 | -0.031 (n.s.) | 339 |
| + Guhr sentiment classifier (the "+0.04" of the first table) | 0.486 | +0.003 (n.s.) | 450 |
| + probe on the NLI model's [CLS] vector | 0.501 | +0.019 (-0.023 to +0.060) | 339 |
| Qwen2.5-3B-Instruct (Q4, llama.cpp), 5 shots, 150 pairs | 0.479 (0.38 to 0.57) | +0.026 (-0.08 to +0.13) | 6,200 * |
| MiniLM-L6 NLI model fine-tuned on the pairs, 1 pass | 0.485 | n.s. (accuracy 0.56) | about 40 |
| LoRA (r 16) or only the top 2 layers of the NLI model | – | too slow to converge on CPU (loss 0.9 after 4 epochs) | |
| **NLI model fully fine-tuned on the pairs (`ft`), 1 pass** | see below | | below `cal` (1 pass instead of 4) |

Nothing that keeps the frozen models gets beyond the noise: better templates, more
hypotheses, a second classifier and a small generative model all stay within +-0.02. The
stance needs a model that is trained on the task.

**Fine-tuned model `ft`** (`scripts/train_sentiment_finetune.py`, `analysis/sentiment.py`
`FineTunedSentiment`, not the default): the NLI model reads the mention's sentence +- one
sentence with the target in [brackets] against "Der Kommentar äußert sich positiv über {Name}.";
its head (entailment / neutral / contradiction) is trained to mean pos / neu / neg. Settings
fixed before the run: all weights, 4 epochs, learning rate 3e-5, batch 16, class weights^0.5.
One fold takes about 14 minutes on 4 CPUs. Result so far: two of the five folds (182 pairs; the run was then killed for lack of memory
on the shared machine, `cv --resume` continues it):

| on the same 182 pairs | macro-F1 | 95 % CI | F1 pos / neu / neg | accuracy | r of score with gold |
|---|---|---|---|---|---|
| `cal` | 0.497 | 0.413 to 0.577 | 0.38 / 0.49 / 0.62 | 0.538 | 0.29 |
| `ft` | 0.548 | 0.454 to 0.628 | 0.47 / 0.49 / 0.69 | 0.593 | 0.43 |

Difference +0.051 (paired 95 % CI -0.048 to +0.147, 16 % of the draws at or below 0): promising,
not shown. The mean score of `ft` (-0.43) is close to the gold mean (-0.39). `ft` is therefore
**not the default**. It ships only if all five folds show at least +0.03 with a paired interval
above 0 and no class losing more than 0.03 (`scripts/eval_sentiment_cv.py` prints exactly
that); its weights (about 560 MB in half precision) then become assets of a GitHub release,
verified by hash (`analysis/finetune.py`). With 455 pairs a true gain of +0.05 is detected in
about half of the cases; about 1,000 gold pairs (roughly 130 pos, 360 neu, 510 neg) would make
+0.03 to +0.05 reliably visible.

## Larger gold set and sent-cal@3 (11 Nov 2026)

**566 new labels.** The new labels come from an **automated annotator** (an AI agent working in
this repository), not from a person, with the guidelines above (stance toward the entity,
sarcasm by its meaning; `linkOk` 0 for wrong links). It labelled blind: no model prediction was
read or shown (the sample query does not read the `label` field of the mentions).

* `labels_extra.csv`: 244 further pairs of the gold comments (other entities in them; 243 usable).
* `labels_extra2.csv`: 335 pairs of 335 new production comments (12 days, 22 Oct to 9 Nov), at
  most one pair per comment, at most 22 % per source, 4 per entity and 38 % per entity type, so
  the newer sources (Bluesky, apollo, computerbase, rbb24, tagesschau …) and products are covered;
  12 are wrong links, 323 usable. Sampling cost 1,334 reads, recorded in `meta/budget`.
* Consistency: a second blind pass over 57 random new pairs (10 %, `relabel3.csv`) agreed on 53
  (93 %, kappa 0.88, macro-F1 0.91). One of the four differences was a slip in the first pass
  (Microsoft, clearly negative, typed as positive) and was corrected; the other three are
  genuine borderline cases and keep the first label. Together with `relabel2.csv` (91 / 100,
  kappa 0.85, against the original labels) this measures one annotator's consistency, not the
  agreement of independent annotators.

All files together: **1,021 pairs from 686 comments, pos 109 / neu 387 / neg 525.**

**Baseline on the larger set** (same protocol, `scripts/eval_sentiment_cv.py`): macro-F1
**0.493** (10 fold seeds 0.487 to 0.500; seed 0: 0.487, 95 % CI 0.449 to 0.522), accuracy 0.529,
F1 pos 0.40 / neu 0.50 / neg 0.59. The interval is now about +-0.04 instead of +-0.05. The shipped
sent-cal@2, scored on the 566 new pairs it never saw, reaches 0.481 (0.437 to 0.526), mean score
-0.12 against a gold mean of -0.43.

**sent-cal@3: unbiased score.** The calibration is refitted on all 1,021 pairs with the same
settings; the label is the strongest of the weighted logits as before, and the score and
confidence come from the same logits with the class weights and the neutral bias subtracted
(`scoreCorrection` = log class weight + bias; a model fitted with class weight w_k estimates
probabilities proportional to w_k p_k). Cross-validated, 10 fold seeds:

| | sent-cal@2 score | **sent-cal@3 score** |
|---|---|---|
| macro-F1 (label) | 0.493 | 0.493 (same label) |
| bias of the mean score against the mean gold polarity | +0.248 | **-0.007** |
| r per mention | 0.294 | 0.308 |
| entity means, 54 entities with >= 5 pairs: r / MAE | 0.653 / 0.317 | 0.669 / **0.263** |
| source means, 11 sources with >= 10 pairs: r / MAE | 0.725 / 0.252 | 0.713 / **0.104** |
| entity type means: r / MAE | 0.838 / 0.246 | 0.843 / **0.077** |

The change of the source-level r (-0.012) is within its noise (bootstrap over comments of the
difference: -0.13 to +0.10; entity level -0.08 to +0.07). The absolute averages drop by about
0.25, including `baseline7d`; `relative7d` (entity minus baseline) changes much less. Existing
mentions are rescored by the `rescore` job, newest day first (README).

## Limitations

* The best free CPU method reaches only **≈ 0.49 macro-F1** on stance toward the entity. Single labels are often wrong. The aggregates over many mentions (daily
  counts, 7-day averages, the relative sentiment against the baseline) are much more reliable
  than any single label (since sent-cal@3 the mean score is unbiased on the gold set). The model under-detects neutral mentions
  (recall 0.42) and confuses them with negative ones, especially when the entity is a victim or
  a comparison.
* Sarcasm, rhetorical questions and implicit criticism are common in comment sections and are
  the main source of errors.
* Each gold pair has one label from one annotator; 566 of the 1,021 come from an automated annotator. Sources outside derstandard, blick, taz and Bluesky have fewer than 60 pairs each.
* Comment sections are negative on average, so absolute averages are mostly below zero. The
  dashboard should compare entities with `relative7d` (entity average minus the overall
  `baseline7d`).
