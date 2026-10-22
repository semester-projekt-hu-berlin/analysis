# Sentiment evaluation

## Protocol

60 (comment, entity) pairs were drawn at random from the real `mentions` in Firestore on
22 Oct 2026 (sources: derstandard.at, blick.ch, taz.de; at most one pair per comment, at most
four per entity). I labelled each pair by hand with the sentiment the comment expresses
**toward that entity** (`pos` / `neu` / `neg`). Sarcasm was labelled by its intended meaning
("Gute Kontakte zur ÖVP reichen vollauf" → neg). `labels.csv` stores only
`mentionId` (= `commentId_entityId`) and the gold label, never comment text.

```bash
python scripts/eval_sentiment.py sample --n 60   # writes eval/sample.local.jsonl (git-ignored, contains text)
python scripts/eval_sentiment.py score --ner bert --models ens guhr news
```

Gold distribution: 39 neg, 17 neu, 4 pos. Comment sections are mostly negative, so always
predicting "neg" already gets 0.65 accuracy. Macro-F1 over the three classes is therefore the
more useful number.

## Results (n = 60)

| model | window | accuracy | macro-F1 |
|---|---|---|---|
| majority class (neg) | – | 0.65 | 0.26 |
| `oliverguhr/german-sentiment-bert` (with its lower-casing normalisation, argmax) | sentence | 0.38 | 0.27 |
| same, raw text (`guhr-raw`), argmax | sentence | 0.45 | 0.43 |
| `mdraw/german-news-sentiment-bert` (`news`), argmax | sentence | 0.43 | 0.42 |
| **ensemble `news` + `guhr-raw`, threshold 0.2 (`ens`, default)** | **target window** | **0.53** | **0.44** |
| `news` | target window | 0.53 | 0.44 |
| `ens` / `news` | whole comment (not target-specific) | 0.65 | 0.47 / 0.50 |

The target window is the mention's sentence plus its neighbours when they mention no other
entity, narrowed to the clause when the same sentence mentions several entities.

Findings:

* Both off-the-shelf models are much too neutral on comments. Most errors are `neg → neu`:
  sarcasm, rhetorical questions and implicit criticism ("Die ÖVP ist doch selbst Rechts.").
  Using the probability margin P(pos) − P(neg) with a threshold instead of argmax, and
  averaging two models, gave the largest gain (macro-F1 0.27 → 0.44).
* The lower-casing normalisation recommended for the Guhr model hurts on this data.
* The whole comment scores slightly higher than the target window on this sample. But many
  comments mention several entities with opposite sentiment, and a whole-comment score would
  give all of them the same label, so the default stays target-specific.
* Linking errors found while labelling: "Linken" (generic "leftists") linked to *Die Linke* on
  Swiss and Austrian sites, and "Harris" (a film character) linked to Kamala Harris. Of the 60
  linked pairs, 3 were wrong (95 % linking precision on this sample).

Caveats: 60 pairs, one annotator, and the choice of threshold was made on the same sample (no
held-out set), so expect lower numbers on new data. Next steps: label more pairs per source,
then fine-tune a small model on comment data (e.g. GermEval 2017 plus our labels), or use the
HU LLM for target-dependent labels as a teacher.
