# Sentiment evaluation

Protocol: pairs (comment, entity) drawn at random from the real `mentions` in Firestore
(at most one pair per comment, at most four per entity), labelled by hand with the sentiment
the comment expresses **toward that entity** (`pos` / `neu` / `neg`). `labels.csv` stores
only `mentionId` (= `commentId_entityId`) and the gold label – never comment text.

```bash
python scripts/eval_sentiment.py sample --n 60   # writes eval/sample.local.jsonl (git-ignored, contains text)
python scripts/eval_sentiment.py score           # compares all model variants
```

Results: pending (no crawled comments yet).
