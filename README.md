# Model files of the analysis pipeline

This branch holds model weights only and shares no history with `main`.

`sentiment-ft-v3/`: what fine-tuning changed in
`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (revision `b5113eb`) for the target
sentiment model `sent-ft@3`, in half precision, split into parts below 100 MB. The pipeline
downloads the parts named in `data/sentiment_ft.json` on `main`, checks their SHA-256 and rebuilds
the model (`analysis/finetune.py`). Training and evaluation: `scripts/train_sentiment_finetune.py`,
`eval/README.md`. No comment texts.

Base model licence: MIT.
