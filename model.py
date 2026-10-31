"""Sentiment inference using a publicly available pretrained model."""
from functools import lru_cache

MODEL_ID = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

@lru_cache(maxsize=1)
def get_classifier():
    from transformers import pipeline
    return pipeline('text-classification', model=MODEL_ID, top_k=None)

def classify(text: str):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Please enter non-empty text.')
    if len(text) > 4000:
        raise ValueError('Text must be at most 4,000 characters.')
    results = get_classifier()(text.strip(), truncation=True, max_length=512)
    if results and isinstance(results[0], list):
        results = results[0]
    scores = {r['label'].lower(): round(float(r['score']), 5) for r in results}
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {'label': ordered[0][0], 'confidence': ordered[0][1], 'scores': scores}
