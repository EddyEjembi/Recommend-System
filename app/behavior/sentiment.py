"""Sentiment scoring utilities.

Wire a real model (transformers, VADER, etc.) into `score_sentiment` when
selected. Until then, the function raises `NotImplementedError` to fail loudly
if accidentally invoked.
"""

from typing import Literal

SentimentLabel = Literal["negative", "neutral", "positive"]


def score_sentiment(text: str) -> tuple[SentimentLabel, float]:
    """Return a sentiment label and confidence score for `text`."""
    raise NotImplementedError("score_sentiment not implemented")


def label_from_stars(stars: float) -> SentimentLabel:
    """Map a 1-5 star rating to a coarse sentiment label."""
    if stars <= 2:
        return "negative"
    if stars >= 4:
        return "positive"
    return "neutral"
