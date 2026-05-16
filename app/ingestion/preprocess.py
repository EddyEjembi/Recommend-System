"""Preprocessing pipeline that turns raw Yelp records into model-ready rows."""

from collections.abc import Iterable, Iterator
from typing import Any

from app.ingestion.cleaners import clean_text


def preprocess_review(record: dict[str, Any]) -> dict[str, Any]:
    """Normalise a single raw review record into a flat, typed dict."""
    return {
        "review_id": record.get("review_id"),
        "user_id": record.get("user_id"),
        "business_id": record.get("business_id"),
        "stars": float(record["stars"]) if "stars" in record else None,
        "text": clean_text(record.get("text", "")),
        "useful": int(record.get("useful", 0)),
        "funny": int(record.get("funny", 0)),
        "cool": int(record.get("cool", 0)),
        "date": record.get("date"),
    }


def preprocess_reviews(records: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Apply `preprocess_review` lazily over an iterable of raw records."""
    for record in records:
        yield preprocess_review(record)
