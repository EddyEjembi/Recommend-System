"""Feature-engineering helpers feeding both retrieval and persona building."""

from collections.abc import Iterable
from typing import Any

from app.behavior.sentiment import label_from_stars


def build_user_features(reviews: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return a feature dict ready for downstream persona prompting."""
    materialised = list(reviews)
    if not materialised:
        return {"sentiment_mix": {}, "avg_review_length": 0}

    labels = [
        label_from_stars(float(r["stars"]))
        for r in materialised
        if r.get("stars") is not None
    ]
    mix = {label: labels.count(label) for label in set(labels)}

    lengths = [len(r.get("text", "")) for r in materialised]
    return {
        "sentiment_mix": mix,
        "avg_review_length": sum(lengths) // len(lengths) if lengths else 0,
        "max_review_length": max(lengths) if lengths else 0,
    }
