"""Derive behavioural traits from a user's review history."""

import math
import re
from collections.abc import Iterable
from statistics import mean, pstdev
from typing import Any

from app.behavior.feature_engineering import build_user_features
from app.types.behavior import UserBehaviorProfile


def summarise_user_activity(reviews: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return aggregate counts and rating statistics for a user.

    `reviews` should be an iterable of preprocessed review dicts (see
    `app.ingestion.preprocess`).
    """
    materialised = list(reviews)
    if not materialised:
        return {"review_count": 0, "avg_stars": None, "total_useful": 0}

    stars = [r["stars"] for r in materialised if r.get("stars") is not None]
    return {
        "review_count": len(materialised),
        "avg_stars": mean(stars) if stars else None,
        "total_useful": sum(int(r.get("useful", 0)) for r in materialised),
        "total_funny": sum(int(r.get("funny", 0)) for r in materialised),
        "total_cool": sum(int(r.get("cool", 0)) for r in materialised),
    }


def detect_review_style(reviews: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Infer stylistic traits (length, punctuation, caps) from past reviews."""
    materialised = [r for r in reviews if str(r.get("text", "")).strip()]
    if not materialised:
        return {
            "review_count": 0,
            "avg_chars": 0,
            "median_chars": 0,
            "p90_chars": 0,
            "chars_stdev": 0.0,
            "avg_words": 0.0,
            "exclamations_per_1k_chars": 0.0,
            "questions_per_1k_chars": 0.0,
            "caps_ratio": 0.0,
        }

    lengths = [len(str(r.get("text", ""))) for r in materialised]
    sorted_lens = sorted(lengths)
    n = len(sorted_lens)
    median = sorted_lens[n // 2] if n % 2 == 1 else (sorted_lens[n // 2 - 1] + sorted_lens[n // 2]) / 2
    p90_idx = min(n - 1, int(math.ceil(0.9 * n) - 1))
    p90 = sorted_lens[p90_idx]

    word_counts: list[int] = []
    exclam_total = 0
    question_total = 0
    caps_chars = 0
    letter_chars = 0

    for r in materialised:
        text = str(r.get("text", ""))
        word_counts.append(len(re.findall(r"\b\w+\b", text)))
        exclam_total += text.count("!")
        question_total += text.count("?")
        for ch in text:
            if ch.isalpha():
                letter_chars += 1
                if ch.isupper():
                    caps_chars += 1

    total_chars = sum(lengths) or 1
    caps_ratio = caps_chars / letter_chars if letter_chars else 0.0

    return {
        "review_count": n,
        "avg_chars": int(mean(lengths)),
        "median_chars": int(median),
        "p90_chars": int(p90),
        "chars_stdev": float(pstdev(lengths)) if n > 1 else 0.0,
        "avg_words": float(mean(word_counts)) if word_counts else 0.0,
        "exclamations_per_1k_chars": exclam_total / total_chars * 1000.0,
        "questions_per_1k_chars": question_total / total_chars * 1000.0,
        "caps_ratio": round(caps_ratio, 4),
    }


def _yelp_meta_subset(user_row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Pick stable columns from a `users.parquet` row for downstream prompts."""
    if not user_row:
        return None
    keys = (
        "user_id",
        "name",
        "review_count",
        "average_stars",
        "is_elite",
        "fans",
        "friends_count",
        "yelping_since",
        "useful",
        "funny",
        "cool",
    )
    return {k: user_row.get(k) for k in keys}


def build_user_behavior_profile(
    user_id: str,
    reviews: Iterable[dict[str, Any]],
    user_row: dict[str, Any] | None = None,
    test_archetype: str | None = None,
) -> UserBehaviorProfile:
    """Assemble a full `UserBehaviorProfile` from subset reviews and optional Yelp user row."""
    materialised = list(reviews)
    activity = summarise_user_activity(materialised)
    style = detect_review_style(materialised)
    features = build_user_features(materialised)
    return UserBehaviorProfile(
        user_id=user_id,
        source="yelp_history",
        activity=activity,
        style=style,
        features=features,
        yelp_user_meta=_yelp_meta_subset(user_row) if user_row else None,
        cold_seed=None,
        test_archetype=test_archetype,
    )
