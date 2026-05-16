"""Map cold-start entries from `test_users.json` into `UserBehaviorProfile`."""

from typing import Any

from app.types.behavior import UserBehaviorProfile


def cold_start_record_to_profile(record: dict[str, Any]) -> UserBehaviorProfile:
    """Turn a `cold_start` object from `test_users.json` into a behaviour profile.

    No review history exists; structured fields become the sole signal for
    downstream persona prompting (Phase 4).
    """
    user_id = str(record["user_id"])
    archetype = str(record.get("archetype", ""))

    cold_seed: dict[str, Any] = {
        "archetype": archetype,
        "demographics": record.get("demographics") or {},
        "preferences": record.get("preferences") or {},
        "service_expectations": record.get("service_expectations") or {},
        "notes": record.get("notes"),
    }

    prefs = record.get("preferences") or {}
    tone = str(prefs.get("tone", ""))
    review_style = str(prefs.get("review_style", ""))
    budget = str(prefs.get("budget", ""))

    activity = {
        "review_count": 0,
        "avg_stars": None,
        "total_useful": 0,
        "total_funny": 0,
        "total_cool": 0,
        "note": "Cold start: no Yelp review history in subset.",
    }
    style = {
        "review_count": 0,
        "avg_chars": 0,
        "median_chars": 0,
        "p90_chars": 0,
        "chars_stdev": 0.0,
        "avg_words": 0.0,
        "exclamations_per_1k_chars": 0.0,
        "questions_per_1k_chars": 0.0,
        "caps_ratio": 0.0,
        "seed_tone": tone,
        "seed_review_style": review_style,
        "seed_budget": budget,
    }
    features = {
        "sentiment_mix": {},
        "avg_review_length": 0,
        "max_review_length": 0,
        "source": "cold_seed",
    }

    return UserBehaviorProfile(
        user_id=user_id,
        source="cold_seed",
        activity=activity,
        style=style,
        features=features,
        yelp_user_meta=None,
        cold_seed=cold_seed,
        test_archetype=archetype or None,
    )
