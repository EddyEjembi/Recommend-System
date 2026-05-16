"""Generates `test_users.json` for evaluating the review-generation pipeline.

Produces two flavors of test users:

  - `existing`   : 5 archetypes sampled from the processed Yelp subset.
                   These exercise the "persona-from-history" code path.
  - `cold_start` : 5 hand-authored Nigerian personas with no Yelp history.
                   These exercise the cold-start code path described in the PRD.

"""

from pathlib import Path
from typing import Any
import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Existing-user archetypes (sampled from the Yelp subset)
# ---------------------------------------------------------------------------


def _augment_users_with_text_length(
    users_df: pd.DataFrame, reviews_df: pd.DataFrame
) -> pd.DataFrame:
    """Return `users_df` enriched with `avg_text_len` and `subset_review_count`."""
    text_lengths = reviews_df.copy()
    text_lengths["text_len"] = text_lengths["text"].fillna("").str.len()

    per_user = text_lengths.groupby("user_id").agg(
        avg_text_len=("text_len", "mean"),
        subset_review_count=("review_id", "count"),
    )

    return (
        users_df.set_index("user_id")
        .join(per_user, how="left")
        .reset_index()
    )


def _row_to_existing_user(row: pd.Series, archetype: str) -> dict[str, Any]:
    """Pack a `users_df` row into the JSON shape we want for test users."""
    return {
        "user_id": row["user_id"],
        "kind": "existing",
        "archetype": archetype,
        "name": row.get("name"),
        "review_count": int(row.get("review_count") or 0),
        "subset_review_count": _safe_int(row.get("subset_review_count")),
        "average_stars": _safe_float(row.get("average_stars")),
        "avg_text_len": _safe_float(row.get("avg_text_len")),
        "is_elite": bool(row.get("is_elite", False)),
        "fans": int(row.get("fans") or 0),
        "friends_count": int(row.get("friends_count") or 0),
        "yelping_since": row.get("yelping_since"),
    }


def _safe_int(value: Any) -> int | None:
    """Convert pandas NA-friendly values into plain ints or None."""
    if value is None or pd.isna(value):
        return None
    return int(value)


def _safe_float(value: Any) -> float | None:
    """Convert pandas NA-friendly values into plain floats or None."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def pick_existing_test_users(
    users_df: pd.DataFrame, reviews_df: pd.DataFrame
) -> list[dict[str, Any]]:
    """Sample five user archetypes from the processed subset.

    The archetypes are chosen to stress different parts of the persona builder:
      - power_reviewer  : elite user with high review_count (rich signal)
      - harsh_critic    : low average_stars (negative-leaning style)
      - generous_rater  : high average_stars (positive-leaning style)
      - verbose_writer  : long average review text (verbose style)
      - terse_writer    : short average review text (laconic style)
    """
    augmented = _augment_users_with_text_length(users_df, reviews_df)
    if augmented.empty:
        return []

    picked: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    def _take(filtered: pd.DataFrame, sort_by: list[tuple[str, bool]], archetype: str) -> None:
        """Pick the first row of `filtered` (after sort) that we haven't used."""
        if filtered.empty:
            return
        sort_cols = [col for col, _ in sort_by]
        ascending = [asc for _, asc in sort_by]
        ranked = filtered.sort_values(by=sort_cols, ascending=ascending, na_position="last")
        for _, row in ranked.iterrows():
            if row["user_id"] in used_ids:
                continue
            picked.append(_row_to_existing_user(row, archetype))
            used_ids.add(row["user_id"])
            return

    elite_pool = augmented[augmented["is_elite"] == True]  # noqa: E712 (pandas mask)
    _take(elite_pool, [("review_count", False)], "power_reviewer")

    harsh_pool = augmented[augmented["average_stars"].fillna(5.0) <= 2.5]
    _take(harsh_pool, [("subset_review_count", False)], "harsh_critic")

    generous_pool = augmented[augmented["average_stars"].fillna(0.0) >= 4.5]
    _take(generous_pool, [("subset_review_count", False)], "generous_rater")

    verbose_pool = augmented[augmented["subset_review_count"].fillna(0) >= 3]
    _take(verbose_pool, [("avg_text_len", False)], "verbose_writer")

    terse_pool = augmented[augmented["subset_review_count"].fillna(0) >= 3]
    _take(terse_pool, [("avg_text_len", True)], "terse_writer")

    return picked


# ---------------------------------------------------------------------------
# Cold-start Nigerian personas (hand-authored)
# ---------------------------------------------------------------------------


def build_cold_start_users() -> list[dict[str, Any]]:
    """Return five hand-authored Nigerian cold-start personas.

    These are intentionally NOT generated from Yelp data. They model the cultural
    context called out in the PRD (Lagos / Abuja / Port Harcourt / Yaba / Ibadan)
    and span different ages, budgets, and review styles. Reference them by
    `user_id` when testing the cold-start persona pipeline.
    """
    return [
        {
            "user_id": "cold_ng_001_lagos_budget_diner",
            "kind": "cold_start",
            "archetype": "lagos_budget_diner",
            "demographics": {
                "city": "Lagos",
                "country": "Nigeria",
                "age_band": "25-34",
                "language": "Nigerian English with mild pidgin",
            },
            "preferences": {
                "budget": "low-to-mid",
                "cuisines": ["Nigerian", "Jollof", "Suya", "Continental"],
                "tone": "casual, friendly",
                "review_style": "short and expressive, occasional pidgin (e.g. 'sha', 'abi')",
            },
            "service_expectations": {
                "wait_time_tolerance": "medium",
                "price_sensitivity": "high",
                "portion_size_importance": "high",
            },
            "notes": "Will mention traffic, value-for-money, and portion sizes naturally.",
        },
        {
            "user_id": "cold_ng_002_abuja_formal_professional",
            "kind": "cold_start",
            "archetype": "abuja_formal_professional",
            "demographics": {
                "city": "Abuja",
                "country": "Nigeria",
                "age_band": "35-44",
                "language": "Formal Nigerian English",
            },
            "preferences": {
                "budget": "high",
                "cuisines": ["Continental", "Asian fusion", "Fine-dining Nigerian"],
                "tone": "measured, articulate",
                "review_style": "longer, structured, balanced critique",
            },
            "service_expectations": {
                "wait_time_tolerance": "low",
                "price_sensitivity": "low",
                "ambience_importance": "high",
                "service_formality_importance": "high",
            },
            "notes": "Rarely uses pidgin. Notices wine list, plating, and service pacing.",
        },
        {
            "user_id": "cold_ng_003_ph_critical_foodie",
            "kind": "cold_start",
            "archetype": "ph_critical_foodie",
            "demographics": {
                "city": "Port Harcourt",
                "country": "Nigeria",
                "age_band": "28-35",
                "language": "Nigerian English with frequent pidgin",
            },
            "preferences": {
                "budget": "mid",
                "cuisines": ["Bole", "Native Rivers cuisine", "Seafood", "Nigerian"],
                "tone": "blunt, opinionated",
                "review_style": "medium-length, calls out issues directly",
            },
            "service_expectations": {
                "wait_time_tolerance": "low",
                "price_sensitivity": "medium",
                "authenticity_importance": "very high",
            },
            "notes": "Will call out inauthentic preparations of native dishes specifically.",
        },
        {
            "user_id": "cold_ng_004_yaba_student_casual",
            "kind": "cold_start",
            "archetype": "yaba_student_casual",
            "demographics": {
                "city": "Lagos (Yaba)",
                "country": "Nigeria",
                "age_band": "18-24",
                "language": "Casual Nigerian English, heavy pidgin and slang",
            },
            "preferences": {
                "budget": "very low",
                "cuisines": ["Suya", "Shawarma", "Indomie spots", "Local rice joints"],
                "tone": "playful, slangy",
                "review_style": "short and punchy, exclamations, light slang",
            },
            "service_expectations": {
                "wait_time_tolerance": "medium",
                "price_sensitivity": "very high",
                "vibe_importance": "high",
            },
            "notes": "Vibe and value matter more than service formality.",
        },
        {
            "user_id": "cold_ng_005_ibadan_family_diner",
            "kind": "cold_start",
            "archetype": "ibadan_family_diner",
            "demographics": {
                "city": "Ibadan",
                "country": "Nigeria",
                "age_band": "40-50",
                "language": "Nigerian English, occasional Yoruba phrases",
            },
            "preferences": {
                "budget": "mid",
                "cuisines": ["Amala", "Ewedu", "Buka-style", "Family restaurants"],
                "tone": "warm and descriptive",
                "review_style": "medium-length, focuses on portion sizes and family suitability",
            },
            "service_expectations": {
                "wait_time_tolerance": "high",
                "price_sensitivity": "medium",
                "portion_size_importance": "very high",
            },
            "notes": "Will praise generous portions and family-friendly seating.",
        },
    ]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def seed_test_users(processed_dir: Path) -> Path:
    """Build `test_users.json` next to the processed Parquet files."""
    users_path = processed_dir / "users.parquet"
    reviews_path = processed_dir / "reviews.parquet"

    if not users_path.exists() or not reviews_path.exists():
        raise FileNotFoundError(
            "Expected users.parquet and reviews.parquet in "
            f"{processed_dir}. Run subset_builder first."
        )

    users_df = pd.read_parquet(users_path)
    reviews_df = pd.read_parquet(reviews_path)

    existing = pick_existing_test_users(users_df, reviews_df)
    cold = build_cold_start_users()

    payload = {"existing": existing, "cold_start": cold}
    out_path = processed_dir / "test_users.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    logger.info(
        "Wrote %s (%d existing archetypes + %d cold-start)",
        out_path,
        len(existing),
        len(cold),
    )
    return out_path


def main() -> None:
    """CLI entrypoint when running this module directly."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    processed_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    out_path = seed_test_users(processed_dir)
    print(f"\nTest users written to: {out_path}")


if __name__ == "__main__":
    main()
