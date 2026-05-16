"""Builds a restaurant-focused subset of the Yelp Open Dataset.

Pipeline:
    1. Filter `business.json` to open restaurants in the top-N cities.
    2. Stream `review.json`, keep reviews on those businesses, cap per business.
    3. Stream `user.json`, keep users that appear in the filtered reviews.
    4. Close the loop: drop users below the min-review threshold and drop their
       reviews. Drop businesses left without reviews.

Outputs the Parquet files into the `app/data/processed/` directory:
    businesses.parquet
    reviews.parquet
    users.parquet

"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import pandas as pd

from app.ingestion.yelp_loader import YelpLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsetConfig:
    """All knobs for the subset-building pipeline."""

    raw_dir: Path
    processed_dir: Path
    required_category: str = "Restaurants"
    top_city_count: int = 3
    min_business_review_count: int = 10
    max_reviews_per_business: int = 50
    min_user_reviews_in_subset: int = 5
    review_scan_limit: int | None = None


def default_config() -> SubsetConfig:
    """Return a `SubsetConfig` rooted at this project's `app/data/` directory."""
    base = Path(__file__).resolve().parents[1] / "data"
    return SubsetConfig(
        raw_dir=base / "raw" / "yelp",
        processed_dir=base / "processed",
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_yelp_file(raw_dir: Path, basename: str) -> Path:
    """Locate a Yelp dump file regardless of whether it has the `yelp_academic_dataset_` prefix."""
    candidates = [
        raw_dir / f"{basename}.json",
        raw_dir / f"yelp_academic_dataset_{basename}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a Yelp {basename} file. Looked in: "
        f"{[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# Step 1: businesses
# ---------------------------------------------------------------------------


def is_in_category(categories: str | None, required: str) -> bool:
    """Return True if `required` is one of the comma-separated tokens in `categories`."""
    if not categories:
        return False
    tokens = [token.strip().lower() for token in categories.split(",")]
    return required.lower() in tokens


def filter_businesses(path: Path, config: SubsetConfig) -> pd.DataFrame:
    """Return the filtered restaurant subset as a DataFrame.

    Filters applied:
      - `required_category` must be present in `categories`
      - `is_open == 1`
      - `review_count >= min_business_review_count`
      - city must be one of the top `top_city_count` cities by restaurant count
    """
    logger.info("Step 1/4: filtering businesses from %s", path)
    candidates: list[dict[str, Any]] = []

    for biz in YelpLoader.stream_json(str(path)):
        if biz.get("is_open") != 1:
            continue
        if int(biz.get("review_count") or 0) < config.min_business_review_count:
            continue
        if not is_in_category(biz.get("categories"), config.required_category):
            continue

        candidates.append(
            {
                "business_id": biz["business_id"],
                "name": biz.get("name"),
                "city": biz.get("city"),
                "state": biz.get("state"),
                "categories": biz.get("categories"),
                "stars": biz.get("stars"),
                "review_count": biz.get("review_count"),
                "price_range": _extract_price_range(biz.get("attributes")),
            }
        )

    df = pd.DataFrame(candidates)
    if df.empty:
        raise RuntimeError(
            "No businesses matched the restaurant filter. "
            "Check that the Yelp dump is present and not corrupted."
        )

    logger.info("Found %d candidate restaurants across all cities", len(df))

    top_cities = (
        df["city"].fillna("").value_counts().head(config.top_city_count).index.tolist()
    )
    logger.info(
        "Keeping top %d cities by restaurant count: %s",
        config.top_city_count,
        top_cities,
    )

    df = df[df["city"].isin(top_cities)].reset_index(drop=True)
    logger.info("After city filter: %d restaurants kept", len(df))
    return df


def _extract_price_range(attributes: dict[str, Any] | None) -> int | None:
    """Pull `RestaurantsPriceRange2` out of the Yelp attributes dict, if present."""
    if not isinstance(attributes, dict):
        return None
    raw = attributes.get("RestaurantsPriceRange2")
    if raw is None or raw == "None":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Step 2: reviews
# ---------------------------------------------------------------------------


def filter_reviews(
    path: Path,
    business_ids: set[str],
    config: SubsetConfig,
) -> pd.DataFrame:
    """Stream reviews, keep only those for subset businesses, and cap per business.

    The per-business cap prefers reviews with higher community engagement
    (useful + funny + cool), breaking ties with the most recent date.
    """
    logger.info("Step 2/4: filtering reviews from %s", path)
    kept: list[dict[str, Any]] = []
    scanned = 0

    for review in YelpLoader.stream_json(str(path)):
        scanned += 1
        if scanned % 500_000 == 0:
            logger.info("  scanned %d reviews, kept %d", scanned, len(kept))

        if (
            config.review_scan_limit is not None
            and scanned >= config.review_scan_limit
        ):
            logger.info("  hit review_scan_limit=%d, stopping", config.review_scan_limit)
            break

        if review.get("business_id") not in business_ids:
            continue

        kept.append(
            {
                "review_id": review["review_id"],
                "user_id": review["user_id"],
                "business_id": review["business_id"],
                "stars": review.get("stars"),
                "useful": int(review.get("useful") or 0),
                "funny": int(review.get("funny") or 0),
                "cool": int(review.get("cool") or 0),
                "text": review.get("text") or "",
                "date": review.get("date"),
            }
        )

    df = pd.DataFrame(kept)
    if df.empty:
        raise RuntimeError(
            "No reviews matched the business subset. "
            "Did Step 1 produce any businesses?"
        )
    logger.info("Scanned %d reviews; matched %d before per-business cap", scanned, len(df))

    df["_engagement"] = df["useful"] + df["funny"] + df["cool"]
    df = df.sort_values(
        by=["business_id", "_engagement", "date"],
        ascending=[True, False, False],
        kind="stable",
    )
    df = df.groupby("business_id", group_keys=False).head(config.max_reviews_per_business)
    df = df.drop(columns=["_engagement"]).reset_index(drop=True)

    logger.info("After per-business cap of %d: %d reviews", config.max_reviews_per_business, len(df))
    return df


# ---------------------------------------------------------------------------
# Step 3: users
# ---------------------------------------------------------------------------


def filter_users(path: Path, user_ids: Iterable[str]) -> pd.DataFrame:
    """Stream the user dump and keep only users whose id is in `user_ids`."""
    user_id_set = set(user_ids)
    logger.info("Step 3/4: filtering users from %s (looking for %d ids)", path, len(user_id_set))
    kept: list[dict[str, Any]] = []

    for user in YelpLoader.stream_json(str(path)):
        if user.get("user_id") not in user_id_set:
            continue
        kept.append(
            {
                "user_id": user["user_id"],
                "name": user.get("name"),
                "review_count": int(user.get("review_count") or 0),
                "yelping_since": user.get("yelping_since"),
                "useful": int(user.get("useful") or 0),
                "funny": int(user.get("funny") or 0),
                "cool": int(user.get("cool") or 0),
                "fans": int(user.get("fans") or 0),
                "average_stars": user.get("average_stars"),
                "is_elite": _is_elite(user.get("elite")),
                "friends_count": _count_friends(user.get("friends")),
            }
        )

    df = pd.DataFrame(kept)
    logger.info("Matched %d users", len(df))
    return df


def _is_elite(elite_raw: Any) -> bool:
    """Yelp stores elite as a comma-separated string of years or the literal "None"."""
    if not elite_raw or elite_raw == "None":
        return False
    if isinstance(elite_raw, str):
        return any(part.strip() for part in elite_raw.split(","))
    return bool(elite_raw)


def _count_friends(friends_raw: Any) -> int:
    """Yelp stores friends as a comma-separated list of user_ids (string)."""
    if not friends_raw or friends_raw == "None":
        return 0
    if isinstance(friends_raw, str):
        return sum(1 for part in friends_raw.split(",") if part.strip())
    if isinstance(friends_raw, list):
        return len(friends_raw)
    return 0


# ---------------------------------------------------------------------------
# Step 4: close the loop
# ---------------------------------------------------------------------------


def close_the_loop(
    businesses_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    users_df: pd.DataFrame,
    min_user_reviews: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop users below the review threshold and any orphaned rows.

    Returns `(businesses, reviews, users)` with consistent foreign keys.
    """
    logger.info("Step 4/4: closing the loop (min %d reviews per user)", min_user_reviews)

    review_counts = reviews_df.groupby("user_id").size()
    valid_user_ids = set(review_counts[review_counts >= min_user_reviews].index)
    logger.info(
        "  users meeting threshold: %d / %d",
        len(valid_user_ids),
        users_df["user_id"].nunique(),
    )

    final_users = users_df[users_df["user_id"].isin(valid_user_ids)].reset_index(drop=True)
    final_reviews = reviews_df[reviews_df["user_id"].isin(valid_user_ids)].reset_index(drop=True)

    surviving_business_ids = set(final_reviews["business_id"].unique())
    final_businesses = (
        businesses_df[businesses_df["business_id"].isin(surviving_business_ids)]
        .reset_index(drop=True)
    )

    logger.info(
        "  final: %d businesses, %d reviews, %d users",
        len(final_businesses),
        len(final_reviews),
        len(final_users),
    )
    return final_businesses, final_reviews, final_users


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def build_subset(config: SubsetConfig) -> dict[str, Path]:
    """Run the full pipeline end-to-end and write three Parquet files.

    Returns a dict mapping logical name -> output path.
    """
    config.processed_dir.mkdir(parents=True, exist_ok=True)

    business_path = find_yelp_file(config.raw_dir, "business")
    review_path = find_yelp_file(config.raw_dir, "review")
    user_path = find_yelp_file(config.raw_dir, "user")

    businesses_df = filter_businesses(business_path, config)
    reviews_df = filter_reviews(
        review_path,
        business_ids=set(businesses_df["business_id"]),
        config=config,
    )
    users_df = filter_users(user_path, user_ids=reviews_df["user_id"].unique())

    final_businesses, final_reviews, final_users = close_the_loop(
        businesses_df=businesses_df,
        reviews_df=reviews_df,
        users_df=users_df,
        min_user_reviews=config.min_user_reviews_in_subset,
    )

    outputs = {
        "businesses": config.processed_dir / "businesses.parquet",
        "reviews": config.processed_dir / "reviews.parquet",
        "users": config.processed_dir / "users.parquet",
    }

    final_businesses.to_parquet(outputs["businesses"], index=False)
    final_reviews.to_parquet(outputs["reviews"], index=False)
    final_users.to_parquet(outputs["users"], index=False)

    for name, path in outputs.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info("Wrote %s -> %s (%.1f MB)", name, path, size_mb)

    return outputs


def main() -> None:
    """CLI entrypoint when running this module directly."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    outputs = build_subset(default_config())
    print("\nSubset build complete:")
    for name, path in outputs.items():
        print(f"  {name:<11}: {path}")


if __name__ == "__main__":
    main()
