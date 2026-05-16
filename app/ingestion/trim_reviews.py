"""Shrink only `reviews.parquet` — businesses, users, and test_users.json stay unchanged.

Example (GitHub-friendly size):

    uv run python -m app.ingestion.trim_reviews --max-per-business 15 --max-total-reviews 20000
    uv run python -m app.retrieval.build_index --only reviews
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from app.ingestion.subset_builder import default_config
from app.retrieval.registry import default_processed_dir

logger = logging.getLogger(__name__)

DEMO_MAX_PER_BUSINESS = 15
DEMO_MAX_TOTAL_REVIEWS = 20_000


def trim_reviews_dataframe(
    reviews_df: pd.DataFrame,
    *,
    max_reviews_per_business: int,
    max_total_reviews: int | None,
    valid_business_ids: set[str] | None = None,
    valid_user_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Cap reviews per business, then optionally cap total rows. Does not touch other tables."""
    df = reviews_df.copy()
    before = len(df)

    if valid_business_ids is not None:
        df = df[df["business_id"].isin(valid_business_ids)]
    if valid_user_ids is not None:
        df = df[df["user_id"].isin(valid_user_ids)]

    if len(df) < before:
        logger.info("Dropped %d reviews with unknown business_id or user_id", before - len(df))

    if df.empty:
        raise RuntimeError("No reviews left after FK filter.")

    df["_engagement"] = df["useful"].fillna(0) + df["funny"].fillna(0) + df["cool"].fillna(0)
    df = df.sort_values(
        by=["business_id", "_engagement", "date"],
        ascending=[True, False, False],
        kind="stable",
    )
    df = df.groupby("business_id", group_keys=False).head(max_reviews_per_business)
    logger.info(
        "After per-business cap of %d: %d reviews (was %d)",
        max_reviews_per_business,
        len(df),
        before,
    )

    if max_total_reviews is not None and len(df) > max_total_reviews:
        df = df.sort_values(
            by=["_engagement", "date"],
            ascending=[False, False],
            kind="stable",
        )
        df = df.head(max_total_reviews)
        logger.info("After global cap of %d: %d reviews", max_total_reviews, len(df))

    return df.drop(columns=["_engagement"]).reset_index(drop=True)


def trim_reviews_parquet(
    processed_dir: Path | None = None,
    *,
    max_reviews_per_business: int,
    max_total_reviews: int | None = None,
    enforce_fk: bool = True,
) -> Path:
    """Rewrite only `reviews.parquet` in `processed_dir`."""
    processed = processed_dir or default_processed_dir()
    reviews_path = processed / "reviews.parquet"
    if not reviews_path.exists():
        raise FileNotFoundError(f"Missing {reviews_path}")

    reviews_df = pd.read_parquet(reviews_path)
    logger.info("Loaded %d reviews from %s", len(reviews_df), reviews_path)

    valid_business_ids: set[str] | None = None
    valid_user_ids: set[str] | None = None
    if enforce_fk:
        businesses_path = processed / "businesses.parquet"
        users_path = processed / "users.parquet"
        if businesses_path.exists():
            valid_business_ids = set(pd.read_parquet(businesses_path)["business_id"].astype(str))
        if users_path.exists():
            valid_user_ids = set(pd.read_parquet(users_path)["user_id"].astype(str))

    trimmed = trim_reviews_dataframe(
        reviews_df,
        max_reviews_per_business=max_reviews_per_business,
        max_total_reviews=max_total_reviews,
        valid_business_ids=valid_business_ids,
        valid_user_ids=valid_user_ids,
    )
    trimmed.to_parquet(reviews_path, index=False)
    size_mb = reviews_path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%d rows, %.1f MB)", reviews_path, len(trimmed), size_mb)
    return reviews_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--demo",
        action="store_true",
        help=f"Preset: --max-per-business {DEMO_MAX_PER_BUSINESS} --max-total-reviews {DEMO_MAX_TOTAL_REVIEWS}",
    )
    parser.add_argument(
        "--max-per-business",
        type=int,
        default=None,
        help=f"Keep at most N reviews per business by engagement (default: {default_config().max_reviews_per_business})",
    )
    parser.add_argument(
        "--max-total-reviews",
        type=int,
        default=None,
        help="Keep at most N reviews overall after the per-business cap.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Override app/data/processed",
    )
    parser.add_argument(
        "--no-fk-check",
        action="store_true",
        help="Do not drop reviews whose business_id/user_id is missing from the other parquet files.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    proc = Path(args.processed_dir) if args.processed_dir else default_processed_dir()

    if args.demo:
        max_per = DEMO_MAX_PER_BUSINESS
        max_total = DEMO_MAX_TOTAL_REVIEWS
    else:
        max_per = args.max_per_business if args.max_per_business is not None else default_config().max_reviews_per_business
        max_total = args.max_total_reviews

    path = trim_reviews_parquet(
        proc,
        max_reviews_per_business=max_per,
        max_total_reviews=max_total,
        enforce_fk=not args.no_fk_check,
    )

    print("\n=== Review trim complete (businesses / users / test_users unchanged) ===")
    print(f"  reviews: {path} ({path.stat().st_size / (1024 * 1024):.1f} MB)")
    print("\nNext (review vector index only):")
    print("  uv run python -m app.retrieval.build_index --only reviews")


if __name__ == "__main__":
    main()
