"""Orchestrates the full Yelp subset pipeline (rebuilds businesses, reviews, and users from raw JSON).

To shrink only `reviews.parquet` + `reviews.faiss` for GitHub, use:

    uv run python -m app.ingestion.trim_reviews --demo
    uv run python -m app.retrieval.build_index --only reviews

Optional flags:

    --top-cities N        keep restaurants from the top N cities (default 3)
    --max-per-business N  cap reviews per business (default 50)
    --min-user-reviews N  drop users below this many in-subset reviews (default 5)
    --review-scan-limit N stop scanning review.json after N rows (for fast smoke tests)
    --skip-test-users     skip the test_users.json seeding step

This script wires together:
    1. `subset_builder.build_subset` -> businesses/reviews/users Parquet files
    2. `test_user_seeder.seed_test_users` -> test_users.json
"""

import argparse
import logging
from dataclasses import replace

from app.ingestion.subset_builder import build_subset, default_config
from app.ingestion.test_user_seeder import seed_test_users


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for the orchestrator."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-cities", type=int, default=None, help="Top N cities by restaurant count (default: 3)")
    parser.add_argument("--max-per-business", type=int, default=None, help="Cap reviews per business (default: 50)")
    parser.add_argument("--min-user-reviews", type=int, default=None, help="Drop users below this threshold (default: 5)")
    parser.add_argument(
        "--review-scan-limit",
        type=int,
        default=None,
        help="Stop scanning review.json after N records. Useful for smoke tests.",
    )
    parser.add_argument(
        "--skip-test-users",
        action="store_true",
        help="Skip writing test_users.json after the Parquet files are built.",
    )
    return parser.parse_args()


def main() -> None:
    """Orchestrate the full pipeline end-to-end."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    config = default_config()
    overrides: dict[str, object] = {}
    if args.top_cities is not None:
        overrides["top_city_count"] = args.top_cities
    if args.max_per_business is not None:
        overrides["max_reviews_per_business"] = args.max_per_business
    if args.min_user_reviews is not None:
        overrides["min_user_reviews_in_subset"] = args.min_user_reviews
    if args.review_scan_limit is not None:
        overrides["review_scan_limit"] = args.review_scan_limit

    if overrides:
        config = replace(config, **overrides)

    outputs = build_subset(config)

    if not args.skip_test_users:
        test_path = seed_test_users(config.processed_dir)
    else:
        test_path = None

    print("\n=== Subset build complete ===")
    for name, path in outputs.items():
        print(f"  {name:<11}: {path}")
    if test_path is not None:
        print(f"  test_users : {test_path}")
    else:
        print("  test_users : skipped (--skip-test-users)")


if __name__ == "__main__":
    main()
