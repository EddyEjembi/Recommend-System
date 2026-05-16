"""CLI entrypoint for Phase 3 behavioural analysis.

Examples:

    # Full run: every user in users.parquet + cold seeds from test_users.json,
    #            every business in businesses.parquet
    python -m app.behavior.build_behavior

    # Fast run: only user_ids listed in test_users.json (existing + cold)
    python -m app.behavior.build_behavior --only-test-users
"""

import argparse
import logging
from pathlib import Path

from app.behavior.pipeline import default_processed_dir, run_behavior_pipeline


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Override processed directory (default: app/data/processed/)",
    )
    parser.add_argument(
        "--only-test-users",
        action="store_true",
        help="Only emit user profiles for ids in test_users.json",
    )
    return parser.parse_args()


def main() -> None:
    """Run the behaviour pipeline and print output paths."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    proc = default_processed_dir() if args.processed_dir is None else Path(args.processed_dir)

    paths = run_behavior_pipeline(proc, only_test_users=args.only_test_users)
    print("\n=== Behaviour pipeline complete ===")
    for key, path in paths.items():
        print(f"  {key:<9}: {path}")


if __name__ == "__main__":
    main()
