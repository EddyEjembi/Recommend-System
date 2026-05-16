"""Orchestrate Phase 3: build user and business behaviour artefacts from Parquet + test users."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app.behavior.business_analyzer import build_business_behavior_profile
from app.behavior.cold_user_from_seed import cold_start_record_to_profile
from app.behavior.user_analyzer import build_user_behavior_profile
from app.types.behavior import BusinessBehaviorProfile, UserBehaviorProfile

logger = logging.getLogger(__name__)


def default_processed_dir() -> Path:
    """Return `app/data/processed/` relative to this package."""
    return Path(__file__).resolve().parents[1] / "data" / "processed"


def load_test_users(path: Path) -> dict[str, Any]:
    """Load `test_users.json` if present; otherwise return empty structure."""
    if not path.exists():
        logger.warning("test_users.json not found at %s - skipping cold-start merge.", path)
        return {"existing": [], "cold_start": []}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _archetype_by_user_id(test_payload: dict[str, Any]) -> dict[str, str]:
    """Map `user_id` -> archetype for existing test users."""
    out: dict[str, str] = {}
    for row in test_payload.get("existing", []) or []:
        uid = row.get("user_id")
        if uid:
            out[str(uid)] = str(row.get("archetype", ""))
    return out


def build_all_user_profiles(
    users_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    test_payload: dict[str, Any],
    only_test_user_ids: set[str] | None = None,
) -> list[UserBehaviorProfile]:
    """Compute `UserBehaviorProfile` for parquet users plus cold-start seeds not in parquet."""
    archetypes = _archetype_by_user_id(test_payload)
    parquet_ids = set(users_df["user_id"].astype(str))

    if only_test_user_ids is not None:
        filtered = reviews_df[reviews_df["user_id"].isin(only_test_user_ids)]
        reviews_by_user = {
            uid: grp.to_dict("records") for uid, grp in filtered.groupby("user_id")
        }
    else:
        reviews_by_user = {
            uid: grp.to_dict("records") for uid, grp in reviews_df.groupby("user_id")
        }

    profiles: list[UserBehaviorProfile] = []

    for _, row in users_df.iterrows():
        uid = str(row["user_id"])
        if only_test_user_ids is not None and uid not in only_test_user_ids:
            continue
        user_reviews = reviews_by_user.get(uid, [])
        user_dict = row.to_dict()
        profiles.append(
            build_user_behavior_profile(
                user_id=uid,
                reviews=user_reviews,
                user_row=user_dict,
                test_archetype=archetypes.get(uid),
            )
        )

    for cold in test_payload.get("cold_start", []) or []:
        uid = str(cold.get("user_id", ""))
        if not uid:
            continue
        if uid in parquet_ids:
            logger.warning(
                "Cold-start user_id %s also appears in users.parquet - skipping cold duplicate.",
                uid,
            )
            continue
        if only_test_user_ids is not None and uid not in only_test_user_ids:
            continue
        profiles.append(cold_start_record_to_profile(cold))

    return profiles


def build_all_business_profiles(
    businesses_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    only_business_ids: set[str] | None = None,
) -> list[BusinessBehaviorProfile]:
    """Compute `BusinessBehaviorProfile` for every business row (optionally filtered)."""
    reviews_by_biz = {bid: grp.to_dict("records") for bid, grp in reviews_df.groupby("business_id")}

    out: list[BusinessBehaviorProfile] = []
    for _, row in businesses_df.iterrows():
        bid = str(row["business_id"])
        if only_business_ids is not None and bid not in only_business_ids:
            continue
        biz_reviews = reviews_by_biz.get(bid, [])
        out.append(build_business_behavior_profile(row.to_dict(), biz_reviews))
    return out


def run_behavior_pipeline(
    processed_dir: Path | None = None,
    *,
    only_test_users: bool = False,
) -> dict[str, Path]:
    """Write `user_behavior.jsonl`, `business_behavior.jsonl`, and a small manifest.

    When `only_test_users` is True, user profiles are built only for user_ids that
    appear in `test_users.json` (existing + cold_start). Business profiles are
    always built for the full `businesses.parquet` unless you later extend this.
    """
    base = processed_dir or default_processed_dir()
    users_path = base / "users.parquet"
    businesses_path = base / "businesses.parquet"
    reviews_path = base / "reviews.parquet"
    test_path = base / "test_users.json"

    if not users_path.exists():
        raise FileNotFoundError(f"Missing {users_path}")
    if not businesses_path.exists():
        raise FileNotFoundError(f"Missing {businesses_path}")
    if not reviews_path.exists():
        raise FileNotFoundError(f"Missing {reviews_path}")

    users_df = pd.read_parquet(users_path)
    businesses_df = pd.read_parquet(businesses_path)
    reviews_df = pd.read_parquet(reviews_path)
    test_payload = load_test_users(test_path)

    only_ids: set[str] | None = None
    if only_test_users:
        only_ids = set()
        for row in test_payload.get("existing", []) or []:
            if row.get("user_id"):
                only_ids.add(str(row["user_id"]))
        for row in test_payload.get("cold_start", []) or []:
            if row.get("user_id"):
                only_ids.add(str(row["user_id"]))
        logger.info(
            "only_test_users=True - building %d user ids from test_users.json",
            len(only_ids),
        )

    logger.info("Building user behaviour profiles...")
    user_profiles = build_all_user_profiles(users_df, reviews_df, test_payload, only_ids)
    logger.info("  users: %d profiles", len(user_profiles))

    logger.info("Building business behaviour profiles...")
    biz_profiles = build_all_business_profiles(businesses_df, reviews_df, None)
    logger.info("  businesses: %d profiles", len(biz_profiles))

    user_out = base / "user_behavior.jsonl"
    biz_out = base / "business_behavior.jsonl"
    manifest_out = base / "behavior_manifest.json"

    with user_out.open("w", encoding="utf-8") as fh:
        for p in user_profiles:
            fh.write(p.model_dump_json() + "\n")

    with biz_out.open("w", encoding="utf-8") as fh:
        for p in biz_profiles:
            fh.write(p.model_dump_json() + "\n")

    manifest = {
        "user_profile_count": len(user_profiles),
        "business_profile_count": len(biz_profiles),
        "only_test_users": only_test_users,
        "outputs": {
            "users": str(user_out),
            "businesses": str(biz_out),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote %s, %s, %s", user_out.name, biz_out.name, manifest_out.name)

    return {"users": user_out, "businesses": biz_out, "manifest": manifest_out}
