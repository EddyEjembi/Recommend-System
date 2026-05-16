"""Register a new cold-start demo user from API input and persist to disk."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from app.behavior.cold_user_from_seed import cold_start_record_to_profile
from app.persona.test_users_store import atomic_write_json, load_test_users_payload
from app.types.behavior import UserBehaviorProfile


def _rewrite_user_behavior_jsonl(path: Path, profiles: list[UserBehaviorProfile]) -> None:
    """Replace `user_behavior.jsonl` with the given profiles (atomic write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    body = "\n".join(p.model_dump_json() for p in profiles) + "\n"
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


def append_user_behavior_profile(path: Path, profile: UserBehaviorProfile) -> None:
    """Append one profile to `user_behavior.jsonl` using a full-file atomic rewrite."""
    existing: list[UserBehaviorProfile] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                existing.append(UserBehaviorProfile.model_validate_json(line))
    if any(p.user_id == profile.user_id for p in existing):
        raise ValueError(f"user_behavior.jsonl already contains user_id={profile.user_id!r}")
    _rewrite_user_behavior_jsonl(path, existing + [profile])


def persist_new_cold_start_demo_user(
    processed_dir: Path,
    *,
    archetype: str,
    demographics: dict[str, Any],
    preferences: dict[str, Any],
    service_expectations: dict[str, Any],
    notes: str | None,
) -> tuple[str, UserBehaviorProfile]:
    """Append a new `cold_start` row to `test_users.json` and matching line to `user_behavior.jsonl`.

    Returns the generated `user_id` and the behaviour profile used by the review pipeline.
    If writing behaviour fails after `test_users.json` was updated, the manifest row is rolled back.
    """
    test_path = processed_dir / "test_users.json"
    behavior_path = processed_dir / "user_behavior.jsonl"
    user_id = f"api_cold_{uuid.uuid4().hex[:16]}"
    record: dict[str, Any] = {
        "user_id": user_id,
        "kind": "cold_start",
        "archetype": archetype.strip(),
        "demographics": demographics,
        "preferences": preferences,
        "service_expectations": service_expectations,
        "notes": notes,
    }
    profile = cold_start_record_to_profile(record)
    payload = load_test_users_payload(test_path)
    cold = list(payload.get("cold_start") or [])
    cold.append(record)
    payload["cold_start"] = cold
    atomic_write_json(test_path, payload)
    try:
        append_user_behavior_profile(behavior_path, profile)
    except Exception:
        cold.pop()
        payload["cold_start"] = cold
        atomic_write_json(test_path, payload)
        raise
    return user_id, profile
