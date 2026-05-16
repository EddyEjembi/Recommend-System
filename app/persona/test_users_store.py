"""Helpers for reading and updating `test_users.json` (demo user manifest)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

TestUserBucket = Literal["existing", "cold_start"]


def load_test_users_payload(path: Path) -> dict[str, Any]:
    """Load the full `test_users.json` object from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Missing test users manifest: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_test_user_bucket(payload: dict[str, Any], user_id: str) -> tuple[TestUserBucket, int]:
    """Return which list (`existing` or `cold_start`) contains `user_id` and its index.

    Raises `ValueError` if the id is not registered (unknown ids must error).
    """
    for bucket in ("existing", "cold_start"):
        rows = payload.get(bucket) or []
        for idx, row in enumerate(rows):
            if str(row.get("user_id", "")) == user_id:
                return bucket, idx
    raise ValueError(
        f"Unknown user_id={user_id!r}: not found under 'existing' or 'cold_start' in test_users.json. "
        "Add the user to that file first."
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so a crash mid-write does not corrupt `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)
