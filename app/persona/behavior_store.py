"""Load `user_behavior.jsonl` into an in-memory index keyed by `user_id`."""

from pathlib import Path

from app.types.behavior import UserBehaviorProfile


def load_user_behavior_index(path: Path) -> dict[str, UserBehaviorProfile]:
    """Parse every line of `user_behavior.jsonl` into `UserBehaviorProfile` objects."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m app.behavior.build_behavior"
        )
    index: dict[str, UserBehaviorProfile] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            profile = UserBehaviorProfile.model_validate_json(line)
            index[profile.user_id] = profile
    return index


def get_profile(index: dict[str, UserBehaviorProfile], user_id: str) -> UserBehaviorProfile | None:
    """Return the profile for `user_id`, or `None` if absent."""
    return index.get(user_id)


def append_user_behavior_line(path: Path, profile: UserBehaviorProfile) -> None:
    """Append one JSON object as a single line to `user_behavior.jsonl` via atomic rewrite."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[str] = []
    if path.exists():
        existing = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for ln in existing:
        if UserBehaviorProfile.model_validate_json(ln).user_id == profile.user_id:
            raise ValueError(f"user_id={profile.user_id!r} already exists in {path}")
    existing.append(profile.model_dump_json())
    body = "\n".join(existing) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
