"""Load `business_behavior.jsonl` into an index keyed by `business_id`."""

from pathlib import Path

from app.types.behavior import BusinessBehaviorProfile


def load_business_behavior_index(path: Path) -> dict[str, BusinessBehaviorProfile]:
    """Parse every line of `business_behavior.jsonl` into profile objects."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m app.behavior.build_behavior"
        )
    index: dict[str, BusinessBehaviorProfile] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            profile = BusinessBehaviorProfile.model_validate_json(line)
            index[profile.business_id] = profile
    return index
