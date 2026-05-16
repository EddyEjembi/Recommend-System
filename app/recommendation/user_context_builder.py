"""Format user behaviour and history blocks for recommendation prompts."""

import json
from typing import Any

from app.types.behavior import UserBehaviorProfile


def format_user_behavior_json(profile: UserBehaviorProfile) -> str:
    """Serialise a user behaviour profile for the prompt."""
    return json.dumps(profile.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=2)


def format_user_history_block(reviews: list[dict[str, Any]], max_snippet: int = 450) -> str:
    """Format recent review dicts into a bounded verbatim block."""
    if not reviews:
        return "(No review history in subset for this user.)"
    parts: list[str] = []
    for r in reviews:
        text = str(r.get("text") or "").strip().replace("\n", " ")
        if not text:
            continue
        if len(text) > max_snippet:
            text = text[: max_snippet - 3] + "..."
        parts.append(
            f"[stars={r.get('stars')} date={r.get('date')} biz={r.get('business_id')}] {text}"
        )
    return "\n\n".join(parts) if parts else "(No non-empty review texts.)"
