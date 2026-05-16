"""Compose system and user prompts for recommendation generation."""

import json

from app.models.schemas import Persona
from app.templates.recommendation import RECOMMEND_JSON_SYSTEM_PROMPT, RECOMMEND_JSON_USER_TEMPLATE


def format_persona_block(persona: Persona) -> str:
    """Render a `Persona` as a stable block for the LLM prompt."""
    lines = [
        f"User id: {persona.user_id}",
        f"Voice & tone: {persona.voice}",
        f"Typical review length (chars): {persona.typical_length}",
    ]
    if persona.preferences:
        lines.append("Preferences: " + "; ".join(persona.preferences))
    if persona.dealbreakers:
        lines.append("Dealbreakers: " + "; ".join(persona.dealbreakers))
    if persona.vocabulary_quirks:
        lines.append("Vocabulary quirks: " + "; ".join(persona.vocabulary_quirks))
    if persona.metadata:
        lines.append("Extra metadata (JSON): " + json.dumps(persona.metadata, ensure_ascii=False))
    return "\n".join(lines)


def build_recommendation_json_prompt(
    persona: Persona,
    user_behavior_json: str,
    user_history_block: str,
    similar_users_block: str,
    candidates_block: str,
    limit: int,
) -> tuple[str, str]:
    """Return `(system_prompt, user_prompt)` for JSON recommendations."""
    persona_block = format_persona_block(persona)
    system = RECOMMEND_JSON_SYSTEM_PROMPT.format(limit=limit).strip()
    user_prompt = RECOMMEND_JSON_USER_TEMPLATE.format(
        persona_block=persona_block,
        user_behavior_json=user_behavior_json,
        user_history_block=user_history_block,
        similar_users_block=similar_users_block,
        candidates_block=candidates_block,
        limit=limit,
    )
    return system, user_prompt.strip()
