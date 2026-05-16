"""Tests for recommendation prompt assembly."""

from app.models.schemas import Persona
from app.recommendation.recommendation_prompt_builder import build_recommendation_json_prompt


def test_build_recommendation_json_prompt_formats_without_key_error() -> None:
    persona = Persona(user_id="u1", voice="Warm and practical.")
    system, user = build_recommendation_json_prompt(
        persona=persona,
        user_behavior_json="{}",
        user_history_block="(none)",
        similar_users_block="(none)",
        candidates_block="1. id=b1 | name='Test'",
        limit=3,
    )
    assert "recommendations" in system
    assert "Return exactly 3 items" in system
    assert "Recommend the best 3 businesses" in user
