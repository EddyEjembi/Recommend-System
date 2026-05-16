"""Tests for recommendation parsing and ranking."""

import json

import pytest

from app.models.schemas import Persona
from app.recommendation.recommendation_output_parser import parse_recommendation_generation_result
from app.recommendation.ranker import score_candidate
from app.types.behavior import UserBehaviorProfile


def test_parse_recommendation_generation_result_valid() -> None:
    raw = json.dumps(
        {
            "recommendations": [
                {
                    "business_id": "biz_a",
                    "business_name": "Mama Kitchen",
                    "score": 0.91,
                    "reason": "Affordable meals and generous portions match your budget.",
                },
                {
                    "business_id": "biz_b",
                    "business_name": "Urban Grill",
                    "score": 0.84,
                    "reason": "Strong reviews for spicy local dishes at fair prices.",
                },
            ]
        }
    )
    out = parse_recommendation_generation_result(
        raw,
        allowed_business_ids={"biz_a", "biz_b"},
        name_by_id={"biz_a": "Mama Kitchen", "biz_b": "Urban Grill"},
        limit=5,
    )
    assert len(out.recommendations) == 2
    assert out.recommendations[0].business_id == "biz_a"
    assert out.recommendations[0].score == pytest.approx(0.91)


def test_parse_recommendation_rejects_unknown_business() -> None:
    raw = json.dumps(
        {
            "recommendations": [
                {
                    "business_id": "not_in_pool",
                    "business_name": "Fake Place",
                    "score": 0.9,
                    "reason": "This should not be accepted as a valid candidate.",
                }
            ]
        }
    )
    with pytest.raises(ValueError):
        parse_recommendation_generation_result(
            raw,
            allowed_business_ids={"biz_a"},
            name_by_id={"biz_a": "Real Place"},
            limit=5,
        )


def test_score_candidate_returns_bounded_retrieval_score() -> None:
    persona = Persona(
        user_id="u1",
        voice="Warm and practical.",
        preferences=["affordable", "large portions"],
        dealbreakers=["slow service"],
    )
    profile = UserBehaviorProfile(
        user_id="u1",
        source="cold_seed",
        cold_seed={
            "archetype": "budget_foodie",
            "preferences": {"budget": "low", "cuisines": ["Buka", "Local"]},
            "service_expectations": {"wait_time_tolerance": "low"},
        },
    )
    row = {
        "business_id": "b1",
        "name": "Test Buka",
        "categories": "Restaurants, Nigerian, Buka",
        "stars": 4.2,
        "review_count": 120,
        "price_range": 1,
        "city": "Lagos",
        "state": "LA",
    }
    ranked = score_candidate(
        business_id="b1",
        name="Test Buka",
        row=row,
        semantic_raw=0.72,
        persona=persona,
        profile=profile,
        business_behavior=None,
    )
    assert 0.0 <= ranked.retrieval_score <= 1.0
    assert ranked.category_score >= 0.5
