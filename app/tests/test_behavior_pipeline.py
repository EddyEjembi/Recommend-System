"""Tests for Phase 3 behaviour profiles and theme extraction."""

from app.behavior.business_analyzer import build_business_behavior_profile
from app.behavior.cold_user_from_seed import cold_start_record_to_profile
from app.behavior.theme_extractor import extract_themes
from app.behavior.user_analyzer import build_user_behavior_profile, detect_review_style


def test_detect_review_style_basic() -> None:
    reviews = [
        {"text": "WOW!!! This place is AMAZING??? So good.", "stars": 5.0},
        {"text": "ok", "stars": 3.0},
    ]
    style = detect_review_style(reviews)
    assert style["review_count"] == 2
    assert style["avg_chars"] > 0
    assert style["exclamations_per_1k_chars"] > 0


def test_extract_themes_praise() -> None:
    reviews = [
        {"text": "The food was delicious and the service was great.", "stars": 5.0},
        {"text": "Amazing food, best meal ever.", "stars": 5.0},
    ]
    praise, complaints = extract_themes(reviews, top_k=3)
    assert any("great_food" in t.theme for t in praise) or any("great_service" in t.theme for t in praise)


def test_build_user_behavior_profile() -> None:
    reviews = [
        {"text": "Great spot. Loved it.", "stars": 5.0, "useful": 1, "funny": 0, "cool": 0},
    ]
    row = {"user_id": "u1", "name": "Test", "review_count": 10, "average_stars": 4.5, "is_elite": False}
    p = build_user_behavior_profile("u1", reviews, user_row=row, test_archetype="tester")
    assert p.user_id == "u1"
    assert p.source == "yelp_history"
    assert p.activity["review_count"] == 1
    assert p.test_archetype == "tester"


def test_cold_start_record_to_profile() -> None:
    record = {
        "user_id": "cold_x",
        "archetype": "demo",
        "demographics": {"city": "Lagos"},
        "preferences": {"tone": "casual", "review_style": "short", "budget": "low"},
        "service_expectations": {},
        "notes": "n",
    }
    p = cold_start_record_to_profile(record)
    assert p.source == "cold_seed"
    assert p.cold_seed is not None
    assert p.cold_seed["demographics"]["city"] == "Lagos"


def test_build_business_behavior_profile() -> None:
    biz = {
        "business_id": "b1",
        "name": "Test Diner",
        "city": "Philadelphia",
        "state": "PA",
        "categories": "Restaurants, Pizza",
        "stars": 4.0,
        "review_count": 100,
        "price_range": 2,
    }
    reviews = [
        {"text": "Slow service but delicious pizza.", "stars": 3.0, "useful": 0, "funny": 0, "cool": 0},
    ]
    p = build_business_behavior_profile(biz, reviews)
    assert p.business_id == "b1"
    assert p.stats["review_count"] == 1
