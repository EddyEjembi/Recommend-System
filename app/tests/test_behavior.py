"""Tests for the behavioural analysis layer."""

from app.behavior.business_analyzer import summarise_business
from app.behavior.sentiment import label_from_stars
from app.behavior.user_analyzer import summarise_user_activity


def test_summarise_user_activity_empty() -> None:
    result = summarise_user_activity([])
    assert result == {"review_count": 0, "avg_stars": None, "total_useful": 0}


def test_summarise_user_activity_aggregates() -> None:
    reviews = [
        {"stars": 5.0, "useful": 2, "funny": 1, "cool": 0},
        {"stars": 3.0, "useful": 0, "funny": 0, "cool": 4},
    ]
    result = summarise_user_activity(reviews)
    assert result["review_count"] == 2
    assert result["avg_stars"] == 4.0
    assert result["total_useful"] == 2
    assert result["total_cool"] == 4


def test_summarise_business_distribution() -> None:
    reviews = [{"stars": 5.0}, {"stars": 4.0}, {"stars": 1.0}]
    result = summarise_business(reviews)
    assert result["review_count"] == 3
    assert result["rating_distribution"][5] == 1
    assert result["rating_distribution"][4] == 1
    assert result["rating_distribution"][1] == 1


def test_label_from_stars_boundaries() -> None:
    assert label_from_stars(1) == "negative"
    assert label_from_stars(2) == "negative"
    assert label_from_stars(3) == "neutral"
    assert label_from_stars(4) == "positive"
    assert label_from_stars(5) == "positive"
