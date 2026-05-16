"""Derive aggregate signals describing a business."""

from collections.abc import Iterable
from statistics import mean
from typing import Any

from app.behavior.theme_extractor import extract_themes
from app.types.behavior import BusinessBehaviorProfile


def summarise_business(reviews: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return rating distribution and engagement stats for a business."""
    materialised = list(reviews)
    if not materialised:
        return {"review_count": 0, "avg_stars": None}

    stars = [r["stars"] for r in materialised if r.get("stars") is not None]
    useful = sum(int(r.get("useful", 0) or 0) for r in materialised)
    funny = sum(int(r.get("funny", 0) or 0) for r in materialised)
    cool = sum(int(r.get("cool", 0) or 0) for r in materialised)
    return {
        "review_count": len(materialised),
        "avg_stars": mean(stars) if stars else None,
        "rating_distribution": _rating_distribution(stars),
        "total_useful": useful,
        "total_funny": funny,
        "total_cool": cool,
    }


def _rating_distribution(stars: list[float]) -> dict[int, int]:
    """Return a histogram keyed by integer star rating."""
    histogram: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for value in stars:
        bucket = int(round(value))
        if bucket in histogram:
            histogram[bucket] += 1
    return histogram


def build_business_behavior_profile(
    business_row: dict[str, Any],
    reviews: Iterable[dict[str, Any]],
    top_theme_k: int = 5,
) -> BusinessBehaviorProfile:
    """Build a full business behaviour profile from a parquet row plus its reviews."""
    materialised = list(reviews)
    stats = summarise_business(materialised)
    praise, complaints = extract_themes(materialised, top_k=top_theme_k)

    bid = str(business_row.get("business_id", ""))
    cats = business_row.get("categories")
    if cats is not None and not isinstance(cats, str):
        cats = str(cats)

    return BusinessBehaviorProfile(
        business_id=bid,
        name=business_row.get("name"),
        city=business_row.get("city"),
        state=business_row.get("state"),
        categories=cats,
        table_stars=_safe_float(business_row.get("stars")),
        table_review_count=_safe_int(business_row.get("review_count")),
        price_range=_safe_int(business_row.get("price_range")),
        stats=stats,
        praise_themes=praise,
        complaint_themes=complaints,
    )


def _safe_int(value: Any) -> int | None:
    """Parse int-like values from parquet rows."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    """Parse float-like values from parquet rows."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
