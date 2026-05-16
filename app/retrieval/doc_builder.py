"""Build the textual documents we feed into the embedder.

Each namespace has its own document format:

  - reviews   : raw review text (no synthesis needed)
  - businesses: synthesized profile string (name, location, categories, stats)
  - users     : synthesized stylistic profile + a few sampled review texts

"""

from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------


def build_review_doc(row: pd.Series | dict[str, Any]) -> str:
    """Return the raw review text, falling back to a placeholder if empty."""
    text = _safe_str(_get(row, "text"))
    return text if text else "Empty review."


# ---------------------------------------------------------------------------
# Businesses
# ---------------------------------------------------------------------------


def build_business_doc(row: pd.Series | dict[str, Any]) -> str:
    """Synthesize a metadata profile string for a business."""
    parts: list[str] = []

    name = _safe_str(_get(row, "name")) or "Unknown business"
    parts.append(name)

    categories = _safe_str(_get(row, "categories"))
    if categories:
        parts.append(f"Categories: {categories}")

    city = _safe_str(_get(row, "city"))
    state = _safe_str(_get(row, "state"))
    location = ", ".join([bit for bit in (city, state) if bit])
    if location:
        parts.append(f"Location: {location}")

    stars = _get(row, "stars")
    review_count = _get(row, "review_count")
    if pd.notna(stars):
        rc = int(review_count) if pd.notna(review_count) else 0
        parts.append(f"Average rating: {float(stars):.1f}/5 over {rc} reviews")

    price_range = _get(row, "price_range")
    if pd.notna(price_range):
        try:
            dollars = "$" * int(price_range)
            parts.append(f"Price range: {dollars}")
        except (TypeError, ValueError):
            pass

    return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def build_user_doc(
    row: pd.Series | dict[str, Any],
    sampled_reviews: list[str] | None = None,
    sample_char_limit: int = 300,
) -> str:
    """Synthesize a stylistic profile + concatenated review samples.

    The profile line captures behavioural signal (avg stars, elite status,
    review volume). The sampled reviews capture voice / vocabulary. Both
    together form a single string that the embedder collapses to one vector.
    """
    parts: list[str] = []

    avg_stars = _get(row, "average_stars")
    if pd.notna(avg_stars):
        avg_stars_f = float(avg_stars)
        if avg_stars_f >= 4.0:
            tone = "tends positive"
        elif avg_stars_f <= 2.5:
            tone = "harshly critical"
        else:
            tone = "balanced"
        parts.append(f"Reviewer who is {tone} (avg {avg_stars_f:.1f}/5)")

    review_count = _get(row, "review_count")
    if pd.notna(review_count):
        parts.append(f"{int(review_count)} lifetime reviews")

    if bool(_get(row, "is_elite", False)):
        parts.append("Elite Yelp member")

    fans = _get(row, "fans")
    if pd.notna(fans) and int(fans) > 0:
        parts.append(f"{int(fans)} fans")

    profile_line = ". ".join(parts) + "." if parts else "Reviewer profile unknown."

    snippets = [
        _safe_str(t)[:sample_char_limit]
        for t in (sampled_reviews or [])
        if _safe_str(t)
    ]
    if snippets:
        joined = " | ".join(snippets)
        return f"{profile_line} Sample reviews: {joined}"
    return profile_line


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get(row: pd.Series | dict[str, Any], key: str, default: Any = None) -> Any:
    """Uniform `.get`-style access across pandas Series and plain dicts."""
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return row.get(key, default)


def _safe_str(value: Any) -> str:
    """Return a stripped string, or '' for None / NaN values."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()
