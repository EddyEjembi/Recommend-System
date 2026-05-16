"""Structured outputs for Phase 3 behavioural analysis (no LLM)."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ThemeSignal(BaseModel):
    """A single extracted theme with a normalised score and optional evidence."""

    theme: str = Field(..., description="Stable snake_case theme id.")
    label: str = Field(..., description="Human-readable label for prompts.")
    score: float = Field(..., ge=0.0, description="Relative strength in [0, ~1+] after normalisation.")
    hit_count: int = Field(..., ge=0, description="How many reviews contributed a hit.")
    snippets: list[str] = Field(
        default_factory=list,
        description="Short verbatim spans (not generated) for grounding.",
    )


class UserBehaviorProfile(BaseModel):
    """Deterministic user behaviour derived from reviews or cold-start seed."""

    user_id: str
    source: Literal["yelp_history", "cold_seed"] = Field(
        ...,
        description="Where the profile came from: subset reviews vs test_users.json seed.",
    )
    activity: dict[str, Any] = Field(default_factory=dict, description="Counts and star aggregates.")
    style: dict[str, Any] = Field(default_factory=dict, description="Writing-style statistics.")
    features: dict[str, Any] = Field(
        default_factory=dict,
        description="Sentiment mix, length stats, etc.",
    )
    yelp_user_meta: dict[str, Any] | None = Field(
        default=None,
        description="Subset of users.parquet row when source is yelp_history.",
    )
    cold_seed: dict[str, Any] | None = Field(
        default=None,
        description="Demographics / preferences when source is cold_seed.",
    )
    test_archetype: str | None = Field(
        default=None,
        description="If this user appears in test_users.json existing list.",
    )


class BusinessBehaviorProfile(BaseModel):
    """Deterministic business context: stats + complaint/praise themes."""

    business_id: str
    name: str | None = None
    city: str | None = None
    state: str | None = None
    categories: str | None = None
    table_stars: float | None = Field(
        default=None,
        description="Stars field from businesses.parquet (Yelp aggregate).",
    )
    table_review_count: int | None = Field(
        default=None,
        description="review_count from businesses.parquet.",
    )
    price_range: int | None = None
    stats: dict[str, Any] = Field(default_factory=dict, description="From-subset review aggregates.")
    praise_themes: list[ThemeSignal] = Field(default_factory=list)
    complaint_themes: list[ThemeSignal] = Field(default_factory=list)
