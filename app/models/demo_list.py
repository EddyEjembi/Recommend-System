"""Pydantic models for demo catalog endpoints (users and businesses lists)."""

from typing import Literal

from pydantic import BaseModel, Field


class TestUserListItem(BaseModel):
    """One demo user from `test_users.json` (large persona blob omitted in this listing)."""

    user_id: str = Field(..., description="Use this id with `POST /reviews`.")
    slot: Literal["existing", "cold_start"] = Field(
        ...,
        description="Which manifest list this user came from.",
    )
    has_persona: bool = Field(
        ...,
        description="Whether a persisted persona exists yet (review flow can build it).",
    )
    kind: str | None = Field(None, description="Manifest `kind` when present.")
    archetype: str | None = None
    name: str | None = None
    subset_review_count: int | None = None


class TestUsersListResponse(BaseModel):
    """All demo users merged from `existing` and `cold_start` in `test_users.json`."""

    users: list[TestUserListItem] = Field(
        ...,
        description="Flattened demo users for the UI or API client.",
    )


class BusinessListItem(BaseModel):
    """One business row from the processed subset (`businesses.parquet`)."""

    business_id: str = Field(..., description="Use with `POST /reviews`.")
    name: str | None = Field(None, description="Display name when present in the subset.")
    city: str | None = None
    state: str | None = None
    categories: list[str] | str | None = None
    stars: float | None = Field(None, description="Aggregate Yelp stars in subset when present.")
    review_count: int | None = Field(None, description="Review count in subset when present.")
    price_range: str | None = None


class BusinessesListResponse(BaseModel):
    """One page of businesses from the demo `businesses.parquet` (see query `limit` / `offset`)."""

    businesses: list[BusinessListItem] = Field(..., description="Slice for this page.")
    total_matching: int = Field(
        ...,
        ge=0,
        description="How many rows match the optional `state` / `city` filters before pagination.",
    )
    limit: int = Field(..., description="Page size used for this response.")
    offset: int = Field(..., description="Number of matching rows skipped from the sorted list.")
    state_filter: str | None = Field(None, description="Echo of `state` query when provided.")
    city_filter: str | None = Field(None, description="Echo of `city` query when provided.")
