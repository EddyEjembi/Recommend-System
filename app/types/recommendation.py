"""Types for recommendation candidate gathering and ranking."""

from typing import Any

from pydantic import BaseModel, Field


class RankedCandidate(BaseModel):
    """A business shortlisted for the LLM with deterministic scores."""

    business_id: str
    name: str
    city: str | None = None
    state: str | None = None
    categories: str | None = None
    stars: float | None = None
    review_count: int | None = None
    price_range: int | None = None
    semantic_score: float = Field(0.0, ge=0.0, le=1.0)
    category_score: float = Field(0.0, ge=0.0, le=1.0)
    price_score: float = Field(0.0, ge=0.0, le=1.0)
    service_score: float = Field(0.0, ge=0.0, le=1.0)
    persona_score: float = Field(0.0, ge=0.0, le=1.0)
    reputation_score: float = Field(0.0, ge=0.0, le=1.0)
    retrieval_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Weighted sum used to order the candidate pool before the LLM.",
    )
    sources: list[str] = Field(default_factory=list)
    summary: str = Field("", description="Short text block for the LLM prompt.")
    metadata: dict[str, Any] = Field(default_factory=dict)
