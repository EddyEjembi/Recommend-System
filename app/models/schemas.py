"""Core domain schemas shared across modules."""

from typing import Any

from pydantic import BaseModel, Field


class Persona(BaseModel):
    """Distilled representation of a Yelp reviewer."""

    user_id: str
    voice: str = Field(..., description="Short prose describing tone and style.")
    preferences: list[str] = Field(default_factory=list)
    dealbreakers: list[str] = Field(default_factory=list)
    typical_length: int = Field(0, ge=0, description="Average review length in characters.")
    vocabulary_quirks: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecommendationItem(BaseModel):
    """One personalized business recommendation."""

    business_id: str = Field(..., description="Subset business id (grounded in candidates).")
    business_name: str = Field(..., description="Display name shown to clients.")
    score: float = Field(..., ge=0.0, le=1.0, description="Fit score from the LLM (0–1).")
    reason: str = Field(..., min_length=8, description="Short explanation of why this fits the user.")


class RecommendationGenerationResult(BaseModel):
    """Structured LLM output for behavioral recommendations."""

    recommendations: list[RecommendationItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Echo fields such as `user_id` and `limit`.",
    )


