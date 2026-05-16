"""Response models used by the API layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.schemas import RecommendationGenerationResult, RecommendationItem


class NewUserUsageHint(BaseModel):
    """How to submit a newly defined user when calling recommendation endpoints."""

    endpoint: str = Field(..., description="HTTP method and path.")
    method: str
    body_field: str = Field(..., description="Place the user object under this key in the JSON body.")
    also_required: list[str] = Field(
        default_factory=list,
        description="Other top-level fields required on the same request.",
    )
    mutually_exclusive_with: list[str] = Field(
        default_factory=list,
        description="Do not send these fields together with `new_user`.",
    )
    auth: str = Field(..., description="Authorization header requirement.")


class NewUserSchemaResponse(BaseModel):
    """JSON Schema and examples for `new_user` on `POST /recommend`."""

    description: str
    json_schema: dict[str, Any] = Field(..., description="JSON Schema (draft 2020-12) for the `new_user` object.")
    required: list[str]
    example: dict[str, Any]
    nested_field_hints: dict[str, dict[str, str]] = Field(
        ...,
        description="Common keys inside optional object fields (not enforced by validation).",
    )
    usage: NewUserUsageHint


class RecommendResponse(BaseModel):
    """Successful response for `POST /recommend`."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "user_id": "Hi10sGSZNxQH3NLyWSZ1oA",
                    "recommendations": [
                        {
                            "business_id": "Pns2l4eNsfO8kk83dixA6A",
                            "business_name": "Example Kitchen",
                            "score": 0.91,
                            "reason": "Affordable meals and generous portions match your budget.",
                        }
                    ],
                }
            ]
        }
    )

    user_id: str = Field(..., description="Resolved user id (existing or newly created).")
    recommendations: list[RecommendationItem] = Field(
        ...,
        description="Personalized recommendations from the LLM agent.",
    )

    @classmethod
    def from_result(cls, user_id: str, result: RecommendationGenerationResult) -> RecommendResponse:
        """Build API response from domain result."""
        return cls(user_id=user_id, recommendations=result.recommendations)
