"""Application-wide prompt and template strings."""

from app.templates.persona import (
    PERSONA_COLD_USER_TEMPLATE,
    PERSONA_SYSTEM_PROMPT,
    PERSONA_WARM_USER_TEMPLATE,
)

__all__ = [
    "PERSONA_COLD_USER_TEMPLATE",
    "PERSONA_SYSTEM_PROMPT",
    "PERSONA_WARM_USER_TEMPLATE",
]
