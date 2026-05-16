"""Prompt templates for Phase 4 persona generation (LLM).

Canonical strings live in `app.templates.persona`; this module re-exports them so
callers can keep importing from `app.persona.prompt_templates`.
"""

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
