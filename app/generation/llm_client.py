"""Thin abstraction over the LLM backend used for generation."""

from typing import Protocol


class LLMClient(Protocol):
    """Interface implemented by any concrete LLM provider client."""

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> str:
        """Return a single completion for the given prompts."""


class NotImplementedLLMClient:
    """Placeholder client used until a concrete provider is wired in."""

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> str:
        """Raise to signal the LLM backend is not configured."""
        raise NotImplementedError("Configure a concrete LLMClient implementation")
