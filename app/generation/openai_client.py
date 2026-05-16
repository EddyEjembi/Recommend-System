"""OpenAI-backed implementation of `LLMClient` for persona and recommendation generation."""

import os
from typing import Any, Final

from openai import APIStatusError, AuthenticationError, OpenAI

_DEFAULT_MODEL: Final[str] = "gpt-4o-mini"


def resolve_api_key(*, header_key: str | None = None, allow_env_fallback: bool = True) -> str:
    """Prefer Bearer token; use env only when header is missing and fallback is allowed."""
    explicit = (header_key or "").strip()
    if explicit:
        return explicit
    if not allow_env_fallback:
        return ""
    return (os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()


def resolve_model(explicit: str | None = None) -> str:
    """Return a non-empty model id (empty env vars are ignored)."""
    if explicit is not None:
        stripped = explicit.strip()
        if stripped:
            return stripped
    for env_name in ("AI_MODEL", "LLM_MODEL"):
        value = (os.getenv(env_name) or "").strip()
        if value:
            return value
    return _DEFAULT_MODEL


def resolve_base_url() -> str | None:
    """Optional OpenAI-compatible base URL (e.g. OpenRouter)."""
    url = (os.getenv("AI_BASE_URL") or "").strip()
    return url or None


class OpenAILLMClient:
    """Thin wrapper around the OpenAI Chat Completions API (default host: api.openai.com)."""

    def __init__(
        self,
        model: str | None = None,
        client: OpenAI | None = None,
        api_key: str | None = None,
        *,
        allow_env_fallback: bool = True,
    ) -> None:
        """Build the SDK client.

        When `api_key` is a non-empty string, **only** that key is used (`allow_env_fallback` is ignored).
        When `api_key` is omitted or blank and `allow_env_fallback` is True, uses `AI_API_KEY` then
        `OPENAI_API_KEY` from the process environment.
        When `api_key` is blank and `allow_env_fallback` is False, raises `ValueError`.
        """
        if client is not None:
            self._client = client
        else:
            resolved_key = resolve_api_key(
                header_key=api_key,
                allow_env_fallback=allow_env_fallback,
            )
            if not resolved_key:
                raise ValueError(
                    "No API key provided. Send Authorization: Bearer <key> on POST /recommend, "
                    "or set AI_API_KEY / OPENAI_API_KEY in the environment."
                )
            base_url = resolve_base_url()
            client_kwargs: dict[str, object] = {"api_key": resolved_key, "max_retries": 5}
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = OpenAI(**client_kwargs)
        self._model = resolve_model(model)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> str:
        """Return the assistant message content for a single turn."""
        kwargs: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            #"extra_body": {"chat_template_kwargs":{"thinking":False}}
        }
        #print(f"Going to call OpenAI with kwargs: {kwargs}")
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            if exc.status_code == 502:
                raise RuntimeError(
                    "LLM provider returned HTTP 502 Bad Gateway. Retry in a few minutes."
                ) from exc
            if exc.status_code == 503:
                raise RuntimeError(
                    "LLM provider returned HTTP 503 Service Unavailable. Retry shortly."
                ) from exc
            if exc.status_code == 504:
                raise RuntimeError(
                    "LLM provider returned HTTP 504 Gateway Timeout. Retry or lower max_tokens."
                ) from exc
            raise
        choice = response.choices[0]
        content = choice.message.content
        if content is None:
            return ""
        return content
