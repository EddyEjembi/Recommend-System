"""Tests for OpenAI client env/header resolution."""

import pytest

from app.generation.openai_client import resolve_api_key, resolve_model


def test_resolve_model_ignores_empty_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_MODEL", "")
    monkeypatch.setenv("LLM_MODEL", "")
    assert resolve_model() == "gpt-4o-mini"


def test_resolve_api_key_prefers_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_API_KEY", "env-key")
    assert resolve_api_key(header_key="header-key", allow_env_fallback=True) == "header-key"


def test_resolve_api_key_uses_env_when_no_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_API_KEY", "env-key")
    assert resolve_api_key(header_key=None, allow_env_fallback=True) == "env-key"


def test_resolve_api_key_openai_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env")
    assert resolve_api_key(header_key=None, allow_env_fallback=True) == "openai-env"
