"""Tolerant JSON helpers used when parsing LLM output and dataset files."""

import re
from pathlib import Path
from typing import Any
import json


def read_json(path: Path) -> Any:
    """Read a JSON document and return the parsed object."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Write `payload` as JSON to `path`, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=indent)


def coerce_json_text(raw: str) -> str:
    """Strip optional ```json fences so `json.loads` can parse LLM output."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def parse_json_object(raw: str) -> Any:
    """Parse a JSON object from raw LLM text, raising `ValueError` on failure."""
    text = coerce_json_text(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from model: {exc}") from exc


def try_parse_json(text: str) -> Any | None:
    """Return the parsed JSON, or `None` if `text` is not valid JSON."""
    try:
        return parse_json_object(text)
    except ValueError:
        return None
