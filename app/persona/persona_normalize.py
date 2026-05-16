"""Coerce LLM persona JSON into shapes that match `app.models.schemas.Persona`."""

from typing import Any


def _coerce_string_list(value: object) -> list[str]:
    """Turn lists, dicts, or prose into a list of short strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, dict):
        items: list[str] = []
        for key, val in value.items():
            label = str(key).strip()
            if val is None or val == "":
                if label:
                    items.append(label)
            elif isinstance(val, list):
                for entry in val:
                    text = str(entry).strip()
                    if text:
                        items.append(f"{label}: {text}")
            else:
                items.append(f"{label}: {val}")
        return items
    text = str(value).strip()
    return [text] if text else []


def normalize_persona_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Map free-form LLM keys onto the `Persona` schema before `model_validate`."""
    out = dict(data)
    meta = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
    for extra_key in ("review_priorities", "emotional_style", "rating_behavior"):
        if extra_key in out and out[extra_key] is not None:
            meta[extra_key] = out.pop(extra_key)
    out["metadata"] = meta
    out["preferences"] = _coerce_string_list(out.get("preferences"))
    out["dealbreakers"] = _coerce_string_list(out.get("dealbreakers"))
    out["vocabulary_quirks"] = _coerce_string_list(out.get("vocabulary_quirks"))
    raw_len = out.get("typical_length")
    if isinstance(raw_len, bool):
        out["typical_length"] = 0
    elif isinstance(raw_len, float):
        out["typical_length"] = max(0, int(round(raw_len)))
    elif isinstance(raw_len, int):
        out["typical_length"] = max(0, raw_len)
    else:
        out["typical_length"] = 0
    voice = out.get("voice")
    if voice is not None and not isinstance(voice, str):
        out["voice"] = str(voice)
    return out
