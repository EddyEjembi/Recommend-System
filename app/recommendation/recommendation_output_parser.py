"""Parse structured JSON recommendation output from the LLM."""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from app.models.schemas import RecommendationGenerationResult, RecommendationItem
from app.utils.json_utils import coerce_json_text, try_parse_json


def _load_first_json_object(raw: str) -> dict[str, Any]:
    """Parse the first JSON object from model text."""
    stripped = (raw or "").strip()
    if not stripped:
        raise ValueError("Empty model output")
    coerced = coerce_json_text(stripped)
    parsed = try_parse_json(coerced)
    if isinstance(parsed, dict):
        return parsed
    idx = stripped.find("{")
    if idx < 0:
        raise ValueError("Expected a JSON object from the model")
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(stripped, idx)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from model: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object from the model")
    return obj


def _coerce_score(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Invalid score type")
    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str) and value.strip():
        score = float(value.strip())
    else:
        raise ValueError("Missing or invalid score")
    return max(0.0, min(1.0, score))


def _normalize_name(value: str) -> str:
    """Lowercase alphanumeric name for fuzzy candidate matching."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _resolve_business_id(
    entry: dict[str, Any],
    allowed_business_ids: set[str],
    name_by_id: dict[str, str],
) -> str:
    """Map LLM entry to a candidate business_id when possible."""
    bid = str(entry.get("business_id") or entry.get("id") or "").strip()
    if bid in allowed_business_ids:
        return bid

    bname = str(entry.get("business_name") or entry.get("name") or "").strip()
    if not bname:
        return ""

    for cid, cname in name_by_id.items():
        if cname.lower() == bname.lower():
            return cid

    norm = _normalize_name(bname)
    if not norm:
        return ""

    for cid, cname in name_by_id.items():
        cnorm = _normalize_name(cname)
        if norm == cnorm or norm in cnorm or cnorm in norm:
            return cid
    return ""


def parse_recommendation_generation_result(
    raw: str,
    *,
    allowed_business_ids: set[str],
    name_by_id: dict[str, str],
    limit: int,
) -> RecommendationGenerationResult:
    """Parse and validate LLM JSON; ensure businesses come from the candidate pool."""
    data = _load_first_json_object(raw)
    raw_list = data.get("recommendations")
    if not isinstance(raw_list, list):
        raise ValueError("Missing required field: recommendations (array)")

    items: list[RecommendationItem] = []
    seen_ids: set[str] = set()

    rejected: list[str] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            rejected.append(f"non-object entry: {entry!r}")
            continue
        bid = _resolve_business_id(entry, allowed_business_ids, name_by_id)
        if not bid:
            rejected.append(str(entry))
            continue
        if bid in seen_ids:
            continue
        seen_ids.add(bid)

        canonical_name = name_by_id.get(bid, str(entry.get("business_name") or bid))
        reason = str(entry.get("reason") or "").strip()
        if len(reason) < 8:
            reason = f"Matches your preferences based on {canonical_name}'s profile and reviews."

        items.append(
            RecommendationItem(
                business_id=bid,
                business_name=canonical_name,
                score=_coerce_score(entry.get("score")),
                reason=reason,
            )
        )
        if len(items) >= limit:
            break

    if not items:
        logger.warning(
            "[recommend] parse rejected %d entries; allowed_ids sample=%s",
            len(rejected),
            list(allowed_business_ids)[:5],
        )
        print(
            "\n--- [recommend] rejected LLM entries (no matching business_id) ---\n"
            + "\n".join(rejected[:10])
            + "\n--- end rejected ---\n"
        )
        raise ValueError(
            "No valid recommendations in model output (business_id must match a candidate)."
        )

    meta: dict[str, Any] = {}
    if isinstance(data.get("metadata"), dict):
        meta = dict(data["metadata"])
    meta.pop("model", None)

    return RecommendationGenerationResult(recommendations=items, metadata=meta)
