"""Orchestrate LLM persona generation for warm and cold-start users."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.generation.llm_client import LLMClient
from app.generation.openai_client import OpenAILLMClient
from app.models.schemas import Persona
from app.persona.behavior_store import load_user_behavior_index
from app.persona.persona_normalize import normalize_persona_payload
from app.persona.cold_query_builder import (
    cold_profile_to_search_query,
    format_neighbor_behaviors_block,
)
from app.persona.prompt_templates import (
    PERSONA_COLD_USER_TEMPLATE,
    PERSONA_SYSTEM_PROMPT,
    PERSONA_WARM_USER_TEMPLATE,
)
from app.persona.test_users_store import TestUserBucket
from app.retrieval.registry import RetrievalRegistry, default_embeddings_dir, default_processed_dir
from app.types.behavior import UserBehaviorProfile
from app.utils.json_utils import parse_json_object
class PersonaBuildConfig:
    """Tuning knobs for persona prompts and neighbour retrieval."""

    history_review_limit: int = 15
    max_review_chars_per_snippet: int = 900
    cold_neighbor_top_k: int = 8
    cold_neighbor_behavior_limit: int = 5
    temperature: float = 0.65
    max_tokens: int = 1100


class PersonaBuildService:
    """Load behaviour + retrieval context and call the LLM once per build."""

    def __init__(
        self,
        llm: LLMClient,
        registry: RetrievalRegistry,
        behavior_by_user: dict[str, UserBehaviorProfile],
        config: PersonaBuildConfig | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._behavior = behavior_by_user
        self._config = config or PersonaBuildConfig()

    @classmethod
    def from_default_paths(
        cls,
        llm: LLMClient | None = None,
        processed_dir: Path | None = None,
        embeddings_dir: Path | None = None,
        config: PersonaBuildConfig | None = None,
    ) -> PersonaBuildService:
        """Load Parquet, FAISS indices, `user_behavior.jsonl`, and default OpenAI client."""
        processed = processed_dir or default_processed_dir()
        embeddings = embeddings_dir or default_embeddings_dir()
        behavior_path = processed / "user_behavior.jsonl"
        behavior = load_user_behavior_index(behavior_path)
        registry = RetrievalRegistry.load(
            processed_dir=processed,
            embeddings_dir=embeddings,
        )
        llm_client: LLMClient = llm if llm is not None else OpenAILLMClient()
        return cls(llm_client, registry, behavior, config)

    def build(self, user_id: str) -> Persona:
        """Build a `Persona` for `user_id` using warm or cold pipeline."""
        profile = self._behavior.get(user_id)
        if profile is None:
            raise KeyError(
                f"No behaviour profile for user_id={user_id!r}. "
                "Run `python -m app.behavior.build_behavior` first, or check the id."
            )
        if profile.source == "cold_seed":
            return self._build_cold(user_id, profile)
        return self._build_warm(user_id, profile)

    def build_for_test_slot(self, user_id: str, bucket: TestUserBucket) -> Persona:
        """Build persona using warm vs cold pipeline based on `test_users.json` list membership.

        The manifest bucket (`existing` vs `cold_start`) is authoritative for routing, not
        only `UserBehaviorProfile.source` (keeps behaviour aligned with your demo file).
        """
        profile = self._behavior.get(user_id)
        if profile is None:
            raise KeyError(
                f"No behaviour profile for user_id={user_id!r} in user_behavior.jsonl. "
                "Run `python -m app.behavior.build_behavior` after adding the user to test_users."
            )
        if bucket == "cold_start":
            return self._build_cold(user_id, profile)
        return self._build_warm(user_id, profile)

    def _build_warm(self, user_id: str, profile: UserBehaviorProfile) -> Persona:
        """Use review history + behaviour JSON for users in the Yelp subset."""
        reviews = self._registry.user_retriever.fetch_user_history(
            user_id,
            top_k=self._config.history_review_limit,
        )
        reviews_block = self._format_reviews_block(reviews)
        behavior_json = json.dumps(
            profile.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            indent=2,
        )
        user_prompt = PERSONA_WARM_USER_TEMPLATE.format(
            user_id=user_id,
            behavior_json=behavior_json,
            reviews_block=reviews_block or "(No review text in subset for this user.)",
        )
        raw = self._llm_complete(user_prompt)
        extra: dict[str, Any] = {}
        if profile.test_archetype:
            extra["test_archetype"] = profile.test_archetype
        return self._parse_persona(raw, user_id, source="warm", extra_meta=extra)

    def _build_cold(self, user_id: str, profile: UserBehaviorProfile) -> Persona:
        """Embed seed text, pull similar indexed users, merge neighbour behaviour into prompt."""
        query = cold_profile_to_search_query(profile)
        matches = self._registry.find_similar_users_by_text(
            query,
            top_k=self._config.cold_neighbor_top_k,
        )
        neighbor_ids = [m.id for m in matches]
        scores = [m.score for m in matches]
        neighbors_block = format_neighbor_behaviors_block(
            neighbor_ids,
            scores,
            self._behavior,
            max_neighbors=self._config.cold_neighbor_behavior_limit,
        )
        cold_seed_json = json.dumps(
            profile.cold_seed or {},
            ensure_ascii=False,
            indent=2,
        )
        user_prompt = PERSONA_COLD_USER_TEMPLATE.format(
            user_id=user_id,
            cold_seed_json=cold_seed_json,
            neighbors_block=neighbors_block,
        )
        raw = self._llm_complete(user_prompt)
        lim = self._config.cold_neighbor_behavior_limit
        extra_meta: dict[str, Any] = {
            "neighbor_user_ids": neighbor_ids[:lim],
            "neighbor_scores": scores[:lim],
            "cold_search_query_excerpt": query[:500],
        }
        if profile.test_archetype:
            extra_meta["test_archetype"] = profile.test_archetype
        return self._parse_persona(raw, user_id, source="cold", extra_meta=extra_meta)

    def _llm_complete(self, user_prompt: str) -> str:
        """Call the LLM; enable OpenAI JSON mode when the client supports it."""
        if isinstance(self._llm, OpenAILLMClient):
            return self._llm.complete(
                PERSONA_SYSTEM_PROMPT,
                user_prompt,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                json_mode=True,
            )
        return self._llm.complete(
            PERSONA_SYSTEM_PROMPT,
            user_prompt,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            json_mode=False,
        )

    def _format_reviews_block(self, reviews: list[dict[str, Any]]) -> str:
        """Join recent reviews into a bounded prompt block."""
        parts: list[str] = []
        cap = self._config.max_review_chars_per_snippet
        for r in reviews:
            text = str(r.get("text") or "").strip().replace("\n", " ")
            if not text:
                continue
            if len(text) > cap:
                text = text[: cap - 3] + "..."
            parts.append(
                f"stars={r.get('stars')} date={r.get('date')} "
                f"business={r.get('business_id')}: {text}"
            )
        return "\n\n".join(parts)

    def _parse_persona(self, raw: str, user_id: str, source: str, extra_meta: dict[str, Any]) -> Persona:
        """Validate LLM JSON into a `Persona`, forcing `user_id` and merging metadata."""
        data = parse_json_object(raw)
        if not isinstance(data, dict):
            raise ValueError("LLM JSON root must be an object")
        data["user_id"] = user_id
        meta = data.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        meta["persona_source"] = source
        meta.update(extra_meta)
        data["metadata"] = meta
        return Persona.model_validate(normalize_persona_payload(data))
