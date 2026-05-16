"""End-to-end recommendation: retrieval, ranking, LLM JSON output."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

from app.core.constants import (
    RECOMMEND_COMPLETION_TEMPERATURE,
    RECOMMEND_DEFAULT_LIMIT,
    RECOMMEND_LLM_MAX_TOKENS,
)
from app.generation.business_behavior_store import load_business_behavior_index
from app.generation.llm_client import LLMClient, NotImplementedLLMClient
from app.generation.openai_client import OpenAILLMClient
from app.recommendation.user_context_builder import format_user_behavior_json, format_user_history_block
from app.models.schemas import Persona, RecommendationGenerationResult
from app.persona.behavior_store import load_user_behavior_index
from app.persona.ensure_persona import ensure_persona_for_test_user
from app.persona.test_users_store import find_test_user_bucket, load_test_users_payload
from app.recommendation.candidate_retriever import CandidateRetriever, CandidateRetrieverConfig
from app.recommendation.recommendation_context_builder import (
    format_candidates_block,
    format_similar_users_block,
)
from app.recommendation.recommendation_output_parser import parse_recommendation_generation_result
from app.recommendation.recommendation_prompt_builder import build_recommendation_json_prompt
from app.recommendation.preference_query_builder import build_user_preference_query
from app.retrieval.registry import RetrievalRegistry, default_embeddings_dir, default_processed_dir
from app.types.behavior import BusinessBehaviorProfile, UserBehaviorProfile


@dataclass
class RecommendGenConfig:
    """Tuning knobs for candidate pool and LLM."""

    candidate_config: CandidateRetrieverConfig | None = None
    user_history_limit: int = 12
    max_llm_tokens: int = RECOMMEND_LLM_MAX_TOKENS


class RecommendationService:
    """Builds personalized recommendations for a registered or cold-start user."""

    def __init__(
        self,
        llm: LLMClient,
        registry: RetrievalRegistry,
        user_behavior: dict[str, UserBehaviorProfile],
        business_behavior: dict[str, BusinessBehaviorProfile],
        processed_dir: Path,
        embeddings_dir: Path,
        config: RecommendGenConfig | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._user_behavior = user_behavior
        self._business_behavior = business_behavior
        self._processed_dir = processed_dir
        self._embeddings_dir = embeddings_dir
        self._config = config or RecommendGenConfig()
        self._candidate_retriever = CandidateRetriever(
            registry,
            business_behavior,
            config=self._config.candidate_config,
        )

    def register_runtime_user_profile(self, profile: UserBehaviorProfile) -> None:
        """Insert or replace a behaviour profile after API cold-start registration."""
        self._user_behavior[profile.user_id] = profile

    @classmethod
    def from_default_paths(
        cls,
        processed_dir: Path | None = None,
        embeddings_dir: Path | None = None,
        llm: LLMClient | None = None,
        config: RecommendGenConfig | None = None,
    ) -> RecommendationService:
        """Load Parquet, FAISS, and behaviour indices."""
        proc = processed_dir or default_processed_dir()
        emb = embeddings_dir or default_embeddings_dir()
        ub = load_user_behavior_index(proc / "user_behavior.jsonl")
        bb = load_business_behavior_index(proc / "business_behavior.jsonl")
        reg = RetrievalRegistry.load(processed_dir=proc, embeddings_dir=emb)
        llm_client: LLMClient = llm if llm is not None else NotImplementedLLMClient()
        return cls(llm_client, reg, ub, bb, proc, emb, config)

    def generate(
        self,
        user_id: str,
        *,
        limit: int = RECOMMEND_DEFAULT_LIMIT,
        temperature: float = RECOMMEND_COMPLETION_TEMPERATURE,
        llm: LLMClient | None = None,
    ) -> RecommendationGenerationResult:
        """Return ranked recommendations with LLM-generated reasons."""
        ensure_persona_for_test_user(
            user_id,
            self._processed_dir,
            self._embeddings_dir,
            llm=llm,
        )

        test_path = self._processed_dir / "test_users.json"
        payload = load_test_users_payload(test_path)
        bucket, index = find_test_user_bucket(payload, user_id)
        persona = Persona.model_validate(payload[bucket][index]["persona"])

        ub = self._user_behavior.get(user_id)
        if ub is None:
            raise KeyError(
                f"No user_behaviour row for user_id={user_id!r}. Re-run `python -m app.behavior.build_behavior`."
            )

        candidates = self._candidate_retriever.gather_ranked(user_id, persona, ub)
        if not candidates:
            raise ValueError(
                f"No candidate businesses found for user_id={user_id!r}. Check embeddings and behaviour data."
            )
        logger.info(
            "[recommend] user_id=%s limit=%d candidate_pool=%d ids=%s",
            user_id,
            limit,
            len(candidates),
            [c.business_id for c in candidates[:8]],
        )

        history = self._registry.user_retriever.fetch_user_history(
            user_id,
            top_k=self._config.user_history_limit,
        )
        query = build_user_preference_query(persona, ub)
        if ub.source == "cold_seed":
            neighbor_matches = self._registry.find_similar_users_by_text(query, top_k=5)
        else:
            neighbor_matches = self._registry.user_retriever.find_similar_users(user_id, top_k=5)

        neighbor_ids = [m.id for m in neighbor_matches]
        neighbor_scores = [m.score for m in neighbor_matches]

        user_behavior_json = format_user_behavior_json(ub)
        user_history_block = format_user_history_block(history)
        similar_users_block = format_similar_users_block(
            self._registry,
            neighbor_ids,
            neighbor_scores,
            self._processed_dir,
        )
        candidates_block = format_candidates_block(candidates)

        system_prompt, user_prompt = build_recommendation_json_prompt(
            persona=persona,
            user_behavior_json=user_behavior_json,
            user_history_block=user_history_block,
            similar_users_block=similar_users_block,
            candidates_block=candidates_block,
            limit=limit,
        )

        completion_client = llm if llm is not None else self._llm
        if isinstance(completion_client, NotImplementedLLMClient):
            raise ValueError(
                "No LLM configured. Pass Authorization: Bearer <api_key> on POST /recommend."
            )
        json_mode = isinstance(completion_client, OpenAILLMClient)
        raw = completion_client.complete(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=self._config.max_llm_tokens,
            json_mode=json_mode,
        )
        print(f"\n--- [recommend] LLM raw response (user_id={user_id}) ---\n{raw}\n--- end LLM response ---\n")
        logger.info("[recommend] LLM raw response (user_id=%s):\n%s", user_id, raw)

        allowed_ids = {c.business_id for c in candidates}
        name_by_id = {c.business_id: c.name for c in candidates}
        result = parse_recommendation_generation_result(
            raw,
            allowed_business_ids=allowed_ids,
            name_by_id=name_by_id,
            limit=limit,
        )
        result.metadata["user_id"] = user_id
        result.metadata["limit"] = limit
        result.metadata["candidate_count"] = len(candidates)
        return result
