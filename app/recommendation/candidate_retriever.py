"""Gather and rank business candidates before the LLM step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.constants import RECOMMEND_CANDIDATE_POOL_SIZE
from app.models.schemas import Persona
from app.recommendation.preference_query_builder import build_user_preference_query
from app.recommendation.ranker import score_candidate
from app.retrieval.registry import RetrievalRegistry
from app.types.behavior import BusinessBehaviorProfile, UserBehaviorProfile
from app.types.recommendation import RankedCandidate


@dataclass
class CandidateRetrieverConfig:
    """Limits for multi-source candidate gathering."""

    business_search_top_k: int = 40
    similar_user_top_k: int = 5
    review_search_top_k: int = 20
    pool_size: int = RECOMMEND_CANDIDATE_POOL_SIZE


@dataclass
class _CandidateAccumulator:
    """Merge candidate ids from several retrieval paths."""

    semantic_scores: dict[str, float] = field(default_factory=dict)
    sources: dict[str, set[str]] = field(default_factory=dict)

    def add(self, business_id: str, semantic: float, source: str) -> None:
        prev = self.semantic_scores.get(business_id, -1.0)
        if semantic > prev:
            self.semantic_scores[business_id] = semantic
        self.sources.setdefault(business_id, set()).add(source)


class CandidateRetriever:
    """Build a ranked shortlist of businesses for one user."""

    def __init__(
        self,
        registry: RetrievalRegistry,
        business_behavior: dict[str, BusinessBehaviorProfile],
        config: CandidateRetrieverConfig | None = None,
    ) -> None:
        self._registry = registry
        self._business_behavior = business_behavior
        self._config = config or CandidateRetrieverConfig()

    def gather_ranked(
        self,
        user_id: str,
        persona: Persona,
        profile: UserBehaviorProfile,
    ) -> list[RankedCandidate]:
        """Return top `pool_size` candidates ordered by retrieval_score."""
        query = build_user_preference_query(persona, profile)
        acc = _CandidateAccumulator()

        for match in self._registry.business_retriever.search_by_query(
            query,
            top_k=self._config.business_search_top_k,
        ):
            acc.add(match.id, match.score, "business_search")

        review_matches = self._registry.review_retriever.search(
            query,
            top_k=self._config.review_search_top_k,
        )
        hydrated = self._registry.review_retriever.hydrate(review_matches)
        for row in hydrated:
            bid = str(row.get("business_id") or "")
            if bid:
                acc.add(bid, float(row.get("score") or 0.0), "review_search")

        similar_users = self._collect_similar_users(user_id, profile, query)
        for neighbor_id in similar_users:
            history = self._registry.user_retriever.fetch_user_history(neighbor_id, top_k=15)
            for rev in history:
                stars = rev.get("stars")
                if stars is not None and int(stars) < 4:
                    continue
                bid = str(rev.get("business_id") or "")
                if bid:
                    acc.add(bid, 0.55, "similar_user")

        exclude = self._reviewed_business_ids(user_id)
        acc.semantic_scores = {k: v for k, v in acc.semantic_scores.items() if k not in exclude}

        ranked = self._score_all(acc, persona, profile)
        ranked.sort(key=lambda c: c.retrieval_score, reverse=True)
        pool = ranked[: self._config.pool_size]
        for c in pool:
            c.sources = sorted(acc.sources.get(c.business_id, set()))
        return pool

    def _collect_similar_users(
        self,
        user_id: str,
        profile: UserBehaviorProfile,
        query: str,
    ) -> list[str]:
        if profile.source == "cold_seed":
            matches = self._registry.find_similar_users_by_text(
                query,
                top_k=self._config.similar_user_top_k,
            )
            return [m.id for m in matches]
        matches = self._registry.user_retriever.find_similar_users(
            user_id,
            top_k=self._config.similar_user_top_k,
        )
        return [m.id for m in matches]

    def _reviewed_business_ids(self, user_id: str) -> set[str]:
        history = self._registry.user_retriever.fetch_user_history(user_id, top_k=500)
        return {str(r.get("business_id")) for r in history if r.get("business_id")}

    def _score_all(
        self,
        acc: _CandidateAccumulator,
        persona: Persona,
        profile: UserBehaviorProfile,
    ) -> list[RankedCandidate]:
        df = self._registry.businesses_df
        out: list[RankedCandidate] = []
        for business_id, sem_raw in acc.semantic_scores.items():
            hit = df[df["business_id"] == business_id]
            if hit.empty:
                continue
            row = hit.iloc[0].to_dict()
            name = str(row.get("name") or business_id)
            bb = self._business_behavior.get(business_id)
            candidate = score_candidate(
                business_id=business_id,
                name=name,
                row=row,
                semantic_raw=sem_raw,
                persona=persona,
                profile=profile,
                business_behavior=bb,
            )
            candidate.sources = sorted(acc.sources.get(business_id, set()))
            out.append(candidate)
        return out
