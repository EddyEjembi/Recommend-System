"""Format context blocks for recommendation LLM prompts."""

from pathlib import Path

from app.persona.behavior_store import load_user_behavior_index
from app.retrieval.registry import RetrievalRegistry
from app.types.recommendation import RankedCandidate


def format_candidates_block(candidates: list[RankedCandidate]) -> str:
    """Serialize ranked candidates with scores for the LLM."""
    if not candidates:
        return "(No candidates — cannot recommend.)"
    lines: list[str] = []
    for rank, c in enumerate(candidates, start=1):
        lines.append(
            f"{rank}. id={c.business_id} | name={c.name!r} | "
            f"retrieval_score={c.retrieval_score:.3f} | "
            f"semantic={c.semantic_score:.2f} category={c.category_score:.2f} "
            f"price={c.price_score:.2f} service={c.service_score:.2f} "
            f"persona={c.persona_score:.2f} reputation={c.reputation_score:.2f} | "
            f"sources={','.join(c.sources)} | {c.summary}"
        )
    return "\n".join(lines)


def format_similar_users_block(
    registry: RetrievalRegistry,
    neighbor_ids: list[str],
    scores: list[float],
    processed_dir: Path,
    max_neighbors: int = 5,
) -> str:
    """Summarise neighbour users for cold-start grounding."""
    if not neighbor_ids:
        return "(No similar users retrieved.)"
    index = load_user_behavior_index(processed_dir / "user_behavior.jsonl")
    lines: list[str] = []
    for rank, (uid, score) in enumerate(zip(neighbor_ids, scores, strict=False), start=1):
        if rank > max_neighbors:
            break
        prof = index.get(uid)
        if prof is None:
            lines.append(f"- user {uid} (similarity={score:.3f})")
            continue
        activity = prof.activity or {}
        lines.append(
            f"- user {uid} (similarity={score:.3f}) reviews={activity.get('review_count')} "
            f"avg_stars={activity.get('avg_stars')}"
        )
    return "\n".join(lines) if lines else "(No similar users retrieved.)"
