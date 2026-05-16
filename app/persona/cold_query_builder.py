"""Build a single embedding-friendly string from a cold-start seed profile."""

import json

from app.types.behavior import UserBehaviorProfile


def cold_profile_to_search_query(profile: UserBehaviorProfile) -> str:
    """Turn `cold_seed` demographics + preferences into text for user-vector search.

    The user FAISS index was built from synthetic docs; this string should read
    like a short reviewer bio so nearest neighbours are semantically relevant.
    """
    cs = profile.cold_seed or {}
    demo = cs.get("demographics") or {}
    prefs = cs.get("preferences") or {}
    svc = cs.get("service_expectations") or {}

    cuisines = prefs.get("cuisines")
    if isinstance(cuisines, list):
        cuisine_str = ", ".join(str(c) for c in cuisines)
    else:
        cuisine_str = str(cuisines or "")

    parts: list[str] = [
        f"Reviewer archetype: {cs.get('archetype', '')}.",
        f"Location: {demo.get('city', '')}, {demo.get('country', '')}.",
        f"Age band: {demo.get('age_band', '')}.",
        f"Language: {demo.get('language', '')}.",
        f"Tone: {prefs.get('tone', '')}.",
        f"Review style: {prefs.get('review_style', '')}.",
        f"Budget: {prefs.get('budget', '')}.",
        f"Cuisines: {cuisine_str}.",
        f"Price sensitivity: {svc.get('price_sensitivity', '')}.",
        f"Wait tolerance: {svc.get('wait_time_tolerance', '')}.",
        str(cs.get("notes") or ""),
    ]
    return " ".join(p for p in parts if p and p != " ")


def format_neighbor_behaviors_block(
    neighbor_ids: list[str],
    scores: list[float],
    index: dict[str, UserBehaviorProfile],
    max_neighbors: int = 5,
) -> str:
    """Serialise similar users' behaviour profiles for the cold-start prompt."""
    lines: list[str] = []
    for rank, (uid, score) in enumerate(zip(neighbor_ids, scores, strict=False), start=1):
        if rank > max_neighbors:
            break
        prof = index.get(uid)
        if prof is None:
            continue
        blob = prof.model_dump(mode="json", exclude={"cold_seed"})
        lines.append(
            f"--- Neighbour {rank}: user_id={uid} (similarity={score:.4f}) ---\n"
            f"{json.dumps(blob, ensure_ascii=False, indent=2)}"
        )
    return "\n\n".join(lines) if lines else "(No neighbour behaviour profiles found in index.)"
