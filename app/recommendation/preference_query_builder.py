"""Build embedding-friendly preference text from persona and behaviour."""

from app.models.schemas import Persona
from app.persona.cold_query_builder import cold_profile_to_search_query
from app.types.behavior import UserBehaviorProfile


def build_user_preference_query(persona: Persona, profile: UserBehaviorProfile) -> str:
    """Combine persona and behaviour into one string for vector search."""
    if profile.source == "cold_seed":
        return cold_profile_to_search_query(profile)

    prefs = "; ".join(persona.preferences) if persona.preferences else ""
    deal = "; ".join(persona.dealbreakers) if persona.dealbreakers else ""
    features = profile.features or {}
    sentiment = features.get("sentiment_mix") or features.get("sentiment") or {}
    sentiment_str = ""
    if isinstance(sentiment, dict):
        sentiment_str = ", ".join(f"{k}:{v}" for k, v in sentiment.items())

    activity = profile.activity or {}
    avg_stars = activity.get("avg_stars")
    archetype = profile.test_archetype or ""

    parts: list[str] = [
        f"Reviewer voice: {persona.voice}",
        f"Preferences: {prefs}",
        f"Dealbreakers: {deal}",
        f"Archetype: {archetype}",
        f"Average stars given: {avg_stars}",
        f"Sentiment mix: {sentiment_str}",
        f"Typical review length: {persona.typical_length} chars",
    ]
    meta = profile.yelp_user_meta or {}
    if meta:
        parts.append(f"Yelp user meta: {meta.get('name', '')} {meta.get('review_count', '')}")

    return " ".join(p for p in parts if p and not p.endswith(": "))
