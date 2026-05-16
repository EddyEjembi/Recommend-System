"""Keyword / phrase buckets for complaint vs praise theme mining.

These are intentionally small and regex-friendly for a hackathon baseline.
Extend with more phrases as you observe misses in qualitative review.
"""

from typing import Final

# theme_id -> list of lowercase substring needles (longest first wins in extractor)
COMPLAINT_PHRASES: Final[dict[str, list[str]]] = {
    "slow_service": [
        "slow service",
        "long wait",
        "waited forever",
        "took forever",
        "took so long",
        "forever to get",
        "never got our",
        "ignored us",
        "waitress was slow",
        "server was slow",
    ],
    "bad_service": [
        "rude",
        "unprofessional",
        "terrible service",
        "awful service",
        "bad service",
        "no service",
        "never checked",
    ],
    "price_value": [
        "overpriced",
        "too expensive",
        "not worth",
        "rip off",
        "ripoff",
        "small portions",
        "tiny portions",
        "portion size",
    ],
    "food_quality": [
        "bland",
        "cold food",
        "undercooked",
        "overcooked",
        "dry",
        "stale",
        "gross",
        "disgusting",
        "worst meal",
        "bad food",
    ],
    "cleanliness": [
        "dirty",
        "filthy",
        "sticky",
        "smelled",
        "smelly",
        "flies",
        "roach",
    ],
    "noise_crowd": [
        "too loud",
        "noisy",
        "crowded",
        "packed",
        "couldn't hear",
    ],
}

PRAISE_PHRASES: Final[dict[str, list[str]]] = {
    "great_food": [
        "delicious",
        "amazing food",
        "best meal",
        "great food",
        "tasty",
        "flavorful",
        "fresh",
        "perfectly cooked",
    ],
    "great_service": [
        "great service",
        "excellent service",
        "friendly staff",
        "attentive",
        "our server was",
        "welcoming",
    ],
    "good_value": [
        "great value",
        "good value",
        "affordable",
        "reasonable price",
        "generous portions",
        "huge portions",
        "worth it",
    ],
    "ambience": [
        "cozy",
        "nice atmosphere",
        "great atmosphere",
        "beautiful decor",
        "vibe",
    ],
    "fast_experience": [
        "quick service",
        "fast service",
        "came out quickly",
        "right away",
    ],
}

THEME_LABELS: Final[dict[str, str]] = {
    "slow_service": "Slow service / long waits",
    "bad_service": "Poor or rude service",
    "price_value": "Price, portions, value",
    "food_quality": "Food quality issues",
    "cleanliness": "Cleanliness / hygiene",
    "noise_crowd": "Noise / crowding",
    "great_food": "Food quality praise",
    "great_service": "Service praise",
    "good_value": "Value / portions praise",
    "ambience": "Atmosphere / ambience",
    "fast_experience": "Speed / efficiency praise",
}
