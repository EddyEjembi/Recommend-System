"""Project-wide constants.

Prefer adding values here when they are referenced from multiple modules.
Single-use literals should stay close to their usage site.
"""

from typing import Final

DEFAULT_TOP_K: Final[int] = 5

USER_NAMESPACE: Final[str] = "users"
BUSINESS_NAMESPACE: Final[str] = "businesses"
REVIEW_NAMESPACE: Final[str] = "reviews"

SENTIMENT_LABELS: Final[tuple[str, ...]] = ("negative", "neutral", "positive")
STAR_RATINGS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)

# Recommendation generation (`POST /recommend`).
RECOMMEND_COMPLETION_TEMPERATURE: Final[float] = 0.55
RECOMMEND_DEFAULT_LIMIT: Final[int] = 5
RECOMMEND_MAX_LIMIT: Final[int] = 10
RECOMMEND_CANDIDATE_POOL_SIZE: Final[int] = 25
RECOMMEND_LLM_MAX_TOKENS: Final[int] = 2400

# `GET /businesses` pagination (Parquet holds thousands of rows).
BUSINESS_LIST_DEFAULT_LIMIT: Final[int] = 50
BUSINESS_LIST_MAX_LIMIT: Final[int] = 200
