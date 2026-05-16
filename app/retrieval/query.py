"""Tiny CLI for smoke-testing the retrieval indices.

Examples:

    # Find reviews semantically similar to a free-text query
    python -m app.retrieval.query reviews "long wait for jollof"

    # Find businesses similar to a given business_id
    python -m app.retrieval.query similar-business <business_id>

    # Find users similar to a given user_id (by review style)
    python -m app.retrieval.query similar-user <user_id>

    # Dump a user's profile + most recent reviews
    python -m app.retrieval.query user <user_id>

    # Dump a business's profile + top reviews
    python -m app.retrieval.query business <business_id>
"""

import argparse
import logging

from app.retrieval.registry import RetrievalRegistry

logger = logging.getLogger(__name__)


def _truncate(text: str | None, limit: int = 200) -> str:
    if not text:
        return ""
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_reviews(registry: RetrievalRegistry, query: str, top_k: int) -> None:
    matches = registry.review_retriever.search(query, top_k=top_k)
    print(f"\nTop {len(matches)} reviews for: {query!r}\n")
    if not matches:
        print("  (no matches)")
        return
    hydrated = registry.review_retriever.hydrate(matches)
    for i, row in enumerate(hydrated, start=1):
        print(
            f"  {i}. score={row['score']:.3f} | stars={row.get('stars')} | "
            f"business={row.get('business_id')} | user={row.get('user_id')}"
        )
        print(f"     {_truncate(row.get('text'))}\n")


def _handle_similar_business(
    registry: RetrievalRegistry, business_id: str, top_k: int
) -> None:
    matches = registry.business_retriever.find_similar_businesses(business_id, top_k=top_k)
    print(f"\nTop {len(matches)} businesses similar to {business_id}\n")
    if not matches:
        print("  (no matches — is the business_id in the subset?)")
        return
    for i, m in enumerate(matches, start=1):
        meta = m.metadata
        print(
            f"  {i}. score={m.score:.3f} | {meta.get('name')} "
            f"({meta.get('city')}, {meta.get('state')}) | {meta.get('categories')}"
        )


def _handle_similar_user(
    registry: RetrievalRegistry, user_id: str, top_k: int
) -> None:
    matches = registry.user_retriever.find_similar_users(user_id, top_k=top_k)
    print(f"\nTop {len(matches)} users similar to {user_id}\n")
    if not matches:
        print("  (no matches — is the user_id in the subset?)")
        return
    for i, m in enumerate(matches, start=1):
        meta = m.metadata
        print(
            f"  {i}. score={m.score:.3f} | {meta.get('name')} | "
            f"{meta.get('review_count')} reviews | avg {meta.get('average_stars')}"
        )


def _handle_user(registry: RetrievalRegistry, user_id: str, top_k: int) -> None:
    profile = registry.user_retriever.fetch_user_profile(user_id)
    if profile is None:
        print(f"\nUser {user_id} not found in subset.")
        return
    print(f"\nUser profile for {user_id}:")
    for key in ("name", "review_count", "average_stars", "is_elite", "fans", "yelping_since"):
        print(f"  {key:<16}: {profile.get(key)}")

    history = registry.user_retriever.fetch_user_history(user_id, top_k=top_k)
    print(f"\nMost recent {len(history)} reviews:")
    for i, row in enumerate(history, start=1):
        print(f"  {i}. stars={row.get('stars')} | business={row.get('business_id')} | date={row.get('date')}")
        print(f"     {_truncate(row.get('text'))}\n")


def _handle_business(
    registry: RetrievalRegistry, business_id: str, top_k: int
) -> None:
    profile = registry.business_retriever.fetch_business_profile(business_id)
    if profile is None:
        print(f"\nBusiness {business_id} not found in subset.")
        return
    print(f"\nBusiness profile for {business_id}:")
    for key in ("name", "city", "state", "categories", "stars", "review_count", "price_range"):
        print(f"  {key:<14}: {profile.get(key)}")

    reviews = registry.business_retriever.fetch_representative_reviews(business_id, top_k=top_k)
    print(f"\nTop {len(reviews)} representative reviews:")
    for i, row in enumerate(reviews, start=1):
        print(f"  {i}. stars={row.get('stars')} | user={row.get('user_id')} | date={row.get('date')}")
        print(f"     {_truncate(row.get('text'))}\n")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


HANDLERS = {
    "reviews": _handle_reviews,
    "similar-business": _handle_similar_business,
    "similar-user": _handle_similar_user,
    "user": _handle_user,
    "business": _handle_business,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "mode",
        choices=sorted(HANDLERS.keys()),
        help="What to query for.",
    )
    parser.add_argument(
        "target",
        help="The query text (for 'reviews') or the id (for everything else).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many results to show (default: 5)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,  # quiet — we print our own output
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    print("Loading retrieval registry (this takes ~5-10s for the embedder)...")
    registry = RetrievalRegistry.load()

    handler = HANDLERS[args.mode]
    handler(registry, args.target, args.top_k)


if __name__ == "__main__":
    main()
