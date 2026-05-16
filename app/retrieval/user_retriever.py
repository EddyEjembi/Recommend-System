"""Retrieve historical context for a given user."""

from typing import Any

import pandas as pd

from app.retrieval.vector_store import FaissVectorStore, VectorMatch


class UserRetriever:
    """Look up a user's history, profile, and find behaviourally-similar users."""

    def __init__(
        self,
        users_df: pd.DataFrame,
        reviews_df: pd.DataFrame,
        store: FaissVectorStore,
    ) -> None:
        self._users_df = users_df
        self._reviews_df = reviews_df
        self._store = store

    def fetch_user_profile(self, user_id: str) -> dict[str, Any] | None:
        """Return the user's row from `users.parquet` as a plain dict.

        Returns `None` if the user is not in the subset (e.g. cold-start users).
        """
        mask = self._users_df["user_id"] == user_id
        if not mask.any():
            return None
        return self._users_df.loc[mask].iloc[0].to_dict()

    def fetch_user_history(self, user_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return the user's most recent reviews (parquet lookup, no vector search)."""
        df = self._reviews_df
        user_reviews = df[df["user_id"] == user_id]
        if user_reviews.empty:
            return []
        user_reviews = user_reviews.sort_values("date", ascending=False).head(top_k)
        return user_reviews.to_dict(orient="records")

    def find_similar_users(self, user_id: str, top_k: int = 5) -> list[VectorMatch]:
        """Return users with similar review patterns to `user_id`.

        Looks up the user's pre-computed vector in the index, then queries.
        Returns an empty list if the user isn't in the index (e.g. cold-start).
        """
        vec = self._store.get_vector(user_id)
        if vec is None:
            return []
        results = self._store.search(vec, top_k=top_k + 1)
        return [r for r in results if r.id != user_id][:top_k]
