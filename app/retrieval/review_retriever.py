"""Retrieve reviews via semantic similarity over the review namespace."""

from typing import Any

import pandas as pd

from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import FaissVectorStore, VectorMatch


class ReviewRetriever:
    """Semantic search over the corpus of reviews."""

    def __init__(
        self,
        reviews_df: pd.DataFrame,
        store: FaissVectorStore,
        embedder: Embedder,
    ) -> None:
        self._reviews_df = reviews_df
        self._store = store
        self._embedder = embedder

    def search(self, query: str, top_k: int = 5) -> list[VectorMatch]:
        """Return reviews semantically similar to `query`."""
        if not query:
            return []
        vector = self._embedder.embed(query)
        return self._store.search(vector, top_k=top_k)

    def hydrate(self, matches: list[VectorMatch]) -> list[dict[str, Any]]:
        """Join match results back to full review rows from parquet."""
        if not matches:
            return []
        ids = [m.id for m in matches]
        score_by_id = {m.id: m.score for m in matches}

        df = self._reviews_df
        hydrated = df[df["review_id"].isin(ids)].copy()
        hydrated["score"] = hydrated["review_id"].map(score_by_id)
        hydrated = hydrated.sort_values("score", ascending=False)
        return hydrated.to_dict(orient="records")
