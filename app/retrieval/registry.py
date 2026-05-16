"""Wires up everything needed for online retrieval.

`RetrievalRegistry.load()` reads the three Parquet files plus the three FAISS
indices and instantiates the embedder + retrievers. Callers (CLI, API, tests)
should rely on this rather than constructing components by hand.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.retrieval.business_retriever import BusinessRetriever
from app.retrieval.embedder import Embedder, SentenceTransformerEmbedder
from app.retrieval.review_retriever import ReviewRetriever
from app.retrieval.user_retriever import UserRetriever
from app.retrieval.vector_store import FaissVectorStore, VectorMatch


def default_processed_dir() -> Path:
    """Return `app/data/processed/` based on this file's location."""
    return Path(__file__).resolve().parents[1] / "data" / "processed"


def default_embeddings_dir() -> Path:
    """Return `app/data/embeddings/` based on this file's location."""
    return Path(__file__).resolve().parents[1] / "data" / "embeddings"


@dataclass
class RetrievalRegistry:
    """All loaded data + components used by the retrieval layer."""

    embedder: Embedder
    businesses_df: pd.DataFrame
    reviews_df: pd.DataFrame
    users_df: pd.DataFrame
    businesses_store: FaissVectorStore
    reviews_store: FaissVectorStore
    users_store: FaissVectorStore
    business_retriever: BusinessRetriever
    review_retriever: ReviewRetriever
    user_retriever: UserRetriever

    @classmethod
    def load(
        cls,
        processed_dir: Path | None = None,
        embeddings_dir: Path | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        embedder: Embedder | None = None,
    ) -> "RetrievalRegistry":
        """Load everything needed for retrieval in one call.

        Pass a pre-built `embedder` to avoid the Sentence-Transformers import
        cost when you only need direct lookups (e.g. unit tests).
        """
        processed = processed_dir or default_processed_dir()
        embeddings = embeddings_dir or default_embeddings_dir()

        businesses_df = pd.read_parquet(processed / "businesses.parquet")
        reviews_df = pd.read_parquet(processed / "reviews.parquet")
        users_df = pd.read_parquet(processed / "users.parquet")

        businesses_store = FaissVectorStore.load(embeddings / "businesses.faiss")
        reviews_store = FaissVectorStore.load(embeddings / "reviews.faiss")
        users_store = FaissVectorStore.load(embeddings / "users.faiss")

        emb: Embedder = embedder or SentenceTransformerEmbedder(model_name=model_name)

        business_retriever = BusinessRetriever(
            businesses_df=businesses_df,
            reviews_df=reviews_df,
            store=businesses_store,
            embedder=emb,
        )
        review_retriever = ReviewRetriever(
            reviews_df=reviews_df,
            store=reviews_store,
            embedder=emb,
        )
        user_retriever = UserRetriever(
            users_df=users_df,
            reviews_df=reviews_df,
            store=users_store,
        )

        return cls(
            embedder=emb,
            businesses_df=businesses_df,
            reviews_df=reviews_df,
            users_df=users_df,
            businesses_store=businesses_store,
            reviews_store=reviews_store,
            users_store=users_store,
            business_retriever=business_retriever,
            review_retriever=review_retriever,
            user_retriever=user_retriever,
        )

    def find_similar_users_by_text(self, query: str, top_k: int = 5) -> list[VectorMatch]:
        """Return Yelp users whose embedded profile is closest to `query` in vector space.

        Used for cold-start persona grounding: embed a synthetic bio built from
        demographics + preferences, then pull nearest indexed neighbours.
        """
        if not query.strip():
            return []
        vector = self.embedder.embed(query)
        return self.users_store.search(vector, top_k=top_k)
