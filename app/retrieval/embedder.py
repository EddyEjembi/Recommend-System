"""Text embedding helpers used across the retrieval layer."""

from collections.abc import Iterable
from typing import Protocol


class Embedder(Protocol):
    """Interface for any embedding backend (OpenAI, SentenceTransformers, etc.)."""

    dim: int

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single piece of text."""

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Return embeddings for a batch of texts."""


class SentenceTransformerEmbedder:
    """Local Sentence-Transformers backend (default: `all-MiniLM-L6-v2`).

    Vectors are L2-normalised so FAISS `IndexFlatIP` returns cosine similarity.
    The model is loaded lazily inside `__init__` so importing this module is
    cheap (no Torch import) — only instantiating the class triggers the load.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        show_progress_bar: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._show_progress = show_progress_bar
        self.model_name = model_name
        if hasattr(self._model, "get_embedding_dimension"):
            self.dim = int(self._model.get_embedding_dimension())
        else:
            self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        """Embed a single string and return its vector."""
        safe_text = text or " "
        vec = self._model.encode(
            [safe_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vec.tolist()

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of strings. Empty inputs are replaced with a single space."""
        materialised = [t if t else " " for t in texts]
        if not materialised:
            return []
        vectors = self._model.encode(
            materialised,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=self._show_progress,
        )
        return vectors.tolist()


class NotImplementedEmbedder:
    """Placeholder embedder kept around for tests and dependency injection."""

    dim: int = 0

    def embed(self, text: str) -> list[float]:
        """Raise to signal the backend is not configured."""
        raise NotImplementedError("Configure a concrete Embedder implementation")

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Raise to signal the backend is not configured."""
        raise NotImplementedError("Configure a concrete Embedder implementation")
