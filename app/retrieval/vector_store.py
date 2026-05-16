"""Abstraction over the underlying vector store, plus a FAISS implementation.

`VectorStore` is the protocol every backend must satisfy. `FaissVectorStore`
is the concrete implementation used by the project.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VectorMatch:
    """A single hit returned by a similarity search."""

    id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    """Interface implemented by every concrete vector-store backend."""

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert a batch of vectors into the index."""

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        namespace: str | None = None,
    ) -> list[VectorMatch]:
        """Return the `top_k` nearest neighbours for `query_vector`."""

    def save(self, path: Path) -> None:
        """Persist the index to disk."""


# ---------------------------------------------------------------------------
# FAISS implementation
# ---------------------------------------------------------------------------


def _meta_path(index_path: Path) -> Path:
    """Sidecar parquet path next to a `.faiss` file."""
    return index_path.parent / f"{index_path.stem}.meta.parquet"


class FaissVectorStore:
    """FAISS `IndexFlatIP` backed by a pandas DataFrame for metadata.

    Vectors must be L2-normalised on insert so inner product == cosine similarity.
    Metadata is stored row-aligned with FAISS's internal ids; saving writes both
    the index and the sidecar parquet so they reload as a single unit.

    The `namespace` argument on `search` is accepted for protocol compatibility
    but ignored — each `FaissVectorStore` represents one logical namespace.
    """

    def __init__(self, dim: int) -> None:
        import faiss

        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._metadata = pd.DataFrame()
        self.dim = dim

    @classmethod
    def load(cls, path: Path) -> "FaissVectorStore":
        """Load a previously-saved FAISS index and its metadata sidecar."""
        import faiss

        index = faiss.read_index(str(path))
        metadata = pd.read_parquet(_meta_path(path))

        store = cls.__new__(cls)
        store._faiss = faiss
        store._index = index
        store._metadata = metadata.reset_index(drop=True)
        store.dim = index.d
        return store

    def __len__(self) -> int:
        return int(self._index.ntotal)

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add a batch of vectors. Caller must ensure vectors are L2-normalised."""
        if not ids:
            return
        if not (len(ids) == len(vectors) == len(metadatas)):
            raise ValueError("ids, vectors, and metadatas must have the same length")

        arr = np.asarray(vectors, dtype="float32")
        if arr.shape[1] != self.dim:
            raise ValueError(
                f"Vector dim {arr.shape[1]} does not match index dim {self.dim}"
            )
        self._index.add(arr)

        rows = [{"id": id_, **meta} for id_, meta in zip(ids, metadatas)]
        new_df = pd.DataFrame(rows)
        if self._metadata.empty:
            self._metadata = new_df.reset_index(drop=True)
        else:
            self._metadata = pd.concat(
                [self._metadata, new_df], ignore_index=True
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        namespace: str | None = None,  # noqa: ARG002 — protocol compatibility
    ) -> list[VectorMatch]:
        """Return up to `top_k` matches sorted by descending score."""
        if len(self) == 0:
            return []
        arr = np.asarray([query_vector], dtype="float32")
        scores, indices = self._index.search(arr, min(top_k, len(self)))

        matches: list[VectorMatch] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = self._metadata.iloc[int(idx)]
            id_ = str(row["id"])
            meta = {col: row[col] for col in self._metadata.columns if col != "id"}
            matches.append(VectorMatch(id=id_, score=float(score), metadata=meta))
        return matches

    def get_vector(self, id_: str) -> list[float] | None:
        """Look up a previously-indexed vector by external id, or return None."""
        if self._metadata.empty:
            return None
        mask = self._metadata["id"] == id_
        if not mask.any():
            return None
        internal_idx = int(mask.idxmax())
        return self._index.reconstruct(internal_idx).tolist()

    def save(self, path: Path) -> None:
        """Write the FAISS index and its parquet sidecar to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path))
        self._metadata.to_parquet(_meta_path(path), index=False)
