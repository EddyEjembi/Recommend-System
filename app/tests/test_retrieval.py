"""Tests for the retrieval layer."""

import pytest

from app.retrieval.embedder import NotImplementedEmbedder


def test_not_implemented_embedder_raises() -> None:
    embedder = NotImplementedEmbedder()
    with pytest.raises(NotImplementedError):
        embedder.embed("anything")


def test_not_implemented_embedder_batch_raises() -> None:
    embedder = NotImplementedEmbedder()
    with pytest.raises(NotImplementedError):
        embedder.embed_batch(["a", "b"])
