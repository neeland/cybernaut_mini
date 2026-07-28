from __future__ import annotations

import numpy as np
import pytest

from cybernaut_mini.config import ConfigError, EmbeddingConfig
from cybernaut_mini.providers.embeddings import HashEmbedder, create_embedding_provider


def test_hash_embedder_is_deterministic(hash_embedder: HashEmbedder) -> None:
    first = hash_embedder.embed_documents(["gene therapy trial"])
    second = hash_embedder.embed_documents(["gene therapy trial"])
    np.testing.assert_array_equal(first, second)


def test_hash_embedder_l2_normalizes(hash_embedder: HashEmbedder) -> None:
    vectors = hash_embedder.embed_documents(["solar panels", "wind farms"])
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_hash_embedder_related_texts_are_closer(hash_embedder: HashEmbedder) -> None:
    vectors = hash_embedder.embed_documents(
        ["gene therapy trial results", "gene therapy study results", "offshore wind turbines"]
    )
    related = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    assert related > unrelated


def test_hash_embedder_handles_empty_text(hash_embedder: HashEmbedder) -> None:
    vectors = hash_embedder.embed_documents([""])
    assert vectors.shape == (1, 64)
    assert not np.isnan(vectors).any()


def test_factory_returns_hash_provider() -> None:
    provider = create_embedding_provider(EmbeddingConfig(provider="hash", dim=32))
    assert provider.identifier == "hash-32"
    assert provider.dim == 32


def test_factory_offline_rejects_sentence_transformers() -> None:
    config = EmbeddingConfig(provider="sentence_transformers")
    with pytest.raises(ConfigError, match="offline"):
        create_embedding_provider(config, offline=True)
