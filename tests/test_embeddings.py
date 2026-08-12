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


def test_factory_offline_rejects_model2vec() -> None:
    """model2vec is a core dependency but still downloads weights on first use."""
    config = EmbeddingConfig(provider="model2vec")
    with pytest.raises(ConfigError, match="offline"):
        create_embedding_provider(config, offline=True)


# --------------------------------------------------------------------------- #
# identifier round-trip                                                        #
#                                                                              #
# `EmbeddingConfig.identifier()` is written into IndexMeta.embedding_model at   #
# build time, and `provider_from_meta` dispatches on that string at query time. #
# If the two disagree, an index reopens under the WRONG provider and produces   #
# query vectors that are not comparable to the stored ones — no exception, just #
# silently meaningless scores. These tests pin the two halves together without  #
# downloading weights.                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (EmbeddingConfig(provider="hash", dim=32), "hash-32"),
        (
            EmbeddingConfig(provider="model2vec"),
            "model2vec:minishlab/potion-multilingual-128M",
        ),
        (
            EmbeddingConfig(provider="model2vec", model="minishlab/potion-base-8M"),
            "model2vec:minishlab/potion-base-8M",
        ),
        (
            EmbeddingConfig(provider="sentence_transformers"),
            "intfloat/multilingual-e5-small",
        ),
    ],
)
def test_config_identifier(config: EmbeddingConfig, expected: str) -> None:
    assert config.identifier() == expected


def test_model2vec_identifier_routes_back_to_model2vec() -> None:
    """A model2vec identifier must not be mistaken for a sentence-transformers repo id.

    Asserted via the offline rejection, whose message names the provider: that
    proves which branch `provider_from_meta` took without loading any weights.
    """
    from cybernaut_mini.models import IndexMeta
    from cybernaut_mini.retrieval import provider_from_meta

    meta = IndexMeta(
        embedding_model=EmbeddingConfig(provider="model2vec").identifier(),
        embedding_dim=256,
        embedding_revision=None,
        n_shards=1,
        n_documents=1,
        seed=42,
    )
    with pytest.raises(ConfigError, match="model2vec"):
        provider_from_meta(meta, offline=True)
