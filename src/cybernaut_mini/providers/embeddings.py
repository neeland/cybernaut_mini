"""Embedding providers behind a common protocol.

``HashEmbedder`` is a first-class deterministic offline provider (feature hashing of
tokens and character trigrams), not a test mock. ``SentenceTransformersEmbedder``
wraps E5-style models and is the quality option; it may download weights on first use.
"""

from __future__ import annotations

import hashlib
import os
from typing import Protocol

import numpy as np
import numpy.typing as npt

from cybernaut_mini.config import ConfigError, EmbeddingConfig
from cybernaut_mini.text import TextProcessor

FloatArray = npt.NDArray[np.float32]


class EmbeddingProvider(Protocol):
    @property
    def identifier(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> FloatArray: ...

    def embed_queries(self, texts: list[str]) -> FloatArray: ...


def l2_normalize(matrix: FloatArray) -> FloatArray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashEmbedder:
    """Deterministic feature-hashing embedder: tokens + char trigrams -> signed buckets."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim
        self._processor = TextProcessor(use_spacy=False)

    @property
    def identifier(self) -> str:
        return f"hash-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> list[str]:
        tokens = self._processor.tokenize(text)
        features = list(tokens)
        for token in tokens:
            padded = f"#{token}#"
            features.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        return features

    def _embed(self, texts: list[str]) -> FloatArray:
        matrix = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                bucket = value % self._dim
                sign = 1.0 if (value >> 63) & 1 else -1.0
                matrix[row, bucket] += sign
        return l2_normalize(matrix)

    def embed_documents(self, texts: list[str]) -> FloatArray:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> FloatArray:
        return self._embed(texts)


class SentenceTransformersEmbedder:
    """sentence-transformers wrapper with E5 ``query:``/``passage:`` prefixes.

    Constructing this downloads the weights on first use. The download is pinned to
    ``revision`` when one is given and authenticated with ``HF_TOKEN`` when present,
    so a private or gated model works without a separate login step.
    """

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            msg = "sentence-transformers is not installed; install the [st] extra"
            raise ConfigError(msg) from exc
        self._model_name = model_name
        self._revision = revision
        self._model = SentenceTransformer(
            model_name,
            revision=revision,
            token=os.environ.get("HF_TOKEN") or None,
        )
        self._is_e5 = "e5" in model_name.lower()

    @property
    def identifier(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def _embed(self, texts: list[str], prefix: str) -> FloatArray:
        prefixed = [f"{prefix}{text}" for text in texts] if self._is_e5 else texts
        vectors = self._model.encode(prefixed, convert_to_numpy=True, show_progress_bar=False)
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> FloatArray:
        return self._embed(texts, "passage: ")

    def embed_queries(self, texts: list[str]) -> FloatArray:
        return self._embed(texts, "query: ")


def create_embedding_provider(
    config: EmbeddingConfig, *, offline: bool = False
) -> EmbeddingProvider:
    if config.provider == "hash":
        return HashEmbedder(dim=config.dim)
    if offline:
        msg = (
            "offline mode: provider 'sentence_transformers' may require a model "
            "download; use provider 'hash'"
        )
        raise ConfigError(msg)
    return SentenceTransformersEmbedder(config.model, revision=config.revision)
