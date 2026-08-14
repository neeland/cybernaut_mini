"""Embedding providers behind a common protocol.

Three providers, in ascending order of cost:

``Model2VecEmbedder`` is the DEFAULT. Static (lookup-table) embeddings distilled
from a real transformer — semantic structure without torch, so it ships in the
core dependencies rather than an extra. It is what shard learning needs: shards
are k-means clusters over document vectors, and clustering only produces coherent
shards if the vector space carries meaning.

``HashEmbedder`` is a deterministic offline provider (feature hashing of tokens and
character trigrams), not a test mock. It is real, fast and needs no download, but
feature hashing has NO semantic structure — documents about the same topic land in
unrelated buckets. It is therefore correct for offline/deterministic tests and
WRONG for learning shards. It is no longer the default for exactly that reason.

``SentenceTransformersEmbedder`` wraps E5-style models and is the quality ceiling;
it needs the optional 'st' extra (torch) and downloads weights on first use.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 5 (Shard
Selection): the dense selector compares question embeddings against shard summary
embeddings, so shard quality is bounded above by embedding quality.

Assumptions: model2vec's distilled vectors are close enough to their teacher to
cluster comparably. Not verified here; the ablation belongs in evals.
Alternatives rejected: keeping the hash embedder as default (cheap, but yields
incoherent shards and would make stages 5-7 look broken when the fault is the
embedder); making sentence-transformers the default (best vectors, but pulls torch
into every install and breaks a bare `kedro run`).
"""

from __future__ import annotations

import hashlib
import os
from typing import Protocol

import numpy as np
import numpy.typing as npt

from cybernaut_mini.config import MODEL2VEC_PREFIX, ConfigError, EmbeddingConfig
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


class Model2VecEmbedder:
    """Static distilled embeddings: real semantics, no torch, CPU-only.

    Weights download on first use and are then cached, so this provider is not
    offline-safe on a cold cache — ``create_embedding_provider`` rejects it under
    ``offline``.

    Pinning is done by us, not by model2vec: ``StaticModel.from_pretrained`` takes
    no ``revision`` argument, so a revision is resolved to an immutable local
    snapshot via ``huggingface_hub`` first. Without that, ``require_reproducible``
    could pass while the weights silently moved.
    """

    #: Marks the provider inside ``IndexMeta.embedding_model`` so an index can be
    #: reopened with the right provider. Mirrors the existing ``hash-`` convention
    #: and avoids an ARTIFACT_VERSION bump for a new metadata field.
    PREFIX = MODEL2VEC_PREFIX

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        try:
            from model2vec import StaticModel
        except ImportError as exc:  # pragma: no cover - core dependency
            msg = (
                "embedding provider 'model2vec' needs the 'model2vec' package, which "
                "is a core dependency — reinstall with `uv sync`."
            )
            raise ConfigError(msg) from exc

        self._model_name = model_name
        self._revision = revision
        token = os.environ.get("HF_TOKEN") or None

        source: str = model_name
        if revision is not None:
            from huggingface_hub import snapshot_download

            source = snapshot_download(model_name, revision=revision, token=token)

        # force_download defaults to True upstream, which would re-fetch the weights
        # on every construction and break offline reuse of a warm cache.
        self._model = StaticModel.from_pretrained(source, token=token, force_download=False)

    @property
    def identifier(self) -> str:
        return f"{self.PREFIX}{self._model_name}"

    @property
    def dim(self) -> int:
        return int(self._model.dim)

    def _embed(self, texts: list[str]) -> FloatArray:
        vectors = self._model.encode(texts, show_progress_bar=False)
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> FloatArray:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> FloatArray:
        # Static models carry no query/passage asymmetry: the same lookup table
        # produces both sides, so no E5-style prefix applies.
        return self._embed(texts)


class SentenceTransformersEmbedder:
    """sentence-transformers wrapper with E5 ``query:``/``passage:`` prefixes.

    Constructing this downloads the weights on first use. The download is pinned to
    ``revision`` when one is given and authenticated with ``HF_TOKEN`` when present,
    so a private or gated model works without a separate login step.
    """

    def __init__(
        self, model_name: str, revision: str | None = None, device: str = "auto"
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            msg = (
                "embedding provider 'sentence_transformers' needs the optional 'st' "
                "extra, which the default install does not include.\n"
                "  install it   : uv sync --extra st       (or: make install-prod)\n"
                "  or stay local: --config configs/tiny.yaml   "
                "(kedro: --params embedding.provider=hash)\n"
                "The hash embedder is a first-class offline provider, not a stub — it "
                "needs no download and keeps builds byte-for-byte reproducible."
            )
            raise ConfigError(msg) from exc
        from cybernaut_mini.accel import resolve_device

        self._model_name = model_name
        self._revision = revision
        # Resolved here rather than left to sentence-transformers' own detection so
        # the choice is inspectable (``self.device``) and so an unavailable request
        # degrades to CPU instead of raising mid-build.
        self._device = resolve_device(device)  # type: ignore[arg-type]
        self._model = SentenceTransformer(
            model_name,
            revision=revision,
            token=os.environ.get("HF_TOKEN") or None,
            device=self._device,
        )
        self._is_e5 = "e5" in model_name.lower()

    @property
    def device(self) -> str:
        """The torch device this embedder actually resolved to."""
        return self._device

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
            f"offline mode: provider {config.provider!r} may require a model "
            "download; use provider 'hash'"
        )
        raise ConfigError(msg)
    if config.provider == "model2vec":
        return Model2VecEmbedder(config.model_name, revision=config.revision)
    return SentenceTransformersEmbedder(config.model_name, revision=config.revision)
