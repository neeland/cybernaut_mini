"""Frozen-embedder policy: Matryoshka truncation plus L2 renormalization.

The WORLD pillar scores millions of events against a handful of anchor sentences,
and every downstream index assumes those cosines live in one fixed vector space.
This module is that space's single gatekeeper: one function that truncates a
matrix to its first ``k`` Matryoshka dimensions and re-normalizes, and one thin
wrapper that applies the identical transform to event text and anchor text so the
two can never drift apart.

Blog ref: https://nosible.com/blog/turning-news-into-a-risk-on-risk-off-equity-signal
    — "the model is trained with Matryoshka representation learning, so the first
    1024 dimensions are themselves a complete embedding. We use those: truncate
    both the event vectors and the anchors to the first 1024 dimensions, then
    L2-renormalise." The same 3072 -> 1024 trick appears in
    https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty
    ("We keep the first 1,024 for speed, which the model's Matryoshka training
    makes safe, and L2-normalize them so cosine similarities stay meaningful").
    The frozen-model argument is from
    https://nosible.com/blog/the-contrastive-geometry-of-risk: the embedder is
    never fine-tuned on the corpus, so no foreknowledge of later events can leak
    backward into earlier vectors. Local copies under ``docs/blog-archive/``.

Assumptions:
    - The posts' embedder is OpenAI ``text-embedding-3-large`` (3,072-d). Offline,
      the repo's providers stand in: the hash embedder for tests, and a
      Matryoshka-capable sentence-transformers model
      (``nomic-ai/nomic-embed-text-v1.5``, 768-d, valid truncated at 256) as the
      opt-in quality path. The model name and revision are pinned as constants so
      a build can prove its space did not move.
    - Truncation before renormalization is exactly the published order. For a
      Matryoshka-trained model the truncated prefix is a complete embedding; for
      the hash embedder it is merely deterministic, which is all the offline
      tests need.
    - ``truncate_dim=None`` means "keep every dimension but still renormalize" so
      one code path serves both truncating and non-truncating configurations.

Alternatives rejected:
    - PCA/random projection to ``k`` dims: changes the geometry run to run and
      needs a fitted artifact; slicing a Matryoshka prefix is deterministic and
      artifact-free, and it is what the posts did.
    - Fine-tuning the embedder on the event corpus: better cosines today, but the
      contrastive-geometry post's point stands — a model trained on the corpus
      encodes the future of that corpus, poisoning any point-in-time claim.
    - Renormalizing only when a truncation happened: silently leaves un-normalized
      vectors in the space when ``truncate_dim`` is None and the provider does not
      normalize; always renormalizing costs one BLAS call and removes the trap.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cybernaut_mini.providers.embeddings import EmbeddingProvider, l2_normalize

__all__ = [
    "MATRYOSHKA_EMBEDDER_NAME",
    "MATRYOSHKA_EMBEDDER_REVISION",
    "MATRYOSHKA_SAFE_DIM",
    "FrozenEmbedder",
    "matryoshka_truncate",
]

FloatArray = npt.NDArray[np.float32]

#: Opt-in Matryoshka-capable open model mirroring the posts' text-embedding-3-large
#: policy: 768-d, trained to remain a complete embedding when truncated to 256.
MATRYOSHKA_EMBEDDER_NAME = "nomic-ai/nomic-embed-text-v1.5"
#: Pinned weights revision (main as of 2025-01; frozen so the space cannot move
#: between builds — see the frozen-model argument in the module docstring).
MATRYOSHKA_EMBEDDER_REVISION = "e5cf08aadaa33385f5990def41f7a23405aec398"
#: The truncation the model is documented to support losslessly-enough.
MATRYOSHKA_SAFE_DIM = 256


def matryoshka_truncate(matrix: FloatArray, dim: int | None) -> FloatArray:
    """First ``dim`` Matryoshka dimensions, L2-renormalized row-wise.

    ``dim=None`` keeps every dimension (renormalizing anyway). ``dim`` larger than
    the matrix width is an error rather than a silent no-op, because it means the
    caller's config disagrees with the provider actually in use.
    """
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        msg = f"expected a 2-d matrix, got shape {array.shape}"
        raise ValueError(msg)
    if dim is not None:
        if dim <= 0:
            msg = f"truncate dim must be positive, got {dim}"
            raise ValueError(msg)
        if dim > array.shape[1]:
            msg = f"cannot truncate {array.shape[1]}-d vectors to {dim} dims"
            raise ValueError(msg)
        array = array[:, :dim]
    return l2_normalize(np.ascontiguousarray(array))


class FrozenEmbedder:
    """An :class:`EmbeddingProvider` with the truncate-and-renorm policy baked in.

    Events and anchors must be embedded by the *same* frozen transform; routing
    both through one instance makes that a type-level guarantee instead of a
    convention. The wrapped provider is treated as frozen — this class never
    trains or adapts it.
    """

    def __init__(self, provider: EmbeddingProvider, *, truncate_dim: int | None = None) -> None:
        if truncate_dim is not None and truncate_dim > provider.dim:
            msg = (
                f"truncate_dim={truncate_dim} exceeds provider dim {provider.dim} "
                f"({provider.identifier})"
            )
            raise ValueError(msg)
        self._provider = provider
        self._truncate_dim = truncate_dim

    @property
    def identifier(self) -> str:
        if self._truncate_dim is None:
            return self._provider.identifier
        return f"{self._provider.identifier}@mrl{self._truncate_dim}"

    @property
    def dim(self) -> int:
        return self._truncate_dim if self._truncate_dim is not None else self._provider.dim

    def embed_documents(self, texts: list[str]) -> FloatArray:
        return matryoshka_truncate(self._provider.embed_documents(texts), self._truncate_dim)

    def embed_queries(self, texts: list[str]) -> FloatArray:
        return matryoshka_truncate(self._provider.embed_queries(texts), self._truncate_dim)

    def transform(self, matrix: FloatArray) -> FloatArray:
        """Apply the frozen truncate-and-renorm policy to already-computed vectors."""
        return matryoshka_truncate(matrix, self._truncate_dim)
