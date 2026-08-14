"""Reranker 4 of 4: a cross-encoder over shard titles and descriptions. Optional.

The post feeds "the LLM-generated titles and descriptions of each selected shard along
with the question" into ``BAAI/bge-reranker-v2-m3``. That model needs torch, which this
repo does not install: ``sentence-transformers`` lives behind the optional ``st`` extra.
So the reranker is a *seam*, not a dependency — :class:`NeuralReranker` holds anything
satisfying :class:`CrossEncoderScorer`, and the sentence-transformers implementation of
that protocol only imports torch when a caller actually constructs it. Importing this
module, or running the stage without it, costs nothing and raises nothing.

It is deliberately absent from :func:`cybernaut_mini.query.s6_rerank.fusion.
default_rerankers`. A default that downloads 2.3 GB of weights on first query is not a
default; it is a surprise.

**Deliberate omission.** The post also reports that LLM reranking of selected shards
"works very well but the added latency is simply too high for a search engine". No LLM
reranker is implemented here for that reason, not for want of a client — this repo ships
an OpenAI-compatible one in ``cybernaut_mini.providers``.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 6, Neural Reranker:
    "We can also feed the LLM-generated titles and descriptions of each selected shard
    along with the question into a neural reranker like bge-reranker-v2-m3 to get a
    reranked set of shards", and: "We have also used LLMs to rerank selected shards.
    That works very well but the added latency is simply too high for a search engine."
    Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions: [inferred] the passage handed to the cross-encoder is ``title`` and
    ``summary`` joined by a newline (:attr:`~cybernaut_mini.query.s6_rerank.base.
    ShardCandidate.document`). The post names both fields but not their arrangement;
    a newline is what bge's own examples use between fields of one passage.
Assumptions: [inferred] the raw cross-encoder logits are used as scores without a
    sigmoid. RRF consumes ranks, and a sigmoid is monotone, so it cannot change the
    ranking — only make the recorded scores prettier and slower.
Assumptions: [inferred] all candidates are scored in one batched call, because a
    cross-encoder's whole cost model is batching; a scorer that ignores the batch is
    still correct, just slow.

Alternatives rejected: importing ``sentence_transformers`` at module import time behind
    a ``try``. Rejected: it makes the *presence* of the extra change what
    ``from ... import NeuralReranker`` does, and a stage that silently reranks
    differently depending on the install is untestable.
Alternatives rejected: a no-op stand-in that returns zeros when the extra is missing.
    Rejected outright — a reranker that scores everything equally looks like it ran.
    Asking for the neural reranker without the extra is an error with an install
    command in it.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from cybernaut_mini.config import ConfigError
from cybernaut_mini.query.s6_rerank.base import RerankQuery, ShardCandidate

__all__ = [
    "DEFAULT_NEURAL_MODEL",
    "CrossEncoderScorer",
    "NeuralReranker",
    "SentenceTransformersCrossEncoder",
    "create_neural_reranker",
    "neural_reranker_available",
]

#: The model the post names for this reranker.
DEFAULT_NEURAL_MODEL = "BAAI/bge-reranker-v2-m3"

_MISSING_EXTRA = (
    "the neural shard reranker needs the optional 'st' extra (sentence-transformers, "
    "which pulls in torch); the default install does not include it.\n"
    "  install it   : uv sync --extra st\n"
    "  or leave it out: the stage-6 default rerankers (bloom, compression, pagerank) "
    "need no model weights and no network — see "
    "cybernaut_mini.query.s6_rerank.fusion.default_rerankers.\n"
    "  or supply your own: NeuralReranker(scorer=...) accepts any object with "
    "score_pairs(query, documents) -> Sequence[float]."
)


@runtime_checkable
class CrossEncoderScorer(Protocol):
    """A cross-encoder: one relevance score per (question, shard passage) pair."""

    def score_pairs(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        """Score every ``document`` against ``query``, in the order given."""
        ...


def neural_reranker_available() -> bool:
    """True when the ``st`` extra is installed, so a real cross-encoder can be built.

    Checks for the module without importing it: importing sentence-transformers pulls
    torch into the process, which costs seconds and hundreds of megabytes.
    """
    return importlib.util.find_spec("sentence_transformers") is not None


class SentenceTransformersCrossEncoder:
    """:class:`CrossEncoderScorer` backed by ``sentence_transformers.CrossEncoder``.

    Constructing this downloads the model on first use. Raises
    :class:`~cybernaut_mini.config.ConfigError` with an install command when the ``st``
    extra is absent — the same failure shape as this repo's other optional providers.
    """

    def __init__(self, model_name: str = DEFAULT_NEURAL_MODEL, revision: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ConfigError(_MISSING_EXTRA) from exc
        self._model_name = model_name
        self._revision = revision
        self._model = CrossEncoder(model_name, revision=revision)

    @property
    def identifier(self) -> str:
        return self._model_name

    def score_pairs(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        if not documents:
            return []
        pairs = [(query, document) for document in documents]
        return [float(score) for score in self._model.predict(pairs)]


@dataclass(frozen=True)
class NeuralReranker:
    """Scores each shard by a cross-encoder over its title and description."""

    scorer: CrossEncoderScorer
    name: str = "neural"

    def score(
        self, query: RerankQuery, candidates: Sequence[ShardCandidate]
    ) -> dict[int, float]:
        """Cross-encoder score per shard, in whatever range the model emits."""
        if not candidates:
            return {}
        documents = [candidate.document for candidate in candidates]
        scores = self.scorer.score_pairs(query.text, documents)
        if len(scores) != len(candidates):
            msg = (
                f"cross-encoder returned {len(scores)} scores for {len(candidates)} "
                f"shards; the pairing would be wrong"
            )
            raise ValueError(msg)
        return {
            candidate.shard_id: float(score)
            for candidate, score in zip(candidates, scores, strict=True)
        }


def create_neural_reranker(
    *,
    scorer: CrossEncoderScorer | None = None,
    model_name: str = DEFAULT_NEURAL_MODEL,
    revision: str | None = None,
) -> NeuralReranker:
    """Build the neural reranker, loading ``model_name`` unless a ``scorer`` is given.

    The ``scorer`` argument is the seam: pass an already-loaded cross-encoder, a remote
    scoring service, or a deterministic test double, and no model is loaded at all.
    """
    if scorer is None:
        scorer = SentenceTransformersCrossEncoder(model_name, revision=revision)
    return NeuralReranker(scorer=scorer)
