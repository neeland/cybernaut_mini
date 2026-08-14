"""The shapes every stage-6 reranker shares: what it is given, and what it returns.

A reranker in this stage is a pure function from (question, intents, prior order) and a
set of shard candidates to a score per shard. It never loads an index, never opens a
socket, and never mutates its inputs, so the four rerankers can be unit-tested
independently and fused without any of them knowing the others exist.

The one piece of shared policy that lives here is *how a score becomes a rank*, because
RRF only consumes ranks and a careless tie-break silently randomises the fusion. Ties
fall back to the prior (stage-5) order and then to ascending shard id, so a reranker
that has no opinion about two shards leaves them exactly as stage 5 had them.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 6, "Shard Reranking":
    "Reranking involves reordering the selected shards by estimating their actual
    relevance to the question, so that the best shards rise to the top and irrelevant
    ones ... get pushed down", and "these ranking factors are then combined using
    reciprocal rank fusion (RRF)". Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions: the post never states a reranker's input contract. [inferred] Every
    reranker here takes the *same* :class:`RerankQuery` and :class:`ShardCandidate`
    objects even though no single reranker uses every field — the bloom reranker
    ignores the centroid, the page reranker ignores the intents — because a uniform
    signature is what lets :mod:`cybernaut_mini.query.s6_rerank.fusion` treat an
    optional reranker (the neural one) as a drop-in.
Assumptions: [inferred] a reranker returns a score per shard, not a ranked list, and
    the ordering is applied centrally by :func:`rank_by_score`. A reranker that
    returned its own order could tie-break differently from its siblings, which is the
    one way to make a deterministic fusion non-deterministic.

Alternatives rejected: passing a ``LoadedIndex`` into each reranker. Rejected because
    it would let a reranker walk all 1,000 manifests (or force-materialise documents)
    when the stage only ever concerns the ~100 shards stage 5 selected;
    :func:`cybernaut_mini.query.s6_rerank.candidates.candidates_from_index` does the
    lazy per-shard fetch once, up front, and hands over plain values.
Alternatives rejected: normalising every reranker's scores onto a common scale and
    summing them. Rejected because the post explicitly fuses with RRF, and RRF exists
    precisely so that a bounded ratio (compression), a fraction (bloom) and a
    probability mass (PageRank) never have to be made commensurable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from cybernaut_mini.query.s3_intents import SearchIntent
from cybernaut_mini.rrf import RankedList
from cybernaut_mini.shard_artifacts import PhraseBloom

__all__ = [
    "RerankQuery",
    "ShardCandidate",
    "ShardReranker",
    "intent_phrases",
    "intent_tokens",
    "rank_by_score",
    "to_ranked_list",
]


@dataclass(frozen=True)
class ShardCandidate:
    """One shard as stage 6 sees it: identity, its LLM text, its centroid, its artifacts.

    Every field except ``shard_id`` is optional, because a shard legitimately arrives
    without some of them: :func:`cybernaut_mini.shard_artifacts.train_zstd_dictionary`
    returns ``None`` for a shard too small to train on, and an index written with
    ``build_artifacts=False`` has no bloom filter at all. Each reranker treats a
    missing input as "no evidence" rather than as an error.
    """

    shard_id: int
    #: LLM-generated shard title — the text the neural reranker scores.
    title: str = ""
    #: LLM-generated shard description, appended to the title for the neural reranker.
    summary: str = ""
    #: Shard centroid, the node feature the page reranker's kNN graph is built from.
    centroid: tuple[float, ...] = ()
    #: This shard's phrase filter, or ``None`` when the index carries no artifacts.
    phrase_bloom: PhraseBloom | None = None
    #: Raw trained Zstandard dictionary bytes, or ``None`` when zstd refused to train.
    zstd_dictionary: bytes | None = None

    @property
    def document(self) -> str:
        """Title and description as one passage, for a cross-encoder pair."""
        return "\n".join(part for part in (self.title, self.summary) if part)


@dataclass(frozen=True)
class RerankQuery:
    """The query side of stage 6: the question, its stage-3 intents, the stage-5 order.

    ``prior`` is the ranking stage 5 produced, best shard first. Two rerankers read it
    for different reasons: the page reranker personalises its restart distribution
    towards its head, and :func:`rank_by_score` uses it to break score ties so that a
    reranker with no opinion is a no-op rather than a shuffle.
    """

    text: str
    intents: tuple[SearchIntent, ...] = ()
    prior: tuple[int, ...] = ()
    #: Cached 1-based rank of every shard in ``prior``; built once in ``__post_init__``.
    prior_rank: Mapping[int, int] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        # Coerce the two sequence fields so a caller can pass a list from stage 3/5
        # without the object silently becoming unhashable or order-unstable.
        object.__setattr__(self, "intents", tuple(self.intents))
        prior = tuple(self.prior)
        if len(set(prior)) != len(prior):
            msg = f"prior order contains duplicate shard ids: {prior!r}"
            raise ValueError(msg)
        object.__setattr__(self, "prior", prior)
        object.__setattr__(
            self, "prior_rank", {shard_id: rank for rank, shard_id in enumerate(prior, start=1)}
        )


@runtime_checkable
class ShardReranker(Protocol):
    """What :func:`cybernaut_mini.query.s6_rerank.fusion.rerank_shards` requires.

    ``name`` labels the ranker inside the fused output (it becomes the key in
    :attr:`cybernaut_mini.rrf.FusedItem.contributions`), so it must be unique within
    one fusion. ``score`` returns a score for *every* candidate — higher is better —
    including the ones it has no evidence about, which score 0.0.
    """

    @property
    def name(self) -> str: ...

    def score(
        self, query: RerankQuery, candidates: Sequence[ShardCandidate]
    ) -> dict[int, float]: ...


def intent_phrases(intents: Iterable[SearchIntent]) -> tuple[str, ...]:
    """The intents' rendered phrases, deduplicated, in the order stage 3 ranked them."""
    seen: dict[str, None] = {}
    for intent in intents:
        if intent.text:
            seen.setdefault(intent.text, None)
    return tuple(seen)


def intent_tokens(intents: Iterable[SearchIntent]) -> tuple[str, ...]:
    """Every token of every intent, deduplicated, in first-seen order.

    Stage 3 emits multi-token phrases ("safer gene-editing medicine"); a shard's bloom
    filter mostly holds single keywords and entity strings. Probing the member tokens
    as well as the whole phrase is what makes the filter answer anything at all for a
    long question — see :class:`cybernaut_mini.query.s6_rerank.bloom.BloomReranker`.
    """
    seen: dict[str, None] = {}
    for intent in intents:
        for token in intent.tokens:
            if token:
                seen.setdefault(token, None)
    return tuple(seen)


def rank_by_score(
    scores: Mapping[int, float],
    *,
    prior_rank: Mapping[int, int] | None = None,
) -> list[int]:
    """Shard ids best-first: descending score, then the prior order, then ascending id.

    The prior tie-break is the important one. Without it, a reranker that scores every
    shard 0.0 (a query whose intents no shard has seen) would reorder the whole
    selection by shard id and hand RRF a strong, meaningless signal.
    """
    ranks = prior_rank or {}
    fallback = len(scores) + 1
    return sorted(
        scores,
        key=lambda shard_id: (-scores[shard_id], ranks.get(shard_id, fallback), shard_id),
    )


def to_ranked_list(
    name: str,
    scores: Mapping[int, float],
    *,
    weight: float = 1.0,
    prior_rank: Mapping[int, int] | None = None,
) -> RankedList:
    """Package one reranker's scores as an RRF input, ordered by :func:`rank_by_score`."""
    order = rank_by_score(scores, prior_rank=prior_rank)
    return RankedList(
        name=name,
        weight=weight,
        ids=tuple(str(shard_id) for shard_id in order),
        scores=tuple(scores[shard_id] for shard_id in order),
    )
