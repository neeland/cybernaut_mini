"""The map-reduce half of blog stage 8: broadcast, collect, group, fuse.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 8: "we finally
    have (1) our instruction-optimized embeddings, (2) our expanded set of search
    terms, (3) our search intents, and (4) a list of 10-30 shards ... This package
    of information is then broadcast to each of the final shards", each of which
    runs the SQL filter, a full lexical search, a full semantic search and a fusion
    of the two; "the results from each shard are combined and grouped by document
    identifier" and about 100 results come back [disclosed]. Stack: Polars, SimSIMD,
    Numba, NumPy [disclosed]. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - The broadcast package is a value, not a call signature [inferred]. Making it a
      dataclass is what lets the map step be tested shard-by-shard, and mirrors the
      post's own framing of "a package of information" being sent to each shard.
    - 10-30 shards and ~100 results are defaults, not constants: both are keyword
      arguments here, defaulted to the post's numbers.
    - Fusion runs on the *reduce* side by default, not inside each shard, even
      though the post lists it as the shard's step 4 [inferred]. An RRF contribution
      is ``w / (k + rank)`` with ranks restarting at 1 in every shard, so fusing
      shard-locally hands the best document of an irrelevant shard the same score as
      the best document of the right one — the pathology
      :func:`cybernaut_mini.retrieval._hybrid_hits_fused_globally` was written to
      fix, and this module preserves that fix. ``fusion="per_shard"`` reproduces the
      literal reading for comparison, and is the only place it is available.
    - Each shard returns at most ``per_shard_limit`` candidates per ranker. The post
      does not say; an uncapped reduce would carry every scored document of every
      shard, which the 1M-document target cannot afford. The default is an order of
      magnitude above the ~100 results returned.

Alternatives rejected:
    - Threads or processes for the map step. The post's shards are separate
      machines; here they are directories, and a thread pool would buy nothing but
      non-determinism in the merge order. The loop is deliberately sequential and
      ordered by shard id.
    - Re-deriving intents or expansions per shard. The post broadcasts them
      precisely once, and re-deriving them would make each shard's answer depend on
      its own vocabulary rather than on the query.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from cybernaut_mini.models import MetadataFilter
from cybernaut_mini.query.s3_intents import SearchIntent
from cybernaut_mini.retrieval import score_shard_components
from cybernaut_mini.rrf import RankedList, rrf_fuse

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from cybernaut_mini.config import RRFConfig
    from cybernaut_mini.indexing import LoadedIndex

    FloatArray = npt.NDArray[np.float32]

__all__ = [
    "DEFAULT_MAX_SHARDS",
    "DEFAULT_MIN_SHARDS",
    "DEFAULT_PER_SHARD_LIMIT",
    "DEFAULT_RESULT_LIMIT",
    "BroadcastRequest",
    "FusedCandidate",
    "FusionScope",
    "RetrievalMode",
    "ShardResponse",
    "broadcast",
    "group_by_document",
    "map_shard",
    "reduce_responses",
    "select_shards",
]

#: The post's shard fan-out: "a list of 10-30 shards".
DEFAULT_MIN_SHARDS = 10
DEFAULT_MAX_SHARDS = 30

#: The post's result count: the worked examples end with "+99 more results".
DEFAULT_RESULT_LIMIT = 100

#: Per-ranker cap on what one shard sends back. Ten times the result limit, so the
#: reduce still sees far more candidates than it can return.
DEFAULT_PER_SHARD_LIMIT = 1000

RetrievalMode = Literal["lexical", "dense", "hybrid"]
FusionScope = Literal["global", "per_shard"]


# ------------------------------------------------------------------ #
# Shard selection                                                     #
# ------------------------------------------------------------------ #


def select_shards(
    routed: Sequence[int],
    *,
    min_shards: int = DEFAULT_MIN_SHARDS,
    max_shards: int = DEFAULT_MAX_SHARDS,
    available: Collection[int] | None = None,
) -> tuple[int, ...]:
    """Clamp a routing order (best shard first) to the post's 10-30 shard fan-out.

    ``routed`` keeps its order — it is stage 6/7's relevance ranking. Duplicates and
    ids missing from ``available`` are dropped. When fewer than ``min_shards``
    survive and ``available`` is known, the shortfall is topped up with the
    remaining ids in ascending order, so a query that routes confidently to three
    shards still searches ten. An index with fewer than ``min_shards`` shards simply
    searches all of them.
    """
    if min_shards < 1:
        msg = f"min_shards must be >= 1, got {min_shards}"
        raise ValueError(msg)
    if max_shards < min_shards:
        msg = f"max_shards ({max_shards}) must be >= min_shards ({min_shards})"
        raise ValueError(msg)

    allowed = set(available) if available is not None else None
    ordered: list[int] = []
    seen: set[int] = set()
    for shard_id in routed:
        if shard_id in seen or (allowed is not None and shard_id not in allowed):
            continue
        seen.add(shard_id)
        ordered.append(shard_id)

    if allowed is not None and len(ordered) < min_shards:
        for shard_id in sorted(allowed):
            if len(ordered) >= min_shards:
                break
            if shard_id not in seen:
                seen.add(shard_id)
                ordered.append(shard_id)

    return tuple(ordered[:max_shards])


# ------------------------------------------------------------------ #
# The broadcast package                                               #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class BroadcastRequest:
    """The package stage 8 sends to every selected shard.

    The four numbered items in the post map onto ``query_vector`` (the
    instruction-optimized embedding), ``expansion_tokens`` (the expanded search
    terms), ``intents`` (stage 3's search intents) and ``shard_ids``.
    """

    query_tokens: tuple[str, ...] = ()
    expansion_tokens: tuple[str, ...] = ()
    query_vector: FloatArray | None = None
    intents: tuple[SearchIntent, ...] = ()
    shard_ids: tuple[int, ...] = ()
    metadata_filter: MetadataFilter | None = None
    mode: RetrievalMode = "hybrid"
    result_limit: int = DEFAULT_RESULT_LIMIT
    per_shard_limit: int = DEFAULT_PER_SHARD_LIMIT
    query_variant: str = ""

    def __post_init__(self) -> None:
        if self.result_limit < 1:
            msg = f"result_limit must be >= 1, got {self.result_limit}"
            raise ValueError(msg)
        if self.per_shard_limit < 1:
            msg = f"per_shard_limit must be >= 1, got {self.per_shard_limit}"
            raise ValueError(msg)
        if self.mode != "lexical" and self.query_vector is None:
            msg = f"mode {self.mode!r} needs a query_vector"
            raise ValueError(msg)
        if self.mode != "dense" and not self.query_tokens and not self.expansion_tokens:
            msg = f"mode {self.mode!r} needs query_tokens or expansion_tokens"
            raise ValueError(msg)

    @property
    def all_query_tokens(self) -> tuple[str, ...]:
        """Query tokens plus expansions, for snippet anchoring."""
        return tuple(self.query_tokens) + tuple(self.expansion_tokens)


@dataclass(frozen=True)
class ShardResponse:
    """What one shard sends back: its lexical and semantic rankings, best first."""

    shard_id: int
    lexical: tuple[tuple[str, float], ...] = ()
    dense: tuple[tuple[str, float], ...] = ()

    @property
    def doc_ids(self) -> tuple[str, ...]:
        """Distinct documents this shard returned, lexical order first."""
        seen: dict[str, None] = {}
        for doc_id, _ in self.lexical:
            seen.setdefault(doc_id, None)
        for doc_id, _ in self.dense:
            seen.setdefault(doc_id, None)
        return tuple(seen)


# ------------------------------------------------------------------ #
# Map                                                                 #
# ------------------------------------------------------------------ #


def map_shard(index: LoadedIndex, shard_id: int, request: BroadcastRequest) -> ShardResponse:
    """Steps 1-3 on one shard: filter, full lexical search, full semantic search.

    Delegates the scoring to :func:`cybernaut_mini.retrieval.score_shard_components`
    rather than duplicating BM25 and cosine here — that function already implements
    the post's steps 1-3 and is the tested path. Only the shard's own manifest and
    BM25 index are touched, both lazily loaded, so a broadcast to 30 shards of a
    1000-shard index never materialises the other 970.
    """
    lexical, dense = score_shard_components(
        index,
        shard_id,
        query_tokens=list(request.query_tokens),
        expansion_tokens=list(request.expansion_tokens),
        query_vector=request.query_vector,
        mode=request.mode,
        metadata_filter=request.metadata_filter,
    )
    cap = request.per_shard_limit
    return ShardResponse(
        shard_id=shard_id,
        lexical=tuple(lexical[:cap]),
        dense=tuple(dense[:cap]),
    )


def broadcast(index: LoadedIndex, request: BroadcastRequest) -> list[ShardResponse]:
    """Send ``request`` to each selected shard and collect the responses.

    Shards are visited in ascending id order so the collected list — and therefore
    every tie-break downstream — is identical run to run. An empty
    ``request.shard_ids`` means "every shard in the index", which is what a small
    index degrades to anyway once :func:`select_shards` tops up to ``min_shards``.
    """
    target_ids = (
        sorted(request.shard_ids) if request.shard_ids else sorted(index.manifests.keys())
    )
    return [map_shard(index, shard_id, request) for shard_id in target_ids]


# ------------------------------------------------------------------ #
# Reduce                                                              #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class FusedCandidate:
    """One document after the reduce, before the intent refinement pass."""

    doc_id: str
    shard_id: int
    shard_ids: tuple[int, ...]
    score: float
    lexical_score: float | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    contributions: dict[str, float] = field(default_factory=dict)


def group_by_document(responses: Iterable[ShardResponse]) -> dict[str, tuple[int, ...]]:
    """Map each returned document id to the shards that returned it, ascending.

    The post's "combined and grouped by document identifier". A document normally
    lives in exactly one shard, but nothing in the index build forbids the same id
    from being present in two (a re-ingested duplicate), and the reduce must emit it
    once, not twice.
    """
    grouped: dict[str, set[int]] = {}
    for response in responses:
        for doc_id in response.doc_ids:
            grouped.setdefault(doc_id, set()).add(response.shard_id)
    return {doc_id: tuple(sorted(shards)) for doc_id, shards in sorted(grouped.items())}


def _best_per_doc(pairs: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    """Keep one entry per doc id (the highest score), sorted by ``(-score, doc_id)``."""
    best: dict[str, float] = {}
    for doc_id, score in pairs:
        current = best.get(doc_id)
        if current is None or score > current:
            best[doc_id] = score
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))


def _single_ranker_candidates(
    ranked: Sequence[tuple[str, float]],
    grouped: dict[str, tuple[int, ...]],
    *,
    ranker: str,
) -> list[FusedCandidate]:
    """Candidates for the lexical-only and dense-only modes: the raw score is the score."""
    candidates: list[FusedCandidate] = []
    for position, (doc_id, score) in enumerate(ranked, start=1):
        shards = grouped.get(doc_id, ())
        candidates.append(
            FusedCandidate(
                doc_id=doc_id,
                shard_id=shards[0] if shards else -1,
                shard_ids=shards,
                score=score,
                lexical_score=score if ranker == "lexical" else None,
                lexical_rank=position if ranker == "lexical" else None,
                dense_score=score if ranker == "dense" else None,
                dense_rank=position if ranker == "dense" else None,
            )
        )
    return candidates


def _fuse_globally(
    responses: Sequence[ShardResponse],
    grouped: dict[str, tuple[int, ...]],
    *,
    rrf_config: RRFConfig,
) -> list[FusedCandidate]:
    """One RRF over the lexical and semantic orders of *all* shards at once."""
    lexical = _best_per_doc(pair for response in responses for pair in response.lexical)
    dense = _best_per_doc(pair for response in responses for pair in response.dense)

    fused = rrf_fuse(
        [
            RankedList(
                name="lexical",
                weight=rrf_config.lexical_weight,
                ids=tuple(doc_id for doc_id, _ in lexical),
                scores=tuple(score for _, score in lexical),
            ),
            RankedList(
                name="dense",
                weight=rrf_config.dense_weight,
                ids=tuple(doc_id for doc_id, _ in dense),
                scores=tuple(score for _, score in dense),
            ),
        ],
        k=rrf_config.k,
    )

    lexical_rank = {doc_id: (r, score) for r, (doc_id, score) in enumerate(lexical, start=1)}
    dense_rank = {doc_id: (r, score) for r, (doc_id, score) in enumerate(dense, start=1)}

    candidates: list[FusedCandidate] = []
    for item in fused:
        l_rank, l_score = lexical_rank.get(item.id, (None, None))
        d_rank, d_score = dense_rank.get(item.id, (None, None))
        shards = grouped.get(item.id, ())
        candidates.append(
            FusedCandidate(
                doc_id=item.id,
                shard_id=shards[0] if shards else -1,
                shard_ids=shards,
                score=item.score,
                lexical_score=l_score,
                lexical_rank=l_rank,
                dense_score=d_score,
                dense_rank=d_rank,
                contributions=dict(item.contributions),
            )
        )
    return candidates


def _fuse_per_shard(
    responses: Sequence[ShardResponse],
    grouped: dict[str, tuple[int, ...]],
    *,
    rrf_config: RRFConfig,
) -> list[FusedCandidate]:
    """The post's literal step 4: fuse inside each shard, then merge on best score.

    Kept because it is what the post describes and because the comparison is the
    evidence for the default. Its known defect is documented on this module: ranks
    restart at 1 per shard, so the top document of every shard ties.
    """
    per_doc: dict[str, FusedCandidate] = {}
    for response in responses:
        fused = rrf_fuse(
            [
                RankedList(
                    name="lexical",
                    weight=rrf_config.lexical_weight,
                    ids=tuple(doc_id for doc_id, _ in response.lexical),
                    scores=tuple(score for _, score in response.lexical),
                ),
                RankedList(
                    name="dense",
                    weight=rrf_config.dense_weight,
                    ids=tuple(doc_id for doc_id, _ in response.dense),
                    scores=tuple(score for _, score in response.dense),
                ),
            ],
            k=rrf_config.k,
        )
        lexical_rank = {
            doc_id: (r, score) for r, (doc_id, score) in enumerate(response.lexical, start=1)
        }
        dense_rank = {
            doc_id: (r, score) for r, (doc_id, score) in enumerate(response.dense, start=1)
        }
        for item in fused:
            l_rank, l_score = lexical_rank.get(item.id, (None, None))
            d_rank, d_score = dense_rank.get(item.id, (None, None))
            candidate = FusedCandidate(
                doc_id=item.id,
                shard_id=response.shard_id,
                shard_ids=grouped.get(item.id, (response.shard_id,)),
                score=item.score,
                lexical_score=l_score,
                lexical_rank=l_rank,
                dense_score=d_score,
                dense_rank=d_rank,
                contributions=dict(item.contributions),
            )
            existing = per_doc.get(item.id)
            if existing is None or (candidate.score, -candidate.shard_id) > (
                existing.score,
                -existing.shard_id,
            ):
                per_doc[item.id] = candidate
    return sorted(per_doc.values(), key=lambda c: (-c.score, c.doc_id))


def reduce_responses(
    responses: Sequence[ShardResponse],
    *,
    mode: RetrievalMode,
    rrf_config: RRFConfig,
    fusion: FusionScope = "global",
    limit: int | None = None,
) -> list[FusedCandidate]:
    """Combine the shard responses, group by document id and rank them.

    ``limit`` truncates the ranked candidates; ``None`` keeps all of them, which is
    what the intent refinement pass wants so that its window is a window on the
    whole ranking.
    """
    grouped = group_by_document(responses)

    if mode == "lexical":
        ranked = _best_per_doc(pair for response in responses for pair in response.lexical)
        candidates = _single_ranker_candidates(ranked, grouped, ranker="lexical")
    elif mode == "dense":
        ranked = _best_per_doc(pair for response in responses for pair in response.dense)
        candidates = _single_ranker_candidates(ranked, grouped, ranker="dense")
    elif fusion == "per_shard":
        candidates = _fuse_per_shard(responses, grouped, rrf_config=rrf_config)
    else:
        candidates = _fuse_globally(responses, grouped, rrf_config=rrf_config)

    return candidates[:limit] if limit is not None else candidates
