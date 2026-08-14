"""Fusing the rerankers: RRF over whichever of the four are available.

Stage 5 already fused (dense, sparse, entity) with RRF; stage 6 fuses again, over a
different set of rankers and over only the shards stage 5 kept. Reusing
:func:`cybernaut_mini.rrf.rrf_fuse` for both is not a shortcut — the post uses RRF at
both points, "once again ... combined using reciprocal rank fusion".

Two behaviours here are worth knowing about because neither is in the post:

* **Abstention.** A reranker that gives every candidate the same score has no opinion.
  Its ranked list would be the tie-break order — the prior — and feeding that to RRF
  would quietly double the weight of stage 5. Such a reranker is dropped and named in
  :attr:`RerankResult.abstained`. This is the common case for the bloom reranker on an
  index built without artifacts, and for the compression reranker on shards too small
  for zstd to train on.
* **The prior is a ranker.** Stage 5's order enters the fusion as a list named
  ``stage5``, so reranking moves shards rather than replacing the ranking outright.
  Set ``include_prior=False`` to see what the rerankers say on their own.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 6: "Once again, these
    ranking factors are then combined using reciprocal rank fusion (RRF)", over the
    bloom, compression, neural and page rerankers. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions: [inferred] equal weights for every reranker, and equal weight for the
    prior. The post gives no weights and this repo has no stage-6 relevance judgements
    to fit them on; an unequal default would be a fabricated number that looked tuned.
Assumptions: [inferred] the neural reranker is not in the default set — it needs the
    optional ``st`` extra and a 2.3 GB download, so it is opt-in via ``rerankers=``.
Assumptions: [inferred] ``k = 60``, the RRF paper's constant and this repo's default
    everywhere else.

Alternatives rejected: cascading the rerankers (bloom filters the set, then compression
    orders the survivors). Rejected because the post fuses rather than cascades, and
    because a bloom false negative is impossible but a bloom *true* zero on a shard
    that is nonetheless relevant is not — a cascade would make that unrecoverable,
    where a fusion lets the other three outvote it.
Alternatives rejected: dropping a candidate whose bloom score is 0.0, which is one
    reading of "should get downranked". Rejected: downranked is not removed, and stage
    8 needs a full ordering of the selected shards to fan out over.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cybernaut_mini.query.s6_rerank.base import (
    RerankQuery,
    ShardCandidate,
    ShardReranker,
    to_ranked_list,
)
from cybernaut_mini.query.s6_rerank.bloom import BloomReranker
from cybernaut_mini.query.s6_rerank.candidates import build_rerank_query, candidates_from_index
from cybernaut_mini.query.s6_rerank.compression import CompressionReranker
from cybernaut_mini.query.s6_rerank.pagerank import PageRankReranker
from cybernaut_mini.rrf import FusedItem, RankedList, rrf_fuse

if TYPE_CHECKING:
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.query.s3_intents import SearchIntent

__all__ = [
    "DEFAULT_RRF_K",
    "PRIOR_RANKER_NAME",
    "RerankResult",
    "default_rerankers",
    "rerank_index_shards",
    "rerank_shards",
]

#: RRF constant, matching the paper and the rest of this repo.
DEFAULT_RRF_K = 60

#: Name the stage-5 ordering carries inside the fused output.
PRIOR_RANKER_NAME = "stage5"

#: Scores within this of each other count as "the same score" for abstention.
_FLAT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class RerankResult:
    """The reordered shards plus every number that produced the order."""

    #: Shard ids, best first. This is what stage 7 and stage 8 consume.
    shard_ids: tuple[int, ...]
    #: Full RRF detail: per-ranker contribution, rank and raw score for each shard.
    fused: tuple[FusedItem, ...]
    #: Raw per-reranker scores, keyed by reranker name then shard id.
    scores: Mapping[str, Mapping[int, float]]
    #: Rankers that voted, in fusion order.
    rankers: tuple[str, ...]
    #: Rankers that had no opinion and were dropped, with their scores still in
    #: :attr:`scores` so the abstention is inspectable.
    abstained: tuple[str, ...] = field(default_factory=tuple)

    def rank_of(self, shard_id: int) -> int | None:
        """1-based position of ``shard_id`` in the reranked order, or ``None``."""
        try:
            return self.shard_ids.index(shard_id) + 1
        except ValueError:
            return None


def default_rerankers() -> list[ShardReranker]:
    """The three rerankers that need no model weights and no network.

    Bloom, compression and page. The neural reranker is built separately with
    :func:`cybernaut_mini.query.s6_rerank.neural.create_neural_reranker` and appended
    by a caller who has the ``st`` extra installed.
    """
    return [BloomReranker(), CompressionReranker(), PageRankReranker()]


def _is_flat(scores: Mapping[int, float]) -> bool:
    """True when every shard got the same score, i.e. the reranker has no opinion."""
    if len(scores) < 2:
        return True
    values = scores.values()
    return (max(values) - min(values)) <= _FLAT_TOLERANCE


def rerank_shards(
    query: RerankQuery,
    candidates: Sequence[ShardCandidate],
    *,
    rerankers: Sequence[ShardReranker] | None = None,
    weights: Mapping[str, float] | None = None,
    k: int = DEFAULT_RRF_K,
    include_prior: bool = True,
    top_n: int | None = None,
) -> RerankResult:
    """Rerank the selected shards, fusing the available rerankers with RRF.

    Parameters
    ----------
    query:
        Question text, stage-3 intents, and the stage-5 order.
    candidates:
        The selected shards, from
        :func:`~cybernaut_mini.query.s6_rerank.candidates.candidates_from_index`.
    rerankers:
        Which rerankers to run; :func:`default_rerankers` when omitted. Names must be
        unique, since the name is the key in the fused contributions.
    weights:
        Per-ranker RRF weight by name, including ``"stage5"`` for the prior. Anything
        unnamed weighs 1.0.
    k:
        RRF constant.
    include_prior:
        Feed the stage-5 order into the fusion as a ranker of its own.
    top_n:
        Truncate the returned shard ids (the full fusion detail is kept).

    Returns
    -------
    RerankResult
        Reordered shard ids plus the per-ranker scores behind them.
    """
    if not candidates:
        return RerankResult(shard_ids=(), fused=(), scores={}, rankers=())

    chosen = list(default_rerankers() if rerankers is None else rerankers)
    names = [reranker.name for reranker in chosen]
    if len(set(names)) != len(names):
        msg = f"reranker names must be unique within one fusion, got {names!r}"
        raise ValueError(msg)
    if include_prior and PRIOR_RANKER_NAME in names:
        msg = f"{PRIOR_RANKER_NAME!r} is reserved for the stage-5 order when include_prior is set"
        raise ValueError(msg)

    weight_of = dict(weights or {})
    ordered_ids = [candidate.shard_id for candidate in candidates]

    lists: list[RankedList] = []
    if include_prior:
        # The prior list is stage 5's order, restricted to the candidates and with any
        # candidate stage 5 never ranked appended in the caller's order.
        candidate_ids = set(ordered_ids)
        prior_ids = [shard_id for shard_id in query.prior if shard_id in candidate_ids]
        ranked_by_prior = set(prior_ids)
        prior_ids += [shard_id for shard_id in ordered_ids if shard_id not in ranked_by_prior]
        lists.append(
            RankedList(
                name=PRIOR_RANKER_NAME,
                weight=weight_of.get(PRIOR_RANKER_NAME, 1.0),
                ids=tuple(str(shard_id) for shard_id in prior_ids),
            )
        )

    all_scores: dict[str, Mapping[int, float]] = {}
    voted: list[str] = []
    abstained: list[str] = []
    for reranker in chosen:
        scores = reranker.score(query, candidates)
        all_scores[reranker.name] = scores
        if _is_flat(scores):
            abstained.append(reranker.name)
            continue
        voted.append(reranker.name)
        lists.append(
            to_ranked_list(
                reranker.name,
                scores,
                weight=weight_of.get(reranker.name, 1.0),
                prior_rank=query.prior_rank,
            )
        )

    if not lists:
        # Every reranker abstained and the prior was excluded: nothing has an opinion,
        # so the honest answer is the order the candidates arrived in.
        return RerankResult(
            shard_ids=tuple(ordered_ids[:top_n] if top_n is not None else ordered_ids),
            fused=(),
            scores=all_scores,
            rankers=(),
            abstained=tuple(abstained),
        )

    fused = rrf_fuse(lists, k=k)
    shard_ids = [int(item.id) for item in fused]
    if top_n is not None:
        shard_ids = shard_ids[:top_n]

    rankers = ([PRIOR_RANKER_NAME] if include_prior else []) + voted
    return RerankResult(
        shard_ids=tuple(shard_ids),
        fused=tuple(fused),
        scores=all_scores,
        rankers=tuple(rankers),
        abstained=tuple(abstained),
    )


def rerank_index_shards(
    index: LoadedIndex,
    question: str,
    *,
    selected: Iterable[int],
    intents: Sequence[SearchIntent] = (),
    rerankers: Sequence[ShardReranker] | None = None,
    weights: Mapping[str, float] | None = None,
    k: int = DEFAULT_RRF_K,
    include_prior: bool = True,
    top_n: int | None = None,
) -> RerankResult:
    """Stage 6 end to end: fetch the selected shards' artifacts, then rerank them.

    ``selected`` is stage 5's output, best shard first; it doubles as the prior order.
    Only those shards are read from the index.
    """
    prior = list(selected)
    candidates = candidates_from_index(index, prior)
    query = build_rerank_query(question, intents=intents, prior=prior)
    return rerank_shards(
        query,
        candidates,
        rerankers=rerankers,
        weights=weights,
        k=k,
        include_prior=include_prior,
        top_n=top_n,
    )
