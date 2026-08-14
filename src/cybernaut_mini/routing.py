"""Shard routing: blog stage 5 (selection) then blog stage 6 (reranking), in one call.

``route()`` is the repo's long-standing entry point for "which shards should this
question be searched in". It no longer *implements* either stage: stage 5 is
:mod:`cybernaut_mini.query.s5_select` and stage 6 is
:mod:`cybernaut_mini.query.s6_rerank`, and this module is the adapter that composes
them and flattens their two result objects into the :class:`RoutingSignals` record the
agent, the trace and the CLI already consume.

What was deleted here, and why
------------------------------
The previous implementation did all of it inline and got two things wrong that the
stage packages get right:

* it scored **every** shard on dense and sparse similarity by scanning the manifest
  dict, which is exactly the O(n_shards) query path the post's stage 5 exists to
  avoid ("if we route the question to the wrong shards, we won't return the best
  document" — but also, at a million shards you cannot look at all of them). Stage 5
  answers the dense factor from a USearch HNSW graph and the sparse factor from a CSC
  traversal that never visits a shard sharing no query term.
* it approximated the post's compression reranker with ``zlib`` over ``summary +
  question``, and its intent reranker with query-bigram coverage of the keyword list.
  The post says "trained Zstandard text compression dictionary" and "bloom filter that
  keeps track of important phrases"; both artifacts were already being built and
  persisted by :mod:`cybernaut_mini.shard_artifacts` and read by nothing. Stage 6 reads
  them, so the zlib proxy is gone rather than kept alongside.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 5 "Shard Selection"
    (dense over HNSW, Bayesian dense, sparse TF-IDF, entity sparse, combined with RRF)
    and stage 6 "Shard Reranking" (bloom, compression, neural, page rerankers, "once
    again ... combined using reciprocal rank fusion"). Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - [inferred] ``route`` runs stage 5 *and* stage 6 because its callers ask one
      question ("give me shards"). The post separates them, and so does the code: the
      two stage packages are independently usable and this module only orders them.
    - [inferred] stage 6 reranks the whole stage-5 selection (``candidate_depth``
      shards, 100 by default), and ``top_n`` truncates afterwards. Reranking only the
      shards you were going to keep anyway cannot change which shards you keep.
    - [inferred] the search intents stage 6 probes the bloom filters with are derived
      here from the question with :func:`~cybernaut_mini.query.s3_intents.predict_intents`
      over a *uniform* IDF table, because a :class:`~cybernaut_mini.indexing.LoadedIndex`
      carries no global IDF statistic and building one would mean reading every shard.
      Uniform IDF still ranks intents by term frequency and phrase length, and a caller
      that has a real table passes ``idf_lookup=``. Pass ``intents=`` to supply stage 3's
      own output.
    - [inferred] the stage-5 structures (HNSW graph, keyword matrix, entity matrix) are
      built once per ``(index, embedding model)`` and cached in a
      :class:`weakref.WeakKeyDictionary`, because the agent calls ``route`` up to 18
      times for one question and rebuilding a 200-shard HNSW graph per call would make
      the amortised cost of the stage worse than the scan it replaced.

Alternatives rejected:
    - Keeping the old inline implementation behind a flag. Two implementations of one
      stage is the confusion this rewrite exists to remove, and the old one is not a
      cheaper approximation of the new one — it is a different, worse answer.
    - Zero-filling ``RoutingSignals.dense``/``.sparse`` so they keep covering every
      shard. That restores the O(n_shards) per-query cost purely to satisfy a trace
      shape, which is the wrong trade at the corpus size this repo targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

from cybernaut_mini.query.s3_intents import IdfLookup, SearchIntent, predict_intents
from cybernaut_mini.query.s5_select import (
    DEFAULT_CANDIDATE_DEPTH,
    DENSE,
    ENTITY,
    SPARSE,
    ShardSelection,
    ShardSelector,
)
from cybernaut_mini.query.s6_rerank import RerankResult, rerank_index_shards
from cybernaut_mini.rrf import FusedItem

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cybernaut_mini.config import RRFConfig
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.providers.embeddings import EmbeddingProvider
    from cybernaut_mini.query.s6_rerank import ShardReranker
    from cybernaut_mini.text import TextProcessor

__all__ = [
    "BLOOM_RERANKER",
    "COMPRESSION_RERANKER",
    "DEFAULT_CANDIDATE_DEPTH",
    "PAGE_RERANKER",
    "RoutingSignals",
    "query_intents",
    "route",
    "shard_selector",
]

#: Stage-6 reranker names, as they appear in :attr:`RoutingSignals.rerank_scores`.
BLOOM_RERANKER = "bloom"
COMPRESSION_RERANKER = "compression"
PAGE_RERANKER = "pagerank"


@dataclass
class RoutingSignals:
    """Per-shard component data produced by :func:`route`.

    The first six fields are the original trace contract and keep their meaning, with
    the two rerank fields now carrying stage 6's *real* rerankers instead of the zlib
    and bigram approximations they used to hold:

    ``dense``, ``sparse``, ``entity``
        Stage-5 factor scores. Unlike the pre-stage-5 router these cover only the
        shards each factor actually ranked (at most ``candidate_depth`` of them), not
        every shard in the index — a factor that never visits a shard has no score for
        it, and inventing a 0.0 would both cost O(n_shards) and lie about what was
        computed. ``entity`` stays ``None`` when the question has no entities, so the
        factor contributes no ranked list at all.
    ``rerank_intent``
        Stage 6's **bloom** reranker: the fraction of the question's search intents the
        shard's phrase filter may have seen. 0.0 is the post's explicit downrank case.
    ``rerank_compression``
        Stage 6's **compression** reranker: the fractional saving the shard's trained
        Zstandard dictionary buys on the question.
    ``fused``
        The final ranking, best first: stage 6's fusion when ``rerank=True``, stage 5's
        when it is False.
    """

    dense: dict[int, float]
    sparse: dict[int, float]
    entity: dict[int, float] | None  # None when query has no entities
    rerank_intent: dict[int, float] | None
    rerank_compression: dict[int, float] | None
    fused: list[FusedItem]
    #: Every stage-6 reranker's raw scores by name, including any that abstained.
    rerank_scores: dict[str, dict[int, float]] = field(default_factory=dict)
    #: Rerankers that scored every candidate identically and were dropped from the fusion.
    rerank_abstained: tuple[str, ...] = ()
    #: Rankers that actually voted in stage 6, in fusion order (``stage5`` first).
    rerank_rankers: tuple[str, ...] = ()
    #: The search intents stage 6 probed the bloom filters with.
    intents: tuple[str, ...] = ()
    #: How many shards each stage-5 factor was allowed to contribute.
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH
    #: Stage-5 order before reranking, best first.
    selected: tuple[int, ...] = ()
    #: Factors the post describes that this replica does not implement, and why.
    omitted_factors: dict[str, str] = field(default_factory=dict)
    #: The full stage-5 result, for callers that want the untruncated factor detail.
    selection: ShardSelection | None = field(default=None, repr=False)
    #: The full stage-6 result, or ``None`` when ``rerank=False``.
    rerank: RerankResult | None = field(default=None, repr=False)


# ------------------------------------------------------------------ #
# Stage-5 selector cache                                              #
# ------------------------------------------------------------------ #

# Keyed by index, then by embedding-model identifier: the HNSW graph is a function of
# the shard summaries and the model that embedded them, and the two sparse matrices are
# a function of the manifests alone. None of the three depends on the question, the
# processor or the RRF weights, which is why those are rebound on a cache hit.
_SELECTORS: WeakKeyDictionary[Any, dict[str, ShardSelector]] = WeakKeyDictionary()


def shard_selector(
    index: LoadedIndex,
    *,
    provider: EmbeddingProvider,
    processor: TextProcessor,
    rrf_config: RRFConfig,
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
) -> ShardSelector:
    """Return the cached :class:`ShardSelector` for ``index``, building it if needed.

    The cache is weak on the index, so an index that goes out of scope takes its HNSW
    graph with it. Callers that want a private selector (different HNSW parameters, an
    injected entity extractor, exact dense search) should construct one directly and
    pass it to :func:`route` as ``selector=``.
    """
    by_model = _SELECTORS.setdefault(index, {})
    selector = by_model.get(provider.identifier)
    if selector is None:
        selector = ShardSelector(
            index,
            provider=provider,
            processor=processor,
            rrf_config=rrf_config,
            candidate_depth=candidate_depth,
        )
        by_model[provider.identifier] = selector
        return selector
    # Query-side settings are cheap and may differ per call; the built structures do not.
    selector.processor = processor
    selector.entity_extractor = processor.entities
    selector.rrf_config = rrf_config
    selector.candidate_depth = candidate_depth
    return selector


def query_intents(
    question: str,
    *,
    idf_lookup: IdfLookup | None = None,
    top_k: int | None = 8,
) -> tuple[SearchIntent, ...]:
    """Stage-3 search intents for ``question``, for stage 6's bloom reranker.

    ``idf_lookup`` defaults to an empty mapping, which
    :func:`~cybernaut_mini.query.s3_intents.resolve_idf` turns into a uniform IDF of
    1.0. That is a real approximation and it is deliberate: an index carries no global
    IDF table, and deriving one would mean reading every shard on every query. With
    uniform IDF the intents are still the question's highest-scoring contiguous runs of
    content words, which is what the bloom filter is probed with.
    """
    return tuple(predict_intents(question, {} if idf_lookup is None else idf_lookup, top_k=top_k))


def route(
    index: LoadedIndex,
    question: str,
    *,
    processor: TextProcessor,
    provider: EmbeddingProvider,
    rrf_config: RRFConfig,
    top_n: int | None = None,
    rerank: bool = True,
    intents: Sequence[SearchIntent] | None = None,
    idf_lookup: IdfLookup | None = None,
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
    selector: ShardSelector | None = None,
    rerankers: Sequence[ShardReranker] | None = None,
    rerank_weights: Mapping[str, float] | None = None,
) -> tuple[list[int], RoutingSignals]:
    """Select shards (stage 5), rerank them (stage 6), return the ids best-first.

    Parameters
    ----------
    index:
        A loaded index. Only shard manifests and — when ``rerank`` is set — the
        selected shards' artifact sidecars are read; no document, token stream or BM25
        index is touched.
    question:
        The user query string.
    processor:
        :class:`~cybernaut_mini.text.TextProcessor` instance (must match build-time
        settings). Supplies the sparse factor's query terms and the entity factor's
        entities.
    provider:
        Embedding provider compatible with the index.
    rrf_config:
        Weights and ``k`` for both fusions.
    top_n:
        If given, cap the returned shard list to the top *n* shards.
    rerank:
        When True, run stage 6 over the stage-5 selection and return its order.
    intents:
        Stage-3 search intents. Derived from the question when omitted; pass ``()`` to
        run stage 6 without the bloom reranker having anything to probe.
    idf_lookup:
        IDF table for the derived intents. See :func:`query_intents`.
    candidate_depth:
        How many shards each stage-5 factor may contribute, and therefore roughly how
        many shards stage 6 reranks.
    selector:
        A pre-built :class:`ShardSelector`; the cached one is used when omitted.
    rerankers, rerank_weights:
        Stage-6 overrides, e.g. appending
        :func:`~cybernaut_mini.query.s6_rerank.create_neural_reranker`.

    Returns
    -------
    tuple[list[int], RoutingSignals]
        Shard ids ordered best-first, plus the full signal trace.
    """
    active = selector or shard_selector(
        index,
        provider=provider,
        processor=processor,
        rrf_config=rrf_config,
        candidate_depth=candidate_depth,
    )
    selection = active.select(question, candidate_depth=candidate_depth)

    dense_factor = selection.signals.factor(DENSE)
    sparse_factor = selection.signals.factor(SPARSE)
    entity_factor = selection.signals.factor(ENTITY)

    resolved_intents = (
        query_intents(question, idf_lookup=idf_lookup) if intents is None else tuple(intents)
    )

    rerank_result: RerankResult | None = None
    fused: list[FusedItem] = list(selection.signals.fused)
    result_ids = list(selection.shard_ids)

    if rerank and result_ids:
        rerank_result = rerank_index_shards(
            index,
            question,
            selected=result_ids,
            intents=resolved_intents,
            rerankers=rerankers,
            weights=rerank_weights,
            k=rrf_config.k,
        )
        fused = list(rerank_result.fused) or fused
        result_ids = list(rerank_result.shard_ids)

    scores = (
        {name: dict(values) for name, values in rerank_result.scores.items()}
        if rerank_result is not None
        else {}
    )

    signals = RoutingSignals(
        dense=dict(dense_factor.scores),
        sparse=dict(sparse_factor.scores),
        entity=dict(entity_factor.scores) if entity_factor.applied else None,
        rerank_intent=scores.get(BLOOM_RERANKER),
        rerank_compression=scores.get(COMPRESSION_RERANKER),
        fused=fused,
        rerank_scores=scores,
        rerank_abstained=() if rerank_result is None else rerank_result.abstained,
        rerank_rankers=() if rerank_result is None else rerank_result.rankers,
        intents=tuple(intent.text for intent in resolved_intents) if rerank else (),
        candidate_depth=selection.signals.candidate_depth,
        selected=tuple(selection.shard_ids),
        omitted_factors=dict(selection.signals.omitted_factors),
        selection=selection,
        rerank=rerank_result,
    )

    if top_n is not None:
        result_ids = result_ids[:top_n]
    return result_ids, signals
