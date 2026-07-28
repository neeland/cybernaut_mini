"""Shard routing: score every shard, fuse with RRF, optionally rerank.

``route()`` returns shard ids ordered best-first plus a ``RoutingSignals`` dataclass
that carries all intermediate scores for inspection and tracing.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cybernaut_mini.rrf import FusedItem, RankedList, rrf_fuse

if TYPE_CHECKING:
    from cybernaut_mini.config import RRFConfig
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.providers.embeddings import EmbeddingProvider
    from cybernaut_mini.text import TextProcessor


@dataclass
class RoutingSignals:
    """Per-shard component data produced by :func:`route`."""

    dense: dict[int, float]
    sparse: dict[int, float]
    entity: dict[int, float] | None  # None when query has no entities
    rerank_intent: dict[int, float] | None
    rerank_compression: dict[int, float] | None
    fused: list[FusedItem]


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors represented as term->weight dicts."""
    vocab = set(a) | set(b)
    dot: float = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in vocab)
    norm_a: float = sum(v * v for v in a.values()) ** 0.5
    norm_b: float = sum(v * v for v in b.values()) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def route(
    index: LoadedIndex,
    question: str,
    *,
    processor: TextProcessor,
    provider: EmbeddingProvider,
    rrf_config: RRFConfig,
    top_n: int | None = None,
    rerank: bool = True,
) -> tuple[list[int], RoutingSignals]:
    """Score every shard, fuse, optionally rerank, and return shard ids best-first.

    Parameters
    ----------
    index:
        A fully-loaded index.
    question:
        The user query string.
    processor:
        :class:`~cybernaut_mini.text.TextProcessor` instance (must match build-time settings).
    provider:
        Embedding provider compatible with the index.
    rrf_config:
        Weights and ``k`` for RRF fusion.
    top_n:
        If given, cap the returned shard list to the top *n* shards.
    rerank:
        When True, apply intent and compression rerankers on top of the initial fuse.

    Returns
    -------
    tuple[list[int], RoutingSignals]
        Shard ids ordered best-first, plus the full signal trace.
    """
    shard_ids = sorted(index.manifests.keys())

    # ------------------------------------------------------------------
    # 1. Dense scores: cosine(query_vector, shard_centroid)
    # ------------------------------------------------------------------
    query_vector = provider.embed_queries([question])[0]

    dense_scores: dict[int, float] = {}
    for sid in shard_ids:
        centroid = np.array(index.manifests[sid].centroid, dtype=np.float32)
        dense_scores[sid] = float(query_vector @ centroid)

    dense_ranked = sorted(shard_ids, key=lambda s: (-dense_scores[s], s))

    # ------------------------------------------------------------------
    # 2. Sparse scores: TF-IDF keyword cosine
    # ------------------------------------------------------------------
    query_tokens = processor.content_tokens(question)
    query_tf: dict[str, float] = {}
    for tok in query_tokens:
        query_tf[tok] = query_tf.get(tok, 0.0) + 1.0

    sparse_scores: dict[int, float] = {}
    for sid in shard_ids:
        shard_kw: dict[str, float] = {
            kw.term: kw.weight for kw in index.manifests[sid].keywords
        }
        sparse_scores[sid] = _cosine(query_tf, shard_kw)

    sparse_ranked = sorted(shard_ids, key=lambda s: (-sparse_scores[s], s))

    # ------------------------------------------------------------------
    # 3. Entity scores (omitted entirely when query has no entities)
    # ------------------------------------------------------------------
    query_entities = set(processor.entities(question))

    entity_scores: dict[int, float] | None = None
    entity_ranked: list[int] | None = None

    if query_entities:
        entity_scores = {}
        for sid in shard_ids:
            manifest = index.manifests[sid]
            overlap = sum(
                e.count for e in manifest.entities if e.text in query_entities
            )
            total_entity_count = sum(e.count for e in manifest.entities)
            entity_scores[sid] = overlap / (1 + total_entity_count)
        _entity_scores = entity_scores
        entity_ranked = sorted(shard_ids, key=lambda s: (-_entity_scores[s], s))

    # ------------------------------------------------------------------
    # 4. Initial RRF fuse
    # ------------------------------------------------------------------
    ranked_lists: list[RankedList] = [
        RankedList(
            name="dense",
            weight=rrf_config.dense_weight,
            ids=tuple(str(s) for s in dense_ranked),
            scores=tuple(dense_scores[s] for s in dense_ranked),
        ),
        RankedList(
            name="sparse",
            weight=rrf_config.lexical_weight,
            ids=tuple(str(s) for s in sparse_ranked),
            scores=tuple(sparse_scores[s] for s in sparse_ranked),
        ),
    ]

    if entity_ranked is not None and entity_scores is not None:
        ranked_lists.append(
            RankedList(
                name="entity",
                weight=rrf_config.entity_weight,
                ids=tuple(str(s) for s in entity_ranked),
                scores=tuple(entity_scores[s] for s in entity_ranked),
            )
        )

    initial_fused = rrf_fuse(ranked_lists, k=rrf_config.k)

    # ------------------------------------------------------------------
    # 5. Rerank (intent + compression), then second fuse
    # ------------------------------------------------------------------
    rerank_intent: dict[int, float] | None = None
    rerank_compression: dict[int, float] | None = None
    final_fused = initial_fused

    if rerank:
        routed_shard_ids = [int(item.id) for item in initial_fused]

        # -- Intent: bigram coverage --
        tokens = processor.content_tokens(question)
        bigrams: list[tuple[str, str]] = []
        for i in range(len(tokens) - 1):
            bigrams.append((tokens[i], tokens[i + 1]))

        intent_ranked: list[int] | None = None
        if bigrams:
            rerank_intent = {}
            for sid in routed_shard_ids:
                manifest = index.manifests[sid]
                kw_terms = {kw.term for kw in manifest.keywords}
                graph = manifest.term_graph
                covered = 0
                for a, b in bigrams:
                    # Covered if both terms in keyword list, or connected in term_graph
                    if (a in kw_terms and b in kw_terms) or (
                        (a in graph and b in graph[a]) or (b in graph and a in graph[b])
                    ):
                        covered += 1
                rerank_intent[sid] = covered / len(bigrams)
            _rerank_intent = rerank_intent
            intent_ranked = sorted(routed_shard_ids, key=lambda s: (-_rerank_intent[s], s))

        # -- Compression: zlib compression gain --
        q_compressed_len = max(1, len(zlib.compress(question.encode())))
        rerank_compression = {}
        for sid in routed_shard_ids:
            summary = index.manifests[sid].summary
            base = len(zlib.compress(summary.encode()))
            combined = len(zlib.compress((summary + " " + question).encode()))
            rerank_compression[sid] = 1.0 - (combined - base) / q_compressed_len
        _rerank_compression = rerank_compression
        compression_ranked = sorted(
            routed_shard_ids, key=lambda s: (-_rerank_compression[s], s)
        )

        # Build second fuse: initial result as "route" ranker + optional intent + compression
        second_lists: list[RankedList] = [
            RankedList(
                name="route",
                weight=1.0,
                ids=tuple(str(s) for s in routed_shard_ids),
            ),
        ]
        if intent_ranked is not None:
            second_lists.append(
                RankedList(
                    name="intent",
                    weight=1.0,
                    ids=tuple(str(s) for s in intent_ranked),
                    scores=tuple(rerank_intent[s] for s in intent_ranked),  # type: ignore[index]
                )
            )
        second_lists.append(
            RankedList(
                name="compression",
                weight=1.0,
                ids=tuple(str(s) for s in compression_ranked),
                scores=tuple(rerank_compression[s] for s in compression_ranked),
            )
        )

        final_fused = rrf_fuse(second_lists, k=rrf_config.k)

    signals = RoutingSignals(
        dense=dense_scores,
        sparse=sparse_scores,
        entity=entity_scores,
        rerank_intent=rerank_intent,
        rerank_compression=rerank_compression,
        fused=final_fused,
    )

    result_ids = [int(item.id) for item in final_fused]
    if top_n is not None:
        result_ids = result_ids[:top_n]

    return result_ids, signals
