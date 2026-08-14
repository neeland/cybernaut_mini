"""Stage 6 of Hybrid-3 — Shard Reranking: making the shard artifacts load-bearing.

Stage 5 selects ~100 shards out of a million on centroid similarity alone, and the
post is blunt about how that goes: "the shards we selected for the Japanese question
were quite bad. The first shard is about quantum computing, the second shard is about
data protection, and the third shard is about digital platforms." Stage 6 reorders that
selection by estimating actual relevance, so the good shards rise and the irrelevant
ones sink.

Four rerankers, fused with RRF:

=================  ===========================================================
:mod:`bloom`       Has this shard's phrase filter ever seen the query's stage-3
                   search intents? A zero is the post's explicit downrank rule.
:mod:`compression` How much does this shard's trained Zstandard dictionary
                   shrink the question? Compression as similarity.
:mod:`neural`      ``BAAI/bge-reranker-v2-m3`` cross-encoder over the shard's
                   title and description. Optional: needs the ``st`` extra, and
                   is *not* in :func:`default_rerankers`.
:mod:`pagerank`    Personalized PageRank over a kNN graph of shard centroids,
                   restarting on the shards stage 5 ranked highest.
=================  ===========================================================

Typical use::

    from cybernaut_mini.query.s3_intents import predict_intents
    from cybernaut_mini.query.s6_rerank import rerank_index_shards

    intents = predict_intents(question, idf)
    result = rerank_index_shards(index, question, selected=stage5_ids, intents=intents)
    result.shard_ids  # reordered, best first

This package is the first consumer of :mod:`cybernaut_mini.shard_artifacts`: before it,
every shard's bloom filter and Zstandard dictionary was built, persisted and read by
nothing.

**Deliberate omission.** The post notes that LLM reranking of shards "works very well
but the added latency is simply too high for a search engine", so no LLM reranker is
implemented — the omission is the post's own finding, not a gap.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 6, "Shard
    Reranking", which names all four rerankers, the RRF fusion over them, the Zstandard
    and rBloom stack, and the worked example in which the shard that actually discusses
    bacteria and yeast in the context of gene editing rises into the top three. Local
    copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions: [inferred] the module split — one file per reranker plus a fusion — is
    this repo's, as is every numeric parameter (kNN k, PageRank alpha, the phrase/token
    mix in the bloom reranker, the RRF weights). Each is justified where it is defined.
Alternatives rejected: putting stage 6 inside ``routing.py`` next to the ``zlib``
    approximation it supersedes. Rejected: stage 5 and stage 6 are separate stages in
    the post with separate inputs, and rewiring the existing router is a later,
    separate change.
"""

from __future__ import annotations

from cybernaut_mini.query.s6_rerank.base import (
    RerankQuery,
    ShardCandidate,
    ShardReranker,
    intent_phrases,
    intent_tokens,
    rank_by_score,
    to_ranked_list,
)
from cybernaut_mini.query.s6_rerank.bloom import DEFAULT_PHRASE_WEIGHT, BloomReranker
from cybernaut_mini.query.s6_rerank.candidates import build_rerank_query, candidates_from_index
from cybernaut_mini.query.s6_rerank.compression import CompressionReranker
from cybernaut_mini.query.s6_rerank.fusion import (
    DEFAULT_RRF_K,
    PRIOR_RANKER_NAME,
    RerankResult,
    default_rerankers,
    rerank_index_shards,
    rerank_shards,
)
from cybernaut_mini.query.s6_rerank.neural import (
    DEFAULT_NEURAL_MODEL,
    CrossEncoderScorer,
    NeuralReranker,
    SentenceTransformersCrossEncoder,
    create_neural_reranker,
    neural_reranker_available,
)
from cybernaut_mini.query.s6_rerank.pagerank import (
    DEFAULT_ALPHA,
    DEFAULT_KNN_K,
    PageRankReranker,
    build_shard_knn_graph,
    restart_distribution,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_KNN_K",
    "DEFAULT_NEURAL_MODEL",
    "DEFAULT_PHRASE_WEIGHT",
    "DEFAULT_RRF_K",
    "PRIOR_RANKER_NAME",
    "BloomReranker",
    "CompressionReranker",
    "CrossEncoderScorer",
    "NeuralReranker",
    "PageRankReranker",
    "RerankQuery",
    "RerankResult",
    "SentenceTransformersCrossEncoder",
    "ShardCandidate",
    "ShardReranker",
    "build_rerank_query",
    "build_shard_knn_graph",
    "candidates_from_index",
    "create_neural_reranker",
    "default_rerankers",
    "intent_phrases",
    "intent_tokens",
    "neural_reranker_available",
    "rank_by_score",
    "rerank_index_shards",
    "rerank_shards",
    "restart_distribution",
    "to_ranked_list",
]
