"""Stage 5 of Hybrid-3 — Shard Selection: route a question to the right shards.

The query pipeline's fifth stage picks which of the corpus's shards a question is worth
searching at all. The post calls it "arguably the single most important stage in our
retrieval pipeline because, if we route the question to the wrong shards, we won't return
the best document" — everything downstream can only reorder what this stage handed it.

Typical use::

    selector = ShardSelector(index, provider=embedder, processor=processor)
    selection = selector.select("How safe is gene editing?", top_n=100)
    selection.shard_ids                       # best-first shard ids
    selection.signals.factor("dense").scores  # per-factor trace
    selector.dense_recall(questions, k=10)    # what HNSW costs in recall

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 5, "Shard Selection".
    Four ranking factors (Vanilla Dense over HNSW, Bayesian Dense, Vanilla Sparse TF-IDF,
    Entity Sparse) combined with reciprocal rank fusion; tech stack NumPy, SciPy, Numba,
    SimSIMD, USearch. Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

What is here, and what is not
-----------------------------
- :mod:`.dense` — factor (a), Vanilla Dense Similarity: a real USearch HNSW graph over
  E5 embeddings of the shard summaries, plus the exact cosine baseline and a recall
  measurement against it. [disclosed]
- Factor (b), Bayesian Dense Similarity, is **omitted**. The post withholds it for a
  patent application and discloses nothing about how it works, so this replica ranks with
  three factors and records the omission in
  :attr:`~.selector.ShardSelectionSignals.omitted_factors` rather than substituting an
  invented method under the post's name. See :mod:`.selector` for the full reasoning.
- :mod:`.sparse` — factors (c) and (d), Vanilla Sparse and Entity Sparse Similarity: one
  shards x terms sparse TF-IDF matrix class, instantiated over shard keywords and over
  shard entities. The entity factor runs only when the question has entities. [disclosed]
- :mod:`.selector` — :class:`~.selector.ShardSelector`, which fuses the surviving factors
  with :func:`cybernaut_mini.rrf.rrf_fuse` and returns ranked shard ids with an
  inspectable per-factor signal record. [disclosed]

Assumptions: the stage reads shard *manifests* only. Documents, tokens and the document
    embedding matrix are never touched, because the point of shard selection is to decide
    what not to open.

Alternatives rejected: extending :func:`cybernaut_mini.routing.route`, which already
    computes dense, sparse and entity scores. It scores every shard by scanning, and it
    folds stage 6's rerankers into the same call — both of which stage 5 is defined
    against. ``routing.py`` is left alone; this package is the sublinear, stage-5-only
    replacement.
"""

from __future__ import annotations

from cybernaut_mini.query.s5_select.dense import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,
    DenseNeighbours,
    HnswCacheError,
    HnswShardIndex,
    RecallReport,
    summary_vectors,
)
from cybernaut_mini.query.s5_select.selector import (
    BAYESIAN_DENSE,
    DEFAULT_CANDIDATE_DEPTH,
    DENSE,
    ENTITY,
    SPARSE,
    FactorRanking,
    ShardSelection,
    ShardSelectionSignals,
    ShardSelector,
)
from cybernaut_mini.query.s5_select.sparse import (
    SparseHits,
    SparseShardMatrix,
    term_frequencies,
)

__all__ = [
    "BAYESIAN_DENSE",
    "DEFAULT_CANDIDATE_DEPTH",
    "DEFAULT_CONNECTIVITY",
    "DEFAULT_EXPANSION_ADD",
    "DEFAULT_EXPANSION_SEARCH",
    "DENSE",
    "ENTITY",
    "SPARSE",
    "DenseNeighbours",
    "FactorRanking",
    "HnswCacheError",
    "HnswShardIndex",
    "RecallReport",
    "ShardSelection",
    "ShardSelectionSignals",
    "ShardSelector",
    "SparseHits",
    "SparseShardMatrix",
    "summary_vectors",
    "term_frequencies",
]
