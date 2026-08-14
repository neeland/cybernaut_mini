"""Stage 8 of the query pipeline: retrieval using map-reduce, with the intent pass.

The broadcast package (embeddings, expansions, intents, 10-30 shard ids) goes out to
each selected shard; each shard filters, searches lexically, searches semantically;
the results are fused, the top results are scanned with Aho-Corasick for the stage-3
search intents, and about 100 snippetted hits come back.

    >>> from cybernaut_mini.query.s8_retrieve import IntentMatcher
    >>> matcher = IntentMatcher(["gene editing"])
    >>> bool(matcher.scan("d1", "A gene editing breakthrough."))
    True
    >>> bool(matcher.scan("d2", "Polygene editingness is not a phrase."))
    False

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 8, "Retrieval
    Using Map-Reduce". Steps 1-4 (SQL filter, full lexical search, full semantic
    search, fusion) are [disclosed] and already implemented in
    :mod:`cybernaut_mini.retrieval`, which this package calls rather than
    duplicates. Step 5 — "then do a full-text search over top results for intents" —
    is [disclosed] but unimplemented until now, and is what makes stage 3's
    :class:`~cybernaut_mini.query.s3_intents.SearchIntent` objects do work at query
    time. The stage-8 stack the post lists is Polars, SimSIMD, Numba, NumPy and
    ahocorasick-rs; ahocorasick-rs is the one this package is built on [disclosed],
    the array work stays in NumPy, and Polars/SimSIMD/Numba are performance
    substitutions this replica does not need at its scale [inferred]. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions (each argued where it is implemented):
    - An intent match reranks; it never filters and never adds an unbounded bonus —
      see :mod:`.refine`.
    - Phrase matches respect word boundaries in spaced scripts and ignore them in
      scriptio-continua ones, because Japanese is one of the post's worked examples
      — see :mod:`.intent_scan`.
    - Fusion happens on the reduce side rather than inside each shard, preserving
      the cross-shard comparability fix in :mod:`cybernaut_mini.retrieval`; the
      post's literal per-shard reading is available as ``fusion="per_shard"`` — see
      :mod:`.mapreduce`.
    - The intent pass reads the full text of the top ``2 * result_limit`` documents
      only — see :mod:`.pipeline`.

Alternatives rejected:
    - Editing :mod:`cybernaut_mini.retrieval` in place. Stage 8 is a superset of it,
      and keeping it a separate package leaves the existing entry point untouched
      for the integration that rewires them.
    - One module. The four concerns (matching, orchestration, reranking policy,
      snippets) are separately testable and separately arguable, and the reranking
      policy in particular is the one inference in this stage worth disagreeing
      with in isolation.
"""

from __future__ import annotations

from cybernaut_mini.query.s8_retrieve.intent_scan import (
    IntentMatch,
    IntentMatcher,
    IntentScan,
    fold,
    intent_patterns,
)
from cybernaut_mini.query.s8_retrieve.mapreduce import (
    DEFAULT_MAX_SHARDS,
    DEFAULT_MIN_SHARDS,
    DEFAULT_PER_SHARD_LIMIT,
    DEFAULT_RESULT_LIMIT,
    BroadcastRequest,
    FusedCandidate,
    FusionScope,
    RetrievalMode,
    ShardResponse,
    broadcast,
    group_by_document,
    map_shard,
    reduce_responses,
    select_shards,
)
from cybernaut_mini.query.s8_retrieve.pipeline import (
    MapReduceResult,
    build_request,
    retrieve_map_reduce,
)
from cybernaut_mini.query.s8_retrieve.refine import (
    DEFAULT_INTENT_WEIGHT,
    RefinedCandidate,
    refine_with_intents,
)
from cybernaut_mini.query.s8_retrieve.snippets import (
    DEFAULT_SNIPPET_MAX_CHARS,
    SNIPPET_HARD_MAX_CHARS,
    make_snippet,
)

__all__ = [
    "DEFAULT_INTENT_WEIGHT",
    "DEFAULT_MAX_SHARDS",
    "DEFAULT_MIN_SHARDS",
    "DEFAULT_PER_SHARD_LIMIT",
    "DEFAULT_RESULT_LIMIT",
    "DEFAULT_SNIPPET_MAX_CHARS",
    "SNIPPET_HARD_MAX_CHARS",
    "BroadcastRequest",
    "FusedCandidate",
    "FusionScope",
    "IntentMatch",
    "IntentMatcher",
    "IntentScan",
    "MapReduceResult",
    "RefinedCandidate",
    "RetrievalMode",
    "ShardResponse",
    "broadcast",
    "build_request",
    "fold",
    "group_by_document",
    "intent_patterns",
    "make_snippet",
    "map_shard",
    "reduce_responses",
    "refine_with_intents",
    "retrieve_map_reduce",
    "select_shards",
]
