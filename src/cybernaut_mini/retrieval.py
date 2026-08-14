"""Retrieval layer: the provider factory, per-shard scoring, and blog stage 8.

Two things live here now, and only two:

* :func:`provider_from_meta`, which reconstructs the build-time embedding provider from
  the index metadata. Nothing to do with the blog pipeline; it is here because every
  query entry point needs it.
* :func:`score_shard_components`, the post's steps 1-3 on a single shard — metadata
  filter, BM25, cosine — and :func:`retrieve`, which is now a thin adapter over
  :mod:`cybernaut_mini.query.s8_retrieve`.

What was deleted here, and why
------------------------------
``search_shard``, ``merge_shard_results``, ``ShardHit``, ``_hybrid_hits_fused_globally``
and ``make_snippet`` are gone. They were this repo's first implementation of stage 8's
map-reduce and snippet steps, and stage 8 now implements all of it —
:func:`~cybernaut_mini.query.s8_retrieve.map_shard` calls the same
:func:`score_shard_components`, :func:`~cybernaut_mini.query.s8_retrieve.reduce_responses`
does the global fusion (carrying over the "fuse across all shards, not inside each one"
fix and its rationale), and
:func:`~cybernaut_mini.query.s8_retrieve.make_snippet` builds the snippet with a
user-chosen maximum length and an intent-aware anchor. Keeping both would have left two
answers to "how does a hybrid query rank".

What retrieval *gained*: the post's step 5. The top results now get an Aho-Corasick
pass for the question's stage-3 search intents, and the intent evidence is fused into
the ranking. It is bounded — only ``2 * top_k`` documents are read — and it is a no-op
when nothing matches.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 8, "Retrieval Using
    Map-Reduce": the broadcast package, the five per-shard steps, the reduce that groups
    by document identifier, and the user-bounded snippet. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - [inferred] ``retrieve`` does **not** apply the post's 10-30 shard fan-out clamp
      (:func:`~cybernaut_mini.query.s8_retrieve.select_shards`). ``shard_ids=None`` still
      means "every shard", because ``retrieve`` is also the evaluation baseline and the
      CLI's non-agent path, both of which are defined as unrouted search. Clamping is
      the caller's decision, and ``route`` is where the caller makes it.
    - [inferred] search intents are derived from the question when the caller does not
      supply stage 3's, using a uniform IDF table for the reason given in
      :func:`cybernaut_mini.routing.query_intents`. Pass ``intents=()`` to switch the
      pass off and reproduce the pre-stage-8 ranking exactly.

Alternatives rejected:
    - Importing :mod:`cybernaut_mini.query.s8_retrieve` at module scope. Stage 8's
      ``mapreduce`` imports :func:`score_shard_components` from here, so a top-level
      import in both directions is a genuine cycle: whichever module is imported first
      would reach a half-initialised partner. The import is function-local and the cycle
      cannot form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from cybernaut_mini.config import ConfigError
from cybernaut_mini.providers.embeddings import EmbeddingProvider, HashEmbedder

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from cybernaut_mini.config import RRFConfig
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.models import IndexMeta, MetadataFilter, SearchHit
    from cybernaut_mini.query.s3_intents import SearchIntent
    from cybernaut_mini.text import TextProcessor

    FloatArray = npt.NDArray[np.float32]

__all__ = [
    "provider_from_meta",
    "retrieve",
    "score_shard_components",
]


# ------------------------------------------------------------------ #
# Provider factory                                                    #
# ------------------------------------------------------------------ #


def provider_from_meta(meta: IndexMeta, *, offline: bool = False) -> EmbeddingProvider:
    """Reconstruct the embedding provider that was used to build the index.

    The provider is recovered from the shape of ``meta.embedding_model``: a
    ``"hash-"`` prefix means :class:`HashEmbedder`, a ``"model2vec:"`` prefix means
    :class:`Model2VecEmbedder`, and a bare repo id means
    :class:`SentenceTransformersEmbedder`. Encoding the provider in the identifier
    rather than in a new ``IndexMeta`` field keeps existing indexes readable.

    Raises :class:`ConfigError` under ``offline`` for the two downloadable providers.
    """
    if meta.embedding_model.startswith("hash-"):
        return HashEmbedder(dim=meta.embedding_dim)

    from cybernaut_mini.providers.embeddings import Model2VecEmbedder

    if meta.embedding_model.startswith(Model2VecEmbedder.PREFIX):
        if offline:
            msg = (
                "offline mode: embedding provider 'model2vec' may require a model "
                "download; use an index built with provider 'hash'"
            )
            raise ConfigError(msg)
        model_name = meta.embedding_model[len(Model2VecEmbedder.PREFIX) :]
        # Reuse the build-time revision so query vectors come from the same weights
        # as the stored document vectors.
        return Model2VecEmbedder(model_name, revision=meta.embedding_revision)

    if offline:
        msg = (
            "offline mode: embedding provider 'sentence_transformers' may require a "
            "model download; use an index built with provider 'hash'"
        )
        raise ConfigError(msg)
    from cybernaut_mini.providers.embeddings import SentenceTransformersEmbedder

    return SentenceTransformersEmbedder(meta.embedding_model, revision=meta.embedding_revision)


# ------------------------------------------------------------------ #
# Per-shard search: the post's steps 1-3                              #
# ------------------------------------------------------------------ #


def score_shard_components(
    index: LoadedIndex,
    shard_id: int,
    *,
    query_tokens: list[str],
    expansion_tokens: list[str],
    query_vector: FloatArray | None,
    mode: str,
    metadata_filter: MetadataFilter | None,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Score one shard's candidates, returning ``(lexical, dense)`` lists, best first.

    The post's steps 1-3 for a single shard: filter the shard's documents by the
    metadata constraints, run a full lexical search, run a full semantic search. Fusion
    (step 4) is deliberately *not* done here — see
    :func:`cybernaut_mini.query.s8_retrieve.reduce_responses`, which fuses over all
    shards at once so the RRF ranks are globally comparable.

    Each list is ``(doc_id, score)`` sorted by ``(-score, doc_id)``. Lexical scores are
    BM25, with expansion-term matches added at 0.3 weight and zero-scoring documents
    dropped; dense scores are cosine. A list is empty when the mode does not use that
    ranker.

    This is stage 8's map step (:func:`cybernaut_mini.query.s8_retrieve.map_shard`
    calls it directly), so its signature is load-bearing for that module.
    """
    manifest = index.manifests[shard_id]
    bm25 = index.bm25[shard_id]
    doc_ids = manifest.document_ids  # list in BM25 corpus order

    # 1. Candidate set (filtered).
    if metadata_filter is not None and not metadata_filter.is_empty():
        candidate_indices = [
            i
            for i, doc_id in enumerate(doc_ids)
            if metadata_filter.matches(index.by_id[doc_id])
        ]
    else:
        candidate_indices = list(range(len(doc_ids)))

    if not candidate_indices:
        return [], []

    candidate_ids = [doc_ids[i] for i in candidate_indices]

    # 2. Lexical scoring.
    lex_ranked: list[tuple[str, float]] = []  # (doc_id, combined_bm25_score)
    if mode in ("lexical", "hybrid"):
        all_scores: npt.NDArray[np.float64] = bm25.get_scores(query_tokens)
        if expansion_tokens:
            exp_scores: npt.NDArray[np.float64] = bm25.get_scores(expansion_tokens)
            all_scores = all_scores + 0.3 * exp_scores

        # Keep only candidates with combined score > 0.
        for i in candidate_indices:
            score = float(all_scores[i])
            if score > 0.0:
                lex_ranked.append((doc_ids[i], score))

        lex_ranked.sort(key=lambda t: (-t[1], t[0]))

    # 3. Dense scoring.
    dense_ranked: list[tuple[str, float]] = []  # (doc_id, cosine)
    if mode in ("dense", "hybrid") and query_vector is not None:
        for doc_id in candidate_ids:
            row = index.row_map[doc_id]
            vec = index.vectors[row]
            cosine = float(vec @ query_vector)
            dense_ranked.append((doc_id, cosine))
        dense_ranked.sort(key=lambda t: (-t[1], t[0]))

    return lex_ranked, dense_ranked


# ------------------------------------------------------------------ #
# Top-level retrieve: blog stage 8                                    #
# ------------------------------------------------------------------ #


def retrieve(
    index: LoadedIndex,
    question: str,
    *,
    mode: Literal["lexical", "dense", "hybrid"],
    processor: TextProcessor,
    provider: EmbeddingProvider,
    metadata_filter: MetadataFilter | None = None,
    rrf_config: RRFConfig,
    top_k: int = 10,
    shard_ids: list[int] | None = None,
    per_shard_limit: int | None = None,
    expansions: list[str] | None = None,
    query_variant: str | None = None,
    intents: Sequence[SearchIntent | str] | None = None,
    fusion: Literal["global", "per_shard"] = "global",
    refine_top_n: int | None = None,
    max_snippet_chars: int = 500,
) -> list[SearchHit]:
    """End-to-end retrieval over a :class:`~cybernaut_mini.indexing.LoadedIndex`.

    Tokenizes the question, embeds it (unless lexical-only), broadcasts the package to
    the selected shards, reduces the responses into one ranking, runs the intent pass
    over the top results, and returns ranked
    :class:`~cybernaut_mini.models.SearchHit` objects.

    Parameters that changed meaning in the move to stage 8
    ------------------------------------------------------
    ``per_shard_limit``
        Now genuinely per shard, in every mode. It used to be applied to the *fused*
        list in hybrid mode, so ``per_shard_limit=3`` capped a 12-shard hybrid query at
        three results in total rather than at three per shard. Defaults to stage 8's
        1000, which is a cap on what a shard sends back, not on what you get.

    New parameters
    --------------
    ``intents``
        Stage-3 search intents (or bare phrases) for the Aho-Corasick pass. Derived from
        the question when omitted; ``()`` disables the pass and reproduces the ranking
        without it.
    ``fusion``
        ``"global"`` (default) fuses lexical and dense over all shards at once;
        ``"per_shard"`` is the post's literal step 4 and is kept only because the
        comparison is the evidence for the default.
    ``refine_top_n``
        How many top results get their full text read and scanned. Defaults to
        ``2 * top_k``; ``0`` disables the pass.
    ``max_snippet_chars``
        The post's user-chosen snippet maximum, 1..500.
    """
    # Imported here, not at module scope: stage 8's map step imports
    # `score_shard_components` from this module, so a top-level import would close a
    # cycle. See the module docstring.
    from cybernaut_mini.query.s8_retrieve import (
        DEFAULT_PER_SHARD_LIMIT,
        BroadcastRequest,
        retrieve_map_reduce,
    )

    if top_k < 1 or (per_shard_limit is not None and per_shard_limit < 1):
        return []

    query_tokens = processor.content_tokens(question)
    if not query_tokens:
        query_tokens = processor.tokenize(question)
    expansion_tokens: list[str] = list(expansions or [])

    effective_mode = mode
    if mode != "dense" and not query_tokens and not expansion_tokens:
        # Nothing lexically searchable. Lexical mode has no answer at all; hybrid
        # degrades to its dense half, which is what the old code computed anyway (an
        # empty BM25 list fused with the dense list is the dense order).
        if mode == "lexical":
            return []
        effective_mode = "dense"

    query_vector: FloatArray | None = None
    if effective_mode != "lexical":
        query_vector = provider.embed_queries([question])[0]

    target_shard_ids = sorted(shard_ids if shard_ids is not None else index.manifests.keys())

    request = BroadcastRequest(
        query_tokens=tuple(query_tokens),
        expansion_tokens=tuple(expansion_tokens),
        query_vector=query_vector,
        intents=_resolve_intents(question, intents),
        shard_ids=tuple(target_shard_ids),
        metadata_filter=metadata_filter,
        mode=effective_mode,
        result_limit=top_k,
        per_shard_limit=(
            DEFAULT_PER_SHARD_LIMIT if per_shard_limit is None else per_shard_limit
        ),
        query_variant=query_variant or question,
    )

    result = retrieve_map_reduce(
        index,
        request,
        rrf_config=rrf_config,
        fusion=fusion,
        refine_top_n=refine_top_n,
        max_snippet_chars=max_snippet_chars,
    )
    return result.hits


def _resolve_intents(
    question: str, intents: Sequence[SearchIntent | str] | None
) -> tuple[SearchIntent, ...]:
    """Stage-3 intents for the scan: the caller's, or derived from the question."""
    from cybernaut_mini.query.s3_intents import SearchIntent as _SearchIntent

    if intents is None:
        from cybernaut_mini.routing import query_intents

        return query_intents(question)
    return tuple(
        intent
        if isinstance(intent, _SearchIntent)
        else _SearchIntent(text=intent, tokens=tuple(intent.split()), score=1.0, span=(0, 0))
        for intent in intents
    )
