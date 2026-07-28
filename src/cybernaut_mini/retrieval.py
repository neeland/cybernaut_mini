"""Retrieval layer: lexical, dense, and hybrid search over a LoadedIndex.

Implements BM25 + cosine dense retrieval, RRF fusion, shard-level search,
cross-shard map-reduce merge, and snippet extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from cybernaut_mini.config import ConfigError, RRFConfig
from cybernaut_mini.models import MetadataFilter, SearchHit
from cybernaut_mini.providers.embeddings import EmbeddingProvider, HashEmbedder
from cybernaut_mini.rrf import RankedList, rrf_fuse
from cybernaut_mini.text import TextProcessor, normalize

if TYPE_CHECKING:
    import numpy.typing as npt

    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.models import IndexMeta

    FloatArray = npt.NDArray[np.float32]


# ------------------------------------------------------------------ #
# Provider factory                                                    #
# ------------------------------------------------------------------ #


def provider_from_meta(meta: IndexMeta, *, offline: bool = False) -> EmbeddingProvider:
    """Reconstruct the embedding provider that was used to build the index.

    If the index was built with a hash provider (``meta.embedding_model`` starts
    with ``"hash-"``), returns a :class:`HashEmbedder` with the correct dim.
    Otherwise returns a :class:`SentenceTransformersEmbedder` unless ``offline``
    is True, in which case raises :class:`ConfigError`.
    """
    if meta.embedding_model.startswith("hash-"):
        return HashEmbedder(dim=meta.embedding_dim)
    if offline:
        msg = (
            "offline mode: embedding provider 'sentence_transformers' may require a "
            "model download; use an index built with provider 'hash'"
        )
        raise ConfigError(msg)
    from cybernaut_mini.providers.embeddings import SentenceTransformersEmbedder

    # Reuse the build-time revision so query vectors come from the same weights
    # as the stored document vectors.
    return SentenceTransformersEmbedder(meta.embedding_model, revision=meta.embedding_revision)


# ------------------------------------------------------------------ #
# Snippet extraction                                                  #
# ------------------------------------------------------------------ #

_WORD_BOUNDARY = re.compile(r"\b")


def make_snippet(text: str, query_tokens: list[str]) -> str:
    """Return a deterministic snippet of <= 500 chars centred on the earliest match.

    Searches ``normalize(text).lower()`` for whole-token matches of any
    ``query_tokens`` (regex word boundaries). Maps the match offset back into
    ``normalize(text)`` (same length) to produce the snippet. Trims both ends to
    whitespace boundaries. Falls back to the leading 500 chars when no token matches.
    """
    norm = normalize(text)
    lower = norm.lower()
    max_len = 500

    match_start: int | None = None
    match_end: int | None = None

    # Find the earliest occurrence of any query token.
    for token in query_tokens:
        if not token:
            continue
        escaped = re.escape(token.lower())
        pattern = re.compile(r"\b" + escaped + r"\b")
        m = pattern.search(lower)
        if m is not None and (match_start is None or m.start() < match_start):
            match_start = m.start()
            match_end = m.end()

    if match_start is None:
        # No match: return the leading <=500 chars trimmed to a word boundary.
        return _trim_to_word_boundary(norm, 0, max_len)

    # Centre a window of max_len around the match.
    match_center = (match_start + match_end) // 2  # type: ignore[operator]
    half = max_len // 2
    win_start = max(0, match_center - half)
    win_end = min(len(norm), win_start + max_len)
    # Adjust start if we hit the end boundary.
    if win_end - win_start < max_len:
        win_start = max(0, win_end - max_len)

    return _trim_to_word_boundary(norm, win_start, win_end)


def _trim_to_word_boundary(text: str, start: int, end: int) -> str:
    """Trim slice [start:end] of ``text`` so it does not split mid-word."""
    snippet = text[start:end]
    if not snippet:
        return snippet

    # If we're not at the very beginning, advance past a leading partial word.
    if start > 0 and snippet and snippet[0] not in (" ", "\t", "\n"):
        space = snippet.find(" ")
        if space == -1:
            # Whole snippet is mid-word; fall back to empty (edge case).
            return ""
        snippet = snippet[space:].lstrip()

    # If we're not at the very end, drop trailing partial word.
    if end < len(text) and snippet and snippet[-1] not in (" ", "\t", "\n"):
        space = snippet.rfind(" ")
        if space == -1:
            return ""
        snippet = snippet[:space].rstrip()

    return snippet.strip()


# ------------------------------------------------------------------ #
# ShardHit dataclass                                                  #
# ------------------------------------------------------------------ #


@dataclass
class ShardHit:
    doc_id: str
    shard_id: int
    bm25_score: float | None
    bm25_rank: int | None
    dense_score: float | None
    dense_rank: int | None
    fused_score: float
    rrf_contributions: dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------ #
# Per-shard search                                                    #
# ------------------------------------------------------------------ #


def search_shard(
    index: LoadedIndex,
    shard_id: int,
    *,
    query_tokens: list[str],
    expansion_tokens: list[str],
    query_vector: FloatArray | None,
    mode: str,
    metadata_filter: MetadataFilter | None,
    rrf_k: int,
    lexical_weight: float,
    dense_weight: float,
    limit: int,
) -> list[ShardHit]:
    """Search one shard and return up to ``limit`` ShardHits.

    Steps:
    1. Filter candidate docs by metadata.
    2. Score lexically (BM25 + expansion) for modes "lexical"/"hybrid".
    3. Score densely (cosine) for modes "dense"/"hybrid".
    4. Fuse with RRF for "hybrid"; use raw score for "lexical"/"dense".
    5. Return top ``limit`` hits.
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
        return []

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

    # 4. Fuse / rank.
    hits: list[ShardHit] = []

    if mode == "lexical":
        for rank_0, (doc_id, score) in enumerate(lex_ranked[:limit]):
            hits.append(
                ShardHit(
                    doc_id=doc_id,
                    shard_id=shard_id,
                    bm25_score=score,
                    bm25_rank=rank_0 + 1,
                    dense_score=None,
                    dense_rank=None,
                    fused_score=score,
                    rrf_contributions={},
                )
            )

    elif mode == "dense":
        for rank_0, (doc_id, score) in enumerate(dense_ranked[:limit]):
            hits.append(
                ShardHit(
                    doc_id=doc_id,
                    shard_id=shard_id,
                    bm25_score=None,
                    bm25_rank=None,
                    dense_score=score,
                    dense_rank=rank_0 + 1,
                    fused_score=score,
                    rrf_contributions={},
                )
            )

    else:  # hybrid
        lex_list = RankedList(
            name="lexical",
            weight=lexical_weight,
            ids=tuple(doc_id for doc_id, _ in lex_ranked),
            scores=tuple(score for _, score in lex_ranked),
        )
        dense_list = RankedList(
            name="dense",
            weight=dense_weight,
            ids=tuple(doc_id for doc_id, _ in dense_ranked),
            scores=tuple(score for _, score in dense_ranked),
        )
        fused = rrf_fuse([lex_list, dense_list], k=rrf_k)

        # Build lookup maps for component ranks/scores.
        lex_rank_map = {doc_id: (r + 1, score) for r, (doc_id, score) in enumerate(lex_ranked)}
        dense_rank_map = {
            doc_id: (r + 1, score) for r, (doc_id, score) in enumerate(dense_ranked)
        }

        for item in fused[:limit]:
            doc_id = item.id
            b_rank, b_score = lex_rank_map.get(doc_id, (None, None))
            d_rank, d_score = dense_rank_map.get(doc_id, (None, None))
            hits.append(
                ShardHit(
                    doc_id=doc_id,
                    shard_id=shard_id,
                    bm25_score=b_score,
                    bm25_rank=b_rank,
                    dense_score=d_score,
                    dense_rank=d_rank,
                    fused_score=item.score,
                    rrf_contributions=dict(item.contributions),
                )
            )

    return hits


# ------------------------------------------------------------------ #
# Cross-shard merge                                                   #
# ------------------------------------------------------------------ #


def merge_shard_results(
    variant_results: list[tuple[str, list[ShardHit]]],
    *,
    top_k: int,
) -> list[tuple[str, ShardHit, float]]:
    """Map-reduce merge of per-variant shard hits.

    For each (variant, hits) pair: per doc keep the BEST fused_score across shards.
    Then SUM the per-variant best scores across variants into the doc's final score.
    Sort by (-final_score, doc_id). Return top_k as (best_variant, best_shard_hit, final_score).
    """
    # variant -> doc_id -> best ShardHit (highest fused_score within that variant)
    variant_best: dict[str, dict[str, ShardHit]] = {}
    for variant, hits in variant_results:
        best = variant_best.setdefault(variant, {})
        for hit in hits:
            existing = best.get(hit.doc_id)
            if existing is None or hit.fused_score > existing.fused_score:
                best[hit.doc_id] = hit

    # doc_id -> (final_score, best_variant, best_shard_hit)
    # Sum the per-variant best scores; track which variant contributed the highest single score.
    doc_final: dict[str, float] = {}
    doc_best_variant: dict[str, str] = {}
    doc_best_hit: dict[str, ShardHit] = {}

    for variant, best in variant_best.items():
        for doc_id, hit in best.items():
            doc_final[doc_id] = doc_final.get(doc_id, 0.0) + hit.fused_score
            existing_hit = doc_best_hit.get(doc_id)
            if existing_hit is None or hit.fused_score > existing_hit.fused_score:
                doc_best_hit[doc_id] = hit
                doc_best_variant[doc_id] = variant

    ranked = sorted(doc_final.keys(), key=lambda d: (-doc_final[d], d))
    return [
        (doc_best_variant[doc_id], doc_best_hit[doc_id], doc_final[doc_id])
        for doc_id in ranked[:top_k]
    ]


# ------------------------------------------------------------------ #
# Top-level retrieve                                                  #
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
) -> list[SearchHit]:
    """End-to-end retrieval over a :class:`~cybernaut_mini.indexing.LoadedIndex`.

    Tokenizes the question, embeds it (unless lexical-only), searches each shard,
    merges results, and returns ranked :class:`~cybernaut_mini.models.SearchHit` objects.
    """
    # Tokenize.
    query_tokens = processor.content_tokens(question)
    if not query_tokens:
        query_tokens = processor.tokenize(question)

    expansion_tokens: list[str] = expansions or []
    all_query_tokens = query_tokens + expansion_tokens

    # Embed query unless lexical-only.
    query_vector: FloatArray | None = None
    if mode != "lexical":
        query_vector = provider.embed_queries([question])[0]

    # Select shards to search.
    target_shard_ids = sorted(shard_ids if shard_ids is not None else index.manifests.keys())
    limit = per_shard_limit if per_shard_limit is not None else top_k

    # Search each shard.
    shard_hits: list[ShardHit] = []
    for sid in target_shard_ids:
        hits = search_shard(
            index,
            sid,
            query_tokens=query_tokens,
            expansion_tokens=expansion_tokens,
            query_vector=query_vector,
            mode=mode,
            metadata_filter=metadata_filter,
            rrf_k=rrf_config.k,
            lexical_weight=rrf_config.lexical_weight,
            dense_weight=rrf_config.dense_weight,
            limit=limit,
        )
        shard_hits.extend(hits)

    # Use a single variant (the question or explicit query_variant label).
    variant_label = query_variant or question
    merged = merge_shard_results([(variant_label, shard_hits)], top_k=top_k)

    # Sanity: no duplicate doc_ids.
    merged_ids = [doc_id for _, hit, _ in merged for doc_id in [hit.doc_id]]
    assert len(merged_ids) == len(set(merged_ids)), "duplicate doc_ids after merge"

    results: list[SearchHit] = []
    for rank, (variant, hit, final_score) in enumerate(merged, start=1):
        document = index.by_id[hit.doc_id]
        snippet = make_snippet(document.text, all_query_tokens)
        results.append(
            SearchHit(
                document=document,
                rank=rank,
                score=final_score,
                shard_id=hit.shard_id,
                bm25_score=hit.bm25_score,
                bm25_rank=hit.bm25_rank,
                dense_score=hit.dense_score,
                dense_rank=hit.dense_rank,
                rrf_contributions=hit.rrf_contributions,
                query_variant=variant,
                expansion_terms=expansion_tokens,
                snippet=snippet,
            )
        )

    return results
