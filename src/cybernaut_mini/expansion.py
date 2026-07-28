"""Query expansion via term-graph neighbour lookup.

Selects candidate expansion terms from the term graphs of the routed shards,
weighted by shard weight and edge weight, filtered by stopwords, length, and
cross-shard ubiquity.  The caller (retrieval layer) applies the expansion
terms at search time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cybernaut_mini.text import STOPWORDS

if TYPE_CHECKING:
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.text import TextProcessor


def expand_query(
    index: LoadedIndex,
    query: str,
    *,
    shard_ids: list[int],
    shard_weights: dict[int, float] | None = None,
    processor: TextProcessor,
    max_terms: int = 5,
) -> list[str]:
    """Return up to *max_terms* expansion terms for *query*.

    Algorithm
    ---------
    1. Derive original terms from ``processor.content_tokens(query)``.
    2. For each selected shard, walk the term graph: for each original term
       present as a node, accumulate ``edge_weight * shard_weight`` on each
       neighbour.
    3. Exclude: original terms, STOPWORDS, tokens shorter than 3 chars, and
       terms that appear in the keyword lists of more than 80 % of ALL shards
       in the index (cross-shard ubiquity filter).
    4. Return the top *max_terms* candidates ordered by ``(-score, term)`` for
       determinism.

    Parameters
    ----------
    index:
        Fully-loaded index.
    query:
        Raw query string.
    shard_ids:
        Shards to draw expansions from (typically the routed shards).
    shard_weights:
        Optional per-shard weight multiplier (default 1.0 for missing entries).
    processor:
        Text processor instance matching the index build settings.
    max_terms:
        Maximum number of expansion terms to return.
    """
    if shard_weights is None:
        shard_weights = {}

    original_terms = set(processor.content_tokens(query))

    # ------------------------------------------------------------------
    # Pre-compute cross-shard document frequency for ubiquity filter.
    # A term is "ubiquitous" when it appears in > 80 % of ALL shards.
    # ------------------------------------------------------------------
    all_shard_ids = list(index.manifests.keys())
    n_total_shards = len(all_shard_ids)
    ubiquity_threshold = 0.8 * n_total_shards  # exclusive: > 80 %

    term_shard_df: dict[str, int] = {}
    for sid in all_shard_ids:
        for kw in index.manifests[sid].keywords:
            term_shard_df[kw.term] = term_shard_df.get(kw.term, 0) + 1

    ubiquitous: set[str] = {
        term for term, df in term_shard_df.items() if df > ubiquity_threshold
    }

    # ------------------------------------------------------------------
    # Accumulate neighbour scores from term graphs of the selected shards.
    # ------------------------------------------------------------------
    candidate_scores: dict[str, float] = {}
    for sid in shard_ids:
        w_shard = shard_weights.get(sid, 1.0)
        graph = index.manifests[sid].term_graph
        for term in original_terms:
            if term not in graph:
                continue
            for neighbour, edge_weight in graph[term].items():
                candidate_scores[neighbour] = (
                    candidate_scores.get(neighbour, 0.0) + edge_weight * w_shard
                )

    # ------------------------------------------------------------------
    # Apply exclusion filters.
    # ------------------------------------------------------------------
    filtered: dict[str, float] = {
        term: score
        for term, score in candidate_scores.items()
        if term not in original_terms
        and term not in STOPWORDS
        and len(term) >= 3
        and term not in ubiquitous
    }

    # ------------------------------------------------------------------
    # Rank and cap.
    # ------------------------------------------------------------------
    ranked = sorted(filtered.items(), key=lambda kv: (-kv[1], kv[0]))
    return [term for term, _ in ranked[:max_terms]]
