"""Query expansion — the repo-level adapter over blog stage 7.

``expand_query`` keeps the signature its callers (``agent/search.py``, the notebooks,
``tests/test_term_graph.py``) already use — a raw query string in, a plain list of new
terms out — and delegates every decision to
:mod:`cybernaut_mini.query.s7_expand`. The hand-rolled neighbour walk, stopword/length
filters and ubiquity table that used to live here are deleted, not duplicated: they
were an earlier, less careful version of the same algorithm.

Two things changed in the move, both improvements and both visible:

* **Activation is a probability distribution.** Stage 7 L1-normalises each ``(shard,
  original term)`` neighbourhood before weighting it by the shard weight, so a shard
  with a densely connected term cannot outvote the others merely by having more edges.
  The manifest term graphs are already row-normalised by the build, so for real indexes
  this is a no-op; for a hand-built graph it is not.
* **The ubiquity table is built once per index, not once per call.** The old code
  recomputed the cross-shard document frequency of every keyword on every single
  expansion — an O(n_shards x keywords) scan inside the query path. It is now cached
  per ``(index, source, threshold)`` behind a :class:`weakref.WeakKeyDictionary`. The
  threshold (0.8, exclusive) and the source (manifest keywords) are unchanged, so the
  filter admits and rejects exactly what it did before.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 7, "Shard-based
    Question Expansion": each shard "has a graph of terms that co-occur", the routed
    shards vote on synonyms for the question's terms, and the post is "very careful not
    to overwhelm the original search words". Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - [inferred] ``max_terms`` is an absolute cap on the *new* terms, which is what this
      function has always meant by it. Stage 7's native knob is a ratio
      (``max_new_per_original``) with ``hard_cap`` as a ceiling, so the ratio is set
      high enough to be non-binding and ``hard_cap`` does the work.
    - [inferred] the ubiquity source stays ``"keywords"`` rather than stage 7's own
      default of ``"graph"``. Under the per-shard term graphs the graph source is the
      better statistic, but switching it silently would change which terms this function
      rejects; ``ubiquity_source=`` is the seam for that experiment.

Alternatives rejected:
    - Returning stage 7's :class:`~cybernaut_mini.query.s7_expand.Expansion` from this
      function. It carries provenance the callers do not consume and would break
      ``agent/search.py``'s ``tuple(expand_query(...))``. Callers that want the marked,
      scored terms should call :func:`cybernaut_mini.query.s7_expand.expand_query`
      directly — this module deliberately adds nothing to it.
    - Keeping a second copy of the filters here "for the string API". The whole point
      of the rewire is that there is one stage 7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

from cybernaut_mini.query.s7_expand import (
    DEFAULT_UBIQUITY_THRESHOLD,
    Expansion,
    ExpansionConfig,
    UbiquityFilter,
    UbiquitySource,
)
from cybernaut_mini.query.s7_expand import expand_query as _expand_terms_from_index

if TYPE_CHECKING:
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.text import TextProcessor

__all__ = ["expand", "expand_query", "ubiquity_filter"]

_UBIQUITY: WeakKeyDictionary[Any, dict[tuple[str, float], UbiquityFilter]] = WeakKeyDictionary()


def ubiquity_filter(
    index: LoadedIndex,
    *,
    source: UbiquitySource = "keywords",
    threshold: float = DEFAULT_UBIQUITY_THRESHOLD,
) -> UbiquityFilter:
    """The cross-shard frequency table for ``index``, built once and cached weakly.

    Building it reads every shard manifest, which is the one O(n_shards) step stage 7
    has; caching it per index keeps that cost off the per-query path. The cache is weak
    on the index, so it is released with the index it describes.
    """
    by_key = _UBIQUITY.setdefault(index, {})
    key = (source, threshold)
    cached = by_key.get(key)
    if cached is None:
        cached = UbiquityFilter.from_index(index, threshold=threshold, source=source)
        by_key[key] = cached
    return cached


def expand(
    index: LoadedIndex,
    query: str,
    *,
    shard_ids: list[int],
    shard_weights: dict[int, float] | None = None,
    processor: TextProcessor,
    max_terms: int = 5,
    ubiquity_source: UbiquitySource | None = "keywords",
    ubiquity_threshold: float = DEFAULT_UBIQUITY_THRESHOLD,
) -> Expansion:
    """Stage 7 over ``query``, returning the full :class:`Expansion` record.

    The originals come from ``processor.content_tokens(query)`` — tokenisation is
    stage 2's job, and stage 7 takes stems. ``ubiquity_source=None`` disables the
    cross-shard ubiquity filter entirely.
    """
    if max_terms < 0:
        msg = f"max_terms must be non-negative, got {max_terms}"
        raise ValueError(msg)
    originals = processor.content_tokens(query)
    ubiquity = (
        None
        if ubiquity_source is None
        else ubiquity_filter(index, source=ubiquity_source, threshold=ubiquity_threshold)
    )
    # max_new_per_original is deliberately non-binding: with a ratio of `max_terms` per
    # original, floor(n_originals * ratio) >= max_terms for any n_originals >= 1, so the
    # hard cap is what decides. hard_cap can only ever *lower* the ratio cap.
    config = ExpansionConfig(
        max_new_per_original=float(max(max_terms, 1)),
        hard_cap=max_terms,
    )
    return _expand_terms_from_index(
        index,
        originals,
        shard_ids=shard_ids,
        shard_weights=shard_weights,
        config=config,
        ubiquity=ubiquity,
    )


def expand_query(
    index: LoadedIndex,
    query: str,
    *,
    shard_ids: list[int],
    shard_weights: dict[int, float] | None = None,
    processor: TextProcessor,
    max_terms: int = 5,
) -> list[str]:
    """Return up to *max_terms* expansion terms for *query*, best first.

    Algorithm (all of it in :mod:`cybernaut_mini.query.s7_expand`)
    -------------------------------------------------------------
    1. Derive original terms from ``processor.content_tokens(query)``.
    2. For each selected shard, L1-normalise each original term's outgoing term-graph
       edges into a transition distribution and add ``shard_weight * probability`` to
       each neighbour; divide by the total contributed weight so the activation is a
       distribution over candidates.
    3. Exclude: the original terms, stopwords, terms shorter than 3 characters,
       all-digit terms, and terms appearing in the keyword lists of more than 80 % of
       ALL shards in the index (cross-shard ubiquity).
    4. Return the top *max_terms* survivors ordered by ``(-score, term)``.

    Parameters
    ----------
    index:
        Loaded index. Only the routed shards' manifests are read.
    query:
        Raw query string.
    shard_ids:
        Shards to draw expansions from (typically the routed shards, best first).
    shard_weights:
        Optional per-shard weight multiplier (default 1.0 for missing entries).
        :func:`cybernaut_mini.query.s7_expand.rank_weights` derives one from rank alone.
    processor:
        Text processor instance matching the index build settings.
    max_terms:
        Maximum number of expansion terms to return.
    """
    expansion = expand(
        index,
        query,
        shard_ids=shard_ids,
        shard_weights=shard_weights,
        processor=processor,
        max_terms=max_terms,
    )
    return list(expansion.new_terms)
