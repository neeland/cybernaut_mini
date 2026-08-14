"""Stage 7 of the query pipeline: shard-based question expansion.

Input: the question's original search words (stage 2's stems) plus the shards stage 6
ranked. Output: those same words, preserved and marked, plus a bounded set of new ones
drawn from the *per-shard* synonym graphs and marked ``[NEW]`` — the shape of the post's
own worked example, in which nine originals gain twenty-one additions.

    >>> from cybernaut_mini.query.s7_expand import ExpansionConfig, ShardGraph, expand_terms
    >>> genetics = ShardGraph(shard_id=11343, weight=1.0, graph={
    ...     "gene": {"crispr": 4.0, "genome": 3.0, "prokaryot": 1.0}})
    >>> expansion = expand_terms(["gene"], [genetics], config=ExpansionConfig(hard_cap=2))
    >>> expansion.render()
    '* "gene"\\n* "crispr" [NEW]\\n* "genome" [NEW]'

The same token in a shard with a different sense expands differently — the claim the post
makes for this stage, and the reason the graphs must be per shard:

    >>> chocolate = ShardGraph(shard_id=90210, weight=1.0, graph={
    ...     "gene": {"wonka": 5.0, "wilder": 2.0}})
    >>> expand_terms(["gene"], [chocolate]).new_terms
    ('wonka', 'wilder')

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 7: "we use the synonym
    graphs in each shard to probabilistically expand our search terms. Because each shard
    is lexically and semantically coherent, the synonyms in each shard are unambiguous.
    Put simply, 'gene' in shard 11,343 only has the genetic meaning. We don't need to
    worry that Gene is also a name and, in some other shards, would co-occur a lot with
    'Willy Wonka'. It's worth mentioning that this step can introduce some noise /
    serendipity, so we are very careful not to overwhelm the original search words."
    Stack: NumPy. Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

What is disclosed and what is inferred
--------------------------------------
[disclosed] Expansion reads a synonym graph *per shard*; higher-ranked shards are the ones
    consulted; originals survive into the expanded set and additions are marked; the
    number of additions is deliberately bounded ("very careful not to overwhelm").
[inferred] Everything numeric. That edges are turned into an L1-normalised distribution
    per ``(shard, term)`` and summed with the shard's rank weight
    (:mod:`~cybernaut_mini.query.s7_expand.graph`); that candidates are filtered by
    stop-word membership, minimum length and cross-shard ubiquity
    (:mod:`~cybernaut_mini.query.s7_expand.filters`); that the cap is a ratio of the
    original term count, defaulting to 2.0 against the example's 2.33
    (:mod:`~cybernaut_mini.query.s7_expand.selection`).

Determinism
-----------
"Probabilistically" is real — the candidate scores are a probability distribution — but
the default selector is deterministic top-k, because this repo requires that the same
input give the same output. ``ExpansionConfig(mode="sampled")`` opts into seeded weighted
sampling without replacement, which is reproducible for a fixed ``seed`` but changes with
it; that is the serendipity knob the post mentions, off by default.

Assumptions:
    - The expansion terms in the post's worked example ("mojica", "raffinos", "bacto")
      are properties of NOSIBLE's 250M-document corpus, not of the technique. Nothing here
      asserts those strings; the tests assert the invariants the example demonstrates.
    - Stage 7 consumes ``ShardManifest.term_graph`` through the resident manifests and
      touches no document, embedding or artifact sidecar, so its cost is a function of the
      number of routed shards rather than of corpus size.

Alternatives rejected:
    - Embedding-based expansion (nearest neighbours of the query vector in a shard's
      vocabulary). Plausible and more powerful, but the post says "synonym graphs", the
      graphs exist in the manifests, and a vector path would need an encoder at query
      time for a stage the post gives one dependency: NumPy.
    - Expanding once against the union of all routed shards' graphs. Cheaper, and exactly
      the ambiguity the post's shard argument exists to avoid: pooling "gene" across a
      genetics shard and a film shard reinstates "Willy Wonka".
    - Keeping ``cybernaut_mini.expansion.expand_query``'s single-function shape. It has
      the right instincts (shard weighting, ubiquity filter, deterministic ordering) and
      they are preserved here, but it returns bare strings with no provenance, no cap
      relative to the question, and no way to test the filters apart from the walk.
"""

from __future__ import annotations

from cybernaut_mini.query.s7_expand.expand import (
    ExpandedTerm,
    Expansion,
    ExpansionConfig,
    expand_query,
    expand_terms,
)
from cybernaut_mini.query.s7_expand.filters import (
    DEFAULT_MIN_LENGTH,
    DEFAULT_UBIQUITY_THRESHOLD,
    TermFilter,
    UbiquityFilter,
    UbiquitySource,
)
from cybernaut_mini.query.s7_expand.graph import (
    ShardGraph,
    TermGraph,
    neighbour_distribution,
    rank_weights,
    shard_graphs_from_index,
)
from cybernaut_mini.query.s7_expand.question import expand_question, question_terms
from cybernaut_mini.query.s7_expand.selection import (
    DEFAULT_MAX_NEW_PER_ORIGINAL,
    Candidate,
    SelectionMode,
    activate,
    expansion_cap,
    filter_candidates,
    select,
)

__all__ = [
    "DEFAULT_MAX_NEW_PER_ORIGINAL",
    "DEFAULT_MIN_LENGTH",
    "DEFAULT_UBIQUITY_THRESHOLD",
    "Candidate",
    "ExpandedTerm",
    "Expansion",
    "ExpansionConfig",
    "SelectionMode",
    "ShardGraph",
    "TermFilter",
    "TermGraph",
    "UbiquityFilter",
    "UbiquitySource",
    "activate",
    "expand_query",
    "expand_question",
    "expand_terms",
    "expansion_cap",
    "filter_candidates",
    "neighbour_distribution",
    "question_terms",
    "rank_weights",
    "select",
    "shard_graphs_from_index",
]
