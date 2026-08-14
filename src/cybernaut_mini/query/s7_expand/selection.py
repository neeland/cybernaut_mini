"""Scoring candidates across the routed shards, and choosing which ones survive the cap.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 7: "we use the synonym
    graphs in each shard to *probabilistically* expand our search terms... we are very
    careful not to overwhelm the original search words." The worked example expands 9
    original stems with 21 new ones. Stack: NumPy. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - "Probabilistically" is implemented as a real distribution over candidates
      [inferred]: each ``(shard, original term)`` pair contributes its L1-normalised
      neighbour distribution, weighted by the shard's rank weight, and the whole thing is
      divided by the total weight actually contributed. The unfiltered activation
      therefore sums to 1.0, which is what makes ``score`` comparable across queries and
      what the sampled selector needs.
    - The default *selector* is deterministic top-k, not sampling [inferred]. The repo
      requires that the same input give the same output, and top-k over a probability
      distribution is the zero-variance limit of sampling from it — the same ranking, with
      the serendipity turned off. ``mode="sampled"`` restores the serendipity through
      seeded weighted sampling without replacement (Efraimidis-Spirakis), which is
      reproducible for a fixed seed but *not* stable under a change of seed, corpus or
      candidate set. Top-k is the tested default; sampling is opt-in and its seed is part
      of the config so a trace can replay it.
    - "Not overwhelm" is given a number: the cap is a multiple of the count of original
      terms, defaulting to 2 new per original. The post's own example is 21 new for 9
      originals (2.33), so 2.0 is the conservative reading of the same shape; the exact
      terms are a function of the corpus and cannot be reproduced here.

Alternatives rejected:
    - An absolute cap ("always 20 expansions"). A two-word question would then be buried
      under 20 synonyms, which is precisely the failure the post warns about; the
      constraint is relative by nature. An absolute ceiling is still available as a second
      limit for very long questions.
    - Sampling with replacement, or independent Bernoulli draws per candidate. Both can
      return duplicates or a wildly variable count, and neither respects the cap without a
      rejection loop.
    - ``random.Random`` for the sampled mode. NumPy is stage 7's disclosed dependency, and
      ``numpy.random.default_rng`` is explicitly reproducible across platforms for a fixed
      seed and bit generator, which ``random`` does not promise across versions.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from cybernaut_mini.query.s7_expand.filters import TermFilter
from cybernaut_mini.query.s7_expand.graph import ShardGraph, neighbour_distribution

#: ``"top_k"`` takes the highest-scoring candidates; ``"sampled"`` draws them without
#: replacement with probability proportional to score, from a seeded generator.
SelectionMode = Literal["top_k", "sampled"]

#: New terms allowed per original term. See the module docstring for the arithmetic.
DEFAULT_MAX_NEW_PER_ORIGINAL = 2.0

__all__ = [
    "DEFAULT_MAX_NEW_PER_ORIGINAL",
    "Candidate",
    "SelectionMode",
    "activate",
    "expansion_cap",
    "filter_candidates",
    "select",
]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One candidate expansion term and the evidence behind it.

    Attributes:
        term: the candidate stem.
        score: its share of the total expansion mass, in ``(0, 1]``.
        sources: the original terms that activated it, sorted.
        shards: the shards that voted for it, in the order they were consulted.
    """

    term: str
    score: float
    sources: tuple[str, ...]
    shards: tuple[int, ...]

    @property
    def support(self) -> int:
        """How many distinct original terms activated this candidate."""
        return len(self.sources)


def activate(
    originals: Sequence[str],
    shard_graphs: Iterable[ShardGraph],
) -> dict[str, Candidate]:
    """Spread mass from ``originals`` onto their neighbours across ``shard_graphs``.

    The returned scores sum to 1.0 whenever at least one original term appears in at least
    one graph with a positive-weight shard, and the mapping is empty otherwise — the
    no-graph, empty-graph and no-overlap cases all degrade to "no expansion" without a
    special case.

    ``originals`` is de-duplicated but its order is irrelevant to the result: mass is
    summed, and summation over a dict in insertion order is not associative in floating
    point, so candidates are accumulated per source term in sorted order to keep the
    arithmetic itself reproducible.
    """
    unique_originals = sorted(set(originals))
    mass: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    shards: dict[str, list[int]] = {}
    total_weight = 0.0

    for shard in shard_graphs:
        if shard.weight <= 0.0:
            continue
        for term in unique_originals:
            distribution = neighbour_distribution(shard.graph, term)
            if not distribution:
                continue
            total_weight += shard.weight
            for neighbour, probability in sorted(distribution.items()):
                mass[neighbour] = mass.get(neighbour, 0.0) + shard.weight * probability
                sources.setdefault(neighbour, set()).add(term)
                bucket = shards.setdefault(neighbour, [])
                if not bucket or bucket[-1] != shard.shard_id:
                    bucket.append(shard.shard_id)

    if total_weight <= 0.0:
        return {}

    return {
        term: Candidate(
            term=term,
            score=score / total_weight,
            sources=tuple(sorted(sources[term])),
            shards=tuple(shards[term]),
        )
        for term, score in mass.items()
    }


def filter_candidates(
    candidates: dict[str, Candidate],
    term_filter: TermFilter,
    *,
    min_score: float = 0.0,
) -> list[Candidate]:
    """Drop inadmissible candidates, returning what is left in ``(-score, term)`` order."""
    kept = [
        candidate
        for candidate in candidates.values()
        if candidate.score > min_score and term_filter.accepts(candidate.term)
    ]
    kept.sort(key=lambda candidate: (-candidate.score, candidate.term))
    return kept


def expansion_cap(
    n_originals: int,
    *,
    max_new_per_original: float = DEFAULT_MAX_NEW_PER_ORIGINAL,
    hard_cap: int | None = None,
) -> int:
    """How many new terms a question with ``n_originals`` original terms may gain.

    ``floor(n_originals * max_new_per_original)``, then clipped by ``hard_cap``. A
    question with no original terms gets no expansions, which is the only sensible
    reading of a ratio applied to zero.

    >>> expansion_cap(9)
    18
    >>> expansion_cap(9, max_new_per_original=2.5, hard_cap=20)
    20
    """
    if n_originals <= 0:
        return 0
    if max_new_per_original < 0.0:
        msg = f"max_new_per_original must be non-negative, got {max_new_per_original!r}"
        raise ValueError(msg)
    cap = math.floor(n_originals * max_new_per_original)
    if hard_cap is not None:
        cap = min(cap, max(hard_cap, 0))
    return cap


def select(
    candidates: Sequence[Candidate],
    limit: int,
    *,
    mode: SelectionMode = "top_k",
    seed: int = 0,
) -> list[Candidate]:
    """Choose at most ``limit`` candidates, in ``(-score, term)`` order.

    ``mode="top_k"`` is deterministic by construction. ``mode="sampled"`` is deterministic
    given ``seed``: it draws without replacement with probability proportional to score
    using Efraimidis-Spirakis keys (``log(u) / w``, largest first) from
    ``numpy.random.default_rng(seed)``, over candidates in a fixed sorted order, so the
    draw does not depend on dict iteration order. Either way the returned list is sorted
    by score so downstream code sees one ordering convention.
    """
    if limit <= 0 or not candidates:
        return []
    ranked = sorted(candidates, key=lambda candidate: (-candidate.score, candidate.term))
    if len(ranked) <= limit:
        return ranked
    if mode == "top_k":
        return ranked[:limit]

    rng = np.random.default_rng(seed)
    # A zero-scored candidate would divide by zero below. `filter_candidates` already
    # drops them; the floor makes `select` safe to call on a hand-built list too, and
    # gives such a candidate the smallest possible key rather than a NaN.
    weights = np.array([candidate.score for candidate in ranked], dtype=np.float64)
    weights = np.maximum(weights, np.finfo(np.float64).tiny)
    uniforms = rng.random(len(ranked))
    # log(u)/w with u in (0, 1]: nextafter keeps log finite when rng returns exactly 0.
    uniforms = np.nextafter(uniforms, 1.0)
    keys = np.log(uniforms) / weights
    chosen = np.argpartition(-keys, limit - 1)[:limit]
    picked = [ranked[int(i)] for i in chosen]
    picked.sort(key=lambda candidate: (-candidate.score, candidate.term))
    return picked
