"""The stage-7 entry points: original terms in, marked expanded term set out.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 7, "Shard-based
    Question Expansion", including the worked example whose *shape* this module
    reproduces: the nine original stems are all still present in the expanded set, the
    twenty-one additions are individually marked ``[NEW]``, and there are roughly two
    additions per original. Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - The originals are preserved, always, and are never subject to the filters
      [disclosed by the example]: they are the user's own words, and the stage is defined
      as adding to them. Only *candidates* can be rejected.
    - Provenance is carried on every term rather than returned as a second list
      [disclosed by the example's ``[NEW]`` markers]. Stage 8 weights an original higher
      than an expansion, so it needs the flag, not just the union.
    - The exact expansion terms in the post's example are a property of NOSIBLE's corpus —
      "mojica", "raffinos" and "bacto" are co-occurrence neighbours in a shard about
      CRISPR history and yeast growth media. No test here asserts those strings; the
      tests assert the invariants (originals preserved, cap respected, provenance marked,
      per-shard senses differ) that the example demonstrates.

Alternatives rejected:
    - Returning ``list[str]``. It loses the provenance the post's own output format
      carries, and forces the caller to re-derive which terms were original by set
      difference — which is wrong the moment an expansion coincides with an original.
    - Expanding the raw question string here. Tokenization is stage 2 and owns a dozen
      language-specific packages; this stage takes stems.
      :mod:`cybernaut_mini.query.s7_expand.question` is the thin bridge for callers that
      have a question rather than terms.
    - Mutating and returning the caller's term list. The expansion is a value that gets
      logged, traced and compared between runs, so it is frozen.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cybernaut_mini.query.s7_expand.filters import (
    DEFAULT_MIN_LENGTH,
    TermFilter,
    UbiquityFilter,
)
from cybernaut_mini.query.s7_expand.graph import ShardGraph, shard_graphs_from_index
from cybernaut_mini.query.s7_expand.selection import (
    DEFAULT_MAX_NEW_PER_ORIGINAL,
    Candidate,
    SelectionMode,
    activate,
    expansion_cap,
    filter_candidates,
    select,
)
from cybernaut_mini.text import STOPWORDS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cybernaut_mini.indexing import LoadedIndex

__all__ = [
    "ExpandedTerm",
    "Expansion",
    "ExpansionConfig",
    "expand_query",
    "expand_terms",
]


@dataclass(frozen=True, slots=True)
class ExpansionConfig:
    """Every knob stage 7 has, in one hashable value.

    Attributes:
        max_new_per_original: new terms allowed per original term.
        hard_cap: absolute ceiling on new terms, or ``None``.
        min_length: shortest admissible expansion term.
        stopwords: function words that can never be expansions.
        allow_numeric: keep all-digit candidates.
        min_score: minimum share of the expansion mass a candidate must hold.
        mode: ``"top_k"`` (deterministic, the default) or ``"sampled"``.
        seed: the sampled mode's seed. Part of the config so a trace can replay a draw.
    """

    max_new_per_original: float = DEFAULT_MAX_NEW_PER_ORIGINAL
    hard_cap: int | None = None
    min_length: int = DEFAULT_MIN_LENGTH
    stopwords: frozenset[str] = STOPWORDS
    allow_numeric: bool = False
    min_score: float = 0.0
    mode: SelectionMode = "top_k"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_new_per_original < 0.0:
            msg = f"max_new_per_original must be non-negative, got {self.max_new_per_original!r}"
            raise ValueError(msg)
        if self.hard_cap is not None and self.hard_cap < 0:
            msg = f"hard_cap must be non-negative, got {self.hard_cap!r}"
            raise ValueError(msg)
        if self.min_length < 1:
            msg = f"min_length must be at least 1, got {self.min_length}"
            raise ValueError(msg)
        if self.mode not in ("top_k", "sampled"):
            msg = f"mode must be 'top_k' or 'sampled', got {self.mode!r}"
            raise ValueError(msg)

    def term_filter(self, ubiquity: UbiquityFilter | None) -> TermFilter:
        """The :class:`TermFilter` this config describes, bound to ``ubiquity``.

        The ubiquity *threshold* is deliberately not a config field: it belongs to the
        frequency table it is compared against, so there is exactly one place to set it
        (``UbiquityFilter(threshold=...)`` or ``dataclasses.replace``) and no way for a
        config and a table to disagree about it.
        """
        return TermFilter(
            stopwords=self.stopwords,
            min_length=self.min_length,
            allow_numeric=self.allow_numeric,
            ubiquity=ubiquity,
        )


@dataclass(frozen=True, slots=True)
class ExpandedTerm:
    """One term in the expanded search-word set.

    Attributes:
        term: the stem itself.
        is_new: False for the question's own terms, True for everything this stage added.
        score: share of the expansion mass, in ``(0, 1]``. Always ``1.0`` for originals —
            they are not competing for the mass, they are the thing it was spread from.
        sources: for a new term, the original terms that activated it. Empty for an
            original.
        shards: for a new term, the shards whose graphs voted for it. Empty for an
            original.
    """

    term: str
    is_new: bool
    score: float = 1.0
    sources: tuple[str, ...] = ()
    shards: tuple[int, ...] = ()

    def render(self) -> str:
        """The post's own notation: ``"term"`` or ``"term" [NEW]``."""
        return f'"{self.term}" [NEW]' if self.is_new else f'"{self.term}"'


@dataclass(frozen=True, slots=True)
class Expansion:
    """The result of stage 7: the originals, preserved, plus the marked additions.

    ``terms`` lists the originals first in the order the question produced them, then the
    new terms by descending score — the layout of the post's worked example.
    """

    terms: tuple[ExpandedTerm, ...] = ()
    #: How many new terms were permitted, whether or not that many were found.
    cap: int = 0
    #: Candidates that survived scoring but lost to the cap. Kept for tracing: an
    #: expansion that hit its cap looks identical to one that ran out of synonyms.
    dropped: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.terms)

    @property
    def originals(self) -> tuple[str, ...]:
        """The question's own terms, in question order."""
        return tuple(term.term for term in self.terms if not term.is_new)

    @property
    def new_terms(self) -> tuple[str, ...]:
        """The added terms, best first."""
        return tuple(term.term for term in self.terms if term.is_new)

    @property
    def all_terms(self) -> tuple[str, ...]:
        """Every search word stage 8 should use, originals first."""
        return tuple(term.term for term in self.terms)

    def weights(
        self, *, original_weight: float = 1.0, expansion_weight: float = 1.0
    ) -> dict[str, float]:
        """Per-term weights for stage 8's lexical scorer.

        Originals get ``original_weight``; a new term gets ``expansion_weight`` scaled by
        its share of the expansion mass, normalised so the best expansion gets the full
        ``expansion_weight``. Turning down ``expansion_weight`` is the second lever
        against overwhelming the original words, after the cap.
        """
        best = max((term.score for term in self.terms if term.is_new), default=0.0)
        weights: dict[str, float] = {}
        for term in self.terms:
            if not term.is_new:
                weights[term.term] = original_weight
            elif best > 0.0:
                weights[term.term] = expansion_weight * (term.score / best)
            else:  # pragma: no cover - unreachable: new terms always carry score > 0
                weights[term.term] = expansion_weight
        return weights

    def render(self) -> str:
        """The post's bullet list, one term per line, ``[NEW]`` on the additions."""
        return "\n".join(f"* {term.render()}" for term in self.terms)


def expand_terms(
    originals: Sequence[str],
    shard_graphs: Iterable[ShardGraph],
    *,
    config: ExpansionConfig | None = None,
    ubiquity: UbiquityFilter | None = None,
) -> Expansion:
    """Expand ``originals`` using the synonym graphs of the shards in ``shard_graphs``.

    This is the whole stage, with no dependency on the index: everything it needs is the
    terms, the per-shard graphs with their weights, and (optionally) the cross-shard
    frequency table used to reject ubiquitous terms.

    Degrades to a no-op — the originals, unchanged, and an empty ``new_terms`` — when
    there are no graphs, when the graphs are empty, when no original appears in any graph,
    or when every candidate is filtered out.

    Duplicate originals are collapsed, preserving first-appearance order, so a question
    that repeats a word does not earn a larger cap for it.
    """
    config = config or ExpansionConfig()
    unique_originals = tuple(dict.fromkeys(originals))
    term_filter = config.term_filter(ubiquity).with_blocked(unique_originals)

    candidates = activate(unique_originals, shard_graphs)
    admissible = filter_candidates(candidates, term_filter, min_score=config.min_score)
    cap = expansion_cap(
        len(unique_originals),
        max_new_per_original=config.max_new_per_original,
        hard_cap=config.hard_cap,
    )
    chosen = select(admissible, cap, mode=config.mode, seed=config.seed)
    chosen_terms = {candidate.term for candidate in chosen}

    terms: list[ExpandedTerm] = [ExpandedTerm(term=term, is_new=False) for term in unique_originals]
    terms.extend(_as_expanded(candidate) for candidate in chosen)
    dropped = tuple(
        candidate.term for candidate in admissible if candidate.term not in chosen_terms
    )
    return Expansion(terms=tuple(terms), cap=cap, dropped=dropped)


def expand_query(
    index: LoadedIndex,
    originals: Sequence[str],
    *,
    shard_ids: Sequence[int],
    shard_weights: Mapping[int, float] | None = None,
    config: ExpansionConfig | None = None,
    ubiquity: UbiquityFilter | None = None,
) -> Expansion:
    """Expand ``originals`` against the term graphs of the routed shards of ``index``.

    ``shard_ids`` is expected in stage-6 rank order, best first. ``shard_weights`` is the
    per-shard contribution weight — pass the reranker's scores, or
    :func:`~cybernaut_mini.query.s7_expand.graph.rank_weights` to derive one from the
    order alone; missing entries default to ``1.0``.

    ``ubiquity`` should be built once per index with
    :meth:`~cybernaut_mini.query.s7_expand.filters.UbiquityFilter.from_index` and reused;
    it is optional because the filter is a refinement, not a correctness requirement, and
    building it per query would make the query path O(n_shards).

    Reads nothing but the resident shard manifests: no document, token list, embedding or
    artifact sidecar is touched.
    """
    graphs = shard_graphs_from_index(index, shard_ids, shard_weights=shard_weights)
    return expand_terms(originals, graphs, config=config, ubiquity=ubiquity)


def _as_expanded(candidate: Candidate) -> ExpandedTerm:
    return ExpandedTerm(
        term=candidate.term,
        is_new=True,
        score=candidate.score,
        sources=candidate.sources,
        shards=candidate.shards,
    )
