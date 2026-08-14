"""Admissibility filters: which candidate expansions carry signal, and which are noise.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 7: "this step can
    introduce some noise / serendipity, so we are very careful not to overwhelm the
    original search words." Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions: the post names the failure mode ("noise") and the response ("very careful")
    but not the mechanism, so this module fixes three cheap, corpus-independent filters
    [inferred]: stop-words, a minimum length, and cross-shard ubiquity. The third is the
    interesting one and follows directly from the post's own argument for sharding — a
    term that occurs in the graph of nearly every shard is exactly a term whose sense the
    shard has failed to disambiguate, so it cannot be the "unambiguous synonym" stage 7
    is supposed to be adding. It is the IDF argument applied to shards instead of
    documents.

    Document frequency is counted over *shards*, not documents, and the default source is
    each shard's term-graph node set, because that is the population candidates are drawn
    from. Counting manifest keywords instead answers a different question ("is this term
    characteristic of many shards"), so both are available and the choice is a parameter.

Alternatives rejected:
    - A global IDF table. It would be a better statistic, but it is a build-time artifact
      this stage does not own, and the whole point of the ubiquity test is that it is
      computable from the manifests already resident in memory.
    - Filtering by part of speech (the machinery exists in ``s3_intents``). The candidates
      here are *stems*, and a stemmer's output is a poor input to a tagger.
    - Dropping the length filter now that stop-words are handled. Stems of two characters
      or fewer are overwhelmingly stemming artifacts ("ha", "wa") and single digits are
      never useful search words; the filter costs one comparison.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from cybernaut_mini.text import STOPWORDS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.query.s7_expand.graph import ShardGraph

#: Shortest admissible expansion term, in characters.
DEFAULT_MIN_LENGTH = 3

#: Fraction of shards a term may appear in before it is considered ubiquitous. 0.8 is the
#: value the repo's original ``expansion.expand_query`` used and is kept so the rewrite is
#: not silently a different filter; it is a keyword argument because the right value is a
#: function of how many shards a corpus has.
DEFAULT_UBIQUITY_THRESHOLD = 0.8

#: Where :meth:`UbiquityFilter.from_index` counts a term as "present in a shard".
UbiquitySource = Literal["graph", "keywords", "both"]

__all__ = [
    "DEFAULT_MIN_LENGTH",
    "DEFAULT_UBIQUITY_THRESHOLD",
    "TermFilter",
    "UbiquityFilter",
    "UbiquitySource",
]


@dataclass(frozen=True, slots=True)
class UbiquityFilter:
    """Cross-shard document frequency, and the threshold above which a term is noise.

    Building one scans every shard manifest, so build it **once per index** and hand it
    to :func:`~cybernaut_mini.query.s7_expand.expand.expand_query`; rebuilding it per
    query is the difference between an O(1) and an O(n_shards) query path, which is the
    thing :class:`~cybernaut_mini.indexing.LoadedIndex` exists to protect.

    Attributes:
        document_frequency: term -> number of shards it was found in.
        n_shards: total shards considered; ``0`` disables the filter entirely.
        threshold: fraction of shards, exclusive. A term is ubiquitous when it appears
            in strictly more than ``threshold * n_shards`` shards.
    """

    document_frequency: Mapping[str, int] = field(default_factory=dict)
    n_shards: int = 0
    threshold: float = DEFAULT_UBIQUITY_THRESHOLD

    def __post_init__(self) -> None:
        if self.n_shards < 0:
            msg = f"n_shards must be non-negative, got {self.n_shards}"
            raise ValueError(msg)
        if not 0.0 <= self.threshold <= 1.0:
            msg = f"threshold must be in [0, 1], got {self.threshold!r}"
            raise ValueError(msg)

    @classmethod
    def from_index(
        cls,
        index: LoadedIndex,
        *,
        threshold: float = DEFAULT_UBIQUITY_THRESHOLD,
        source: UbiquitySource = "graph",
    ) -> UbiquityFilter:
        """Count, over every shard manifest in ``index``, how many shards hold each term.

        Uses only the resident manifests: no sidecar, document or vector is read.
        """
        frequency: dict[str, int] = {}
        n_shards = 0
        for manifest in index.manifests.values():
            n_shards += 1
            terms: set[str] = set()
            if source in ("graph", "both"):
                terms.update(manifest.term_graph)
                for neighbours in manifest.term_graph.values():
                    terms.update(neighbours)
            if source in ("keywords", "both"):
                terms.update(keyword.term for keyword in manifest.keywords)
            for term in terms:
                frequency[term] = frequency.get(term, 0) + 1
        return cls(document_frequency=frequency, n_shards=n_shards, threshold=threshold)

    @classmethod
    def from_graphs(
        cls,
        graphs: Iterable[ShardGraph],
        *,
        threshold: float = DEFAULT_UBIQUITY_THRESHOLD,
    ) -> UbiquityFilter:
        """Count over an explicit set of shard graphs, for tests and offline analysis.

        Note this is a *narrower* population than :meth:`from_index`: if it is given only
        the routed shards, "appears everywhere" means "appears in every routed shard",
        which for a well-routed query is most good expansions. Prefer ``from_index``.
        """
        frequency: dict[str, int] = {}
        n_shards = 0
        for shard in graphs:
            n_shards += 1
            terms = set(shard.graph)
            for neighbours in shard.graph.values():
                terms.update(neighbours)
            for term in terms:
                frequency[term] = frequency.get(term, 0) + 1
        return cls(document_frequency=frequency, n_shards=n_shards, threshold=threshold)

    def fraction(self, term: str) -> float:
        """Fraction of shards ``term`` appears in; ``0.0`` when no shards were counted."""
        if self.n_shards == 0:
            return 0.0
        return self.document_frequency.get(term, 0) / self.n_shards

    def is_ubiquitous(self, term: str) -> bool:
        """True when ``term`` appears in strictly more than ``threshold`` of all shards."""
        if self.n_shards == 0:
            return False
        return self.document_frequency.get(term, 0) > self.threshold * self.n_shards


@dataclass(frozen=True, slots=True)
class TermFilter:
    """Everything that can disqualify a candidate expansion term.

    Attributes:
        stopwords: function words, excluded outright.
        min_length: shortest admissible term, in characters.
        allow_numeric: keep candidates that are all digits. Off by default: a bare
            number is a document detail, not a synonym.
        ubiquity: cross-shard frequency filter, or ``None`` to skip it.
        blocked: an explicit deny-list, e.g. the query's own original terms.
    """

    stopwords: frozenset[str] = STOPWORDS
    min_length: int = DEFAULT_MIN_LENGTH
    allow_numeric: bool = False
    ubiquity: UbiquityFilter | None = None
    blocked: frozenset[str] = frozenset()

    def rejection(self, term: str) -> str | None:
        """The reason ``term`` is inadmissible, or ``None`` if it is admissible.

        Returning the reason rather than a bool is what lets the trace output explain an
        expansion that came back empty, which is otherwise indistinguishable from a shard
        that simply had no synonyms.
        """
        if term in self.blocked:
            return "original"
        if len(term) < self.min_length:
            return "too-short"
        if term in self.stopwords:
            return "stopword"
        if not self.allow_numeric and term.isdigit():
            return "numeric"
        if self.ubiquity is not None and self.ubiquity.is_ubiquitous(term):
            return "ubiquitous"
        return None

    def accepts(self, term: str) -> bool:
        """True when ``term`` survives every filter."""
        return self.rejection(term) is None

    def with_blocked(self, terms: Iterable[str]) -> TermFilter:
        """A copy whose deny-list is ``terms`` (used to block the originals)."""
        return TermFilter(
            stopwords=self.stopwords,
            min_length=self.min_length,
            allow_numeric=self.allow_numeric,
            ubiquity=self.ubiquity,
            blocked=frozenset(terms),
        )
