"""Why the loop works: occupancy sparsity and scoped-vs-global tagging precision.

Two measurements that quantify the post's "Why Does It Work?" section. Occupancy
shows that the entity x collection matrix is extremely sparse — most entities exist
in only a few collections — which is what makes per-collection automata cheap.
Scoped-vs-global precision shows what that sparsity buys: an ambiguous unigram like
"chase" fires on sports and crime prose across the whole corpus, but inside the
collection whose documents actually concern the entity, the same pattern is precise.

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "Not all entities exist in every collection. And the more coherent your
    collections, the truer this becomes. In fact, the distribution of entities is
    extremely sparse in the average case. So, learning collection-specific models
    saves a tonne of compute cost and minimizes spurious matches." Local copy:
    ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions:
    - Occupancy is measured over the tags table: an ``(entity, collection)`` cell is
      occupied when at least one chunk of that collection carries a tag for that
      entity. Density is occupied cells over ``n_entities * n_collections`` and
      sparsity is its complement — the post gives no formula, this is the plain
      reading of "distribution of entities across collections" [inferred].
    - Precision is measured against a caller-supplied relevance oracle over real
      chunks. The tests derive the oracle mechanically from the text (does the chunk
      contain an unambiguous longer pattern of the entity), never from invented
      judgments.
    - A "match" is a boundary-respecting automaton hit of the single ambiguous
      pattern, counted per chunk (a chunk either matches or it does not): tagging is
      per-chunk, so chunk-level precision is the quantity that predicts spurious
      tags.
    - Zero matches yields a precision of 0.0 rather than NaN: the report carries the
      raw counts, so a caller who needs to distinguish "no matches" from "all wrong"
      has them.

Alternatives rejected:
    - Measuring sparsity over the patterns table instead of tags: patterns record
      where an entity *may* be matched, tags record where it *is* — the compute-cost
      argument is about real matches in real chunks.
    - Recall alongside precision: computing recall needs exhaustive relevance over
      all chunks, which the mechanical oracle provides only for the matched subset
      honestly; precision is also the quantity the post's "minimizes spurious
      matches" claims.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass

from cybernaut_mini.entities.store import EntityStore
from cybernaut_mini.entities.tagger import CollectionTagger

__all__ = ["OccupancyReport", "PrecisionReport", "ambiguous_term_precision", "occupancy"]


@dataclass(frozen=True)
class OccupancyReport:
    """How sparsely entities occupy collections, from the tags table."""

    n_entities: int
    n_collections: int
    occupied_cells: int

    @property
    def total_cells(self) -> int:
        return self.n_entities * self.n_collections

    @property
    def density(self) -> float:
        return self.occupied_cells / self.total_cells if self.total_cells else 0.0

    @property
    def sparsity(self) -> float:
        return 1.0 - self.density


def occupancy(store: EntityStore) -> OccupancyReport:
    """Entity x collection occupancy over everything the loop has tagged so far."""
    cells = store.occupancy_cells()
    entities = {entity_id for entity_id, _ in cells}
    collections = {collection_id for _, collection_id in cells}
    return OccupancyReport(
        n_entities=len(entities),
        n_collections=len(collections),
        occupied_cells=len(cells),
    )


@dataclass(frozen=True)
class PrecisionReport:
    """Chunk-level precision of one ambiguous pattern, globally and scoped."""

    term: str
    global_matches: int
    global_relevant: int
    scoped_matches: int
    scoped_relevant: int

    @property
    def global_precision(self) -> float:
        return self.global_relevant / self.global_matches if self.global_matches else 0.0

    @property
    def scoped_precision(self) -> float:
        return self.scoped_relevant / self.scoped_matches if self.scoped_matches else 0.0


def ambiguous_term_precision(
    term: str,
    chunks: Iterable[tuple[str, str, str]],
    scoped_collections: Collection[str],
    is_relevant: Callable[[str, str, str], bool],
) -> PrecisionReport:
    """Precision of matching *term* everywhere vs only inside *scoped_collections*.

    *chunks* yields ``(collection_id, chunk_id, text)`` triples of real corpus text;
    *is_relevant* is the caller's mechanical relevance oracle over the same triple.
    The scoped numbers describe what a per-collection automaton would have done: the
    pattern simply does not exist in the other collections, so their spurious
    matches never happen.
    """
    tagger = CollectionTagger([(term, "probe")])
    global_matches = global_relevant = scoped_matches = scoped_relevant = 0
    for collection_id, chunk_id, text in chunks:
        if not tagger.tag(text):
            continue
        relevant = is_relevant(collection_id, chunk_id, text)
        global_matches += 1
        global_relevant += int(relevant)
        if collection_id in scoped_collections:
            scoped_matches += 1
            scoped_relevant += int(relevant)
    return PrecisionReport(
        term=term,
        global_matches=global_matches,
        global_relevant=global_relevant,
        scoped_matches=scoped_matches,
        scoped_relevant=scoped_relevant,
    )
