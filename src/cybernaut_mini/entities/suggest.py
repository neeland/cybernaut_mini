"""Suggestion accumulation: unmatched discovery surfaces, counted, promoted at 3.

The bridge between the slow discovery tagger and the expensive Resolver. Every
discovered surface that the collection's production automaton does *not* already
match is normalised and counted; a surface whose count reaches the promotion
threshold becomes ready for resolution. Everything the production tagger already
covers is filtered out here, which is what keeps the Resolver's workload sublinear
in corpus size.

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "The slow, universal tagger suggests entities for each collection. Those
    suggestions are accumulated in a small database and, once a certain threshold is
    met, the suggestion is sent to the Resolver Agent." Local copy:
    ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions:
    - The threshold is 3 [inferred]: the post says only "a certain threshold". Three
      independent sightings inside one coherent collection is the smallest count
      that cannot be produced by a single noisy chunk, and it is the value this
      repo's gap analysis fixes for laptop scale (``count>=3``).
    - Surfaces are normalised with the tagger's own fold before counting, so
      "JPMorgan", "jpmorgan" and "JPMORGAN" accumulate as one row — the automaton
      they may eventually become patterns for is case-folded the same way.
    - "Already matched" is decided by running the automaton over the surface itself:
      a discovery that *contains* a known pattern ("JPMorgan Chase & Co." when
      "jpmorgan chase" is a pattern) is skipped too, since the production tagger
      would already tag any chunk that surface appears in [inferred].
    - Surfaces shorter than 3 folded characters are dropped: at 1-2 characters the
      false-match rate of an exact-substring automaton swamps any evidence three
      sightings can provide.

Alternatives rejected:
    - Promoting inside :func:`EntityStore.upsert_suggestion` the moment a count hits
      the threshold: fewer round trips, but promotion becomes a side effect of
      counting and the caller can no longer batch a discovery pass and then ask
      "what became ready?" — which is the shape the flush loop wants.
    - Counting matched surfaces anyway (status ``matched``) for telemetry: honest
      bookkeeping, but it doubles the table's write volume for rows the loop never
      reads; the tagger's tags table already records what was matched.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from cybernaut_mini.entities.store import EntityStore
from cybernaut_mini.entities.tagger import CollectionTagger, normalise_surface

__all__ = ["MIN_SURFACE_CHARS", "PROMOTION_THRESHOLD", "SuggestionUpdate", "accumulate"]

#: Sightings needed before a surface is sent to the Resolver.
PROMOTION_THRESHOLD = 3

#: Folded surfaces shorter than this are noise for an exact-substring matcher.
MIN_SURFACE_CHARS = 3


@dataclass(frozen=True)
class SuggestionUpdate:
    """The outcome of accumulating one batch of discoveries for one collection."""

    collection_id: str
    accumulated: tuple[str, ...]
    skipped_covered: tuple[str, ...]
    promoted: tuple[str, ...]


def accumulate(
    store: EntityStore,
    collection_id: str,
    surfaces: Iterable[str],
    tagger: CollectionTagger,
    threshold: int = PROMOTION_THRESHOLD,
) -> SuggestionUpdate:
    """Count unmatched discovery surfaces and promote the ones that crossed *threshold*.

    Returns which surfaces were counted, which were skipped because the production
    automaton already covers them, and which are now ready for the Resolver.
    Duplicates within one batch each count — three sightings in one chunk are three
    sightings.
    """
    accumulated: list[str] = []
    skipped: list[str] = []
    for surface in surfaces:
        norm = normalise_surface(surface)
        if len(norm) < MIN_SURFACE_CHARS:
            continue
        if tagger.covers(norm):
            skipped.append(norm)
            continue
        store.upsert_suggestion(collection_id, norm)
        accumulated.append(norm)
    promoted = store.promote_ready(collection_id, threshold)
    return SuggestionUpdate(
        collection_id=collection_id,
        accumulated=tuple(accumulated),
        skipped_covered=tuple(skipped),
        promoted=tuple(promoted),
    )
