"""Self-organising faceted entities: dual tagger → suggestions → Resolver → patterns.

The loop that lets collections discover their own entity facets. A fast per-collection
Aho-Corasick tagger tags everything; a slow sampled discovery tagger suggests; the
suggestion accumulator promotes at three sightings; the Resolver identifies the
entity with free tools under a two-source acceptance rule; distillation turns the
record into ~100 literal patterns; and flush propagates new pattern versions by
rescanning only stale chunks.

Public surface::

    from cybernaut_mini.entities import (
        EntityStore, CollectionTagger, DualTagger, accumulate,
        DeterministicResolver, distill, FlushBuffer, FlushScheduler,
        flush_collection, occupancy, ambiguous_term_precision,
        # query-time and stream-time surfaces on top of the loop:
        FacetDetector, DocAllowlistFilter, two_stage_disambiguator,
        StreamingIndex, simulate_stream,
    )

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    the whole package is that post's "Distilling Agents" pipeline, one module per
    moving part. Local copy:
    ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions: the split follows the post's own component list (two taggers, a
suggestion database, a Resolver, pattern distillation, flush) so each claim in the
post has exactly one module answering for it.

Alternatives rejected: a single ``entities.py`` module (the Resolver alone carries
three tools and an agent loop; one file would bury the loop's shape); folding the
loop into the query package (this is index-side machinery — the query path only
ever consumes its tags).
"""

from __future__ import annotations

from cybernaut_mini.entities.disambig import (
    ContextEnsemble,
    NewsStory,
    SeedSet,
    anchor_seed_labels,
    anchors_from_wikidata,
    country_mentions,
    story_from_document,
    two_stage_disambiguator,
)
from cybernaut_mini.entities.distill import DEFAULT_PATTERN_CAP, distill, name_variants
from cybernaut_mini.entities.facets import (
    DEFAULT_CENTROID_THRESHOLD,
    DocAllowlistFilter,
    FacetComparison,
    FacetDetector,
    FacetHit,
    FacetIndex,
    compare_facet_vs_global,
    facet_centroids,
)
from cybernaut_mini.entities.flush import (
    FlushBuffer,
    FlushReport,
    FlushScheduler,
    PendingChunk,
    flush_collection,
    run_flush_cycle,
    texts_from,
)
from cybernaut_mini.entities.ingest_stream import (
    DEFAULT_DOCS_PER_DAY,
    DayReport,
    EntityStreamState,
    StreamingIndex,
    dump_curve,
    route_to_centroids,
    simulate_stream,
    usable_judgments,
)
from cybernaut_mini.entities.metrics import (
    OccupancyReport,
    PrecisionReport,
    ambiguous_term_precision,
    occupancy,
)
from cybernaut_mini.entities.resolver import (
    ACCEPTANCE_MIN_SOURCES,
    DeterministicResolver,
    EdgarCompanyTickers,
    NetworkDisabledError,
    Resolution,
    ResolverAgent,
    ResolverError,
    ResolverRecord,
    SourceEvidence,
    WikidataTool,
    WikipediaTool,
    entity_id_for,
    resolve_and_store,
    sources_in_agreement,
)
from cybernaut_mini.entities.store import EntityStore, StoredEntity, Suggestion
from cybernaut_mini.entities.suggest import PROMOTION_THRESHOLD, SuggestionUpdate, accumulate
from cybernaut_mini.entities.tagger import (
    DEFAULT_SAMPLE_RATE,
    CapitalizedSpanDiscovery,
    CollectionTagger,
    DiscoveryTagger,
    DualTagger,
    DualTagResult,
    GlinerDiscoveryTagger,
    TagMatch,
    normalise_surface,
)

__all__ = [
    "ACCEPTANCE_MIN_SOURCES",
    "DEFAULT_CENTROID_THRESHOLD",
    "DEFAULT_DOCS_PER_DAY",
    "DEFAULT_PATTERN_CAP",
    "DEFAULT_SAMPLE_RATE",
    "PROMOTION_THRESHOLD",
    "CapitalizedSpanDiscovery",
    "CollectionTagger",
    "ContextEnsemble",
    "DayReport",
    "DeterministicResolver",
    "DiscoveryTagger",
    "DocAllowlistFilter",
    "DualTagResult",
    "DualTagger",
    "EdgarCompanyTickers",
    "EntityStore",
    "EntityStreamState",
    "FacetComparison",
    "FacetDetector",
    "FacetHit",
    "FacetIndex",
    "FlushBuffer",
    "FlushReport",
    "FlushScheduler",
    "GlinerDiscoveryTagger",
    "NetworkDisabledError",
    "NewsStory",
    "OccupancyReport",
    "PendingChunk",
    "PrecisionReport",
    "Resolution",
    "ResolverAgent",
    "ResolverError",
    "ResolverRecord",
    "SeedSet",
    "SourceEvidence",
    "StoredEntity",
    "StreamingIndex",
    "Suggestion",
    "SuggestionUpdate",
    "TagMatch",
    "WikidataTool",
    "WikipediaTool",
    "accumulate",
    "ambiguous_term_precision",
    "anchor_seed_labels",
    "anchors_from_wikidata",
    "compare_facet_vs_global",
    "country_mentions",
    "distill",
    "dump_curve",
    "entity_id_for",
    "facet_centroids",
    "flush_collection",
    "name_variants",
    "normalise_surface",
    "occupancy",
    "resolve_and_store",
    "route_to_centroids",
    "run_flush_cycle",
    "simulate_stream",
    "sources_in_agreement",
    "story_from_document",
    "texts_from",
    "two_stage_disambiguator",
    "usable_judgments",
]
