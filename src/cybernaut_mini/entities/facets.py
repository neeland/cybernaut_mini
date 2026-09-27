"""Faceted pre-retrieval routing: map a query to facets, restrict retrieval to them.

The entity loop's flush output is a tags table — chunk x entity x pattern. This
module turns those rows into a query-time capability: detect which entity facets a
question concerns (the same Aho-Corasick automata the production tagger uses, run
over the *query* string, plus cosine similarity of the query embedding against
per-facet centroid vectors), and hand retrieval a docid-allowlist restriction
through the :class:`~cybernaut_mini.models.MetadataFilter` seam it already has.
When no facet fires, the answer is ``None`` and the caller searches globally —
facet routing may only ever narrow a search, never lose one.

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "Then, when a query arrives, you map the query to the most relevant facets and
    search within them for the most relevant documents. If your classifier is good
    and your index supports pre-retrieval, you can unlock higher precision and lower
    latency." Local copy:
    ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions:
    - Chunk ids equal document ids at this repo's granularity [inferred]: the index
      stores one vector and one BM25 row per document, so the entity loop's "chunks"
      are the documents themselves. A ``chunk_to_doc`` hook is kept for callers that
      tag at a finer grain than they retrieve.
    - The query-side classifier is the *same* automaton discipline as the production
      tagger — :class:`~cybernaut_mini.entities.tagger.CollectionTagger` over the
      store's active patterns — so a facet can only be detected by patterns that
      also produced its tags. The post says "you map the query to the most relevant
      facets" without naming the classifier; reusing the tagger is the reading that
      adds no second model [inferred].
    - A facet centroid is the L2-normalised mean of the tagged documents' index
      vectors, exactly how :mod:`cybernaut_mini.sharding` builds shard centroids;
      cosine is then a dot product. The acceptance threshold defaults to 0.30
      [inferred — the post gives none]; automaton hits carry score 1.0 and always
      outrank centroid hits, because an exact pattern in the query is stronger
      evidence than embedding proximity.
    - The restriction object is a subclass of
      :class:`~cybernaut_mini.models.MetadataFilter` adding a ``doc_ids`` allowlist.
      ``score_shard_components`` only ever calls ``is_empty()`` and ``matches()``,
      so the subclass rides the existing seam and retrieval.py needs no edit — this
      workstream produces the filter object; wiring it into the CLI/retrieval
      call-sites is the integration step's.
    - Global fallback is expressed as ``None`` (no filter), not as an empty
      allowlist: an empty allowlist would return zero results, which is the one
      behaviour facet routing must never introduce.

Alternatives rejected:
    - Extending ``MetadataFilter.metadata_equals`` to smuggle doc ids through the
      existing fields: ``matches`` compares against ``document.metadata``, and doc
      ids are not metadata; a typed subclass keeps the operator explicit and
      validated (``extra="forbid"`` still applies).
    - Editing ``retrieval.py`` to take a ``doc_allowlist`` parameter: WS1 owns that
      file; the ``MetadataFilter`` seam already reaches the exact candidate-set
      code path the post's pre-retrieval describes.
    - Classifying the query against facet centroids only (no automaton): loses the
      post's precision argument — "chase field" in a query is an exact facet signal
      the embedding may blur into generic sports/geography neighbourhoods.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import PrivateAttr

from cybernaut_mini.entities.store import EntityStore
from cybernaut_mini.entities.tagger import CollectionTagger
from cybernaut_mini.models import Document, MetadataFilter

if TYPE_CHECKING:
    import numpy.typing as npt

    from cybernaut_mini.config import RRFConfig
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.providers.embeddings import EmbeddingProvider
    from cybernaut_mini.text import TextProcessor

    FloatArray = npt.NDArray[np.float32]

__all__ = [
    "DEFAULT_CENTROID_THRESHOLD",
    "DocAllowlistFilter",
    "FacetComparison",
    "FacetDetector",
    "FacetHit",
    "FacetIndex",
    "compare_facet_vs_global",
    "facet_centroids",
]

#: Minimum cosine between query embedding and a facet centroid for a centroid hit.
#: [inferred] — the post names no threshold; 0.30 is well above the near-zero cosine
#: of unrelated hash/E5 embeddings while below same-topic similarity.
DEFAULT_CENTROID_THRESHOLD = 0.30


class DocAllowlistFilter(MetadataFilter):
    """A :class:`MetadataFilter` with a document-id allowlist, AND-combined.

    Rides the existing ``metadata_filter`` seam of
    :func:`cybernaut_mini.retrieval.score_shard_components`, which only calls
    ``is_empty()`` and ``matches()`` — no retrieval change needed. The inherited
    fields (language, published window, metadata equality) still apply; a document
    passes only if it is in the allowlist *and* satisfies them.
    """

    doc_ids: list[str] | None = None

    _allowed: frozenset[str] | None = PrivateAttr(default=None)

    def is_empty(self) -> bool:
        return self.doc_ids is None and super().is_empty()

    def matches(self, document: Document) -> bool:
        if self.doc_ids is not None:
            if self._allowed is None:
                self._allowed = frozenset(self.doc_ids)
            if document.id not in self._allowed:
                return False
        return super().matches(document)


class FacetIndex:
    """Doc-id -> entity-id tag rows, inverted to answer "which docs carry facet X?".

    Normally built from the entity store's tags table (the flush output) via
    :meth:`from_store`; accepts any ``doc -> entity ids`` mapping so callers with
    parquet-side tags can feed it directly.
    """

    def __init__(self, doc_entities: Mapping[str, Iterable[str]]) -> None:
        by_entity: dict[str, set[str]] = {}
        by_doc: dict[str, tuple[str, ...]] = {}
        for doc_id, entity_ids in doc_entities.items():
            ids = tuple(sorted(set(entity_ids)))
            by_doc[doc_id] = ids
            for entity_id in ids:
                by_entity.setdefault(entity_id, set()).add(doc_id)
        self._by_doc = by_doc
        self._by_entity: dict[str, tuple[str, ...]] = {
            entity_id: tuple(sorted(doc_ids)) for entity_id, doc_ids in sorted(by_entity.items())
        }

    @classmethod
    def from_store(
        cls,
        store: EntityStore,
        collection_ids: Sequence[str] | None = None,
        chunk_to_doc: Callable[[str], str] | None = None,
    ) -> FacetIndex:
        """Invert the store's tags table; chunk ids are doc ids unless mapped."""
        collections = list(collection_ids) if collection_ids is not None else store.collections()
        doc_entities: dict[str, set[str]] = {}
        for collection_id in collections:
            for chunk_id, entity_id, _pattern in store.tags_for(collection_id):
                doc_id = chunk_to_doc(chunk_id) if chunk_to_doc is not None else chunk_id
                doc_entities.setdefault(doc_id, set()).add(entity_id)
        return cls(doc_entities)

    def entity_ids(self) -> tuple[str, ...]:
        return tuple(self._by_entity)

    def doc_ids_for(self, entity_id: str) -> tuple[str, ...]:
        return self._by_entity.get(entity_id, ())

    def entities_for(self, doc_id: str) -> tuple[str, ...]:
        return self._by_doc.get(doc_id, ())

    def __len__(self) -> int:
        return len(self._by_entity)


def facet_centroids(
    facet_index: FacetIndex, index: LoadedIndex
) -> dict[str, FloatArray]:
    """L2-normalised mean index vector of each facet's tagged documents.

    Documents the index does not hold (tags can outlive an index subset) are
    skipped; a facet with no resident documents gets no centroid.
    """
    centroids: dict[str, FloatArray] = {}
    for entity_id in facet_index.entity_ids():
        rows = [
            index.row_map[doc_id]
            for doc_id in facet_index.doc_ids_for(entity_id)
            if doc_id in index.row_map
        ]
        if not rows:
            continue
        mean = np.mean([index.vectors[row] for row in rows], axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean))
        if norm > 0.0:
            mean = (mean / norm).astype(np.float32)
        centroids[entity_id] = mean
    return centroids


@dataclass(frozen=True)
class FacetHit:
    """One detected facet: which entity, which detector fired, how strongly."""

    entity_id: str
    source: Literal["automaton", "centroid"]
    score: float


class FacetDetector:
    """Query -> facets, via the production automaton plus facet-centroid cosine."""

    def __init__(
        self,
        facet_index: FacetIndex,
        patterns: Iterable[tuple[str, str]],
        centroids: Mapping[str, FloatArray] | None = None,
        centroid_threshold: float = DEFAULT_CENTROID_THRESHOLD,
    ) -> None:
        self.facet_index = facet_index
        self._tagger = CollectionTagger(patterns)
        self._centroids = dict(centroids) if centroids is not None else {}
        self.centroid_threshold = centroid_threshold

    @classmethod
    def from_store(
        cls,
        store: EntityStore,
        facet_index: FacetIndex | None = None,
        collection_ids: Sequence[str] | None = None,
        centroids: Mapping[str, FloatArray] | None = None,
        centroid_threshold: float = DEFAULT_CENTROID_THRESHOLD,
    ) -> FacetDetector:
        """Build from the store: active patterns of every (or the given) collection."""
        collections = list(collection_ids) if collection_ids is not None else store.collections()
        patterns: list[tuple[str, str]] = []
        for collection_id in collections:
            patterns.extend(store.active_patterns(collection_id))
        if facet_index is None:
            facet_index = FacetIndex.from_store(store, collection_ids)
        return cls(facet_index, patterns, centroids, centroid_threshold)

    def detect(
        self, question: str, query_vector: FloatArray | None = None
    ) -> tuple[FacetHit, ...]:
        """Facets the question concerns, strongest first.

        Automaton hits (an entity pattern occurs verbatim in the question) score 1.0.
        Centroid hits (cosine of ``query_vector`` against a facet centroid at or
        above the threshold) score their cosine and never displace an automaton hit
        for the same entity. No hits means "search globally".
        """
        hits: dict[str, FacetHit] = {}
        for match in self._tagger.tag(question):
            hits.setdefault(
                match.entity_id,
                FacetHit(entity_id=match.entity_id, source="automaton", score=1.0),
            )
        if query_vector is not None:
            for entity_id, centroid in self._centroids.items():
                if entity_id in hits:
                    continue
                cosine = float(centroid @ query_vector)
                if cosine >= self.centroid_threshold:
                    hits[entity_id] = FacetHit(
                        entity_id=entity_id, source="centroid", score=cosine
                    )
        return tuple(sorted(hits.values(), key=lambda h: (-h.score, h.entity_id)))

    def filter_for(
        self,
        question: str,
        query_vector: FloatArray | None = None,
        base: MetadataFilter | None = None,
    ) -> DocAllowlistFilter | None:
        """The docid-allowlist restriction for the question, or ``None`` for global.

        The allowlist is the union of the hit facets' tagged documents; *base*
        constraints (language, dates, metadata) are carried into the returned
        filter. ``None`` — never an empty allowlist — signals the global fallback,
        both when no facet fires and when the hit facets have no tagged documents.
        """
        hits = self.detect(question, query_vector)
        allowed = sorted(
            {
                doc_id
                for hit in hits
                for doc_id in self.facet_index.doc_ids_for(hit.entity_id)
            }
        )
        if not allowed:
            return None
        fields = base.model_dump() if base is not None else {}
        return DocAllowlistFilter(doc_ids=allowed, **fields)


@dataclass(frozen=True)
class FacetComparison:
    """Facet-restricted vs global retrieval on one question, same everything else."""

    question: str
    used_facet: bool
    restricted_precision: float
    global_precision: float
    restricted_seconds: float
    global_seconds: float


def compare_facet_vs_global(
    index: LoadedIndex,
    question: str,
    detector: FacetDetector,
    *,
    relevant_ids: set[str],
    mode: Literal["lexical", "dense", "hybrid"],
    processor: TextProcessor,
    provider: EmbeddingProvider,
    rrf_config: RRFConfig,
    top_k: int = 10,
) -> FacetComparison:
    """Precision@k and latency of facet-restricted retrieval against the global run.

    The post's promised win is "higher precision and lower latency"; this measures
    both on one real question with real judgments. When no facet fires the
    restricted run *is* the global run (the fallback), and ``used_facet`` is False.
    """
    from cybernaut_mini.retrieval import retrieve

    query_vector: FloatArray | None = None
    if mode != "lexical":
        query_vector = provider.embed_queries([question])[0]
    metadata_filter = detector.filter_for(question, query_vector)

    def _run(filter_: MetadataFilter | None) -> tuple[float, float]:
        started = time.perf_counter()
        hits = retrieve(
            index,
            question,
            mode=mode,
            processor=processor,
            provider=provider,
            metadata_filter=filter_,
            rrf_config=rrf_config,
            top_k=top_k,
        )
        seconds = time.perf_counter() - started
        if not hits:
            return 0.0, seconds
        relevant = sum(1 for hit in hits if hit.document.id in relevant_ids)
        return relevant / len(hits), seconds

    global_precision, global_seconds = _run(None)
    if metadata_filter is None:
        return FacetComparison(
            question=question,
            used_facet=False,
            restricted_precision=global_precision,
            global_precision=global_precision,
            restricted_seconds=global_seconds,
            global_seconds=global_seconds,
        )
    restricted_precision, restricted_seconds = _run(metadata_filter)
    return FacetComparison(
        question=question,
        used_facet=True,
        restricted_precision=restricted_precision,
        global_precision=global_precision,
        restricted_seconds=restricted_seconds,
        global_seconds=global_seconds,
    )
