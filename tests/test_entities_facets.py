"""Faceted pre-retrieval routing over real fixture text and a real built index.

The tags come from flushing the post's own JPMorgan patterns ("chase",
"chase field") over the committed CC-News fixture documents, the facet centroids
come from a hash-embedded index built over the same real documents, and the
restricted-vs-global comparison runs the repo's actual ``retrieve`` path.

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "when a query arrives, you map the query to the most relevant facets and search
    within them for the most relevant documents. If your classifier is good and your
    index supports pre-retrieval, you can unlock higher precision and lower
    latency." Local copy:
    ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions:
    - The allowlist filter must ride the existing ``MetadataFilter`` seam untouched,
      so the integration test calls ``retrieve`` itself with a
      :class:`DocAllowlistFilter` and asserts containment — no retrieval edits.
    - Global fallback is ``None``, never an empty allowlist; asserted directly.

Alternatives rejected:
    - Wall-clock latency assertions for the "lower latency" claim: flaky on shared
      machines; the comparison object carries the timings and the test asserts they
      are measured, while precision — the deterministic half of the claim — is
      pinned numerically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cybernaut_mini.config import RRFConfig
from cybernaut_mini.entities.facets import (
    DocAllowlistFilter,
    FacetDetector,
    FacetIndex,
    compare_facet_vs_global,
    facet_centroids,
)
from cybernaut_mini.entities.flush import FlushBuffer, flush_collection, texts_from
from cybernaut_mini.entities.ingest_stream import StreamingIndex
from cybernaut_mini.entities.store import EntityStore
from cybernaut_mini.ingest import load_documents
from cybernaut_mini.models import Document, MetadataFilter
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.text import TextProcessor

FIXTURE_DOCS = Path("data/01_raw/fixtures/documents.jsonl")

#: Real fixture documents (same ids the flush tests pin): two match the standalone
#: unigram "chase"; the Phoenix hiking doc also contains the bigram "chase field".
CHASE_ARRIETA = "ccn-4fff8c81dee64495"
CHASE_FIELD = "ccn-ecd900247927d44d"
NO_MATCH = "ccn-0a7e4496a9cac29e"

JPM = "Q192314"


@pytest.fixture(scope="module")
def fixture_documents() -> list[Document]:
    return load_documents(FIXTURE_DOCS)


@pytest.fixture(scope="module")
def tagged_store(fixture_documents: list[Document]) -> EntityStore:
    """Flush the JPMorgan patterns over the real chase trio to produce tags."""
    store = EntityStore()
    buffer = FlushBuffer()
    store.add_patterns("news", JPM, ["chase", "chase field"])
    texts = {
        doc.id: doc.text
        for doc in fixture_documents
        if doc.id in {CHASE_ARRIETA, CHASE_FIELD, NO_MATCH}
    }
    for chunk_id, text in sorted(texts.items()):
        buffer.add("news", chunk_id, text)
    flush_collection(store, buffer, "news", texts_from(texts))
    return store


def test_facet_index_inverts_tags_by_doc_id(tagged_store: EntityStore) -> None:
    facet_index = FacetIndex.from_store(tagged_store)
    assert facet_index.entity_ids() == (JPM,)
    assert facet_index.doc_ids_for(JPM) == (CHASE_ARRIETA, CHASE_FIELD)
    assert facet_index.entities_for(NO_MATCH) == ()
    assert len(facet_index) == 1


def test_automaton_detects_facet_in_query_and_builds_allowlist(
    tagged_store: EntityStore,
) -> None:
    detector = FacetDetector.from_store(tagged_store)
    hits = detector.detect("10 best hikes near Chase Field")
    assert [(h.entity_id, h.source, h.score) for h in hits] == [(JPM, "automaton", 1.0)]

    metadata_filter = detector.filter_for("10 best hikes near Chase Field")
    assert metadata_filter is not None
    assert metadata_filter.doc_ids == [CHASE_ARRIETA, CHASE_FIELD]


def test_no_facet_hit_means_global_fallback_not_empty_allowlist(
    tagged_store: EntityStore,
) -> None:
    detector = FacetDetector.from_store(tagged_store)
    assert detector.detect("solar panel efficiency records") == ()
    assert detector.filter_for("solar panel efficiency records") is None


def test_filter_carries_base_metadata_constraints(tagged_store: EntityStore) -> None:
    detector = FacetDetector.from_store(tagged_store)
    base = MetadataFilter(language=["en"])
    metadata_filter = detector.filter_for("chase earnings", base=base)
    assert metadata_filter is not None
    assert metadata_filter.language == ["en"]
    assert metadata_filter.doc_ids == [CHASE_ARRIETA, CHASE_FIELD]


def test_doc_allowlist_filter_semantics(fixture_documents: list[Document]) -> None:
    by_id = {doc.id: doc for doc in fixture_documents}
    allow = DocAllowlistFilter(doc_ids=[CHASE_FIELD])
    assert allow.matches(by_id[CHASE_FIELD])
    assert not allow.matches(by_id[CHASE_ARRIETA])
    assert not allow.is_empty()
    assert DocAllowlistFilter().is_empty()
    # Inherited constraints still AND with the allowlist.
    narrowed = DocAllowlistFilter(doc_ids=[CHASE_FIELD], language=["de"])
    assert not narrowed.matches(by_id[CHASE_FIELD])  # fixture doc is English


# ------------------------------------------------------------------ #
# Centroid detection and end-to-end retrieval over a real index       #
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def small_index(fixture_documents: list[Document]) -> StreamingIndex:
    """A hash-embedded index over the chase trio plus real filler documents."""
    wanted = {CHASE_ARRIETA, CHASE_FIELD, NO_MATCH}
    chosen = [doc for doc in fixture_documents if doc.id in wanted]
    chosen += [doc for doc in fixture_documents if doc.id not in wanted][:37]
    embedder = HashEmbedder(dim=64)
    texts = [f"{doc.title}\n{doc.text}" for doc in chosen]
    return StreamingIndex.bootstrap(
        chosen,
        embedder.embed_documents(texts),
        n_shards=4,
        seed=42,
        embedding_model=embedder.identifier,
        processor=TextProcessor(use_spacy=False),
    )


def test_centroid_detection_thresholds(
    tagged_store: EntityStore, small_index: StreamingIndex
) -> None:
    index = small_index.to_loaded_index()
    facet_index = FacetIndex.from_store(tagged_store)
    centroids = facet_centroids(facet_index, index)
    assert set(centroids) == {JPM}

    embedder = HashEmbedder(dim=64)
    # No pattern occurs in this query: only the centroid path can fire.
    query_vector = embedder.embed_queries(["hiking trails around metro Phoenix"])[0]

    permissive = FacetDetector.from_store(
        tagged_store, centroids=centroids, centroid_threshold=-1.0
    )
    hits = permissive.detect("hiking trails around metro Phoenix", query_vector)
    assert [(h.entity_id, h.source) for h in hits] == [(JPM, "centroid")]
    assert -1.0 <= hits[0].score <= 1.0

    strict = FacetDetector.from_store(
        tagged_store, centroids=centroids, centroid_threshold=1.1
    )
    assert strict.detect("hiking trails around metro Phoenix", query_vector) == ()


def test_allowlist_restricts_retrieval_through_existing_seam(
    small_index: StreamingIndex,
) -> None:
    from cybernaut_mini.retrieval import retrieve

    index = small_index.to_loaded_index()
    hits = retrieve(
        index,
        "best hikes chase field phoenix",
        mode="hybrid",
        processor=TextProcessor(use_spacy=False),
        provider=HashEmbedder(dim=64),
        metadata_filter=DocAllowlistFilter(doc_ids=[CHASE_ARRIETA, CHASE_FIELD]),
        rrf_config=RRFConfig(),
        top_k=10,
    )
    assert hits, "restricted retrieval should still return the allowlisted documents"
    assert {hit.document.id for hit in hits} <= {CHASE_ARRIETA, CHASE_FIELD}


def test_compare_facet_vs_global_precision(
    tagged_store: EntityStore, small_index: StreamingIndex
) -> None:
    index = small_index.to_loaded_index()
    detector = FacetDetector.from_store(tagged_store)
    comparison = compare_facet_vs_global(
        index,
        "10 best hikes near chase field",
        detector,
        relevant_ids={CHASE_FIELD},
        mode="hybrid",
        processor=TextProcessor(use_spacy=False),
        provider=HashEmbedder(dim=64),
        rrf_config=RRFConfig(),
        top_k=10,
    )
    assert comparison.used_facet
    assert comparison.restricted_precision >= comparison.global_precision
    assert comparison.restricted_precision > 0.0
    assert comparison.restricted_seconds >= 0.0
    assert comparison.global_seconds >= 0.0

    # A query with no facet falls back: both sides are the same global run.
    fallback = compare_facet_vs_global(
        index,
        "solar panel efficiency records",
        detector,
        relevant_ids={CHASE_FIELD},
        mode="hybrid",
        processor=TextProcessor(use_spacy=False),
        provider=HashEmbedder(dim=64),
        rrf_config=RRFConfig(),
        top_k=10,
    )
    assert not fallback.used_facet
    assert fallback.restricted_precision == fallback.global_precision
