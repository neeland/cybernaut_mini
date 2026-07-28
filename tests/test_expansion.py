"""Tests for query expansion: expansion.expand_query()."""

from __future__ import annotations

import numpy as np
import pytest

from cybernaut_mini.expansion import expand_query
from cybernaut_mini.indexing import LoadedIndex
from cybernaut_mini.models import IndexMeta, ShardKeyword, ShardManifest
from cybernaut_mini.text import STOPWORDS, TextProcessor

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def processor() -> TextProcessor:
    return TextProcessor(use_spacy=False)


# ---------------------------------------------------------------------------
# Helper: build a minimal LoadedIndex from scratch using custom manifests.
# We re-use the session-scoped built_index where possible, but for the
# shard_weights and ubiquity tests we need full control over manifests.
# ---------------------------------------------------------------------------


def _make_manifest(
    shard_id: int,
    *,
    term_graph: dict[str, dict[str, float]],
    keyword_terms: list[str],
) -> ShardManifest:
    keywords = [ShardKeyword(term=t, weight=0.5) for t in keyword_terms]
    return ShardManifest(
        shard_id=shard_id,
        document_ids=[f"doc-{shard_id}"],
        centroid=[0.0] * 4,
        title=f"shard-{shard_id}",
        summary=f"summary for shard {shard_id}",
        keywords=keywords,
        entities=[],
        term_graph=term_graph,
        document_count=1,
        embedding_model="hash-4",
    )


def _make_index(manifests: list[ShardManifest]) -> LoadedIndex:
    """Build a LoadedIndex shell with minimal data for expansion tests."""
    from cybernaut_mini.models import Document

    docs = [
        Document(id=f"doc-{m.shard_id}", title=f"T{m.shard_id}", text=f"text {m.shard_id}")
        for m in manifests
    ]
    vectors = np.zeros((len(docs), 4), dtype=np.float32)
    row_map = {doc.id: i for i, doc in enumerate(docs)}
    manifests_dict = {m.shard_id: m for m in manifests}
    meta = IndexMeta(
        embedding_model="hash-4",
        embedding_dim=4,
        n_shards=len(manifests),
        n_documents=len(docs),
        seed=0,
    )
    # BM25Okapi requires non-empty token lists; provide one token per doc.
    doc_tokens = {doc.id: ["placeholder"] for doc in docs}
    return LoadedIndex(
        meta=meta,
        documents=docs,
        vectors=vectors,
        row_map=row_map,
        manifests=manifests_dict,
        doc_tokens=doc_tokens,
    )


# ---------------------------------------------------------------------------
# 1. Real graph edge: 'accelerate' -> 'feedback' exists in the built index.
#    Expanding a query containing 'accelerate' must return 'feedback'.
# ---------------------------------------------------------------------------


def test_expansion_returns_graph_neighbour(
    built_index: LoadedIndex,
    processor: TextProcessor,
) -> None:
    """Expanding a term with a known graph edge returns its neighbour."""
    # The built index has a global term_graph where 'accelerate' -> 'feedback' (w≈0.257).
    terms = expand_query(
        built_index,
        "accelerate",
        shard_ids=list(built_index.manifests.keys()),
        processor=processor,
        max_terms=10,
    )
    assert "feedback" in terms, f"expected 'feedback' in expansions, got {terms}"


def test_expansions_exclude_original_terms(
    built_index: LoadedIndex,
    processor: TextProcessor,
) -> None:
    query = "accelerate feedback"
    original = set(processor.content_tokens(query))
    terms = expand_query(
        built_index,
        query,
        shard_ids=list(built_index.manifests.keys()),
        processor=processor,
        max_terms=10,
    )
    for t in terms:
        assert t not in original, f"original term {t!r} leaked into expansions"


def test_expansions_exclude_stopwords(
    built_index: LoadedIndex,
    processor: TextProcessor,
) -> None:
    terms = expand_query(
        built_index,
        "accelerate climate",
        shard_ids=list(built_index.manifests.keys()),
        processor=processor,
        max_terms=20,
    )
    for t in terms:
        assert t not in STOPWORDS, f"stopword {t!r} in expansions"


def test_expansions_exclude_short_tokens(
    built_index: LoadedIndex,
    processor: TextProcessor,
) -> None:
    terms = expand_query(
        built_index,
        "accelerate climate",
        shard_ids=list(built_index.manifests.keys()),
        processor=processor,
        max_terms=20,
    )
    for t in terms:
        assert len(t) >= 3, f"short token {t!r} in expansions"


# ---------------------------------------------------------------------------
# 2. max_terms cap
# ---------------------------------------------------------------------------


def test_max_terms_cap(
    built_index: LoadedIndex,
    processor: TextProcessor,
) -> None:
    for cap in (1, 3):
        terms = expand_query(
            built_index,
            "accelerate climate feedback",
            shard_ids=list(built_index.manifests.keys()),
            processor=processor,
            max_terms=cap,
        )
        assert len(terms) <= cap


# ---------------------------------------------------------------------------
# 3. Empty result when query terms absent from all selected term graphs
# ---------------------------------------------------------------------------


def test_empty_result_for_unknown_query(
    built_index: LoadedIndex,
    processor: TextProcessor,
) -> None:
    terms = expand_query(
        built_index,
        "xyzzy quuxfrobnitz",  # tokens that won't be graph nodes
        shard_ids=list(built_index.manifests.keys()),
        processor=processor,
        max_terms=5,
    )
    assert terms == []


# ---------------------------------------------------------------------------
# 4. shard_weights scale scores: weighting one shard 10x makes its unique
#    neighbour rank first.  Uses custom manifests with different term graphs.
# ---------------------------------------------------------------------------


def test_shard_weights_affect_ranking(processor: TextProcessor) -> None:
    """Weighting shard 0 by 10x makes its neighbour 'beta' rank above shard 1's 'gamma'."""
    # Shard 0: "alpha" -> {"beta": 0.5}
    # Shard 1: "alpha" -> {"gamma": 0.5}
    m0 = _make_manifest(0, term_graph={"alpha": {"beta": 0.5}}, keyword_terms=["alpha"])
    m1 = _make_manifest(1, term_graph={"alpha": {"gamma": 0.5}}, keyword_terms=["alpha"])
    idx = _make_index([m0, m1])

    # Equal weights: beta and gamma score equally; tie-break by term ('beta' < 'gamma')
    terms_equal = expand_query(
        idx, "alpha", shard_ids=[0, 1], shard_weights={0: 1.0, 1: 1.0},
        processor=processor, max_terms=5,
    )
    assert "beta" in terms_equal and "gamma" in terms_equal

    # Shard 0 x10: beta_score = 0.5*10 = 5.0 > gamma_score = 0.5*1 = 0.5
    terms_s0_heavy = expand_query(
        idx, "alpha", shard_ids=[0, 1], shard_weights={0: 10.0, 1: 1.0},
        processor=processor, max_terms=5,
    )
    assert terms_s0_heavy.index("beta") < terms_s0_heavy.index("gamma"), (
        f"expected 'beta' before 'gamma' when shard 0 is weighted 10x; got {terms_s0_heavy}"
    )

    # Shard 1 x10: gamma_score = 0.5*10 = 5.0 > beta_score = 0.5*1 = 0.5
    terms_s1_heavy = expand_query(
        idx, "alpha", shard_ids=[0, 1], shard_weights={0: 1.0, 1: 10.0},
        processor=processor, max_terms=5,
    )
    assert terms_s1_heavy.index("gamma") < terms_s1_heavy.index("beta"), (
        f"expected 'gamma' before 'beta' when shard 1 is weighted 10x; got {terms_s1_heavy}"
    )


# ---------------------------------------------------------------------------
# 5. Ubiquity exclusion: term in keywords of ALL shards is excluded (>80% rule)
# ---------------------------------------------------------------------------


def test_ubiquity_exclusion(processor: TextProcessor) -> None:
    """A term in keywords of ALL 4 shards is excluded from expansions."""
    # 4 shards; "common" appears in all 4 keyword lists (4/4 = 100% > 80%)
    # "alpha" -> {"common": 0.9, "rare": 0.3}
    # "common" also appears in graph as neighbour of "alpha"
    graph: dict[str, dict[str, float]] = {"alpha": {"common": 0.9, "rare": 0.3}}
    manifests = [
        _make_manifest(i, term_graph=graph, keyword_terms=["alpha", "common"])
        for i in range(4)
    ]
    idx = _make_index(manifests)

    terms = expand_query(
        idx, "alpha", shard_ids=[0, 1, 2, 3],
        processor=processor, max_terms=10,
    )
    assert "common" not in terms, (
        f"'common' appears in all 4 shard keyword lists and should be excluded; got {terms}"
    )
    assert "rare" in terms, (
        f"'rare' is not ubiquitous and should be included; got {terms}"
    )


def test_near_ubiquity_not_excluded(processor: TextProcessor) -> None:
    """A term in exactly 80% (not strictly more) of shard keyword lists is NOT excluded."""
    # 5 shards; "boundary" in 4/5 = 80% keyword lists (not strictly > 80%)
    graph: dict[str, dict[str, float]] = {"alpha": {"boundary": 0.9}}
    # Put "boundary" in only 4 of the 5 shard keyword lists
    manifests = [
        _make_manifest(
            i,
            term_graph=graph,
            keyword_terms=["alpha", "boundary"] if i < 4 else ["alpha"],
        )
        for i in range(5)
    ]
    idx = _make_index(manifests)

    terms = expand_query(
        idx, "alpha", shard_ids=list(range(5)),
        processor=processor, max_terms=10,
    )
    # 4/5 = 80.0%, threshold is strictly > 80%, so "boundary" should still be included
    assert "boundary" in terms, (
        f"'boundary' at exactly 80% should NOT be excluded; got {terms}"
    )


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


def test_expand_query_is_deterministic(
    built_index: LoadedIndex,
    processor: TextProcessor,
) -> None:
    kwargs = dict(
        shard_ids=list(built_index.manifests.keys()),
        processor=processor,
        max_terms=5,
    )
    result1 = expand_query(built_index, "accelerate climate feedback loops", **kwargs)  # type: ignore[call-overload]
    result2 = expand_query(built_index, "accelerate climate feedback loops", **kwargs)  # type: ignore[call-overload]
    assert result1 == result2
