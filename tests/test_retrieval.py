"""Tests for the retrieval module (M3).

All tests use HashEmbedder(dim=64) and TextProcessor(use_spacy=False):
no network, no spaCy.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cybernaut_mini.config import RRFConfig
from cybernaut_mini.indexing import LoadedIndex
from cybernaut_mini.models import MetadataFilter
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.retrieval import (
    ShardHit,
    make_snippet,
    merge_shard_results,
    retrieve,
)
from cybernaut_mini.text import TextProcessor, normalize

# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def processor() -> TextProcessor:
    return TextProcessor(use_spacy=False)


@pytest.fixture(scope="module")
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=64)


@pytest.fixture(scope="module")
def rrf_config() -> RRFConfig:
    return RRFConfig()


# ------------------------------------------------------------------ #
# Lexical mode                                                        #
# ------------------------------------------------------------------ #


def test_lexical_zylophristine_rank1(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    hits = retrieve(
        built_index,
        "zylophristine assay",
        mode="lexical",
        processor=processor,
        provider=embedder,
        rrf_config=rrf_config,
        top_k=10,
    )
    assert hits, "Expected at least one result"
    assert hits[0].document.id == "doc-lex", (
        f"Expected doc-lex at rank 1, got {hits[0].document.id}"
    )


# ------------------------------------------------------------------ #
# Dense mode                                                          #
# ------------------------------------------------------------------ #


def test_dense_doc_para_in_top5(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    hits = retrieve(
        built_index,
        "gene edit",
        mode="dense",
        processor=processor,
        provider=embedder,
        rrf_config=rrf_config,
        top_k=10,
    )
    doc_ids = [h.document.id for h in hits]
    assert "doc-para" in doc_ids[:5], (
        f"Expected doc-para in top 5 dense hits; got {doc_ids}"
    )


def test_lexical_excludes_doc_para(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    """No exact token overlap between 'gene edit' and doc-para → BM25 score 0 → excluded."""
    hits = retrieve(
        built_index,
        "gene edit",
        mode="lexical",
        processor=processor,
        provider=embedder,
        rrf_config=rrf_config,
        top_k=10,
    )
    doc_ids = [h.document.id for h in hits]
    assert "doc-para" not in doc_ids, (
        f"doc-para should be absent from lexical results; got {doc_ids}"
    )


# ------------------------------------------------------------------ #
# Hybrid mode                                                         #
# ------------------------------------------------------------------ #


def test_hybrid_no_duplicate_ids(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    hits = retrieve(
        built_index,
        "zylophristine assay",
        mode="hybrid",
        processor=processor,
        provider=embedder,
        rrf_config=rrf_config,
        top_k=10,
    )
    doc_ids = [h.document.id for h in hits]
    assert len(doc_ids) == len(set(doc_ids)), f"Duplicate doc ids in hybrid results: {doc_ids}"


def test_hybrid_rrf_contributions_populated(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    """Hits that appear in both BM25 and dense lists have rrf_contributions with both rankers."""
    hits = retrieve(
        built_index,
        "zylophristine assay",
        mode="hybrid",
        processor=processor,
        provider=embedder,
        rrf_config=rrf_config,
        top_k=10,
    )
    # Find a hit that appears in both lexical and dense (has both bm25_rank and dense_rank).
    dual_hits = [h for h in hits if h.bm25_rank is not None and h.dense_rank is not None]
    assert dual_hits, "Expected at least one hit present in both lexical and dense lists"
    for hit in dual_hits:
        assert "lexical" in hit.rrf_contributions, (
            f"Missing 'lexical' key in rrf_contributions for {hit.document.id}"
        )
        assert "dense" in hit.rrf_contributions, (
            f"Missing 'dense' key in rrf_contributions for {hit.document.id}"
        )


# ------------------------------------------------------------------ #
# Metadata filters                                                    #
# ------------------------------------------------------------------ #


def test_filter_metadata_equals_biotech(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    mf = MetadataFilter(metadata_equals={"category": "biotech"})
    hits = retrieve(
        built_index,
        "gene editing molecular",
        mode="hybrid",
        processor=processor,
        provider=embedder,
        metadata_filter=mf,
        rrf_config=rrf_config,
        top_k=10,
    )
    assert hits, "Expected at least one biotech hit"
    for hit in hits:
        assert hit.document.metadata.get("category") == "biotech", (
            f"Expected category=biotech, got {hit.document.metadata}"
        )


def test_filter_language_excludes_de(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    mf = MetadataFilter(language=["en"])
    hits = retrieve(
        built_index,
        "solar energy",
        mode="hybrid",
        processor=processor,
        provider=embedder,
        metadata_filter=mf,
        rrf_config=rrf_config,
        top_k=24,
    )
    de_hits = [h for h in hits if h.document.language == "de"]
    assert not de_hits, f"German doc(s) should be excluded by language filter: {de_hits}"


def test_filter_published_from(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    cutoff = datetime(2025, 1, 1, tzinfo=UTC)
    mf = MetadataFilter(published_from=cutoff)
    hits = retrieve(
        built_index,
        "research",
        mode="hybrid",
        processor=processor,
        provider=embedder,
        metadata_filter=mf,
        rrf_config=rrf_config,
        top_k=24,
    )
    for hit in hits:
        assert hit.document.published_at is not None
        assert hit.document.published_at >= cutoff, (
            f"Hit {hit.document.id} published at {hit.document.published_at} is before cutoff"
        )


def test_filter_published_to(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    cutoff = datetime(2024, 1, 1, tzinfo=UTC)
    mf = MetadataFilter(published_to=cutoff)
    hits = retrieve(
        built_index,
        "research",
        mode="hybrid",
        processor=processor,
        provider=embedder,
        metadata_filter=mf,
        rrf_config=rrf_config,
        top_k=24,
    )
    for hit in hits:
        assert hit.document.published_at is not None
        assert hit.document.published_at < cutoff, (
            f"Hit {hit.document.id} published at {hit.document.published_at} is not before cutoff"
        )


# ------------------------------------------------------------------ #
# make_snippet                                                        #
# ------------------------------------------------------------------ #


def test_snippet_max_500_chars() -> None:
    text = "word " * 200  # 1000 chars
    snippet = make_snippet(text, ["word"])
    assert len(snippet) <= 500


def test_snippet_deterministic() -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 20
    tokens = ["fox"]
    s1 = make_snippet(text, tokens)
    s2 = make_snippet(text, tokens)
    assert s1 == s2


def test_snippet_centers_on_earliest_match() -> None:
    # Build text where match token appears late (use space-separated words so \b works).
    prefix = "unrelated content repeated here again and again " * 30
    suffix = " remarkable appears right here in the text"
    text = prefix + suffix
    snippet = make_snippet(text, ["remarkable"])
    # Snippet must contain the query token.
    assert "remarkable" in snippet, (
        f"Snippet should contain query token; got: {snippet!r}"
    )
    # Snippet must NOT start with the beginning of the text (match is late).
    norm = normalize(text)
    assert not snippet.startswith(norm[:20]), (
        "Snippet should be centered near match, not at text start"
    )


def test_snippet_no_match_falls_back_to_leading() -> None:
    text = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 5
    snippet = make_snippet(text, ["nonexistenttoken123"])
    norm = normalize(text)
    # Should start from beginning.
    assert norm.startswith(snippet) or snippet in norm[:600]
    assert len(snippet) <= 500


def test_snippet_no_word_split() -> None:
    text = "word " * 200
    tokens = ["word"]
    snippet = make_snippet(text, tokens)
    norm = normalize(text)
    # Snippet must be a substring of the normalized text.
    assert snippet in norm or snippet == norm[: len(snippet)], (
        "Snippet is not a substring of the normalized text"
    )
    # Boundaries must align with spaces or string ends.
    if snippet:
        # Must not start or end mid-word (first/last char not a partial token).
        # Check that the snippet boundary is at a word boundary in the norm text.
        start_idx = norm.find(snippet)
        end_idx = start_idx + len(snippet) if start_idx != -1 else -1
        if start_idx > 0:
            assert norm[start_idx - 1] == " ", (
                "Snippet starts after a non-space character (word split)"
            )
        if end_idx != -1 and end_idx < len(norm):
            assert norm[end_idx] == " ", (
                "Snippet ends before a non-space character (word split)"
            )


# ------------------------------------------------------------------ #
# merge_shard_results                                                 #
# ------------------------------------------------------------------ #


def _make_shard_hit(doc_id: str, shard_id: int, score: float) -> ShardHit:
    return ShardHit(
        doc_id=doc_id,
        shard_id=shard_id,
        bm25_score=None,
        bm25_rank=None,
        dense_score=score,
        dense_rank=1,
        fused_score=score,
        rrf_contributions={},
    )


def test_merge_same_doc_two_shards_keeps_best() -> None:
    """Same doc in two shards of one variant: keep best score (not sum)."""
    hit_a = _make_shard_hit("doc-x", shard_id=0, score=0.8)
    hit_b = _make_shard_hit("doc-x", shard_id=1, score=0.3)
    result = merge_shard_results([("v1", [hit_a, hit_b])], top_k=5)
    assert len(result) == 1
    _variant, best_hit, final_score = result[0]
    # best score is 0.8 (not 0.8 + 0.3)
    assert abs(final_score - 0.8) < 1e-6, f"Expected 0.8, got {final_score}"
    assert best_hit.shard_id == 0


def test_merge_same_doc_two_variants_sums() -> None:
    """Same doc in two variants: final score = sum of per-variant bests."""
    hit_v1 = _make_shard_hit("doc-x", shard_id=0, score=0.5)
    hit_v2 = _make_shard_hit("doc-x", shard_id=0, score=0.4)
    result = merge_shard_results([("v1", [hit_v1]), ("v2", [hit_v2])], top_k=5)
    assert len(result) == 1
    _variant, _hit, final_score = result[0]
    assert abs(final_score - 0.9) < 1e-6, f"Expected 0.9, got {final_score}"


def test_merge_deterministic_order_on_ties() -> None:
    """Tied scores sort by doc_id ascending."""
    hit_b = _make_shard_hit("doc-b", shard_id=0, score=0.5)
    hit_a = _make_shard_hit("doc-a", shard_id=0, score=0.5)
    result = merge_shard_results([("v1", [hit_b, hit_a])], top_k=5)
    doc_ids = [hit.doc_id for _, hit, _ in result]
    assert doc_ids == sorted(doc_ids), f"Expected ascending doc_id order; got {doc_ids}"


# ------------------------------------------------------------------ #
# Per-shard limit and shard_ids restriction                          #
# ------------------------------------------------------------------ #


def test_per_shard_limit_respected(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    limit = 2
    hits = retrieve(
        built_index,
        "solar energy wind",
        mode="lexical",
        processor=processor,
        provider=embedder,
        rrf_config=rrf_config,
        top_k=20,
        per_shard_limit=limit,
    )
    # Each shard contributes at most `limit` hits before merge.
    # After merge we can have at most n_shards * limit total.
    n_shards = built_index.meta.n_shards
    assert len(hits) <= n_shards * limit


def test_shard_ids_restriction(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    allowed = [0, 1]
    hits = retrieve(
        built_index,
        "research data",
        mode="hybrid",
        processor=processor,
        provider=embedder,
        rrf_config=rrf_config,
        top_k=20,
        shard_ids=allowed,
    )
    for hit in hits:
        assert hit.shard_id in allowed, (
            f"Hit shard_id {hit.shard_id} not in restricted set {allowed}"
        )
