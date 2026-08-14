"""Tests for the retrieval module (M3).

All tests use HashEmbedder(dim=64) and TextProcessor(use_spacy=False):
no network, no spaCy.

Stage 8 moved two of this module's functions without changing what they must do:

* ``retrieval.make_snippet(text, query_tokens=tokens)`` is now
  :func:`cybernaut_mini.query.s8_retrieve.make_snippet`, which takes the query tokens
  by keyword and accepts intent matches as a higher-priority anchor. The snippet tests
  below are the originals, repointed at the new home and unchanged in what they assert.
* ``retrieval.merge_shard_results`` / ``ShardHit`` are now
  :func:`cybernaut_mini.query.s8_retrieve.reduce_responses` over
  :class:`~cybernaut_mini.query.s8_retrieve.ShardResponse`. The two properties the old
  merge tests pinned — a document found in two shards keeps its *best* score rather
  than the sum, and ties break on ascending doc id — are re-asserted against the new
  function. The old third case (summing across query *variants*) is gone with the
  feature: ``merge_shard_results`` was only ever called with a single variant, so that
  branch had no production caller.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cybernaut_mini.config import RRFConfig
from cybernaut_mini.indexing import LoadedIndex
from cybernaut_mini.models import MetadataFilter
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.query.s8_retrieve import (
    ShardResponse,
    make_snippet,
    reduce_responses,
)
from cybernaut_mini.retrieval import retrieve
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
    snippet = make_snippet(text, query_tokens=["word"])
    assert len(snippet) <= 500


def test_snippet_deterministic() -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 20
    tokens = ["fox"]
    s1 = make_snippet(text, query_tokens=tokens)
    s2 = make_snippet(text, query_tokens=tokens)
    assert s1 == s2


def test_snippet_centers_on_earliest_match() -> None:
    # Build text where match token appears late (use space-separated words so \b works).
    prefix = "unrelated content repeated here again and again " * 30
    suffix = " remarkable appears right here in the text"
    text = prefix + suffix
    snippet = make_snippet(text, query_tokens=["remarkable"])
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
    snippet = make_snippet(text, query_tokens=["nonexistenttoken123"])
    norm = normalize(text)
    # Should start from beginning.
    assert norm.startswith(snippet) or snippet in norm[:600]
    assert len(snippet) <= 500


def test_snippet_no_word_split() -> None:
    text = "word " * 200
    tokens = ["word"]
    snippet = make_snippet(text, query_tokens=tokens)
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
# Cross-shard reduce (was merge_shard_results)                        #
# ------------------------------------------------------------------ #


def test_reduce_same_doc_two_shards_keeps_best(rrf_config: RRFConfig) -> None:
    """Same doc returned by two shards: keep the best score, never the sum."""
    responses = [
        ShardResponse(shard_id=0, lexical=(("doc-x", 0.8),)),
        ShardResponse(shard_id=1, lexical=(("doc-x", 0.3),)),
    ]
    candidates = reduce_responses(responses, mode="lexical", rrf_config=rrf_config)
    assert len(candidates) == 1, "the duplicated document must be emitted once"
    assert candidates[0].score == pytest.approx(0.8), (
        f"expected the best score 0.8, not the sum 1.1; got {candidates[0].score}"
    )
    # The reduce still records that both shards returned it.
    assert candidates[0].shard_ids == (0, 1)


def test_reduce_deterministic_order_on_ties(rrf_config: RRFConfig) -> None:
    """Tied scores sort by doc_id ascending, whatever order the shards answered in."""
    responses = [
        ShardResponse(shard_id=0, lexical=(("doc-b", 0.5), ("doc-a", 0.5))),
    ]
    candidates = reduce_responses(responses, mode="lexical", rrf_config=rrf_config)
    doc_ids = [candidate.doc_id for candidate in candidates]
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
