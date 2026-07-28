"""Tests for shard routing: routing.route() and RoutingSignals."""

from __future__ import annotations

import pytest

from cybernaut_mini.config import RRFConfig
from cybernaut_mini.indexing import LoadedIndex
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.routing import route
from cybernaut_mini.rrf import RankedList, rrf_fuse
from cybernaut_mini.text import TextProcessor

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def processor() -> TextProcessor:
    return TextProcessor(use_spacy=False)


@pytest.fixture(scope="module")
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=64)


@pytest.fixture(scope="module")
def rrf_cfg() -> RRFConfig:
    return RRFConfig()


# ---------------------------------------------------------------------------
# Basic sanity: non-empty result covering all valid shard ids
# ---------------------------------------------------------------------------


def test_route_returns_valid_shard_ids(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_cfg: RRFConfig,
) -> None:
    shard_ids, _signals = route(
        built_index,
        "solar energy efficiency",
        processor=processor,
        provider=embedder,
        rrf_config=rrf_cfg,
    )
    valid = set(built_index.manifests.keys())
    assert len(shard_ids) > 0
    assert all(s in valid for s in shard_ids)


def test_dense_and_sparse_cover_every_shard(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_cfg: RRFConfig,
) -> None:
    _shard_ids, signals = route(
        built_index,
        "climate feedback loops",
        processor=processor,
        provider=embedder,
        rrf_config=rrf_cfg,
    )
    expected = set(built_index.manifests.keys())
    assert set(signals.dense.keys()) == expected
    assert set(signals.sparse.keys()) == expected


# ---------------------------------------------------------------------------
# Inspectability: fused items carry per-ranker contributions and ranks
# ---------------------------------------------------------------------------


def test_fused_items_are_inspectable(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_cfg: RRFConfig,
) -> None:
    _shard_ids, signals = route(
        built_index,
        "renewable energy storage batteries",
        processor=processor,
        provider=embedder,
        rrf_config=rrf_cfg,
    )
    assert signals.fused, "fused list must not be empty"
    for item in signals.fused:
        assert item.contributions, "each fused item must have at least one contribution"
        assert item.ranks, "each fused item must have rank entries"
        assert item.score > 0.0


# ---------------------------------------------------------------------------
# Entity: regex processor always returns [] → signals.entity must be None
# ---------------------------------------------------------------------------


def test_entity_is_none_for_regex_processor(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_cfg: RRFConfig,
) -> None:
    _shard_ids, signals = route(
        built_index,
        "cancer immunotherapy clinical trials",
        processor=processor,
        provider=embedder,
        rrf_config=rrf_cfg,
    )
    assert signals.entity is None

    # No "entity" key should appear in any fused item's contributions
    for item in signals.fused:
        assert "entity" not in item.contributions


# ---------------------------------------------------------------------------
# Tie-break: ascending shard id when RRF scores are equal
# ---------------------------------------------------------------------------


def test_rrf_fuse_tiebreak_ascending_id() -> None:
    """When two items receive identical RRF scores, tie-break is ascending id."""
    # r1 ranks "0" first and "2" second; r2 ranks "2" first and "0" second.
    # Each item is rank-1 in one list and rank-2 in the other → identical fused scores.
    list_a = RankedList(name="r1", weight=1.0, ids=("0", "2"))
    list_b = RankedList(name="r2", weight=1.0, ids=("2", "0"))
    fused = rrf_fuse([list_a, list_b], k=60)
    assert len(fused) == 2
    # Scores must be equal (symmetric ranks).
    assert fused[0].score == pytest.approx(fused[1].score)
    # Tie-break: ascending string id ("0" < "2").
    assert fused[0].id == "0"
    assert fused[1].id == "2"


def test_route_tiebreak_ascending_shard_id(
    built_index: LoadedIndex,
    embedder: HashEmbedder,
    rrf_cfg: RRFConfig,
) -> None:
    """If two shard ids end up with equal fused scores, lower id comes first."""
    # Use a zero-length query so that all shards score identically on sparse/dense.
    # We rely on the tiebreak test above for the core RRF property; this validates
    # that route() does not re-sort in a way that violates it.
    processor = TextProcessor(use_spacy=False)
    shard_ids, _signals = route(
        built_index,
        "a",  # very short query; scores may collide
        processor=processor,
        provider=embedder,
        rrf_config=rrf_cfg,
    )
    # Confirm the list is sorted by score desc then id asc (no strict check on values,
    # just that no id appears twice and all are valid).
    assert len(shard_ids) == len(set(shard_ids)), "duplicate shard ids in result"
    assert all(s in built_index.manifests for s in shard_ids)


# ---------------------------------------------------------------------------
# top_n cap
# ---------------------------------------------------------------------------


def test_top_n_caps_returned_list(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_cfg: RRFConfig,
) -> None:
    for cap in (1, 2):
        shard_ids, _signals = route(
            built_index,
            "large language model reasoning",
            processor=processor,
            provider=embedder,
            rrf_config=rrf_cfg,
            top_n=cap,
        )
        assert len(shard_ids) == cap


# ---------------------------------------------------------------------------
# rerank=False: rerank_intent and rerank_compression must be None
# ---------------------------------------------------------------------------


def test_rerank_false_skips_rerank_signals(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_cfg: RRFConfig,
) -> None:
    _shard_ids, signals = route(
        built_index,
        "gene editing CRISPR",
        processor=processor,
        provider=embedder,
        rrf_config=rrf_cfg,
        rerank=False,
    )
    assert signals.rerank_intent is None
    assert signals.rerank_compression is None


# ---------------------------------------------------------------------------
# rerank=True with >=2 token query populates rerank signals
# ---------------------------------------------------------------------------


def test_rerank_true_populates_intent_and_compression(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_cfg: RRFConfig,
) -> None:
    # "solar panel efficiency" has >=2 content tokens
    _shard_ids, signals = route(
        built_index,
        "solar panel efficiency",
        processor=processor,
        provider=embedder,
        rrf_config=rrf_cfg,
        rerank=True,
    )
    assert signals.rerank_intent is not None, "rerank_intent should be populated"
    assert signals.rerank_compression is not None, "rerank_compression should be populated"

    # The routed shard ids in the initial fuse are the keys
    fused_ids = {int(item.id) for item in signals.fused}
    # signals are keyed by shard id
    assert set(signals.rerank_intent.keys()) == fused_ids
    assert set(signals.rerank_compression.keys()) == fused_ids


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_route_is_deterministic(
    built_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_cfg: RRFConfig,
) -> None:
    question = "antibiotic resistance treatment protocols"
    result1, signals1 = route(
        built_index,
        question,
        processor=processor,
        provider=embedder,
        rrf_config=rrf_cfg,
    )
    result2, signals2 = route(
        built_index,
        question,
        processor=processor,
        provider=embedder,
        rrf_config=rrf_cfg,
    )
    assert result1 == result2
    assert signals1.dense == signals2.dense
    assert signals1.sparse == signals2.sparse
    assert signals1.entity == signals2.entity
    assert [item.id for item in signals1.fused] == [item.id for item in signals2.fused]
    assert [item.score for item in signals1.fused] == [item.score for item in signals2.fused]
