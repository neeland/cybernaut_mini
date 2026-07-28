"""Tests for cybernaut_mini.sharding."""

from __future__ import annotations

import numpy as np
import pytest

from cybernaut_mini.providers.embeddings import l2_normalize
from cybernaut_mini.sharding import ShardingResult, shard_documents


def _random_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return l2_normalize(v)


# ------------------------------------------------------------------ #
# Validation                                                          #
# ------------------------------------------------------------------ #


def test_n_shards_greater_than_n_docs_raises_before_clustering() -> None:
    vectors = _random_vectors(5, 32, seed=0)
    with pytest.raises(ValueError, match=r"n_shards.*n_docs|n_docs.*n_shards"):
        shard_documents(vectors, n_shards=6, seed=42)


def test_n_shards_zero_raises() -> None:
    vectors = _random_vectors(5, 32, seed=0)
    with pytest.raises(ValueError):
        shard_documents(vectors, n_shards=0, seed=42)


def test_n_shards_negative_raises() -> None:
    vectors = _random_vectors(5, 32, seed=0)
    with pytest.raises(ValueError):
        shard_documents(vectors, n_shards=-1, seed=42)


# ------------------------------------------------------------------ #
# All shards non-empty after repair                                   #
# ------------------------------------------------------------------ #


def test_all_shards_non_empty_synthetic() -> None:
    """40 vectors, 6 shards — every shard must be non-empty after repair."""
    vectors = _random_vectors(40, 32, seed=7)
    result = shard_documents(vectors, n_shards=6, seed=42)
    counts = [0] * 6
    for label in result.labels:
        counts[label] += 1
    assert all(c > 0 for c in counts), f"Empty shards detected: {counts}"
    assert len(result.labels) == 40


def test_all_shards_non_empty_tight() -> None:
    """Exactly as many docs as shards — each must have exactly 1."""
    n = 8
    vectors = _random_vectors(n, 16, seed=99)
    result = shard_documents(vectors, n_shards=n, seed=0)
    counts = [0] * n
    for label in result.labels:
        counts[label] += 1
    assert all(c == 1 for c in counts)


# ------------------------------------------------------------------ #
# Balancing                                                           #
# ------------------------------------------------------------------ #


def test_balancing_terminates_and_respects_threshold() -> None:
    """After balancing, no shard should exceed 1.5 * mean — or we stop safely."""
    n_docs = 40
    n_shards = 6
    vectors = _random_vectors(n_docs, 32, seed=7)
    result = shard_documents(vectors, n_shards=n_shards, seed=42)

    counts = [0] * n_shards
    for label in result.labels:
        counts[label] += 1

    mean_size = n_docs / n_shards
    # Either fully balanced or capped at the guard
    assert max(counts) <= max(int(1.5 * mean_size) + 1, 2), (
        f"Oversized shard detected: {counts}, mean={mean_size}"
    )
    assert sum(counts) == n_docs


def test_balancing_respects_shard_non_empty() -> None:
    """Balancing must never empty a shard."""
    vectors = _random_vectors(20, 16, seed=5)
    result = shard_documents(vectors, n_shards=4, seed=11)
    counts = [0] * 4
    for label in result.labels:
        counts[label] += 1
    assert all(c > 0 for c in counts)


# ------------------------------------------------------------------ #
# Determinism                                                         #
# ------------------------------------------------------------------ #


def test_determinism_same_seed() -> None:
    """Same inputs + seed must produce identical labels and centroids."""
    vectors = _random_vectors(40, 32, seed=7)
    r1 = shard_documents(vectors, n_shards=6, seed=42)
    r2 = shard_documents(vectors, n_shards=6, seed=42)
    assert r1.labels == r2.labels
    assert np.array_equal(r1.centroids, r2.centroids)


def test_determinism_different_seeds_differ() -> None:
    """Different seeds should (very likely) produce different results."""
    vectors = _random_vectors(40, 32, seed=7)
    r1 = shard_documents(vectors, n_shards=6, seed=42)
    r2 = shard_documents(vectors, n_shards=6, seed=999)
    # Not guaranteed, but overwhelmingly likely with real data
    assert r1.labels != r2.labels or not np.array_equal(r1.centroids, r2.centroids)


# ------------------------------------------------------------------ #
# Return type                                                         #
# ------------------------------------------------------------------ #


def test_result_type() -> None:
    vectors = _random_vectors(20, 16, seed=1)
    result = shard_documents(vectors, n_shards=3, seed=0)
    assert isinstance(result, ShardingResult)
    assert isinstance(result.labels, list)
    assert len(result.labels) == 20
    assert result.centroids.shape == (3, 16)
    assert result.centroids.dtype == np.float32


def test_single_shard() -> None:
    """n_shards=1 should always work and put everything in shard 0."""
    vectors = _random_vectors(10, 8, seed=3)
    result = shard_documents(vectors, n_shards=1, seed=0)
    assert all(label == 0 for label in result.labels)
    assert result.centroids.shape == (1, 8)
