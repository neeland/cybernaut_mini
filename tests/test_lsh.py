"""Random Hyperplanes LSH — offline, over the committed real fixture embeddings.

Everything runs against ``artifacts/fixture/embeddings.npy`` (460 real MIRACL +
CC-News documents embedded with the deterministic hash-256 provider) — no network,
no model downloads, no synthetic corpora. Where a test needs pure-math inputs (scale
invariance, validation errors) it uses seeded numeric arrays only.

Blog ref: https://nosible.com/blog/using-vector-search-to-see-signals-in-company-news
    — the assertions pin the post's contract: hashing a corpus row and a query
    through the same path yields identical codes (self-similarity == ``max_codes``),
    candidate ranking is integers end to end, ``get_signal``'s
    ``lb = index.max_codes * 0.70`` floor works verbatim on ``full_search`` output,
    and the five-step timing anatomy is reported. Local copy:
    ``docs/blog-archive/using-vector-search-to-see-signals-in-company-news.md``.

Assumptions:
    - With the candidate set widened to the whole corpus, LSH search *must* equal
      exact cosine search (the rerank is exact over all rows), so that equality is
      asserted bit-for-bit. With a narrow candidate set (multiplier 8) recall of the
      exact top-1 is probabilistic; 100 leave-one-out queries measured 89/100 on
      this frozen fixture + seed, so the test asserts a safety-margined >= 80.
    - The fixture's hash embeddings bunch neighbors tightly (background cosine
      ~0.66), which makes the recall bound *harder* than it would be on a trained
      embedder — a conservative fixture, not a flattering one.

Alternatives rejected:
    - Mocking the hasher to force collisions: would test numpy, not the contract.
    - Asserting exact matching-code values for real pairs: they are stable for a
      frozen seed but opaque; the floor inequality is the semantics that must hold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from cybernaut_mini.lsh import (
    DEFAULT_N_BITS,
    LSHIndex,
    RandomHyperplanesLSH,
    benchmark_search,
    codes_sidecar_path,
    format_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = REPO_ROOT / "artifacts" / "fixture" / "embeddings.npy"


@pytest.fixture(scope="module")
def embeddings() -> npt.NDArray[np.float32]:
    return np.load(EMBEDDINGS_PATH).astype(np.float32)


@pytest.fixture(scope="module")
def index(embeddings: npt.NDArray[np.float32]) -> LSHIndex:
    return LSHIndex(embeddings, seed=42)


# ------------------------------------------------------------------ #
# Hasher construction and encoding                                   #
# ------------------------------------------------------------------ #


def test_fit_is_deterministic_for_a_seed() -> None:
    first = RandomHyperplanesLSH.fit(64, seed=7)
    second = RandomHyperplanesLSH.fit(64, seed=7)
    assert np.array_equal(first.hyperplanes, second.hyperplanes)
    other = RandomHyperplanesLSH.fit(64, seed=8)
    assert not np.array_equal(first.hyperplanes, other.hyperplanes)


def test_codes_pack_to_uint64_words(embeddings: npt.NDArray[np.float32]) -> None:
    hasher = RandomHyperplanesLSH.fit(embeddings.shape[1], seed=42)
    codes = hasher.encode(embeddings)
    assert codes.shape == (embeddings.shape[0], DEFAULT_N_BITS // 64)
    assert codes.dtype == np.uint64


def test_encoding_is_scale_invariant() -> None:
    rng = np.random.default_rng(0)
    vector = rng.normal(size=64).astype(np.float32)
    hasher = RandomHyperplanesLSH.fit(64, seed=1)
    assert np.array_equal(hasher.encode(vector), hasher.encode(vector * 3.0))
    assert np.array_equal(hasher.encode_query(vector), hasher.encode(vector)[0])


def test_construction_rejects_bad_shapes() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="multiple of 64"):
        RandomHyperplanesLSH(rng.normal(size=(8, 100)).astype(np.float32))
    with pytest.raises(ValueError, match="2-D"):
        RandomHyperplanesLSH(rng.normal(size=8).astype(np.float32))
    hasher = RandomHyperplanesLSH.fit(8, n_bits=64, seed=0)
    with pytest.raises(ValueError, match="expected dim 8"):
        hasher.encode(rng.normal(size=(2, 9)).astype(np.float32))


def test_index_rejects_mismatched_hasher_and_codes(
    embeddings: npt.NDArray[np.float32],
) -> None:
    with pytest.raises(ValueError, match="hasher dim"):
        LSHIndex(embeddings, hasher=RandomHyperplanesLSH.fit(embeddings.shape[1] + 1))
    with pytest.raises(ValueError, match="codes shape"):
        LSHIndex(embeddings, codes=np.zeros((3, 4), dtype=np.uint64))


# ------------------------------------------------------------------ #
# The query path is the corpus path                                  #
# ------------------------------------------------------------------ #


def test_self_similarity_is_max_codes(index: LSHIndex) -> None:
    sims = index.full_search(index.embeddings[10])
    assert int(sims[10]) == index.max_codes == DEFAULT_N_BITS


def test_ranking_is_pure_integer(index: LSHIndex) -> None:
    """No floats between the query hash and the matching-code counts."""
    assert index.codes.dtype == np.uint64
    sims = index.full_search(index.embeddings[0])
    assert sims.dtype == np.int64
    assert sims.min() >= 0
    assert sims.max() <= index.max_codes


def test_similarity_floor_transfers_verbatim(index: LSHIndex) -> None:
    """The post's ``lb = index.max_codes * 0.70`` filter, code for code."""
    assert index.similarity_floor() == pytest.approx(179.2)
    all_sims = index.full_search(index.embeddings[10])
    lb = index.max_codes * 0.70
    valid_ixs = np.where(all_sims >= lb)[0]
    assert 10 in valid_ixs  # An identical vector always survives its own floor.


# ------------------------------------------------------------------ #
# Search quality against exact cosine                                #
# ------------------------------------------------------------------ #


def test_full_candidate_set_equals_exact_search(
    index: LSHIndex, embeddings: npt.NDArray[np.float32]
) -> None:
    """Rerank over all rows == flat exact cosine search, indices and scores."""
    n_rows = embeddings.shape[0]
    for query_row in (0, 137, 288):
        query = index.embeddings[query_row]
        result = index.search(query, top_k=10, candidate_multiplier=n_rows)
        exact = np.argsort(-(index.embeddings @ query), kind="stable")[:10]
        assert np.array_equal(result.indices, exact)
        expected = (index.embeddings @ query)[exact].astype(np.float32)
        assert np.allclose(result.cosines, expected)


def test_narrow_candidate_recall_of_exact_top1(index: LSHIndex) -> None:
    """Leave-one-out: the exact nearest neighbour survives a top-5, x8 search.

    Measured 89/100 on this frozen fixture and seed; asserted with margin.
    """
    hits = 0
    for query_row in range(100):
        sims = index.embeddings @ index.embeddings[query_row]
        sims[query_row] = -1.0
        exact_top1 = int(np.argmax(sims))
        result = index.search(index.embeddings[query_row], top_k=5, candidate_multiplier=8)
        others = [int(i) for i in result.indices if int(i) != query_row]
        hits += exact_top1 in others[:5]
    assert hits >= 80


def test_matching_codes_track_cosine_for_the_real_syndicated_pair(index: LSHIndex) -> None:
    """The fixture's real near-duplicate pair (cosine ~0.915) clears the 0.70 floor."""
    sims = index.embeddings @ index.embeddings.T
    np.fill_diagonal(sims, -1.0)
    i, j = np.unravel_index(int(np.argmax(sims)), sims.shape)
    matching = index.full_search(index.embeddings[i])
    assert float(matching[j]) >= index.similarity_floor()


# ------------------------------------------------------------------ #
# locs restriction (metadata prefilter seam)                         #
# ------------------------------------------------------------------ #


def test_full_search_aligns_with_locs(index: LSHIndex) -> None:
    locs = np.array([5, 10, 400], dtype=np.int64)
    restricted = index.full_search(index.embeddings[10], locs)
    assert restricted.shape == (3,)
    assert int(restricted[1]) == index.max_codes  # locs[1] is the query row itself.
    unrestricted = index.full_search(index.embeddings[10])
    assert np.array_equal(restricted, unrestricted[locs])


def test_search_never_leaves_the_filter(index: LSHIndex) -> None:
    locs = np.arange(0, 100, dtype=np.int64)
    result = index.search(index.embeddings[250], top_k=10, locs=locs)
    assert set(int(i) for i in result.indices) <= set(range(100))
    assert result.indices.shape == (10,)


def test_search_rejects_bad_top_k(index: LSHIndex) -> None:
    with pytest.raises(ValueError, match="top_k"):
        index.search(index.embeddings[0], top_k=0)


# ------------------------------------------------------------------ #
# Determinism and the codes sidecar                                  #
# ------------------------------------------------------------------ #


def test_two_builds_are_identical(embeddings: npt.NDArray[np.float32]) -> None:
    first = LSHIndex(embeddings, seed=42)
    second = LSHIndex(embeddings, seed=42)
    assert np.array_equal(first.codes, second.codes)
    result_a = first.search(embeddings[3], top_k=7)
    result_b = second.search(embeddings[3], top_k=7)
    assert np.array_equal(result_a.indices, result_b.indices)
    assert np.array_equal(result_a.matching_codes, result_b.matching_codes)


def test_codes_sidecar_roundtrip(
    index: LSHIndex, embeddings: npt.NDArray[np.float32], tmp_path: Path
) -> None:
    sidecar = codes_sidecar_path(tmp_path / "embeddings.npy")
    assert sidecar.name == "embeddings.lsh256.npy"
    index.save_codes(sidecar)
    loaded = LSHIndex.load_codes(sidecar)
    assert np.array_equal(loaded, index.codes)
    rebuilt = LSHIndex(embeddings, hasher=index.hasher, codes=loaded)
    assert np.array_equal(
        rebuilt.full_search(embeddings[0]), index.full_search(embeddings[0])
    )


# ------------------------------------------------------------------ #
# Benchmark helper                                                   #
# ------------------------------------------------------------------ #


def test_benchmark_reports_the_five_steps(
    index: LSHIndex, embeddings: npt.NDArray[np.float32]
) -> None:
    result = benchmark_search(index, embeddings[:5], top_k=10)
    assert result.n_vectors == embeddings.shape[0]
    assert result.n_queries == 5
    for value in (
        result.filter_seconds,
        result.encode_seconds,
        result.lsh_seconds,
        result.reconstruct_seconds,
        result.exact_cosine_seconds,
        result.exact_baseline_seconds,
    ):
        assert value >= 0.0
    assert result.lsh_vectors_per_second > 0.0
    assert result.exact_vectors_per_second > 0.0
    report = format_benchmark(result)
    assert "LSH Search over the" in report
    assert "Reconstruct the vectors" in report
    assert report.count("\n") == 5  # Five steps plus the throughput line.


def test_benchmark_respects_locs(index: LSHIndex, embeddings: npt.NDArray[np.float32]) -> None:
    locs = np.arange(50, dtype=np.int64)
    result = benchmark_search(index, embeddings[0], locs=locs)
    assert result.n_vectors == 50
    assert result.n_queries == 1
