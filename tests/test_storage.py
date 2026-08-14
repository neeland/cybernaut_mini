"""Tests for cybernaut_mini.storage."""

from __future__ import annotations

import json
import shutil
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cybernaut_mini.storage import (
    JsonlStore,
    JsonlStoreError,
    LruCache,
    MappedEmbeddings,
    StaleStoreError,
    dequantise_int8,
    quantise_int8,
    sidecar_path,
    write_embeddings,
    write_int8_embeddings,
)

# ------------------------------------------------------------------ #
# Corpus generation helpers                                           #
# ------------------------------------------------------------------ #

_VOCAB = [f"tok{index:03d}" for index in range(512)]

#: Big enough that the "load it all" comparison is unambiguous, small enough
#: that generating and scanning it stays a couple of seconds.
_BIG_N = 50_000

#: ~45 tokens averages ~315 characters, close to the 303-char sample corpus and
#: within an order of magnitude of a real ~3,000-char CC-News article.
_TOKENS_PER_DOC = 45


def _write_corpus(path: Path, n: int, *, seed: int = 20250814) -> None:
    """Write ``n`` synthetic JSONL records to ``path`` without holding them in memory."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(_VOCAB), size=(n, _TOKENS_PER_DOC))
    with path.open("w", encoding="utf-8") as fh:
        for row in range(n):
            text = " ".join(_VOCAB[int(index)] for index in picks[row])
            record = {"id": f"doc-{row:07d}", "text": text, "row": row}
            fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            fh.write("\n")


def _write_records(path: Path, records: list[dict[str, Any]]) -> Path:
    lines = [json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records]
    path.write_text("".join(lines), encoding="utf-8")
    return path


@pytest.fixture()
def small_store(tmp_path: Path) -> JsonlStore:
    path = _write_records(
        tmp_path / "documents.jsonl",
        [
            {"id": "doc-a", "text": "alpha"},
            {"id": "doc-b", "text": "beta"},
            {"id": "doc-c", "text": "gamma"},
            {"id": "doc-d", "text": "delta"},
        ],
    )
    return JsonlStore.build(path)


@pytest.fixture(scope="module")
def big_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 50,000-record JSONL file with its sidecar index already built."""
    directory = tmp_path_factory.mktemp("big_corpus")
    path = directory / "documents.jsonl"
    _write_corpus(path, _BIG_N)
    JsonlStore.build(path).close()
    return path


def _unit_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# ------------------------------------------------------------------ #
# JsonlStore — basic reads                                            #
# ------------------------------------------------------------------ #


def test_get_returns_the_right_record(small_store: JsonlStore) -> None:
    assert small_store.get("doc-c") == {"id": "doc-c", "text": "gamma"}


def test_len_counts_records(small_store: JsonlStore) -> None:
    assert len(small_store) == 4


def test_ids_streams_in_file_order(small_store: JsonlStore) -> None:
    assert list(small_store.ids()) == ["doc-a", "doc-b", "doc-c", "doc-d"]


def test_get_row_reads_by_line_number(small_store: JsonlStore) -> None:
    assert small_store.get_row(1)["text"] == "beta"


def test_row_of_matches_file_order(small_store: JsonlStore) -> None:
    assert small_store.row_of("doc-d") == 3


def test_id_at_round_trips_row_of(small_store: JsonlStore) -> None:
    assert small_store.id_at(small_store.row_of("doc-b")) == "doc-b"


def test_get_row_out_of_range_raises(small_store: JsonlStore) -> None:
    with pytest.raises(IndexError):
        small_store.get_row(4)


def test_missing_id_raises_key_error(small_store: JsonlStore) -> None:
    with pytest.raises(KeyError):
        small_store.get("doc-zzz")


def test_contains(small_store: JsonlStore) -> None:
    assert "doc-a" in small_store
    assert "doc-zzz" not in small_store


def test_contains_non_string_is_false(small_store: JsonlStore) -> None:
    assert 7 not in small_store


def test_get_many_preserves_requested_order(small_store: JsonlStore) -> None:
    got = small_store.get_many(["doc-d", "doc-a", "doc-c"])
    assert [record["id"] for record in got] == ["doc-d", "doc-a", "doc-c"]


def test_get_many_reports_missing_ids(small_store: JsonlStore) -> None:
    with pytest.raises(KeyError, match="doc-nope"):
        small_store.get_many(["doc-a", "doc-nope"])


def test_get_many_of_nothing_is_empty(small_store: JsonlStore) -> None:
    assert small_store.get_many([]) == []


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text('{"id":"a","v":1}\n\n   \n{"id":"b","v":2}\n', encoding="utf-8")
    store = JsonlStore.build(path)
    assert len(store) == 2
    assert store.get("b") == {"id": "b", "v": 2}
    store.close()


def test_unicode_ids_and_text_survive(tmp_path: Path) -> None:
    path = _write_records(
        tmp_path / "documents.jsonl",
        [{"id": "文書-1", "text": "細菌と酵母"}, {"id": "doc-é", "text": "café"}],
    )
    store = JsonlStore.build(path)
    assert store.get("文書-1")["text"] == "細菌と酵母"
    assert store.get("doc-é")["text"] == "café"
    store.close()


def test_multibyte_text_does_not_shift_offsets(tmp_path: Path) -> None:
    """Offsets are byte offsets, so a 3-byte character must not desynchronise them."""
    path = _write_records(
        tmp_path / "documents.jsonl",
        [{"id": f"doc-{i}", "text": "細菌" * (i + 1)} for i in range(20)],
    )
    store = JsonlStore.build(path)
    for i in range(20):
        assert store.get(f"doc-{i}")["text"] == "細菌" * (i + 1)
    store.close()


def test_empty_file_builds_an_empty_store(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text("", encoding="utf-8")
    store = JsonlStore.build(path)
    assert len(store) == 0
    assert list(store.ids()) == []
    with pytest.raises(KeyError):
        store.get("anything")
    store.close()


def test_context_manager_closes_the_handle(small_store: JsonlStore) -> None:
    with small_store as store:
        store.get("doc-a")
    with pytest.raises(ValueError, match="closed"):
        small_store.get("doc-a")


def test_reads_after_close_raise_instead_of_segfaulting(small_store: JsonlStore) -> None:
    """close() unmaps the digest tables, so every read must be gated on the flag.

    Regression: ``get`` consults the memory-mapped id-digest table *before* it
    touches the file handle, so guarding only the handle let a post-close read
    dereference a released mapping and crash the interpreter.
    """
    small_store.close()
    small_store.close()  # idempotent
    for call in (
        lambda: small_store.get("doc-a"),
        lambda: small_store.get_many(["doc-a"]),
        lambda: small_store.get_row(0),
        lambda: small_store.row_of("doc-a"),
        lambda: small_store.id_at(0),
        lambda: list(small_store.ids()),
    ):
        with pytest.raises(ValueError, match="closed"):
            call()


def test_mapped_embeddings_rows_after_close_raises(tmp_path: Path) -> None:
    """Same use-after-free hazard on the embedding mapping."""
    path = write_embeddings(tmp_path / "embeddings.npy", _unit_vectors(10, 4, seed=59))
    mapped = MappedEmbeddings.open(path)
    mapped.close()
    mapped.close()  # idempotent
    with pytest.raises(ValueError, match="closed"):
        mapped.rows([0])


# ------------------------------------------------------------------ #
# JsonlStore — build-time validation                                  #
# ------------------------------------------------------------------ #


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = _write_records(
        tmp_path / "documents.jsonl",
        [{"id": "doc-a", "v": 1}, {"id": "doc-a", "v": 2}],
    )
    with pytest.raises(JsonlStoreError, match="duplicate"):
        JsonlStore.build(path)


def test_missing_id_field_is_rejected_with_a_line_number(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text('{"id":"a"}\n{"nope":"b"}\n', encoding="utf-8")
    with pytest.raises(JsonlStoreError, match=r":2 has no non-empty string 'id'"):
        JsonlStore.build(path)


def test_invalid_json_is_rejected_with_a_line_number(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text('{"id":"a"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(JsonlStoreError, match=r":2 is not valid JSON"):
        JsonlStore.build(path)


def test_non_object_line_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text('{"id":"a"}\n[1,2,3]\n', encoding="utf-8")
    with pytest.raises(JsonlStoreError, match="expected an object"):
        JsonlStore.build(path)


def test_build_over_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(JsonlStoreError, match="missing file"):
        JsonlStore.build(tmp_path / "nope.jsonl")


def test_custom_id_field(tmp_path: Path) -> None:
    path = _write_records(
        tmp_path / "shards.jsonl",
        [{"shard_id": "s-0", "n": 1}, {"shard_id": "s-1", "n": 2}],
    )
    store = JsonlStore.build(path, id_field="shard_id")
    assert store.get("s-1")["n"] == 2
    store.close()


# ------------------------------------------------------------------ #
# JsonlStore — sidecar lifecycle                                      #
# ------------------------------------------------------------------ #


def test_open_without_a_sidecar_raises(tmp_path: Path) -> None:
    path = _write_records(tmp_path / "documents.jsonl", [{"id": "a"}])
    with pytest.raises(StaleStoreError, match="no sidecar index"):
        JsonlStore.open(path)


def test_open_can_build_the_sidecar_on_demand(tmp_path: Path) -> None:
    path = _write_records(tmp_path / "documents.jsonl", [{"id": "a", "v": 1}])
    store = JsonlStore.open(path, build_if_missing=True)
    assert store.get("a")["v"] == 1
    store.close()


def test_appending_to_the_source_invalidates_the_sidecar(tmp_path: Path) -> None:
    path = _write_records(tmp_path / "documents.jsonl", [{"id": "a", "v": 1}])
    JsonlStore.build(path).close()
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"id":"b","v":2}\n')
    with pytest.raises(StaleStoreError, match="rebuild it"):
        JsonlStore.open(path)


def test_stale_sidecar_can_be_rebuilt_in_place(tmp_path: Path) -> None:
    path = _write_records(tmp_path / "documents.jsonl", [{"id": "a", "v": 1}])
    JsonlStore.build(path).close()
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"id":"b","v":2}\n')
    store = JsonlStore.open(path, build_if_missing=True)
    assert len(store) == 2
    assert store.get("b")["v"] == 2
    store.close()


def test_sidecar_keyed_on_another_field_is_rejected(tmp_path: Path) -> None:
    path = _write_records(tmp_path / "documents.jsonl", [{"id": "a", "key": "k"}])
    JsonlStore.build(path, id_field="key").close()
    with pytest.raises(StaleStoreError, match="keyed on 'key'"):
        JsonlStore.open(path, id_field="id")


def test_sidecar_with_a_foreign_format_version_is_rejected(tmp_path: Path) -> None:
    path = _write_records(tmp_path / "documents.jsonl", [{"id": "a"}])
    JsonlStore.build(path).close()
    meta_file = sidecar_path(path) / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["format_version"] = "99"
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(StaleStoreError, match="format version"):
        JsonlStore.open(path)


def test_sidecar_is_byte_identical_across_rebuilds(tmp_path: Path) -> None:
    """Determinism: the same JSONL must always produce the same sidecar bytes."""
    path = tmp_path / "documents.jsonl"
    _write_corpus(path, 2_000)

    JsonlStore.build(path).close()
    first = tmp_path / "first"
    shutil.copytree(sidecar_path(path), first)
    shutil.rmtree(sidecar_path(path))

    JsonlStore.build(path).close()
    second = sidecar_path(path)

    names = sorted(entry.name for entry in first.iterdir())
    assert names == sorted(entry.name for entry in second.iterdir())
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_offsets_are_int64_so_files_beyond_2gb_are_addressable(small_store: JsonlStore) -> None:
    offsets = np.load(sidecar_path(small_store.path) / "offsets.npy")
    assert offsets.dtype == np.int64
    assert offsets.shape == (4, 2)
    # Offsets are byte positions, and lengths cover the whole line including "\n".
    text = small_store.path.read_bytes()
    for row in range(4):
        start, length = (int(value) for value in offsets[row])
        assert text[start : start + length].endswith(b"\n")


# ------------------------------------------------------------------ #
# JsonlStore — the memory claim, measured                             #
# ------------------------------------------------------------------ #


def test_reading_by_id_costs_orders_of_magnitude_less_than_loading_the_file(
    big_corpus: Path,
) -> None:
    """The whole point of the layer: open + 100 seeks must not cost the file.

    Measured with :mod:`tracemalloc`, which since NumPy 1.22 also accounts for
    NumPy's own allocations, so a sidecar that was loaded rather than mapped
    would show up here.
    """
    sample = [f"doc-{row:07d}" for row in range(0, _BIG_N, _BIG_N // 100)][:100]
    assert len(sample) == 100

    tracemalloc.start()
    tracemalloc.reset_peak()
    store = JsonlStore.open(big_corpus)
    records = store.get_many(sample)
    store_current, store_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(records) == 100
    assert [record["id"] for record in records] == sample
    store.close()
    del records

    tracemalloc.start()
    tracemalloc.reset_peak()
    with big_corpus.open(encoding="utf-8") as fh:
        everything = [json.loads(line) for line in fh if line.strip()]
    eager_current, eager_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(everything) == _BIG_N
    del everything

    file_bytes = big_corpus.stat().st_size
    ratio = eager_peak / max(store_peak, 1)
    print(
        f"\n[storage] {_BIG_N} records, {file_bytes / 1e6:.1f} MB on disk"
        f"\n[storage] JsonlStore.open + 100 get_many : "
        f"peak {store_peak / 1e6:.3f} MB, retained {store_current / 1e6:.3f} MB"
        f"\n[storage] json.loads over the whole file : "
        f"peak {eager_peak / 1e6:.3f} MB, retained {eager_current / 1e6:.3f} MB"
        f"\n[storage] ratio: {ratio:.0f}x less heap for random access"
    )

    # "Orders of magnitude", asserted conservatively so the test is not flaky.
    assert ratio > 100, f"expected >100x, got {ratio:.1f}x"
    assert store_peak < 1_000_000, f"opening the store cost {store_peak} bytes"


def test_random_access_agrees_with_a_full_scan(big_corpus: Path) -> None:
    """The cheap path must return exactly what the expensive path returns."""
    store = JsonlStore.open(big_corpus)
    wanted = {f"doc-{row:07d}" for row in (0, 1, 7, 999, 25_000, _BIG_N - 1)}
    expected: dict[str, dict[str, Any]] = {}
    with big_corpus.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record["id"] in wanted:
                expected[record["id"]] = record
    assert len(expected) == len(wanted)
    for record_id, record in expected.items():
        assert store.get(record_id) == record
    store.close()


def test_store_over_the_big_corpus_reports_every_id(big_corpus: Path) -> None:
    store = JsonlStore.open(big_corpus)
    assert len(store) == _BIG_N
    ids = store.ids()
    assert next(ids) == "doc-0000000"
    assert sum(1 for _ in ids) == _BIG_N - 1
    store.close()


# ------------------------------------------------------------------ #
# int8 quantisation                                                   #
# ------------------------------------------------------------------ #


def test_quantise_int8_shrinks_the_matrix_fourfold() -> None:
    vectors = _unit_vectors(1_000, 256, seed=3)
    codes, _scale = quantise_int8(vectors)
    assert codes.dtype == np.int8
    assert codes.shape == vectors.shape
    assert codes.nbytes * 4 == vectors.nbytes


def test_quantise_int8_accuracy_cost_is_measured_and_bounded() -> None:
    """Report the real reconstruction and cosine error on L2-normalised vectors."""
    corpus = _unit_vectors(5_000, 256, seed=7)
    queries = corpus[:500]

    codes, scale = quantise_int8(corpus)
    restored = dequantise_int8(codes, scale)
    component_err = np.abs(corpus - restored)

    query_codes, query_scale = quantise_int8(queries)
    exact = queries @ corpus.T
    approx = dequantise_int8(query_codes, query_scale) @ restored.T
    cosine_err = np.abs(exact - approx)

    top_exact = np.argsort(-exact, axis=1)[:, :10]
    top_approx = np.argsort(-approx, axis=1)[:, :10]
    recall = float(
        np.mean(
            [
                len(set(a.tolist()) & set(b.tolist())) / 10
                for a, b in zip(top_exact, top_approx, strict=True)
            ]
        )
    )

    print(
        f"\n[storage] int8 quantisation, 5000x256 L2-normalised float32"
        f"\n[storage]   scale                : {scale}"
        f"\n[storage]   component |err|      : max {component_err.max():.3e}, "
        f"mean {component_err.mean():.3e}"
        f"\n[storage]   cosine |err|         : max {cosine_err.max():.3e}, "
        f"mean {cosine_err.mean():.3e}"
        f"\n[storage]   recall@10 vs float32 : {recall:.4f}"
        f"\n[storage]   bytes                : {corpus.nbytes} -> {codes.nbytes}"
    )

    # The theoretical bound is half a quantisation step, i.e. scale / 2.
    assert component_err.max() <= scale / 2 + 1e-6
    assert cosine_err.max() < 0.02
    assert cosine_err.mean() < 0.002
    assert recall >= 0.95


def test_dequantise_recovers_l2_norms_closely() -> None:
    vectors = _unit_vectors(200, 256, seed=11)
    restored = dequantise_int8(*quantise_int8(vectors))
    norms = np.linalg.norm(restored, axis=1)
    assert np.abs(norms - 1.0).max() < 0.01


def test_quantisation_is_deterministic() -> None:
    vectors = _unit_vectors(300, 64, seed=13)
    first_codes, first_scale = quantise_int8(vectors)
    second_codes, second_scale = quantise_int8(vectors.copy())
    assert first_scale == second_scale
    assert np.array_equal(first_codes, second_codes)


def test_quantisation_scale_survives_canonical_json_exactly(tmp_path: Path) -> None:
    """The scale written to disk must be the scale used to quantise, bit for bit."""
    vectors = _unit_vectors(100, 64, seed=17)
    _codes, scale = quantise_int8(vectors)
    written = write_int8_embeddings(tmp_path / "embeddings.npy", vectors)
    assert written == scale
    payload = json.loads((tmp_path / "embeddings.npy.scale.json").read_text(encoding="utf-8"))
    assert payload["scale"] == scale


def test_quantise_all_zero_matrix_does_not_divide_by_zero() -> None:
    codes, scale = quantise_int8(np.zeros((10, 8), dtype=np.float32))
    assert scale == 1.0
    assert not codes.any()


def test_quantise_rejects_non_finite_values() -> None:
    vectors = np.ones((2, 4), dtype=np.float32)
    vectors[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        quantise_int8(vectors)


def test_quantise_rejects_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        quantise_int8(np.ones(8, dtype=np.float32))


# ------------------------------------------------------------------ #
# MappedEmbeddings                                                    #
# ------------------------------------------------------------------ #


def test_open_maps_instead_of_loading(tmp_path: Path) -> None:
    vectors = _unit_vectors(500, 32, seed=19)
    path = write_embeddings(tmp_path / "embeddings.npy", vectors)
    with MappedEmbeddings.open(path) as mapped:
        assert isinstance(mapped._array, np.memmap)
        assert mapped.shape == (500, 32)
        assert mapped.dim == 32
        assert len(mapped) == 500
        assert not mapped.is_quantised


def test_rows_returns_a_real_in_memory_array(tmp_path: Path) -> None:
    vectors = _unit_vectors(500, 32, seed=23)
    path = write_embeddings(tmp_path / "embeddings.npy", vectors)
    with MappedEmbeddings.open(path) as mapped:
        block = mapped.rows([3, 100, 499])
    assert not isinstance(block, np.memmap)
    assert block.dtype == np.float32
    assert np.allclose(block, vectors[[3, 100, 499]])


def test_rows_preserves_order_and_repetition(tmp_path: Path) -> None:
    vectors = _unit_vectors(50, 8, seed=29)
    path = write_embeddings(tmp_path / "embeddings.npy", vectors)
    with MappedEmbeddings.open(path) as mapped:
        block = mapped.rows([9, 1, 9])
    assert np.allclose(block, vectors[[9, 1, 9]])


def test_rows_of_nothing_keeps_the_dimension(tmp_path: Path) -> None:
    path = write_embeddings(tmp_path / "embeddings.npy", _unit_vectors(10, 16, seed=31))
    with MappedEmbeddings.open(path) as mapped:
        assert mapped.rows([]).shape == (0, 16)


def test_rows_out_of_range_raises(tmp_path: Path) -> None:
    path = write_embeddings(tmp_path / "embeddings.npy", _unit_vectors(10, 4, seed=37))
    with MappedEmbeddings.open(path) as mapped:
        with pytest.raises(IndexError):
            mapped.rows([10])
        with pytest.raises(IndexError):
            mapped.rows([-1])


def test_int8_payload_is_dequantised_transparently(tmp_path: Path) -> None:
    vectors = _unit_vectors(400, 128, seed=41)
    path = tmp_path / "embeddings.npy"
    scale = write_int8_embeddings(path, vectors)
    assert path.stat().st_size < vectors.nbytes / 3
    with MappedEmbeddings.open(path) as mapped:
        assert mapped.is_quantised
        assert mapped.scale == scale
        block = mapped.rows([0, 399])
    assert block.dtype == np.float32
    assert np.abs(block - vectors[[0, 399]]).max() <= scale / 2 + 1e-6


def test_write_embeddings_clears_a_stale_scale_sidecar(tmp_path: Path) -> None:
    """Overwriting an int8 matrix with float32 must not leave the old scale behind."""
    path = tmp_path / "embeddings.npy"
    write_int8_embeddings(path, _unit_vectors(10, 8, seed=43))
    assert (tmp_path / "embeddings.npy.scale.json").exists()
    write_embeddings(path, _unit_vectors(10, 8, seed=43))
    assert not (tmp_path / "embeddings.npy.scale.json").exists()
    with MappedEmbeddings.open(path) as mapped:
        assert not mapped.is_quantised


def test_int8_payload_without_its_scale_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.npy"
    write_int8_embeddings(path, _unit_vectors(10, 8, seed=47))
    (tmp_path / "embeddings.npy.scale.json").unlink()
    with pytest.raises(JsonlStoreError, match=r"scale\.json is missing"):
        MappedEmbeddings.open(path)


def test_open_missing_matrix_raises(tmp_path: Path) -> None:
    with pytest.raises(JsonlStoreError, match="no embedding matrix"):
        MappedEmbeddings.open(tmp_path / "nope.npy")


def test_one_dimensional_matrix_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "flat.npy"
    np.save(path, np.zeros(10, dtype=np.float32))
    with pytest.raises(ValueError, match="2-D"):
        MappedEmbeddings.open(path)


def test_mapping_a_matrix_costs_far_less_than_loading_it(tmp_path: Path) -> None:
    """A 100,000 x 64 float32 matrix is 25 MB; mapping it must cost ~nothing."""
    rows, dim = 100_000, 64
    path = tmp_path / "embeddings.npy"
    write_embeddings(path, _unit_vectors(rows, dim, seed=53))
    del_bytes = rows * dim * 4

    tracemalloc.start()
    tracemalloc.reset_peak()
    with MappedEmbeddings.open(path) as mapped:
        block = mapped.rows(list(range(0, rows, rows // 100))[:100])
    mapped_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert block.shape == (100, dim)

    tracemalloc.start()
    tracemalloc.reset_peak()
    eager = np.load(path)
    eager_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert eager.shape == (rows, dim)
    del eager

    print(
        f"\n[storage] embeddings {rows}x{dim} float32 = {del_bytes / 1e6:.1f} MB on disk"
        f"\n[storage]   mmap open + 100 rows : peak {mapped_peak / 1e6:.3f} MB"
        f"\n[storage]   np.load whole matrix : peak {eager_peak / 1e6:.3f} MB"
        f"\n[storage]   ratio: {eager_peak / max(mapped_peak, 1):.0f}x"
    )
    assert eager_peak / max(mapped_peak, 1) > 100


# ------------------------------------------------------------------ #
# LruCache                                                            #
# ------------------------------------------------------------------ #


def test_factory_runs_once_per_key() -> None:
    cache: LruCache[int, str] = LruCache(maxsize=4)
    calls: list[int] = []

    def build(shard_id: int) -> str:
        calls.append(shard_id)
        return f"bm25-{shard_id}"

    assert cache.get_or_build(1, lambda: build(1)) == "bm25-1"
    assert cache.get_or_build(1, lambda: build(1)) == "bm25-1"
    assert calls == [1]
    assert cache.stats() == {"hits": 1, "misses": 1, "evictions": 0, "size": 1, "maxsize": 4}


def test_least_recently_used_key_is_evicted_first() -> None:
    cache: LruCache[str, int] = LruCache(maxsize=2)
    cache.get_or_build("a", lambda: 1)
    cache.get_or_build("b", lambda: 2)
    cache.get_or_build("a", lambda: 1)  # refreshes "a", so "b" is now oldest
    cache.get_or_build("c", lambda: 3)
    assert cache.keys() == ["a", "c"]
    assert "b" not in cache
    assert cache.evictions == 1


def test_peek_does_not_move_a_key_or_count_a_hit() -> None:
    cache: LruCache[str, int] = LruCache(maxsize=2)
    cache.get_or_build("a", lambda: 1)
    cache.get_or_build("b", lambda: 2)
    assert cache.peek("a") == 1
    assert cache.hits == 0
    cache.get_or_build("c", lambda: 3)
    assert "a" not in cache  # peek left "a" as the least recently used


def test_peek_of_an_absent_key_is_none() -> None:
    cache: LruCache[str, int] = LruCache(maxsize=2)
    assert cache.peek("nope") is None


def test_put_refreshes_an_existing_key_without_evicting() -> None:
    cache: LruCache[str, int] = LruCache(maxsize=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)
    assert cache.keys() == ["b", "a"]
    assert cache.peek("a") == 10
    assert cache.evictions == 0


def test_invalidate_removes_without_counting_an_eviction() -> None:
    cache: LruCache[str, int] = LruCache(maxsize=2)
    cache.put("a", 1)
    assert cache.invalidate("a") is True
    assert cache.invalidate("a") is False
    assert len(cache) == 0
    assert cache.evictions == 0


def test_clear_keeps_counters_and_reset_stats_keeps_values() -> None:
    cache: LruCache[str, int] = LruCache(maxsize=2)
    cache.get_or_build("a", lambda: 1)
    cache.get_or_build("a", lambda: 1)
    cache.clear()
    assert cache.stats()["hits"] == 1
    assert len(cache) == 0

    cache.get_or_build("b", lambda: 2)
    cache.reset_stats()
    assert cache.stats()["hits"] == 0
    assert cache.stats()["misses"] == 0
    assert cache.peek("b") == 2


def test_maxsize_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        LruCache(maxsize=0)


def test_a_query_over_1000_shards_builds_only_the_shards_it_touches() -> None:
    """The shape the retrieval path needs: 1,000 shards, 30 touched, 32 resident."""
    built: list[int] = []
    cache: LruCache[int, list[int]] = LruCache(maxsize=32)

    def build_shard(shard_id: int) -> list[int]:
        built.append(shard_id)
        return [shard_id] * 4

    routed = list(range(120, 150))
    for shard_id in routed:
        cache.get_or_build(shard_id, lambda sid=shard_id: build_shard(sid))

    assert built == routed  # 30 of 1,000 shards, in routing order
    assert len(cache) == 30
    assert cache.stats() == {"hits": 0, "misses": 30, "evictions": 0, "size": 30, "maxsize": 32}

    # A second, overlapping query re-uses what is resident and evicts the oldest.
    for shard_id in [125, 126, 900, 901, 902]:
        cache.get_or_build(shard_id, lambda sid=shard_id: build_shard(sid))
    assert built == [*routed, 900, 901, 902]
    assert cache.hits == 2
    assert cache.evictions == 1
    assert len(cache) == 32


def test_eviction_sequence_is_deterministic() -> None:
    """Same key sequence, same residency and same counters, every time."""
    sequence = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]

    def run() -> tuple[list[int], dict[str, int]]:
        cache: LruCache[int, int] = LruCache(maxsize=4)
        for key in sequence:
            cache.get_or_build(key, lambda k=key: k * 10)
        return cache.keys(), cache.stats()

    first = run()
    for _ in range(5):
        assert run() == first

    keys, stats = first
    # Hand-traced: 3 1 4 [1 hit] 5 | 9 evicts 3 | 2 evicts 4 | 6 evicts 1 |
    # [5 hit] | 3 evicts 9 | [5 hit] | 8 evicts 2 | 9 evicts 6 | 7 evicts 3 | [9 hit]
    assert keys == [5, 8, 7, 9]
    assert stats == {"hits": 4, "misses": 11, "evictions": 7, "size": 4, "maxsize": 4}
