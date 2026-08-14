from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from kedro.io.core import DatasetError

from cybernaut_mini.datasets import (
    CanonicalJsonDataset,
    HuggingFaceDataset,
    JsonlDataset,
    NpyDataset,
    ShardIndexDataset,
)
from cybernaut_mini.indexing import LoadedIndex

# ------------------------------------------------------------------ #
# Helpers for streaming tests                                         #
# ------------------------------------------------------------------ #

_PINNED_SHA = "0123456789abcdef0123456789abcdef01234567"


class _FakeIterableDataset:
    """Minimal stand-in for datasets.IterableDataset in streaming mode."""

    def __init__(self, rows: list[dict[str, Any]], features: dict[str, Any] | None) -> None:
        self._rows = rows
        # Mirrors IterableDataset.features: a dict-like object or None.
        self.features = features

    def __iter__(self) -> Any:
        return iter(self._rows)


def _install_fake_datasets(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, Any]],
    features: dict[str, Any] | None,
) -> None:
    """Inject a fake `datasets` module so load() never hits the network."""
    iterable = _FakeIterableDataset(rows, features)
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **kw: iterable  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_canonical_json_round_trip_and_bytes(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    dataset = CanonicalJsonDataset(str(path))
    dataset.save({"b": 1.0, "a": [0.123456789123]})
    assert dataset.load() == {"a": [0.12345679], "b": 1.0}
    assert path.read_text(encoding="utf-8") == '{"a":[0.12345679],"b":1.0}\n'


def test_jsonl_round_trip_preserves_order(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    dataset = JsonlDataset(str(path))
    records = [{"id": "b"}, {"id": "a"}]
    dataset.save(records)
    assert dataset.load() == records
    assert path.read_text(encoding="utf-8") == '{"id":"b"}\n{"id":"a"}\n'


def test_npy_round_trip_float32(tmp_path: Path) -> None:
    path = tmp_path / "vectors.npy"
    dataset = NpyDataset(str(path))
    dataset.save(np.array([[1.0, 2.0]], dtype=np.float64))  # type: ignore[arg-type]
    loaded = dataset.load()
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded, [[1.0, 2.0]])


def test_datasets_report_existence(tmp_path: Path) -> None:
    dataset = CanonicalJsonDataset(str(tmp_path / "missing.json"))
    assert not dataset.exists()
    dataset.save(1)
    assert dataset.exists()


# ------------------------------------------------------------------ #
# Fixture protection                                                  #
# ------------------------------------------------------------------ #


def test_save_into_committed_fixture_dir_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the corpus_ingest pipeline once overwrote data/sample.

    `documents` is both an ingest output and a build input, so a default pointing
    into data/sample made an acquisition run silently rewrite the golden corpus the
    determinism tests compare against.
    """
    project = tmp_path / "project"
    (project / "data" / "sample").mkdir(parents=True)
    fixture = project / "data" / "sample" / "documents.jsonl"
    fixture.write_text('{"id":"original"}\n', encoding="utf-8")
    monkeypatch.chdir(project)

    dataset = JsonlDataset("data/sample/documents.jsonl")
    with pytest.raises(DatasetError, match="committed fixtures"):
        dataset.save([{"id": "replacement"}])

    # The original bytes must survive the refused write.
    assert fixture.read_text(encoding="utf-8") == '{"id":"original"}\n'


def test_reading_a_committed_fixture_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard is write-only: --input data/sample/documents.jsonl must still load."""
    project = tmp_path / "project"
    (project / "data" / "sample").mkdir(parents=True)
    (project / "data" / "sample" / "documents.jsonl").write_text(
        '{"id":"original"}\n', encoding="utf-8"
    )
    monkeypatch.chdir(project)

    assert JsonlDataset("data/sample/documents.jsonl").load() == [{"id": "original"}]


def test_generated_layers_are_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    dataset = JsonlDataset("data/03_primary/corpus.jsonl")
    dataset.save([{"id": "a"}])
    assert dataset.load() == [{"id": "a"}]


# ------------------------------------------------------------------ #
# HuggingFaceDataset — pinning guarantees                             #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("revision", ["main", "master", "HEAD", "refs/heads/main", ""])
def test_unpinned_revision_is_rejected(revision: str) -> None:
    with pytest.raises(DatasetError, match="immutable"):
        HuggingFaceDataset(repo_id="NOSIBLE/prediction", revision=revision)


def test_placeholder_revision_is_rejected_with_resolution_hint() -> None:
    with pytest.raises(DatasetError, match="placeholder"):
        HuggingFaceDataset(
            repo_id="NOSIBLE/prediction", revision="REPLACE_WITH_COMMIT_SHA"
        )


def test_pinned_revision_is_accepted_and_described() -> None:
    dataset = HuggingFaceDataset(
        repo_id="NOSIBLE/prediction",
        revision="0123456789abcdef0123456789abcdef01234567",
        columns=["text", "url"],
        max_rows=10,
    )
    described = dataset._describe()
    assert described["repo_id"] == "NOSIBLE/prediction"
    assert described["revision"] == "0123456789abcdef0123456789abcdef01234567"
    assert described["max_rows"] == 10


def test_hugging_face_dataset_is_read_only() -> None:
    """A pipeline must not be able to write back to the Hub.

    Kedro rejects a `None` save on its own, so the guard is exercised with a real
    payload — the case where a node genuinely wires an output to this entry.
    """
    dataset = HuggingFaceDataset(
        repo_id="NOSIBLE/prediction",
        revision="0123456789abcdef0123456789abcdef01234567",
    )
    with pytest.raises(DatasetError, match="read-only"):
        dataset.save([{"text": "anything"}])  # type: ignore[arg-type]


def test_hub_existence_is_never_probed() -> None:
    """`exists()` must not make a network call during a local pipeline run."""
    dataset = HuggingFaceDataset(
        repo_id="NOSIBLE/prediction",
        revision="0123456789abcdef0123456789abcdef01234567",
    )
    assert dataset.exists() is False


# ------------------------------------------------------------------ #
# ShardIndexDataset                                                   #
# ------------------------------------------------------------------ #


def test_shard_index_round_trip(built_index_path: Path, tmp_path: Path) -> None:
    """Payload -> save -> load reconstructs a queryable index."""
    source = LoadedIndex.load(built_index_path)
    target = tmp_path / "index"
    dataset = ShardIndexDataset(str(target))

    assert dataset.exists() is False
    dataset.save(
        {
            "meta": source.meta.model_dump(mode="json"),
            "documents": [d.model_dump(mode="json") for d in source.documents],
            "vectors": source.vectors.tolist(),
            "manifests": [
                m.model_dump(mode="json")
                for m in sorted(source.manifests.values(), key=lambda m: m.shard_id)
            ],
            "doc_tokens": source.doc_tokens,
        }
    )
    assert dataset.exists() is True

    loaded = dataset.load()
    assert loaded.meta.n_documents == source.meta.n_documents
    assert loaded.meta.n_shards == source.meta.n_shards
    assert [d.id for d in loaded.documents] == [d.id for d in source.documents]


def test_shard_index_reports_missing_payload_key(tmp_path: Path) -> None:
    dataset = ShardIndexDataset(str(tmp_path / "index"))
    with pytest.raises(DatasetError, match="missing key"):
        dataset.save({"documents": [], "vectors": [], "manifests": []})


def test_shard_index_absent_without_valid_marker(tmp_path: Path) -> None:
    """A directory of files without `_VALID` is an incomplete write, not an index."""
    target = tmp_path / "index"
    target.mkdir()
    (target / "documents.jsonl").write_text("", encoding="utf-8")
    assert ShardIndexDataset(str(target)).exists() is False


def test_missing_input_reports_how_to_produce_it(tmp_path: Path) -> None:
    """A missing pipeline input must explain itself, not raise a bare FileNotFoundError.

    Regression: in a fresh clone `kedro run` died with an unannotated traceback out of
    `io.open`, giving no hint that the corpus is produced by an upstream pipeline.
    """
    dataset = JsonlDataset(str(tmp_path / "absent" / "corpus.jsonl"))
    with pytest.raises(DatasetError, match="does not exist"):
        dataset.load()


def test_missing_input_error_names_both_recoveries(tmp_path: Path) -> None:
    dataset = JsonlDataset(str(tmp_path / "absent.jsonl"))
    try:
        dataset.load()
    except DatasetError as exc:
        message = str(exc)
    else:  # pragma: no cover - the call above must raise
        pytest.fail("expected DatasetError")

    assert "corpus_ingest" in message, "should name the pipeline that produces it"
    assert "input_path" in message, "should name the override for an existing corpus"


# ------------------------------------------------------------------ #
# HuggingFaceDataset — streaming mode                                 #
# ------------------------------------------------------------------ #


def test_streaming_revision_guards_still_fire_for_mutable_ref() -> None:
    """The SHA-pinning check is at __init__ time and must not be skipped for streaming."""
    with pytest.raises(DatasetError, match="immutable"):
        HuggingFaceDataset(repo_id="test/repo", revision="main", streaming=True)


def test_streaming_placeholder_revision_is_rejected() -> None:
    with pytest.raises(DatasetError, match="placeholder"):
        HuggingFaceDataset(
            repo_id="test/repo", revision="REPLACE_WITH_COMMIT_SHA", streaming=True
        )


def test_streaming_appears_in_describe() -> None:
    ds = HuggingFaceDataset(repo_id="test/repo", revision=_PINNED_SHA, streaming=True)
    described = ds._describe()
    assert described["streaming"] is True


def test_non_streaming_describe_includes_streaming_false() -> None:
    ds = HuggingFaceDataset(repo_id="test/repo", revision=_PINNED_SHA)
    assert ds._describe()["streaming"] is False


def test_streaming_returns_all_rows_when_no_max(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"text": f"doc {i}", "url": f"https://x.com/{i}"} for i in range(8)]
    _install_fake_datasets(monkeypatch, rows, features={"text": None, "url": None})

    ds = HuggingFaceDataset(repo_id="test/repo", revision=_PINNED_SHA, streaming=True)
    result = ds.load()
    assert len(result) == 8
    assert result[0] == {"text": "doc 0", "url": "https://x.com/0"}


def test_streaming_slices_stop_at_max_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"text": f"doc {i}"} for i in range(20)]
    _install_fake_datasets(monkeypatch, rows, features={"text": None})

    ds = HuggingFaceDataset(
        repo_id="test/repo", revision=_PINNED_SHA, max_rows=5, streaming=True
    )
    result = ds.load()
    assert len(result) == 5
    assert [r["text"] for r in result] == ["doc 0", "doc 1", "doc 2", "doc 3", "doc 4"]


def test_streaming_column_projection_via_features(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"text": "hello", "url": "https://x.com", "extra": "drop me"}]
    _install_fake_datasets(
        monkeypatch, rows, features={"text": None, "url": None, "extra": None}
    )

    ds = HuggingFaceDataset(
        repo_id="test/repo",
        revision=_PINNED_SHA,
        columns=["text", "url"],
        streaming=True,
    )
    result = ds.load()
    assert result == [{"text": "hello", "url": "https://x.com"}]


def test_streaming_missing_column_raises_via_features(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"text": "hello"}]
    _install_fake_datasets(monkeypatch, rows, features={"text": None})

    ds = HuggingFaceDataset(
        repo_id="test/repo",
        revision=_PINNED_SHA,
        columns=["text", "no_such_col"],
        streaming=True,
    )
    with pytest.raises(DatasetError, match="absent"):
        ds.load()


def test_streaming_missing_column_falls_back_to_first_row_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When features is None the loader peeks at the first row for column discovery."""
    rows = [{"text": "hello"}]
    _install_fake_datasets(monkeypatch, rows, features=None)  # force first-row fallback

    ds = HuggingFaceDataset(
        repo_id="test/repo",
        revision=_PINNED_SHA,
        columns=["text", "no_such_col"],
        streaming=True,
    )
    with pytest.raises(DatasetError, match="absent"):
        ds.load()


def test_streaming_first_row_not_dropped_when_features_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row consumed to discover the schema must be re-inserted into the result."""
    rows = [{"text": "first"}, {"text": "second"}]
    _install_fake_datasets(monkeypatch, rows, features=None)

    ds = HuggingFaceDataset(
        repo_id="test/repo",
        revision=_PINNED_SHA,
        columns=["text"],
        streaming=True,
    )
    result = ds.load()
    assert [r["text"] for r in result] == ["first", "second"]


def test_streaming_empty_dataset_with_features_none_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_datasets(monkeypatch, rows=[], features=None)

    ds = HuggingFaceDataset(
        repo_id="test/repo",
        revision=_PINNED_SHA,
        columns=["text"],
        streaming=True,
    )
    assert ds.load() == []


def test_streaming_no_columns_no_max_returns_all(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"a": i, "b": i * 2} for i in range(3)]
    _install_fake_datasets(monkeypatch, rows, features=None)

    ds = HuggingFaceDataset(repo_id="test/repo", revision=_PINNED_SHA, streaming=True)
    result = ds.load()
    assert result == rows
