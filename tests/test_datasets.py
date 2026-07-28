from __future__ import annotations

from pathlib import Path

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
