from __future__ import annotations

import gzip
import io
import json
import sys
import time as _time_module
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from kedro.io.core import DatasetError

from cybernaut_mini.datasets import (
    CanonicalJsonDataset,
    HfFileDataset,
    HuggingFaceDataset,
    JsonlDataset,
    MiraclTsvDataset,
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
    """Regression: the corpus_ingest pipeline once overwrote committed fixture data.

    `documents` is both an ingest output and a build input, so a default pointing
    into the fixture directory would make an acquisition run silently rewrite the
    golden corpus the determinism tests compare against.
    """
    project = tmp_path / "project"
    (project / "data" / "01_raw" / "fixtures").mkdir(parents=True)
    fixture = project / "data" / "01_raw" / "fixtures" / "documents.jsonl"
    fixture.write_text('{"id":"original"}\n', encoding="utf-8")
    monkeypatch.chdir(project)

    dataset = JsonlDataset("data/01_raw/fixtures/documents.jsonl")
    with pytest.raises(DatasetError, match="committed fixtures"):
        dataset.save([{"id": "replacement"}])

    # The original bytes must survive the refused write.
    assert fixture.read_text(encoding="utf-8") == '{"id":"original"}\n'


def test_reading_a_committed_fixture_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard is write-only: --input data/01_raw/fixtures/documents.jsonl must still load."""
    project = tmp_path / "project"
    (project / "data" / "01_raw" / "fixtures").mkdir(parents=True)
    (project / "data" / "01_raw" / "fixtures" / "documents.jsonl").write_text(
        '{"id":"original"}\n', encoding="utf-8"
    )
    monkeypatch.chdir(project)

    assert JsonlDataset("data/01_raw/fixtures/documents.jsonl").load() == [{"id": "original"}]


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


# ------------------------------------------------------------------ #
# MiraclTsvDataset — offline tests                                    #
# ------------------------------------------------------------------ #

_MIRACL_SHA = "5be20db9509754dadad47689368639fcec739c00"

# Canonical path components built by MiraclTsvDataset for language="en", split="dev"
_TOPICS_PATH = (
    f"datasets/miracl/miracl@{_MIRACL_SHA}"
    "/miracl-v1.0-en/topics/topics.miracl-v1.0-en-dev.tsv"
)
_QRELS_PATH = (
    f"datasets/miracl/miracl@{_MIRACL_SHA}"
    "/miracl-v1.0-en/qrels/qrels.miracl-v1.0-en-dev.tsv"
)

_SAMPLE_TOPICS = "0\tIs Creole a pidgin of French?\n20\tWhat percentage of oxygen?\n"
# query 0 has one positive + one negative; query 20 has no qrels (should be dropped)
_SAMPLE_QRELS = "0\tQ0\t462221#4\t1\n0\tQ0\t1170520#2\t0\n"


class _FakeFile:
    """Context-manager that iterates over pre-loaded lines."""

    def __init__(self, content: str) -> None:
        self._lines = content.splitlines(keepends=True)

    def __enter__(self) -> Any:
        return iter(self._lines)

    def __exit__(self, *a: Any) -> None:
        pass


def _install_fake_hf_filesystem(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, str],
) -> None:
    """Patch huggingface_hub so MiraclTsvDataset never hits the network."""

    class _FakeFs:
        def __init__(self, token: Any = None) -> None:
            pass

        def open(self, path: str, mode: str = "r", encoding: str = "utf-8") -> _FakeFile:
            if path not in files:
                raise FileNotFoundError(f"fake fs: {path!r} not found")
            return _FakeFile(files[path])

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.HfFileSystem = _FakeFs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)


def _default_files() -> dict[str, str]:
    return {_TOPICS_PATH: _SAMPLE_TOPICS, _QRELS_PATH: _SAMPLE_QRELS}


@pytest.mark.parametrize("revision", ["main", "master", "HEAD", ""])
def test_miracl_tsv_revision_must_be_pinned(revision: str) -> None:
    with pytest.raises(DatasetError, match="immutable"):
        MiraclTsvDataset(repo_id="miracl/miracl", revision=revision, language="en")


def test_miracl_tsv_placeholder_revision_is_rejected() -> None:
    with pytest.raises(DatasetError, match="placeholder"):
        MiraclTsvDataset(
            repo_id="miracl/miracl", revision="REPLACE_WITH_COMMIT_SHA", language="en"
        )


def test_miracl_tsv_describe_contains_key_fields() -> None:
    ds = MiraclTsvDataset(repo_id="miracl/miracl", revision=_MIRACL_SHA, language="en")
    described = ds._describe()
    assert described["repo_id"] == "miracl/miracl"
    assert described["revision"] == _MIRACL_SHA
    assert described["language"] == "en"
    assert described["split"] == "dev"


def test_miracl_tsv_exists_returns_false_without_network() -> None:
    ds = MiraclTsvDataset(repo_id="miracl/miracl", revision=_MIRACL_SHA, language="en")
    assert ds.exists() is False


def test_miracl_tsv_is_read_only() -> None:
    ds = MiraclTsvDataset(repo_id="miracl/miracl", revision=_MIRACL_SHA, language="en")
    with pytest.raises(DatasetError, match="read-only"):
        ds.save([])  # type: ignore[arg-type]


def test_miracl_tsv_load_returns_structured_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_hf_filesystem(monkeypatch, _default_files())
    ds = MiraclTsvDataset(repo_id="miracl/miracl", revision=_MIRACL_SHA, language="en")
    rows = ds.load()
    # query 20 has no qrels → dropped; only query 0 survives
    assert len(rows) == 1
    row = rows[0]
    assert row["query_id"] == "0"
    assert row["query"] == "Is Creole a pidgin of French?"
    assert "positive_passages" in row
    assert "negative_passages" in row


def test_miracl_tsv_positive_passages_have_grade_1_docids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_hf_filesystem(monkeypatch, _default_files())
    ds = MiraclTsvDataset(repo_id="miracl/miracl", revision=_MIRACL_SHA, language="en")
    rows = ds.load()
    pos = rows[0]["positive_passages"]
    assert len(pos) == 1
    assert pos[0]["docid"] == "462221#4"


def test_miracl_tsv_negative_passages_have_grade_0_docids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_hf_filesystem(monkeypatch, _default_files())
    ds = MiraclTsvDataset(repo_id="miracl/miracl", revision=_MIRACL_SHA, language="en")
    rows = ds.load()
    neg = rows[0]["negative_passages"]
    assert len(neg) == 1
    assert neg[0]["docid"] == "1170520#2"


def test_miracl_tsv_drops_topics_with_no_qrels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Topics that appear in the topics file but have zero qrel rows are silently dropped."""
    topics = "99\tOrphan question with no assessments.\n"
    qrels = ""  # no rows at all
    _install_fake_hf_filesystem(monkeypatch, {_TOPICS_PATH: topics, _QRELS_PATH: qrels})
    ds = MiraclTsvDataset(repo_id="miracl/miracl", revision=_MIRACL_SHA, language="en")
    assert ds.load() == []


def test_miracl_tsv_multiple_positives_collected(monkeypatch: pytest.MonkeyPatch) -> None:
    topics = "5\tMulti-positive query.\n"
    qrels = "5\tQ0\tdoc1#0\t1\n5\tQ0\tdoc2#0\t1\n5\tQ0\tdoc3#0\t0\n"
    _install_fake_hf_filesystem(monkeypatch, {_TOPICS_PATH: topics, _QRELS_PATH: qrels})
    ds = MiraclTsvDataset(repo_id="miracl/miracl", revision=_MIRACL_SHA, language="en")
    rows = ds.load()
    assert len(rows) == 1
    assert len(rows[0]["positive_passages"]) == 2
    assert len(rows[0]["negative_passages"]) == 1


# ------------------------------------------------------------------ #
# HfFileDataset — offline tests                                       #
# ------------------------------------------------------------------ #


_CORPUS_SHA = "d921ec7e349ce0d28daf30b2da9da5ee698bef0d"
_REPO_ID = "miracl/miracl-corpus"
_PATH_PAT = "miracl-corpus-v1.0-en/*.jsonl.gz"
_HF_GLOB = f"datasets/{_REPO_ID}@{_CORPUS_SHA}/miracl-corpus-v1.0-en"


def _gz_bytes(rows: list[dict]) -> bytes:
    """Return gzip-compressed JSONL bytes for a fake shard."""
    buf = io.BytesIO()
    with gzip.open(buf, "wt", encoding="utf-8") as gz:
        for row in rows:
            gz.write(json.dumps(row) + "\n")
    return buf.getvalue()


class _GzFile:
    """Binary context-manager wrapping bytes — mimics fs.open(path, 'rb')."""

    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    def __enter__(self) -> io.BytesIO:
        return self._buf

    def __exit__(self, *a: object) -> None:
        pass


def _install_fake_hf_file_fs(
    monkeypatch: pytest.MonkeyPatch,
    shards: dict[str, bytes],
    *,
    fail_shard: str | None = None,
    fail_error: Exception | None = None,
    fail_on_attempts: int = 1,
    created_fs: list | None = None,
) -> None:
    """Patch ``huggingface_hub.HfFileSystem`` for HfFileDataset without network.

    ``shards``: mapping from full HF path to gzip-JSONL bytes.
    ``fail_shard``: basename of shard to make transiently fail.
    ``fail_error``: exception to raise; defaults to the closed-client RuntimeError.
    ``fail_on_attempts``: number of consecutive attempts that raise before success.
    ``created_fs``: if provided, each HfFileSystem() construction appends
        ``{"skip_instance_cache": bool}`` so tests can verify re-creation.
    """
    call_counts: dict[str, int] = {}
    _err = fail_error or RuntimeError(
        "Cannot send a request, as the client has been closed"
    )

    class _FakeFs:
        def __init__(
            self, token: object = None, skip_instance_cache: bool = False
        ) -> None:
            if created_fs is not None:
                created_fs.append({"skip_instance_cache": skip_instance_cache})

        def glob(self, pattern: str) -> list[str]:
            return sorted(shards)

        def open(self, path: str, mode: str = "rb") -> _GzFile:
            if fail_shard and path.endswith(fail_shard):
                call_counts[path] = call_counts.get(path, 0) + 1
                if call_counts[path] <= fail_on_attempts:
                    raise _err
            if path not in shards:
                raise FileNotFoundError(f"fake fs: {path!r}")
            return _GzFile(shards[path])

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.HfFileSystem = _FakeFs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)


def _make_shards(n: int, rows_each: int = 2) -> dict[str, bytes]:
    """Return ``n`` named fake shards, each with ``rows_each`` docid rows."""
    shards = {}
    for i in range(n):
        name = f"docs-{i:02d}.jsonl.gz"
        path = f"datasets/{_REPO_ID}@{_CORPUS_SHA}/miracl-corpus-v1.0-en/{name}"
        rows = [
            {"docid": f"{i*rows_each+j}#0", "title": f"T{i}{j}", "text": f"body {i} {j}"}
            for j in range(rows_each)
        ]
        shards[path] = _gz_bytes(rows)
    return shards


@pytest.mark.parametrize("revision", ["main", "master", "HEAD", ""])
def test_hf_file_dataset_unpinned_revision_is_rejected(revision: str) -> None:
    with pytest.raises(DatasetError, match="immutable"):
        HfFileDataset(repo_id=_REPO_ID, revision=revision, path_pattern=_PATH_PAT)


def test_hf_file_dataset_placeholder_revision_is_rejected() -> None:
    with pytest.raises(DatasetError, match="placeholder"):
        HfFileDataset(
            repo_id=_REPO_ID,
            revision="REPLACE_WITH_COMMIT_SHA",
            path_pattern=_PATH_PAT,
        )


def test_hf_file_dataset_is_read_only() -> None:
    ds = HfFileDataset(repo_id=_REPO_ID, revision=_CORPUS_SHA, path_pattern=_PATH_PAT)
    with pytest.raises(DatasetError, match="read-only"):
        ds.save([])  # type: ignore[arg-type]


def test_hf_file_dataset_exists_never_probes_network() -> None:
    ds = HfFileDataset(repo_id=_REPO_ID, revision=_CORPUS_SHA, path_pattern=_PATH_PAT)
    assert ds.exists() is False


def test_hf_file_dataset_describe_contains_key_fields() -> None:
    ds = HfFileDataset(
        repo_id=_REPO_ID,
        revision=_CORPUS_SHA,
        path_pattern=_PATH_PAT,
        max_rows=500,
    )
    d = ds._describe()
    assert d["repo_id"] == _REPO_ID
    assert d["revision"] == _CORPUS_SHA
    assert d["path_pattern"] == _PATH_PAT
    assert d["max_rows"] == 500


def test_hf_file_dataset_loads_all_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy-path: 3 shards x 2 rows = 6 total rows in alphabetical shard order."""
    shards = _make_shards(3, rows_each=2)
    _install_fake_hf_file_fs(monkeypatch, shards)
    monkeypatch.setattr(_time_module, "sleep", lambda _: None)

    ds = HfFileDataset(repo_id=_REPO_ID, revision=_CORPUS_SHA, path_pattern=_PATH_PAT)
    result = ds.load()
    assert len(result) == 6
    assert result[0]["docid"] == "0#0"  # first row of first shard


def test_hf_file_dataset_max_rows_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    shards = _make_shards(4, rows_each=3)
    _install_fake_hf_file_fs(monkeypatch, shards)
    monkeypatch.setattr(_time_module, "sleep", lambda _: None)

    ds = HfFileDataset(
        repo_id=_REPO_ID, revision=_CORPUS_SHA, path_pattern=_PATH_PAT, max_rows=5
    )
    result = ds.load()
    assert len(result) == 5


# ── Retry tests ───────────────────────────────────────────────────────────────


def test_hf_file_dataset_retries_transient_error_and_returns_full_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shard 2 raises closed-client RuntimeError once; retry succeeds → full result.

    This is the production failure mode: MIRACL shard 55/66,
    ``RuntimeError: Cannot send a request, as the client has been closed``.
    """
    shards = _make_shards(3, rows_each=2)
    shard2_name = sorted(shards)[1].rsplit("/", 1)[-1]  # docs-01.jsonl.gz

    created_fs: list[dict] = []
    _install_fake_hf_file_fs(
        monkeypatch,
        shards,
        fail_shard=shard2_name,
        fail_on_attempts=1,
        created_fs=created_fs,
    )
    sleeps: list[float] = []
    monkeypatch.setattr(_time_module, "sleep", sleeps.append)

    ds = HfFileDataset(repo_id=_REPO_ID, revision=_CORPUS_SHA, path_pattern=_PATH_PAT)
    result = ds.load()

    # Full 6 rows across all 3 shards — nothing was lost.
    assert len(result) == 6

    # sleep() was called once (2-second first retry delay).
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(2.0)

    # The retry created a new HfFileSystem with skip_instance_cache=True.
    assert any(c["skip_instance_cache"] for c in created_fs), (
        "expected HfFileSystem(skip_instance_cache=True) on retry"
    )


def test_hf_file_dataset_permanent_failure_raises_dataset_error_naming_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shard that fails all 4 attempts → DatasetError that names the shard file."""
    shards = _make_shards(2, rows_each=2)
    shard2_name = sorted(shards)[1].rsplit("/", 1)[-1]  # docs-01.jsonl.gz

    _install_fake_hf_file_fs(
        monkeypatch,
        shards,
        fail_shard=shard2_name,
        fail_on_attempts=999,  # always fails
    )
    monkeypatch.setattr(_time_module, "sleep", lambda _: None)

    ds = HfFileDataset(repo_id=_REPO_ID, revision=_CORPUS_SHA, path_pattern=_PATH_PAT)
    with pytest.raises(DatasetError, match=shard2_name):
        ds.load()


def test_hf_file_dataset_non_retryable_error_raises_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-transient error (e.g. JSON decode) must not be retried."""
    shards = _make_shards(2, rows_each=2)
    shard2_name = sorted(shards)[1].rsplit("/", 1)[-1]

    _install_fake_hf_file_fs(
        monkeypatch,
        shards,
        fail_shard=shard2_name,
        fail_error=ValueError("bad json"),
        fail_on_attempts=999,
    )
    sleeps: list[float] = []
    monkeypatch.setattr(_time_module, "sleep", sleeps.append)

    ds = HfFileDataset(repo_id=_REPO_ID, revision=_CORPUS_SHA, path_pattern=_PATH_PAT)
    with pytest.raises(DatasetError):
        ds.load()

    # No sleep means no retry was attempted.
    assert sleeps == [], "non-retryable error must not trigger retry sleeps"


def test_hf_file_dataset_per_shard_progress_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An INFO log line mentioning the shard name must appear for every shard."""
    import logging

    shards = _make_shards(3, rows_each=2)
    _install_fake_hf_file_fs(monkeypatch, shards)
    monkeypatch.setattr(_time_module, "sleep", lambda _: None)

    ds = HfFileDataset(repo_id=_REPO_ID, revision=_CORPUS_SHA, path_pattern=_PATH_PAT)
    with caplog.at_level(logging.INFO, logger="cybernaut_mini.datasets"):
        ds.load()

    shard_names = [p.rsplit("/", 1)[-1] for p in sorted(shards)]
    for name in shard_names:
        matching = [r for r in caplog.records if name in r.message]
        assert matching, f"no INFO log line found for shard {name!r}"
