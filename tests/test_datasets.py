from __future__ import annotations

from pathlib import Path

import numpy as np

from cybernaut_mini.datasets import CanonicalJsonDataset, JsonlDataset, NpyDataset


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
