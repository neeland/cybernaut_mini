"""Custom Kedro datasets for the artifact formats used by this project.

All JSON-bearing datasets serialize through :func:`cybernaut_mini.models.canonical_dumps`
so pipeline outputs keep the byte-for-byte determinism guarantee.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from kedro.io import AbstractDataset

from cybernaut_mini.models import canonical_dumps


class CanonicalJsonDataset(AbstractDataset[Any, Any]):
    """A single JSON value written with the canonical writer."""

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def load(self) -> Any:
        import json

        return json.loads(self._filepath.read_text(encoding="utf-8"))

    def save(self, data: Any) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._filepath.write_text(canonical_dumps(data) + "\n", encoding="utf-8")

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._filepath)}


class JsonlDataset(AbstractDataset[list[Any], list[Any]]):
    """A list of JSON records, one canonical-JSON object per line."""

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def load(self) -> list[Any]:
        import json

        records: list[Any] = []
        with self._filepath.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def save(self, data: list[Any]) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        lines = "".join(canonical_dumps(record) + "\n" for record in data)
        self._filepath.write_text(lines, encoding="utf-8")

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._filepath)}


class NpyDataset(AbstractDataset[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    """A float32 NumPy array stored as .npy."""

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def load(self) -> npt.NDArray[np.float32]:
        array: npt.NDArray[np.float32] = np.load(self._filepath).astype(np.float32)
        return array

    def save(self, data: npt.NDArray[np.float32]) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._filepath, data.astype(np.float32))

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._filepath)}
