"""Custom Kedro datasets for the artifact formats used by this project.

All JSON-bearing datasets serialize through :func:`cybernaut_mini.models.canonical_dumps`
so pipeline outputs keep the byte-for-byte determinism guarantee.

Every path the corpus and the index travel goes through a dataset in this module,
so the catalog — not a node body — owns all file and network I/O. That is what makes
``kedro-viz`` show real lineage and what lets ``conf/prod`` swap a local corpus for a
Hugging Face one without touching pipeline code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from kedro.io import AbstractDataset
from kedro.io.core import DatasetError

from cybernaut_mini.models import canonical_dumps

#: Committed fixture directories. A pipeline writing here would silently rewrite the
#: golden corpus that the determinism tests compare against, so saves are refused.
#: Reads are unaffected — `--input data/sample/documents.jsonl` stays supported.
PROTECTED_DIRS = ("data/sample", "data/00_reference")


def _guard_protected(filepath: Path) -> None:
    """Refuse to write over a committed fixture."""
    try:
        relative = filepath.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        # Outside the project (e.g. a pytest tmp_path); nothing to protect.
        return
    as_posix = relative.as_posix()
    for protected in PROTECTED_DIRS:
        if as_posix == protected or as_posix.startswith(f"{protected}/"):
            msg = (
                f"refusing to write {as_posix}: {protected} holds committed fixtures. "
                f"Point this catalog entry at a generated layer "
                f"(data/01_raw, data/02_intermediate, data/03_primary) instead."
            )
            raise DatasetError(msg)


class CanonicalJsonDataset(AbstractDataset[Any, Any]):
    """A single JSON value written with the canonical writer."""

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def load(self) -> Any:
        import json

        return json.loads(self._filepath.read_text(encoding="utf-8"))

    def save(self, data: Any) -> None:
        _guard_protected(self._filepath)
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

        if not self._filepath.exists():
            # A bare traceback here is unhelpful: the usual cause is running a build
            # before the pipeline that produces its input has ever run.
            msg = (
                f"{self._filepath} does not exist.\n"
                f"  If it is a pipeline output, produce it first: "
                f"`kedro run --pipeline corpus_ingest` (see conf/base/catalog.yml).\n"
                f"  If you meant an existing corpus, point at it: "
                f"`--params input_path=data/sample/documents.jsonl` "
                f"(CLI: `--input data/sample/documents.jsonl`)."
            )
            raise DatasetError(msg)

        records: list[Any] = []
        with self._filepath.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def save(self, data: list[Any]) -> None:
        _guard_protected(self._filepath)
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
        _guard_protected(self._filepath)
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._filepath, data.astype(np.float32))

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._filepath)}


class HuggingFaceDataset(AbstractDataset[None, list[dict[str, Any]]]):
    """Read-only corpus source backed by a Hugging Face Hub dataset repo.

    ``revision`` is **required and must be a commit SHA or tag**, never a branch
    name: a production shard is only reproducible if the corpus it was built from
    is pinned. ``main`` is rejected outright.

    Requires the ``hf`` extra (``uv sync --extra hf``). The import is deferred to
    :meth:`load` so the core package — and the whole offline test suite — never
    depends on it.
    """

    #: Refs that move over time and so cannot pin a reproducible build.
    _MUTABLE_REFS = frozenset({"main", "master", "head", "refs/heads/main"})

    #: Shipped in conf/prod as a deliberate tripwire — a real SHA must replace it.
    _PLACEHOLDER_REF = "replace_with_commit_sha"

    def __init__(
        self,
        repo_id: str,
        revision: str,
        split: str = "train",
        columns: list[str] | None = None,
        max_rows: int | None = None,
        token_env: str = "HF_TOKEN",
    ) -> None:
        normalized_ref = (revision or "").strip().lower()
        if normalized_ref == self._PLACEHOLDER_REF:
            msg = (
                f"HuggingFaceDataset({repo_id!r}): revision is still the placeholder. "
                f"Resolve the real SHA and paste it into conf/prod/catalog.yml:\n"
                f"  uv run python -c \"from huggingface_hub import HfApi; "
                f"print(HfApi().dataset_info('{repo_id}').sha)\""
            )
            raise DatasetError(msg)
        if not normalized_ref or normalized_ref in self._MUTABLE_REFS:
            msg = (
                f"HuggingFaceDataset({repo_id!r}): revision must pin an immutable "
                f"commit SHA or tag, got {revision!r}. A moving ref makes the built "
                f"index unreproducible."
            )
            raise DatasetError(msg)
        self._repo_id = repo_id
        self._revision = revision
        self._split = split
        self._columns = columns
        self._max_rows = max_rows
        self._token_env = token_env

    def load(self) -> list[dict[str, Any]]:
        try:
            # The Hugging Face `datasets` package, not this module: Python 3 has no
            # implicit relative imports, so the absolute name wins despite the clash.
            from datasets import load_dataset
        except ImportError as exc:
            msg = (
                "HuggingFaceDataset needs the 'hf' extra; install it with "
                "`uv sync --extra hf`"
            )
            raise DatasetError(msg) from exc

        # A private repo needs a token; a public one must still work without it.
        token = os.environ.get(self._token_env) or None

        split = self._split
        if self._max_rows is not None:
            # Slice in the split spec so only the requested rows are materialised.
            split = f"{split}[:{self._max_rows}]"

        dataset = load_dataset(
            self._repo_id,
            revision=self._revision,
            split=split,
            token=token,
        )
        if self._columns is not None:
            missing = set(self._columns) - set(dataset.column_names)
            if missing:
                msg = (
                    f"HuggingFaceDataset({self._repo_id!r}): requested columns "
                    f"{sorted(missing)} are absent; available: {sorted(dataset.column_names)}"
                )
                raise DatasetError(msg)
            dataset = dataset.select_columns(self._columns)

        rows: list[dict[str, Any]] = [dict(row) for row in dataset]
        return rows

    def save(self, data: None) -> None:
        msg = (
            f"HuggingFaceDataset({self._repo_id!r}) is read-only; write the "
            f"normalised corpus to a JsonlDataset in data/01_raw instead."
        )
        raise DatasetError(msg)

    def _exists(self) -> bool:
        # Existence would require a network round-trip; report False so Kedro
        # never treats a Hub lookup as a cheap local check.
        return False

    def _describe(self) -> dict[str, Any]:
        return {
            "repo_id": self._repo_id,
            "revision": self._revision,
            "split": self._split,
            "columns": self._columns,
            "max_rows": self._max_rows,
        }


class ShardIndexDataset(AbstractDataset[dict[str, Any], Any]):
    """The built shard index directory, as a single catalog-addressable artifact.

    ``save`` accepts the plain-dict payload emitted by the ``build_index`` node and
    performs the canonical write; ``load`` returns a query-ready
    :class:`~cybernaut_mini.indexing.LoadedIndex`. Keeping the conversion here means
    nodes stay pure functions over plain data and never touch the filesystem.
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def load(self) -> Any:
        from cybernaut_mini.indexing import LoadedIndex

        return LoadedIndex.load(self._filepath)

    def save(self, data: dict[str, Any]) -> None:
        from cybernaut_mini.indexing import write_index
        from cybernaut_mini.models import Document, IndexMeta, ShardManifest

        try:
            meta = IndexMeta.model_validate(data["meta"])
            documents = [Document.model_validate(d) for d in data["documents"]]
            manifests = [ShardManifest.model_validate(m) for m in data["manifests"]]
            vectors = np.array(data["vectors"], dtype=np.float32)
            doc_tokens: dict[str, list[str]] = data["doc_tokens"]
        except KeyError as exc:
            msg = f"ShardIndexDataset payload is missing key {exc}"
            raise DatasetError(msg) from exc

        write_index(
            self._filepath,
            meta=meta,
            documents=documents,
            vectors=vectors,
            manifests=manifests,
            doc_tokens=doc_tokens,
        )

    def _exists(self) -> bool:
        # `_VALID` is written last, so its presence is the completeness signal.
        return (self._filepath / "_VALID").exists()

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._filepath)}
