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
#: Reads are unaffected — `--input data/01_raw/fixtures/documents.jsonl` stays supported.
PROTECTED_DIRS = ("data/01_raw/fixtures", "data/00_reference")


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
        # Atomic write: crash mid-save leaves no partial file at the real path.
        tmp = self._filepath.with_suffix(".tmp")
        tmp.write_text(canonical_dumps(data) + "\n", encoding="utf-8")
        tmp.replace(self._filepath)

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
                f"`--params input_path=data/01_raw/fixtures/documents.jsonl` "
                f"(CLI: `--input data/01_raw/fixtures/documents.jsonl`)."
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
        # Atomic write: crash mid-save leaves no partial file at the real path.
        tmp = self._filepath.with_suffix(".tmp")
        tmp.write_text(lines, encoding="utf-8")
        tmp.replace(self._filepath)

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
        # Atomic write: np.save to a .tmp.npy sidecar then rename.
        tmp = self._filepath.parent / (self._filepath.stem + ".tmp.npy")
        np.save(tmp, data.astype(np.float32))
        tmp.replace(self._filepath)

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._filepath)}


class HuggingFaceDataset(AbstractDataset[None, list[dict[str, Any]]]):
    """Read-only corpus source backed by a Hugging Face Hub dataset repo.

    ``revision`` is **required and must be a commit SHA or tag**, never a branch
    name: a production shard is only reproducible if the corpus it was built from
    is pinned. ``main`` is rejected outright.

    Set ``streaming=True`` for large datasets (CC-News is hundreds of GB in full):
    the loader opens an :class:`~datasets.IterableDataset` over Arrow shards and
    uses :func:`itertools.islice` so only ``max_rows`` rows are downloaded rather
    than materialising a whole shard boundary.

    ``filter_equals`` pre-filters rows in streaming mode **before** the
    ``max_rows`` cap is applied, so the cap counts only rows that pass the
    filter. Use this to restrict a multilingual corpus to a single language
    without downloading rows that will be dropped — e.g. for CC-News:
    ``filter_equals: {language: en}``.

    Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — CC-News as a
        corpus source; 200k-document builds are the target scale (Phase 1.2).

    Assumptions:
        - ``IterableDataset.features`` is populated from the dataset's Arrow
          schema metadata without iterating rows, so the missing-column check
          can inspect it before consuming any data. When ``features`` is
          ``None`` (a dynamically typed transform was chained upstream), the
          check falls back to the first row's keys and re-inserts that row so
          nothing is lost.
        - ``filter_equals`` is applied in streaming mode only. The eager path
          materialises the whole split first; for large multilingual corpora
          ``streaming=True`` is therefore mandatory.

    Alternatives rejected:
        - ``IterableDataset.select_columns`` for projection in streaming mode.
          Supported since ``datasets>=2.4`` but adds another lazy wrapper; a
          per-row dict comprehension is simpler and independent of the exact
          ``datasets`` API version.
        - Filtering in ``select_documents`` rather than here: would download
          millions of non-English CC-News rows into the raw snapshot, wasting
          disk and network just to drop them.

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
        config: str | None = None,
        columns: list[str] | None = None,
        max_rows: int | None = None,
        token_env: str = "HF_TOKEN",
        streaming: bool = False,
        filter_equals: dict[str, Any] | None = None,
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
        self._config = config
        self._columns = columns
        self._max_rows = max_rows
        self._token_env = token_env
        self._streaming = streaming
        self._filter_equals = filter_equals

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

        if self._streaming:
            return self._load_streaming(load_dataset, token)
        return self._load_eager(load_dataset, token)

    def _load_eager(self, load_dataset: Any, token: str | None) -> list[dict[str, Any]]:
        """Non-streaming path: byte-identical to the original implementation."""
        split = self._split
        if self._max_rows is not None:
            # Slice in the split spec so only the requested rows are materialised.
            split = f"{split}[:{self._max_rows}]"

        dataset = load_dataset(
            self._repo_id,
            self._config,
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

    def _load_streaming(self, load_dataset: Any, token: str | None) -> list[dict[str, Any]]:
        """Streaming path: opens an IterableDataset and slices with islice."""
        import itertools

        dataset = load_dataset(
            self._repo_id,
            self._config,
            revision=self._revision,
            split=self._split,
            token=token,
            streaming=True,
        )

        it: Any = iter(dataset)

        if self._columns is not None:
            # Prefer features (populated from Arrow schema, no rows consumed).
            features = getattr(dataset, "features", None)
            if features is not None:
                available = set(features.keys())
            else:
                # Fall back to the first row's keys; re-insert it so nothing is lost.
                try:
                    first_row = next(it)
                except StopIteration:
                    return []
                available = set(first_row.keys())
                it = itertools.chain([first_row], it)

            missing = set(self._columns) - available
            if missing:
                msg = (
                    f"HuggingFaceDataset({self._repo_id!r}): requested columns "
                    f"{sorted(missing)} are absent; available: {sorted(available)}"
                )
                raise DatasetError(msg)

        # Apply filter_equals BEFORE max_rows so the cap counts only matching rows.
        if self._filter_equals is not None:
            fe = self._filter_equals
            it = (row for row in it if all(row.get(k) == v for k, v in fe.items()))

        if self._max_rows is not None:
            it = itertools.islice(it, self._max_rows)

        if self._columns is not None:
            cols = self._columns
            return [{col: row[col] for col in cols} for row in it]
        return [dict(row) for row in it]

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
            "config": self._config,
            "columns": self._columns,
            "max_rows": self._max_rows,
            "streaming": self._streaming,
            "filter_equals": self._filter_equals,
        }


class HfFileDataset(AbstractDataset[None, list[dict[str, Any]]]):
    """Read-only corpus source backed by JSONL.gz files on the Hugging Face Hub.

    For datasets whose ``load_dataset`` loading script is deprecated (e.g.
    ``miracl/miracl-corpus``), individual shard files are accessed directly via
    :class:`huggingface_hub.HfFileSystem` rather than the ``datasets`` library.
    The caller supplies a glob pattern relative to the repo root; matching shards
    are read in alphabetical order so the result is deterministic.

    Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — MIRACL corpus
        as the fidelity substrate for eval; every docid in the qrels must land in
        the built index. Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

    Assumptions:
        - All files matched by ``path_pattern`` are JSONL compressed with gzip
          (``.jsonl.gz``). Mixed or uncompressed shards are not supported.
        - ``filter_column`` + ``filter_values`` are designed for docid-set filters
          (thousands of values, O(1) set lookup per row). Pass the full qrel docid
          set to guarantee the fidelity rule: every qrels-positive doc is in the
          raw snapshot, so recall is never artificially capped. When both are
          ``None`` the whole dataset streams through; ``max_rows`` caps it.
        - Row counting across shards follows shard alphabetical order. A partial
          last shard is consumed only to the cap — its remaining bytes are not
          downloaded.

    Alternatives rejected:
        - Extending :class:`HuggingFaceDataset` with a ``file_mode`` flag: the
          two loading paths (``load_dataset`` vs ``HfFileSystem + gzip``) import
          entirely different libraries and share zero logic. A flag would produce
          a class with two unrelated personalities. A dedicated class keeps each
          path independently followable.
        - ``hf_hub_download`` per shard (requires enumerating all shard names
          up front): ``HfFileSystem.glob`` lists the directory lazily without
          materialising every path, which is what streaming decompression needs.

    Requires the ``hf`` extra (``uv sync --extra hf``).
    """

    _MUTABLE_REFS = frozenset({"main", "master", "head", "refs/heads/main"})
    _PLACEHOLDER_REF = "replace_with_commit_sha"

    def __init__(
        self,
        repo_id: str,
        revision: str,
        path_pattern: str,
        max_rows: int | None = None,
        filter_column: str | None = None,
        filter_values: list[str] | None = None,
        token_env: str = "HF_TOKEN",
    ) -> None:
        normalized_ref = (revision or "").strip().lower()
        if normalized_ref == self._PLACEHOLDER_REF:
            msg = (
                f"HfFileDataset({repo_id!r}): revision is still the placeholder. "
                f"Resolve the real SHA and paste it into conf/prod/catalog.yml:\n"
                f"  uv run python -c \"from huggingface_hub import HfApi; "
                f"print(HfApi().dataset_info('{repo_id}').sha)\""
            )
            raise DatasetError(msg)
        if not normalized_ref or normalized_ref in self._MUTABLE_REFS:
            msg = (
                f"HfFileDataset({repo_id!r}): revision must pin an immutable "
                f"commit SHA or tag, got {revision!r}."
            )
            raise DatasetError(msg)
        self._repo_id = repo_id
        self._revision = revision
        self._path_pattern = path_pattern
        self._max_rows = max_rows
        self._filter_column = filter_column
        # Convert to a set once so per-row lookup is O(1).
        self._filter_values: frozenset[str] | None = (
            frozenset(filter_values) if filter_values is not None else None
        )
        self._token_env = token_env

    def load(self) -> list[dict[str, Any]]:
        import gzip
        import json

        try:
            from huggingface_hub import HfFileSystem
        except ImportError as exc:
            msg = (
                "HfFileDataset needs the 'hf' extra; install it with "
                "`uv sync --extra hf`"
            )
            raise DatasetError(msg) from exc

        token = os.environ.get(self._token_env) or None
        fs = HfFileSystem(token=token)

        # HfFileSystem uses path format  datasets/{repo_id}@{revision}/{path}
        hf_glob = f"datasets/{self._repo_id}@{self._revision}/{self._path_pattern}"
        try:
            shard_paths = sorted(fs.glob(hf_glob))
        except Exception as exc:
            msg = (
                f"HfFileDataset({self._repo_id!r}): failed to list shards "
                f"matching {self._path_pattern!r} at revision {self._revision!r}: {exc}"
            )
            raise DatasetError(msg) from exc

        if not shard_paths:
            msg = (
                f"HfFileDataset({self._repo_id!r}): no files matched "
                f"{self._path_pattern!r} at revision {self._revision!r}"
            )
            raise DatasetError(msg)

        rows: list[dict[str, Any]] = []
        filter_col = self._filter_column
        filter_vals = self._filter_values  # frozenset or None

        for shard_path in shard_paths:
            if self._max_rows is not None and len(rows) >= self._max_rows:
                break
            try:
                with fs.open(shard_path, "rb") as raw_fh:
                    with gzip.open(raw_fh, "rt", encoding="utf-8") as gz:
                        for line in gz:
                            line = line.strip()
                            if not line:
                                continue
                            row = json.loads(line)
                            if filter_col is not None and filter_vals is not None:
                                if row.get(filter_col) not in filter_vals:
                                    continue
                            rows.append(row)
                            if self._max_rows is not None and len(rows) >= self._max_rows:
                                break
            except DatasetError:
                raise
            except Exception as exc:
                msg = (
                    f"HfFileDataset({self._repo_id!r}): error reading "
                    f"{shard_path!r}: {exc}"
                )
                raise DatasetError(msg) from exc

        return rows

    def save(self, data: None) -> None:
        msg = (
            f"HfFileDataset({self._repo_id!r}) is read-only; write the "
            f"normalised corpus to a JsonlDataset in data/01_raw instead."
        )
        raise DatasetError(msg)

    def _exists(self) -> bool:
        return False

    def _describe(self) -> dict[str, Any]:
        return {
            "repo_id": self._repo_id,
            "revision": self._revision,
            "path_pattern": self._path_pattern,
            "max_rows": self._max_rows,
            "filter_column": self._filter_column,
            "filter_values_count": (
                len(self._filter_values) if self._filter_values is not None else None
            ),
        }


class MiraclTsvDataset(AbstractDataset[None, list[dict[str, Any]]]):
    """Read MIRACL topics + qrels from Hugging Face Hub TSV files.

    ``load_dataset("miracl/miracl", ...)`` is blocked by a deprecated loading
    script (``miracl.py``). This dataset reads the underlying TSV files directly
    via :class:`huggingface_hub.HfFileSystem` and reassembles them into the same
    structured row format that ``load_dataset`` would have returned::

        {
            "query_id": str,
            "query": str,
            "positive_passages": [{"docid": str}, ...],   # grade 1
            "negative_passages": [{"docid": str}, ...],   # grade 0
        }

    This shape is exactly what :func:`cybernaut_mini.corpus.load_miracl_judgments`
    expects, so no downstream code changes are needed.

    Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — MIRACL as the
        qrels source for eval; docid stability requires using the same TSV files
        that define the qrel identifiers, not a post-hoc HF conversion.
        Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

    Assumptions:
        - Topics TSV format: ``{query_id}\\t{question}`` (no header, UTF-8).
        - Qrels TSV format: ``{query_id}\\tQ0\\t{docid}\\t{grade}`` (TREC-style,
          no header). Only grades 0 and 1 appear in MIRACL en-dev.
        - Topics with no matching qrel rows are silently dropped (they carry no
          judgment signal and ``load_miracl_judgments`` would skip them anyway).

    Alternatives rejected:
        - Fixing the ``miracl/miracl`` loading script upstream: out of scope for
          this educational replica.
        - Parsing the dataset via ``HuggingFaceDataset`` with a custom config:
          ``datasets`` refuses to load *any* dataset with a deprecated script.
        - Fetching the topics + qrels as two separate Kedro datasets and merging
          in a node: doubles the catalog boilerplate and adds a merge node just
          to satisfy HF's file layout. Wrapping both files in one dataset class
          keeps the Kedro graph identical to the ``load_dataset`` design.

    Requires the ``hf`` extra (``uv sync --extra hf``).
    """

    _MUTABLE_REFS = frozenset({"main", "master", "head", "refs/heads/main"})
    _PLACEHOLDER_REF = "replace_with_commit_sha"

    def __init__(
        self,
        repo_id: str,
        revision: str,
        language: str,
        split: str = "dev",
        token_env: str = "HF_TOKEN",
    ) -> None:
        normalized_ref = (revision or "").strip().lower()
        if normalized_ref == self._PLACEHOLDER_REF:
            msg = (
                f"MiraclTsvDataset({repo_id!r}): revision is still the placeholder. "
                f"Resolve the real SHA and paste it into conf/prod/catalog.yml:\n"
                f"  uv run python -c \"from huggingface_hub import HfApi; "
                f"print(HfApi().dataset_info('{repo_id}').sha)\""
            )
            raise DatasetError(msg)
        if not normalized_ref or normalized_ref in self._MUTABLE_REFS:
            msg = (
                f"MiraclTsvDataset({repo_id!r}): revision must pin an immutable "
                f"commit SHA or tag, got {revision!r}."
            )
            raise DatasetError(msg)
        self._repo_id = repo_id
        self._revision = revision
        self._language = language
        self._split = split
        self._token_env = token_env

    def load(self) -> list[dict[str, Any]]:
        try:
            from huggingface_hub import HfFileSystem
        except ImportError as exc:
            msg = (
                "MiraclTsvDataset needs the 'hf' extra; install it with "
                "`uv sync --extra hf`"
            )
            raise DatasetError(msg) from exc

        token = os.environ.get(self._token_env) or None
        fs = HfFileSystem(token=token)

        lang = self._language
        split = self._split
        rev = self._revision
        base = f"datasets/{self._repo_id}@{rev}/miracl-v1.0-{lang}"
        topics_path = f"{base}/topics/topics.miracl-v1.0-{lang}-{split}.tsv"
        qrels_path = f"{base}/qrels/qrels.miracl-v1.0-{lang}-{split}.tsv"

        # --- topics: query_id → question ---
        topics: dict[str, str] = {}
        try:
            with fs.open(topics_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        topics[parts[0]] = parts[1]
        except DatasetError:
            raise
        except Exception as exc:
            msg = (
                f"MiraclTsvDataset({self._repo_id!r}): failed to read topics "
                f"from {topics_path!r}: {exc}"
            )
            raise DatasetError(msg) from exc

        # --- qrels: query_id → (positives, negatives) ---
        positives: dict[str, list[dict[str, str]]] = {}
        negatives: dict[str, list[dict[str, str]]] = {}
        try:
            with fs.open(qrels_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 4:
                        continue
                    qid, _q0, docid, grade_str = parts[0], parts[1], parts[2], parts[3]
                    entry = {"docid": docid}
                    if grade_str.strip() == "1":
                        positives.setdefault(qid, []).append(entry)
                    else:
                        negatives.setdefault(qid, []).append(entry)
        except DatasetError:
            raise
        except Exception as exc:
            msg = (
                f"MiraclTsvDataset({self._repo_id!r}): failed to read qrels "
                f"from {qrels_path!r}: {exc}"
            )
            raise DatasetError(msg) from exc

        # --- assemble rows (only topics that have qrel rows) ---
        rows: list[dict[str, Any]] = []
        for qid, question in topics.items():
            pos = positives.get(qid, [])
            neg = negatives.get(qid, [])
            if not pos and not neg:
                continue
            rows.append(
                {
                    "query_id": qid,
                    "query": question,
                    "positive_passages": pos,
                    "negative_passages": neg,
                }
            )
        return rows

    def save(self, data: None) -> None:
        msg = (
            f"MiraclTsvDataset({self._repo_id!r}) is read-only; write the "
            f"normalised judgments to a JsonlDataset in data/02_intermediate instead."
        )
        raise DatasetError(msg)

    def _exists(self) -> bool:
        return False

    def _describe(self) -> dict[str, Any]:
        return {
            "repo_id": self._repo_id,
            "revision": self._revision,
            "language": self._language,
            "split": self._split,
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
