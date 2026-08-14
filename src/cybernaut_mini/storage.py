"""Disk-backed storage primitives so a large index is opened, not loaded.

The index artifacts written by :mod:`cybernaut_mini.indexing` are read today by
materialising every one of them into the Python heap: every ``Document`` as a
pydantic model, every document's token list into one dict, and a ``BM25Okapi``
per shard at load time. At the project's target scale — 1,000,000 documents over
1,000 shards, the ratio the post discloses — that is roughly 20 GB of heap on a
box with 5 GB free, so the index simply cannot be opened. This module supplies
the three primitives that replace eager loading: random access into a JSONL file
by record id, memory-mapped embeddings, and a bounded cache for the per-shard
objects a query actually touches.

Nothing here rewires the loader; it is the layer the loader is meant to sit on.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — "NOSIBLE is actually
    made up of a federation of 250,000 smaller search engines called shards", and
    the retrieval walkthrough only ever opens the handful of shards a question is
    routed to, never all of them. NOSIBLE's own research writing goes further and
    describes collections being flushed to disk periodically, with "larger or most
    important collections ... flushed more frequently and ... also more likely to
    be cached in RAM" — so a RAM/disk tier with an eviction policy is part of the
    design being replicated, not an optimisation bolted onto it. Local copy:
    docs/blog-archive/the-road-to-cybernaut-1.md

Assumptions:
    - The post does not describe an on-disk record format, so this module keeps
      the repo's existing JSONL artifacts byte-for-byte unchanged and adds a
      *sidecar* index beside them. An index written by an older build stays
      readable, and the sidecar can be deleted and rebuilt at any time.
    - Record ids are unique within a file and contain no newline. Both hold for
      ids produced by :func:`cybernaut_mini.corpus.make_document_id`; both are
      checked at build time rather than assumed.
    - The id lookup table stores a 64-bit BLAKE2b digest per id, not the id
      string, because 1,000,000 Python strings in a dict cost ~80 MB while the
      digest table costs 16 MB. A digest collision would be a correctness bug, so
      every hit is verified against the record's real id after the seek; the
      digest is an accelerator, never the authority.
    - Quantisation is symmetric and per-matrix. That is only appropriate because
      the vectors reaching it are L2-normalised, so every component already lies
      in [-1, 1] and no single row can stretch the scale for the rest.
    - The LRU cache is not thread-safe. Callers are the query path, which is
      single-threaded here; a lock would make the hit/miss counters depend on
      interleaving, and determinism is a hard requirement in this repo.

Alternatives rejected:
    - SQLite for the document store: gives random access, transactions, and an
      id index for free, and would be the right call for a mutable corpus.
      Rejected because it replaces a plain-text, diffable, canonically-ordered
      artifact with an opaque binary whose page layout is not byte-stable across
      libsqlite versions — which would break this repo's byte-identical rebuild
      guarantee for a feature (mutation) the build does not need.
    - Keeping the offsets in JSON: readable, but a million int64 offsets cost
      ~8 MB as ``.npy`` and ~20 MB of text that then has to be parsed into a
      million Python ints (~30 MB of heap) every time the index is opened. The
      ``.npy`` sidecars are instead memory-mapped, so opening costs kilobytes.
    - A single ``.npz`` bundle for the sidecar: one file instead of five, but
      ``np.savez`` writes a ZIP whose entries carry the current wall-clock time,
      so the same input would produce different bytes on every run.
    - Product quantisation (PQ/OPQ) instead of scalar int8: a far better bytes-
      per-vector ratio and what a billion-scale index would really use. Rejected
      because it needs a trained codebook — another artifact to version, seed,
      and keep deterministic — for a 4x saving that int8 already delivers with
      a single scalar and a measurable, explainable error bound.
"""

from __future__ import annotations

import hashlib
import json
from array import array
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from cybernaut_mini.models import FLOAT_DECIMALS, canonical_dumps

__all__ = [
    "SIDECAR_SUFFIX",
    "STORE_FORMAT_VERSION",
    "JsonlStore",
    "JsonlStoreError",
    "LruCache",
    "MappedEmbeddings",
    "StaleStoreError",
    "dequantise_int8",
    "quantise_int8",
    "sidecar_path",
    "write_embeddings",
    "write_int8_embeddings",
]

Int8Array = npt.NDArray[np.int8]
Int64Array = npt.NDArray[np.int64]
UInt64Array = npt.NDArray[np.uint64]
FloatArray = npt.NDArray[np.float32]

#: Bumped whenever the sidecar layout changes; an older sidecar is rejected
#: rather than misread.
STORE_FORMAT_VERSION = "1"

#: The sidecar lives in a sibling directory, so deleting it is one ``rmtree``
#: and it never collides with an artifact name inside the index directory.
SIDECAR_SUFFIX = ".jsonlidx"

_OFFSETS_NAME = "offsets.npy"
_ID_HASHES_NAME = "id_hashes.npy"
_ID_ROWS_NAME = "id_rows.npy"
_IDS_NAME = "ids.txt"
#: Written last, like the index's own ``_VALID`` marker, so an interrupted build
#: is never mistaken for a complete sidecar.
_META_NAME = "meta.json"

#: int8 payloads carry their scale beside them; without it the codes are noise.
_SCALE_SUFFIX = ".scale.json"

_INT8_MAX = 127


class JsonlStoreError(ValueError):
    """Raised when a JSONL store or its sidecar index cannot be built or read."""


class StaleStoreError(JsonlStoreError):
    """Raised when a sidecar index does not describe the file it sits beside."""


def sidecar_path(path: Path) -> Path:
    """Directory holding the sidecar index for the JSONL file at ``path``."""
    return path.with_name(path.name + SIDECAR_SUFFIX)


def _id_digest(record_id: str) -> int:
    """64-bit BLAKE2b digest of an id, as an unsigned int.

    Deterministic across processes and platforms — unlike :func:`hash`, which is
    salted per process — so the sidecar is reproducible.
    """
    return int.from_bytes(hashlib.blake2b(record_id.encode("utf-8"), digest_size=8).digest(), "big")


# ------------------------------------------------------------------ #
# Random-access JSONL                                                 #
# ------------------------------------------------------------------ #


class JsonlStore:
    """Read any record of a JSONL file by id with one seek, without loading it.

    :meth:`build` scans the file once and writes a sidecar holding a ``(n, 2)``
    int64 array of ``(byte offset, byte length)`` per record, a sorted table of
    id digests mapping to line numbers, and the ids themselves as newline-
    delimited text. :meth:`open` memory-maps those arrays, so opening a store
    over a multi-gigabyte file costs kilobytes of heap regardless of its size.
    """

    def __init__(
        self,
        path: Path,
        *,
        offsets: Int64Array,
        id_hashes: UInt64Array,
        id_rows: Int64Array,
        id_field: str,
    ) -> None:
        self.path = path
        self.id_field = id_field
        self._offsets = offsets
        self._id_hashes = id_hashes
        self._id_rows = id_rows
        self._handle = path.open("rb")
        self._closed = False

    def _check_open(self) -> None:
        """Guard every read against a closed store.

        Once :meth:`close` has released the memory maps, touching the offset or
        digest arrays is a use-after-free that segfaults the interpreter rather
        than raising. Reads therefore check the flag before touching anything.
        """
        if self._closed:
            msg = f"JSONL store over {self.path} is closed"
            raise ValueError(msg)

    # -------------------------------------------------------------- #
    # Construction                                                    #
    # -------------------------------------------------------------- #

    @classmethod
    def build(cls, path: Path, *, id_field: str = "id") -> JsonlStore:
        """Scan ``path`` once, write its sidecar index, and return an open store.

        Streams the file, so peak memory is the offset tables (24 bytes per
        record) rather than the file itself. Blank lines are skipped, matching
        how the rest of the repo reads its JSONL artifacts.

        Raises :class:`JsonlStoreError` for a missing file, a line that is not a
        JSON object, a record with no usable ``id_field``, an id containing a
        newline, or a duplicate id.
        """
        path = Path(path)
        if not path.is_file():
            msg = f"cannot build a JSONL store over a missing file: {path}"
            raise JsonlStoreError(msg)

        sidecar = sidecar_path(path)
        sidecar.mkdir(parents=True, exist_ok=True)
        # meta.json is the completeness marker; drop any older one up front so an
        # interrupted rebuild cannot leave a stale-but-valid-looking sidecar.
        meta_file = sidecar / _META_NAME
        meta_file.unlink(missing_ok=True)

        starts = array("q")
        lengths = array("q")
        digests = array("Q")

        offset = 0
        line_no = 0
        with path.open("rb") as source, (sidecar / _IDS_NAME).open("w", encoding="utf-8") as ids_fh:
            for raw in source:
                start = offset
                offset += len(raw)
                line_no += 1
                stripped = raw.strip()
                if not stripped:
                    continue
                record_id = cls._extract_id(stripped, id_field=id_field, line_no=line_no, path=path)
                starts.append(start)
                lengths.append(len(raw))
                digests.append(_id_digest(record_id))
                ids_fh.write(record_id)
                ids_fh.write("\n")

        count = len(starts)
        offsets = np.empty((count, 2), dtype=np.int64)
        if count:
            offsets[:, 0] = np.frombuffer(starts, dtype=np.int64)
            offsets[:, 1] = np.frombuffer(lengths, dtype=np.int64)
        hashes = np.frombuffer(digests, dtype=np.uint64).copy()

        cls._reject_repeated_ids(hashes, sidecar=sidecar, path=path)

        # A stable sort keeps the sidecar byte-identical across rebuilds even if
        # two ids ever share a digest.
        order = np.argsort(hashes, kind="stable").astype(np.int64)

        np.save(sidecar / _OFFSETS_NAME, offsets)
        np.save(sidecar / _ID_HASHES_NAME, hashes[order])
        np.save(sidecar / _ID_ROWS_NAME, order)
        meta = {
            "format_version": STORE_FORMAT_VERSION,
            "id_field": id_field,
            "record_count": count,
            "source_bytes": path.stat().st_size,
        }
        meta_file.write_text(canonical_dumps(meta) + "\n", encoding="utf-8")

        return cls.open(path, id_field=id_field)

    @staticmethod
    def _extract_id(raw: bytes, *, id_field: str, line_no: int, path: Path) -> str:
        """Parse one line and pull out its id, with the line number in any error."""
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"{path}:{line_no} is not valid JSON: {exc}"
            raise JsonlStoreError(msg) from exc
        if not isinstance(record, dict):
            msg = f"{path}:{line_no} is a JSON {type(record).__name__}, expected an object"
            raise JsonlStoreError(msg)
        value = record.get(id_field)
        if not isinstance(value, str) or not value:
            msg = f"{path}:{line_no} has no non-empty string {id_field!r} field"
            raise JsonlStoreError(msg)
        if "\n" in value:
            msg = f"{path}:{line_no} has an id containing a newline: {value!r}"
            raise JsonlStoreError(msg)
        return value

    @staticmethod
    def _reject_repeated_ids(hashes: UInt64Array, *, sidecar: Path, path: Path) -> None:
        """Fail the build if two records share an id (or, absurdly, a digest).

        The cheap check is on digests alone. It only reads the id list back when
        that check trips, which keeps the common path free of a set of a million
        strings while still naming the offending id when something is wrong.
        """
        if hashes.size == 0 or np.unique(hashes).size == hashes.size:
            return
        seen: set[str] = set()
        duplicates: set[str] = set()
        with (sidecar / _IDS_NAME).open(encoding="utf-8") as fh:
            for line in fh:
                record_id = line.rstrip("\n")
                if record_id in seen:
                    duplicates.add(record_id)
                seen.add(record_id)
        if duplicates:
            sample = sorted(duplicates)[:3]
            msg = f"{path} contains {len(duplicates)} duplicate id(s), e.g. {sample}"
            raise JsonlStoreError(msg)
        msg = (
            f"{path} contains two distinct ids sharing a 64-bit digest; "
            "rebuild with a wider digest before using this file"
        )
        raise JsonlStoreError(msg)

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        id_field: str = "id",
        mmap: bool = True,
        build_if_missing: bool = False,
    ) -> JsonlStore:
        """Open an already-built store over ``path``.

        With ``mmap`` (the default) the offset tables stay on disk and are paged
        in on demand, so opening is O(1) in the number of records.

        Raises :class:`StaleStoreError` if the sidecar is absent, was written by
        a different format version, indexes a different ``id_field``, or
        describes a file of a different size than the one on disk.
        ``build_if_missing`` turns any of those into a rebuild instead.
        """
        path = Path(path)
        try:
            return cls._open_strict(path, id_field=id_field, mmap=mmap)
        except StaleStoreError:
            if not build_if_missing:
                raise
        return cls.build(path, id_field=id_field)

    @classmethod
    def _open_strict(cls, path: Path, *, id_field: str, mmap: bool) -> JsonlStore:
        if not path.is_file():
            msg = f"cannot open a JSONL store over a missing file: {path}"
            raise JsonlStoreError(msg)
        sidecar = sidecar_path(path)
        meta_file = sidecar / _META_NAME
        if not meta_file.is_file():
            msg = f"no sidecar index for {path} (expected {meta_file})"
            raise StaleStoreError(msg)
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("format_version") != STORE_FORMAT_VERSION:
            msg = (
                f"sidecar index for {path} has format version "
                f"{meta.get('format_version')!r}, expected {STORE_FORMAT_VERSION!r}"
            )
            raise StaleStoreError(msg)
        if meta.get("id_field") != id_field:
            msg = (
                f"sidecar index for {path} is keyed on {meta.get('id_field')!r}, "
                f"not the requested {id_field!r}"
            )
            raise StaleStoreError(msg)
        actual_bytes = path.stat().st_size
        if meta.get("source_bytes") != actual_bytes:
            msg = (
                f"sidecar index for {path} describes a {meta.get('source_bytes')}-byte "
                f"file but the file is now {actual_bytes} bytes; rebuild it"
            )
            raise StaleStoreError(msg)

        mode: Literal["r"] | None = "r" if mmap else None
        offsets = np.load(sidecar / _OFFSETS_NAME, mmap_mode=mode)
        id_hashes = np.load(sidecar / _ID_HASHES_NAME, mmap_mode=mode)
        id_rows = np.load(sidecar / _ID_ROWS_NAME, mmap_mode=mode)
        return cls(
            path,
            offsets=offsets,
            id_hashes=id_hashes,
            id_rows=id_rows,
            id_field=id_field,
        )

    # -------------------------------------------------------------- #
    # Reading                                                         #
    # -------------------------------------------------------------- #

    def __len__(self) -> int:
        return int(self._offsets.shape[0])

    def ids(self) -> Iterator[str]:
        """Yield every id in file order, streaming — never a list of a million."""
        self._check_open()
        with (sidecar_path(self.path) / _IDS_NAME).open(encoding="utf-8") as fh:
            for line in fh:
                yield line.rstrip("\n")

    def id_at(self, row: int) -> str:
        """Return the id of record ``row`` (its line number among non-blank lines)."""
        return str(self.get_row(row)[self.id_field])

    def get_row(self, row: int) -> dict[str, Any]:
        """Read record ``row`` by seeking straight to its byte offset."""
        self._check_open()
        count = len(self)
        if not 0 <= row < count:
            msg = f"row {row} is out of range for a store of {count} records"
            raise IndexError(msg)
        start, length = (int(value) for value in self._offsets[row])
        self._handle.seek(start)
        raw = self._handle.read(length)
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            # The offsets said a record started here and it does not. The sidecar
            # is describing a different file than the one on disk — the staleness
            # check is on byte size alone, so an edit that preserves the total size
            # gets this far. Name that cause rather than letting a bare
            # JSONDecodeError escape from what looks like a plain dict lookup.
            msg = (
                f"{self.path} row {row} does not start a valid JSON record at byte "
                f"{start} (length {length}): {exc}. The offset sidecar at "
                f"{sidecar_path(self.path)} is stale — {self.path} was edited without "
                f"changing its size. Delete the sidecar to force a rebuild."
            )
            raise JsonlStoreError(msg) from exc
        if not isinstance(record, dict):
            msg = f"{self.path} row {row} is not a JSON object"
            raise JsonlStoreError(msg)
        return record

    def _candidate_rows(self, record_id: str) -> list[int]:
        """Rows whose id digest matches — normally exactly one, never verified here."""
        self._check_open()
        digest = np.uint64(_id_digest(record_id))
        lo = int(np.searchsorted(self._id_hashes, digest, side="left"))
        hi = int(np.searchsorted(self._id_hashes, digest, side="right"))
        return [int(self._id_rows[position]) for position in range(lo, hi)]

    def row_of(self, record_id: str) -> int:
        """Line number of ``record_id``, verified against the record's real id."""
        for row in self._candidate_rows(record_id):
            if self.get_row(row).get(self.id_field) == record_id:
                return row
        raise KeyError(record_id)

    def get(self, record_id: str) -> dict[str, Any]:
        """Return the record with this id. Raises :class:`KeyError` if absent."""
        for row in self._candidate_rows(record_id):
            record = self.get_row(row)
            if record.get(self.id_field) == record_id:
                return record
        raise KeyError(record_id)

    def get_many(self, record_ids: Iterable[str]) -> list[dict[str, Any]]:
        """Return records for ``record_ids``, in the order asked for.

        Reads are issued in ascending byte-offset order so a batch walks the file
        forwards instead of seeking back and forth; the returned list is still
        aligned with the input. Raises :class:`KeyError` naming the missing ids.
        """
        self._check_open()
        wanted = list(record_ids)
        plan: list[tuple[int, int, int]] = []
        for position, record_id in enumerate(wanted):
            for row in self._candidate_rows(record_id):
                plan.append((int(self._offsets[row][0]), row, position))
        plan.sort()

        found: list[dict[str, Any] | None] = [None] * len(wanted)
        for _start, row, position in plan:
            if found[position] is not None:
                continue
            record = self.get_row(row)
            if record.get(self.id_field) == wanted[position]:
                found[position] = record

        missing = [wanted[i] for i, record in enumerate(found) if record is None]
        if missing:
            raise KeyError(f"{len(missing)} id(s) not in {self.path}, e.g. {missing[:3]}")
        return [record for record in found if record is not None]

    def __contains__(self, record_id: object) -> bool:
        if not isinstance(record_id, str):
            return False
        try:
            self.row_of(record_id)
        except KeyError:
            return False
        return True

    # -------------------------------------------------------------- #
    # Lifecycle                                                       #
    # -------------------------------------------------------------- #

    def close(self) -> None:
        """Close the file handle and release the memory-mapped offset tables.

        Idempotent. After this, every read raises :class:`ValueError` rather than
        dereferencing a released mapping.
        """
        if self._closed:
            return
        self._closed = True
        if not self._handle.closed:
            self._handle.close()
        for name in ("_offsets", "_id_hashes", "_id_rows"):
            array_obj = getattr(self, name)
            underlying = getattr(array_obj, "_mmap", None)
            if underlying is not None:
                underlying.close()

    def __enter__(self) -> JsonlStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# ------------------------------------------------------------------ #
# int8 quantisation                                                   #
# ------------------------------------------------------------------ #


def quantise_int8(vectors: FloatArray) -> tuple[Int8Array, float]:
    """Quantise L2-normalised vectors to int8 with one symmetric scale.

    ``scale = max(|v|) / 127``, so a row of 256 float32 components drops from
    1024 to 256 bytes. Symmetric (zero-point-free) quantisation is the right
    choice here precisely because the input is L2-normalised: the component
    distribution is centred on zero and bounded by 1, so no asymmetric offset
    would buy anything.

    The scale is rounded to the repo's canonical float precision so that the
    value used for quantisation is bit-identical to the value later read back
    from disk — otherwise a round trip through the artifact would dequantise
    against a subtly different number.
    """
    if vectors.ndim != 2:
        msg = f"expected a 2-D array of vectors, got shape {vectors.shape}"
        raise ValueError(msg)
    values = np.asarray(vectors, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        msg = "cannot quantise vectors containing NaN or inf"
        raise ValueError(msg)

    max_abs = float(np.abs(values).max()) if values.size else 0.0
    scale = max_abs / _INT8_MAX
    rounded = round(scale, FLOAT_DECIMALS)
    # An all-zero (or vanishingly small) matrix would otherwise divide by zero.
    scale = rounded if rounded > 0.0 else 1.0

    codes = np.rint(values / np.float32(scale))
    codes = np.clip(codes, -_INT8_MAX, _INT8_MAX)
    return codes.astype(np.int8), scale


def dequantise_int8(codes: Int8Array, scale: float) -> FloatArray:
    """Reverse :func:`quantise_int8`, returning float32 in the original units."""
    return np.asarray(codes, dtype=np.float32) * np.float32(scale)


# ------------------------------------------------------------------ #
# Memory-mapped embeddings                                            #
# ------------------------------------------------------------------ #


def _npy_path(path: Path) -> Path:
    """Mirror ``np.save``'s habit of appending ``.npy``, so sidecar names line up."""
    path = Path(path)
    return path if path.suffix == ".npy" else path.with_name(path.name + ".npy")


def write_embeddings(path: Path, vectors: FloatArray) -> Path:
    """Write float32 embeddings that :class:`MappedEmbeddings` can map."""
    target = _npy_path(path)
    np.save(target, np.asarray(vectors, dtype=np.float32))
    Path(str(target) + _SCALE_SUFFIX).unlink(missing_ok=True)
    return target


def write_int8_embeddings(path: Path, vectors: FloatArray) -> float:
    """Write int8 embeddings plus their scale sidecar; returns the scale used.

    The payload is a quarter the size of the float32 form. Readers do not need
    to know which form was written — :meth:`MappedEmbeddings.rows` dequantises
    transparently and returns float32 either way.
    """
    target = _npy_path(path)
    codes, scale = quantise_int8(vectors)
    np.save(target, codes)
    scale_file = Path(str(target) + _SCALE_SUFFIX)
    payload = {"dtype": "int8", "format_version": STORE_FORMAT_VERSION, "scale": scale}
    scale_file.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")
    return scale


class MappedEmbeddings:
    """The embedding matrix, left on disk and paged in a row at a time.

    ``np.load(..., mmap_mode="r")`` means a 1,000,000 x 256 float32 matrix (1 GB)
    never enters the heap; only the rows a query asks for are ever materialised.
    """

    def __init__(self, matrix: np.ndarray[Any, Any], *, scale: float | None = None) -> None:
        if matrix.ndim != 2:
            msg = f"expected a 2-D embedding matrix, got shape {matrix.shape}"
            raise ValueError(msg)
        self._array = matrix
        self.scale = scale
        self._closed = False

    @classmethod
    def open(cls, path: Path, *, mmap: bool = True) -> MappedEmbeddings:
        """Map the ``.npy`` at ``path``, picking up an int8 scale sidecar if present."""
        path = Path(path)
        if not path.is_file():
            msg = f"no embedding matrix at {path}"
            raise JsonlStoreError(msg)
        matrix = np.load(path, mmap_mode="r" if mmap else None)
        scale: float | None = None
        scale_file = Path(str(path) + _SCALE_SUFFIX)
        if scale_file.is_file():
            scale = float(json.loads(scale_file.read_text(encoding="utf-8"))["scale"])
        elif matrix.dtype == np.int8:
            msg = f"{path} holds int8 codes but {scale_file.name} is missing"
            raise JsonlStoreError(msg)
        return cls(matrix, scale=scale)

    @property
    def shape(self) -> tuple[int, int]:
        rows, dim = self._array.shape
        return int(rows), int(dim)

    @property
    def dim(self) -> int:
        return self.shape[1]

    @property
    def dtype(self) -> np.dtype[Any]:
        return cast("np.dtype[Any]", self._array.dtype)

    @property
    def is_quantised(self) -> bool:
        return bool(self._array.dtype == np.int8)

    def __len__(self) -> int:
        return self.shape[0]

    def rows(self, indices: Sequence[int] | npt.NDArray[np.integer[Any]]) -> FloatArray:
        """Return the requested rows as a real, in-memory float32 array.

        Order and repetition follow ``indices`` exactly. Only the selected rows
        are touched, so a query over 30 shards pays for those shards' vectors and
        nothing else.
        """
        if self._closed:
            msg = "embedding matrix is closed"
            raise ValueError(msg)
        wanted = np.asarray(indices, dtype=np.int64).reshape(-1)
        total = len(self)
        if wanted.size and (wanted.min() < 0 or wanted.max() >= total):
            msg = f"row index out of range for {total} rows"
            raise IndexError(msg)
        if wanted.size == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        block = np.asarray(self._array[wanted])
        if self.is_quantised:
            return dequantise_int8(block, self.scale if self.scale is not None else 1.0)
        return block.astype(np.float32, copy=False)

    def close(self) -> None:
        """Release the mapping. Idempotent; :meth:`rows` raises afterwards."""
        if self._closed:
            return
        self._closed = True
        underlying = getattr(self._array, "_mmap", None)
        if underlying is not None:
            underlying.close()

    def __enter__(self) -> MappedEmbeddings:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# ------------------------------------------------------------------ #
# Bounded cache for per-shard objects                                 #
# ------------------------------------------------------------------ #


class LruCache[K, V]:
    """Count-bounded LRU cache with inspectable hit/miss/eviction counters.

    The point of use is per-shard state that is expensive to construct and huge
    in aggregate — a ``BM25Okapi`` per shard, say. A query routed to 30 of 1,000
    shards should build 30 of them and keep only the most recently used handful
    resident, which is exactly the "most important collections ... more likely to
    be cached in RAM" behaviour the design calls for.

    Eviction is strict least-recently-used with no randomness anywhere, so the
    same sequence of keys always produces the same residency and the same
    counters — which is what makes the counters testable.
    """

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            msg = f"maxsize must be at least 1, got {maxsize}"
            raise ValueError(msg)
        self.maxsize = maxsize
        self._entries: OrderedDict[K, V] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def evictions(self) -> int:
        return self._evictions

    def stats(self) -> dict[str, int]:
        """Counters as a plain dict, for assertions and for trace output."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "size": len(self._entries),
            "maxsize": self.maxsize,
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def keys(self) -> list[K]:
        """Cached keys, least-recently-used first."""
        return list(self._entries)

    def peek(self, key: K) -> V | None:
        """Look at a value without counting a hit or changing recency."""
        return self._entries.get(key)

    def put(self, key: K, value: V) -> None:
        """Insert or refresh ``key``, evicting the least-recently-used if full."""
        if key in self._entries:
            self._entries[key] = value
            self._entries.move_to_end(key)
            return
        self._entries[key] = value
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)
            self._evictions += 1

    def get_or_build(self, key: K, factory: Callable[[], V]) -> V:
        """Return the cached value for ``key``, calling ``factory`` only on a miss.

        ``factory`` takes no arguments so the caller decides — and can be tested
        on — exactly what work a miss costs.
        """
        if key in self._entries:
            self._hits += 1
            self._entries.move_to_end(key)
            return self._entries[key]
        self._misses += 1
        value = factory()
        self.put(key, value)
        return value

    def invalidate(self, key: K) -> bool:
        """Drop one entry. Returns whether it was present; not counted as eviction."""
        return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        """Drop every entry. Counters survive, so a test can span a clear."""
        self._entries.clear()

    def reset_stats(self) -> None:
        """Zero the counters without touching the cached values."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0
