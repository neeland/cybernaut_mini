"""Random Hyperplanes LSH: 256-bit sign codes, integer Hamming search, exact re-rank.

The 2024 post's search anatomy spends its time almost anywhere but on floats: the
backbone is a set of random hyperplanes whose sign pattern turns every embedding into
a fixed-width bit code, and candidate ranking is then XOR plus population count on
packed integers. That is the lesson this module preserves: between hashing the query
and gathering the final handful of rows for an exact cosine, *no floating-point
operation touches the corpus*. Similarity at the ranking stage is "number of matching
codes" out of ``max_codes`` (= the bit width), which is exactly the unit the post's
``get_signal`` uses for its ``0.70 * index.max_codes`` similarity floor — that floor
transfers verbatim through :meth:`LSHIndex.full_search` and
:meth:`LSHIndex.similarity_floor`.

Blog ref: https://nosible.com/blog/using-vector-search-to-see-signals-in-company-news
    — "we can avoid computing cosine similarity on floats and instead compute hamming
    distances on integers. Because this involves no floating-point operations, most
    integer operations can be cached, and it is trivial to parallelize, this search
    can be done extremely quickly on CPUs", and the 0.28 s anatomy: "1. Execute the
    SQL query ... 2. Encode the search term ... 3. LSH Search ... 4. Reconstruct the
    vectors ... 5. Compute the cosine similarity of the top N x M results." Local
    copy: ``docs/blog-archive/using-vector-search-to-see-signals-in-company-news.md``.

Assumptions:
    - 256 hyperplanes. The post does not disclose its bit width; 256 divides evenly
      into four ``uint64`` words per row (one cache line), and at 256 bits the
      expected matching count for cosine ``c`` is ``256 * (1 - arccos(c)/pi)`` with a
      standard deviation under 8 bits, so a 0.9-cosine near-duplicate (~219 expected
      matches) sits far above the 0.70 floor (179.2). The width is a parameter, but
      it must stay a multiple of 64 so codes always pack to whole words.
    - Hyperplanes are drawn once from ``rng.normal(size=(dim, n_bits))`` with a fixed
      seed. A spherically-symmetric Gaussian is the textbook Charikar construction:
      each hyperplane's normal is uniform on the sphere, giving the
      ``P(bit match) = 1 - theta/pi`` collision probability the ranking relies on.
    - Sign encoding ignores vector norms (``sign(x @ H)`` is scale-invariant), so
      queries and corpus rows may arrive unnormalised; only the exact-cosine re-rank
      needs unit rows, and the index normalises its copy of the embeddings once at
      construction.
    - ``np.packbits`` output viewed as ``uint64`` is endian-dependent, but both the
      corpus and every query pass through the *same* ``encode`` path in the same
      process, so XOR popcounts are internally consistent on any platform. A codes
      sidecar written on one endianness is only readable on the same endianness —
      acceptable for a laptop artifact that can be re-encoded in milliseconds.
    - Ties in matching-code counts are broken by row position (stable sort), so the
      candidate set — and therefore the whole search — is deterministic.

Alternatives rejected:
    - usearch / FAISS HNSW (both installed, and HNSW already serves shard summaries
      in ``routing.py``): graph indexes answer top-k but cannot cheaply answer "score
      *every* row in this filtered subset", which is the contract ``get_signal``
      needs — the post filters first with SQL, then scores all matching locs. A flat
      code array indexed by ``locs`` composes with any metadata prefilter for free.
    - MinHash/SimHash over token sets: preserve Jaccard, not angular distance; the
      post names Random Hyperplanes as the angular-preserving choice for embeddings.
    - ``np.unpackbits`` + float dot for Hamming: numerically identical, but it
      reintroduces the float path the post's optimisation story exists to remove;
      ``np.bitwise_count`` on packed ``uint64`` words is the pure-integer route.
    - Multi-probe bucketing (hash tables keyed by code prefixes): the classic way to
      avoid scanning all codes, but at laptop scale the full XOR scan of four words
      per row is already memory-bandwidth-bound and beats table overhead; bucketing
      is an optimisation for the 55-million-row regime, not the 200-thousand one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = [
    "DEFAULT_N_BITS",
    "DEFAULT_SIMILARITY_FLOOR_FRACTION",
    "LSHBenchmark",
    "LSHIndex",
    "LSHSearchResult",
    "RandomHyperplanesLSH",
    "codes_sidecar_path",
    "format_benchmark",
]

FloatArray = npt.NDArray[np.float32]
CodeArray = npt.NDArray[np.uint64]
IntArray = npt.NDArray[np.int64]

#: Bit width of a code: 4 x uint64 words per row.
DEFAULT_N_BITS = 256

#: The post's ``get_signal`` keeps results with ``sims >= index.max_codes * 0.70``.
DEFAULT_SIMILARITY_FLOOR_FRACTION = 0.70

_BITS_PER_WORD = 64


def _l2_normalize_rows(matrix: FloatArray) -> FloatArray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


class RandomHyperplanesLSH:
    """The hasher: a frozen ``(dim, n_bits)`` Gaussian matrix and its sign encoder.

    Corpus rows and queries must go through the *same* instance (or one rebuilt from
    the same ``(dim, n_bits, seed)``) — the codes are meaningless across hashers.
    """

    def __init__(self, hyperplanes: FloatArray) -> None:
        planes = np.asarray(hyperplanes, dtype=np.float32)
        if planes.ndim != 2:
            msg = f"hyperplanes must be 2-D (dim, n_bits), got shape {planes.shape}"
            raise ValueError(msg)
        if planes.shape[1] % _BITS_PER_WORD != 0:
            msg = f"n_bits must be a multiple of {_BITS_PER_WORD}, got {planes.shape[1]}"
            raise ValueError(msg)
        self.hyperplanes: FloatArray = planes

    @classmethod
    def fit(cls, dim: int, *, n_bits: int = DEFAULT_N_BITS, seed: int = 42) -> RandomHyperplanesLSH:
        """Draw the hyperplanes: ``H = rng.normal(size=(dim, n_bits))``, seeded."""
        rng = np.random.default_rng(seed)
        planes = rng.normal(size=(dim, n_bits)).astype(np.float32)
        return cls(planes)

    @property
    def dim(self) -> int:
        return int(self.hyperplanes.shape[0])

    @property
    def n_bits(self) -> int:
        return int(self.hyperplanes.shape[1])

    @property
    def n_words(self) -> int:
        return self.n_bits // _BITS_PER_WORD

    def encode(self, vectors: FloatArray) -> CodeArray:
        """Sign-encode rows: ``packbits((X @ H) > 0)`` viewed as uint64 words.

        Accepts a single vector or a matrix; always returns ``(n, n_words)`` uint64.
        Scale-invariant — no normalisation is required before encoding.
        """
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.shape[1] != self.dim:
            msg = f"expected dim {self.dim}, got {matrix.shape[1]}"
            raise ValueError(msg)
        bits = (matrix @ self.hyperplanes) > 0
        packed = np.packbits(bits, axis=1)
        codes: CodeArray = np.ascontiguousarray(packed).view(np.uint64)
        return codes

    def encode_query(self, vector: FloatArray) -> CodeArray:
        """Hash one query through the identical path the corpus took: 1-D code row."""
        code: CodeArray = self.encode(vector)[0]
        return code


@dataclass(frozen=True)
class LSHSearchResult:
    """Top-k rows after integer candidate ranking and exact-cosine re-rank.

    Attributes
    ----------
    indices:
        Global row locs into the indexed embedding matrix, best first.
    cosines:
        Exact cosine similarity of each returned row against the query.
    matching_codes:
        Integer matching-code count (out of ``max_codes``) for each returned row —
        the quantity the candidate ranking was computed on.
    """

    indices: IntArray
    cosines: npt.NDArray[np.float32]
    matching_codes: IntArray


class LSHIndex:
    """Flat LSH index over an embedding matrix: integer rank, exact re-rank.

    The embeddings are copied and L2-normalised once so the re-rank dot product *is*
    cosine similarity. Codes are either computed at construction or supplied from a
    sidecar written by :meth:`save_codes`.
    """

    def __init__(
        self,
        embeddings: FloatArray,
        *,
        hasher: RandomHyperplanesLSH | None = None,
        codes: CodeArray | None = None,
        seed: int = 42,
    ) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            msg = f"embeddings must be 2-D, got shape {matrix.shape}"
            raise ValueError(msg)
        self.embeddings: FloatArray = _l2_normalize_rows(matrix)
        self.hasher: RandomHyperplanesLSH = hasher or RandomHyperplanesLSH.fit(
            matrix.shape[1], seed=seed
        )
        if self.hasher.dim != matrix.shape[1]:
            msg = f"hasher dim {self.hasher.dim} != embedding dim {matrix.shape[1]}"
            raise ValueError(msg)
        if codes is None:
            self.codes: CodeArray = self.hasher.encode(self.embeddings)
        else:
            code_arr = np.asarray(codes, dtype=np.uint64)
            if code_arr.shape != (matrix.shape[0], self.hasher.n_words):
                msg = (
                    f"codes shape {code_arr.shape} != "
                    f"({matrix.shape[0]}, {self.hasher.n_words})"
                )
                raise ValueError(msg)
            self.codes = code_arr

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def max_codes(self) -> int:
        """Maximum matching-code count: the bit width. Named after the post's field."""
        return self.hasher.n_bits

    def similarity_floor(self, fraction: float = DEFAULT_SIMILARITY_FLOOR_FRACTION) -> float:
        """The post's ``lb = index.max_codes * 0.70`` similarity floor."""
        return self.max_codes * fraction

    def full_search(
        self, vector: FloatArray, locs: npt.NDArray[np.int64] | None = None
    ) -> IntArray:
        """Score every (filtered) row against the query — in matching codes.

        Returns one integer per loc, aligned with ``locs`` (or with all rows when
        ``locs`` is ``None``). This is the ``index.full_search(vector, locs)``
        contract ``get_signal`` consumes: callers threshold the result against
        :meth:`similarity_floor` and index back into their loc array. Between the
        query hash and the return there are only XORs and popcounts — no floats.
        """
        qcode = self.hasher.encode_query(np.asarray(vector, dtype=np.float32))
        codes = self.codes if locs is None else self.codes[np.asarray(locs, dtype=np.int64)]
        xor = np.bitwise_xor(codes, qcode[None, :])
        hamming = np.bitwise_count(xor).sum(axis=1, dtype=np.int64)
        matching: IntArray = self.max_codes - hamming
        return matching

    def search(
        self,
        vector: FloatArray,
        top_k: int = 10,
        *,
        candidate_multiplier: int = 4,
        locs: npt.NDArray[np.int64] | None = None,
    ) -> LSHSearchResult:
        """Rank by matching codes, then exact-cosine re-rank the top ``N x M``.

        ``top_k * candidate_multiplier`` candidates survive the integer ranking (the
        post's "top N x M results"); only those rows are gathered ("reconstructed")
        for the float cosine. Ties are broken by row position, so results are
        deterministic.
        """
        if top_k < 1:
            msg = f"top_k must be >= 1, got {top_k}"
            raise ValueError(msg)
        loc_arr: IntArray = (
            np.arange(len(self), dtype=np.int64)
            if locs is None
            else np.asarray(locs, dtype=np.int64)
        )
        matching = self.full_search(vector, loc_arr)
        n_candidates = min(loc_arr.shape[0], top_k * max(1, candidate_multiplier))
        order = np.argsort(-matching, kind="stable")[:n_candidates]
        candidate_locs = loc_arr[order]

        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm > 0.0:
            query = (query / norm).astype(np.float32)
        cosines = self.embeddings[candidate_locs] @ query
        rerank = np.argsort(-cosines, kind="stable")[:top_k]
        return LSHSearchResult(
            indices=candidate_locs[rerank],
            cosines=cosines[rerank].astype(np.float32),
            matching_codes=matching[order][rerank],
        )

    def save_codes(self, path: Path) -> None:
        """Write the packed codes as an ``.npy`` sidecar next to ``embeddings.npy``."""
        np.save(path, self.codes)

    @staticmethod
    def load_codes(path: Path) -> CodeArray:
        codes: CodeArray = np.load(path).astype(np.uint64)
        return codes


def codes_sidecar_path(embeddings_path: Path, n_bits: int = DEFAULT_N_BITS) -> Path:
    """Sidecar naming: ``embeddings.npy`` -> ``embeddings.lsh256.npy``."""
    return embeddings_path.with_suffix(f".lsh{n_bits}.npy")


@dataclass(frozen=True)
class LSHBenchmark:
    """Wall-clock anatomy of one batch of searches, in the post's five steps.

    Times are total seconds across all queries. ``lsh_vectors_per_second`` counts
    integer-ranked rows per second (steps 3 only); ``exact_vectors_per_second`` is
    the flat float-dot baseline over the same rows.
    """

    n_vectors: int
    n_queries: int
    filter_seconds: float
    encode_seconds: float
    lsh_seconds: float
    reconstruct_seconds: float
    exact_cosine_seconds: float
    exact_baseline_seconds: float

    @property
    def lsh_vectors_per_second(self) -> float:
        total = self.n_vectors * self.n_queries
        return total / self.lsh_seconds if self.lsh_seconds > 0 else float("inf")

    @property
    def exact_vectors_per_second(self) -> float:
        total = self.n_vectors * self.n_queries
        return total / self.exact_baseline_seconds if self.exact_baseline_seconds > 0 else float(
            "inf"
        )


def benchmark_search(
    index: LSHIndex,
    queries: FloatArray,
    *,
    locs: npt.NDArray[np.int64] | None = None,
    top_k: int = 10,
    candidate_multiplier: int = 4,
) -> LSHBenchmark:
    """Time the five search steps and a flat exact-cosine baseline.

    Mirrors the post's anatomy: (1) filter to locs, (2) encode the query, (3) LSH
    scan, (4) reconstruct candidate vectors, (5) exact cosine on the top N x M. The
    baseline scores every filtered row with a float dot product, which is what the
    LSH scan replaces.
    """
    query_matrix = np.asarray(queries, dtype=np.float32)
    if query_matrix.ndim == 1:
        query_matrix = query_matrix[None, :]

    t0 = time.perf_counter()
    loc_arr: IntArray = (
        np.arange(len(index), dtype=np.int64) if locs is None else np.asarray(locs, dtype=np.int64)
    )
    filter_seconds = time.perf_counter() - t0

    encode_seconds = 0.0
    lsh_seconds = 0.0
    reconstruct_seconds = 0.0
    exact_cosine_seconds = 0.0
    n_candidates = min(loc_arr.shape[0], top_k * max(1, candidate_multiplier))

    for row in query_matrix:
        t0 = time.perf_counter()
        qcode = index.hasher.encode_query(row)
        encode_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        xor = np.bitwise_xor(index.codes[loc_arr], qcode[None, :])
        hamming = np.bitwise_count(xor).sum(axis=1, dtype=np.int64)
        matching = index.max_codes - hamming
        order = np.argsort(-matching, kind="stable")[:n_candidates]
        lsh_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        candidates = index.embeddings[loc_arr[order]]
        reconstruct_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        norm = float(np.linalg.norm(row))
        unit = (row / norm).astype(np.float32) if norm > 0.0 else row
        cosines = candidates @ unit
        np.argsort(-cosines, kind="stable")
        exact_cosine_seconds += time.perf_counter() - t0

    t0 = time.perf_counter()
    subset = index.embeddings[loc_arr]
    for row in query_matrix:
        norm = float(np.linalg.norm(row))
        unit = (row / norm).astype(np.float32) if norm > 0.0 else row
        np.argsort(-(subset @ unit), kind="stable")
    exact_baseline_seconds = time.perf_counter() - t0

    return LSHBenchmark(
        n_vectors=int(loc_arr.shape[0]),
        n_queries=int(query_matrix.shape[0]),
        filter_seconds=filter_seconds,
        encode_seconds=encode_seconds,
        lsh_seconds=lsh_seconds,
        reconstruct_seconds=reconstruct_seconds,
        exact_cosine_seconds=exact_cosine_seconds,
        exact_baseline_seconds=exact_baseline_seconds,
    )


def format_benchmark(result: LSHBenchmark) -> str:
    """Render the post's five-step timing printout, plus the throughput comparison."""
    lines = [
        f"1. {result.filter_seconds:.4f} seconds - Filter to {result.n_vectors:,} locations.",
        f"2. {result.encode_seconds:.4f} seconds - Encode {result.n_queries} search term(s).",
        f"3. {result.lsh_seconds:.4f} seconds - LSH Search over the "
        f"{result.n_vectors:,} matching vectors.",
        f"4. {result.reconstruct_seconds:.4f} seconds - Reconstruct the vectors to "
        f"calculate cosine similarity.",
        f"5. {result.exact_cosine_seconds:.4f} seconds - Compute the cosine similarity "
        f"of the top N x M results.",
        f"LSH throughput: {result.lsh_vectors_per_second:,.0f} vectors/s; "
        f"exact cosine baseline: {result.exact_vectors_per_second:,.0f} vectors/s.",
    ]
    return "\n".join(lines)
