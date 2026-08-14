"""Vanilla Dense Similarity: an HNSW index over shard-summary embeddings.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 5, "Shard Selection":
    "**Vanilla Dense Similarity -** This selector computes the cosine similarity between
    the E5 embedding of the question and E5 embeddings of the LLM-generated summaries for
    each shard ([uses HNSW])." The stage's tech stack names USearch, NumPy, SciPy, Numba
    and SimSIMD. Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Techniques
----------
- HNSW over shard summary vectors [disclosed] — the post links the HNSW article and names
  USearch, so the graph index is not an inference.
- Cosine over L2-normalised vectors [disclosed] — the post says "cosine similarity".
- Single-threaded construction and search [inferred] — the post says nothing about
  reproducibility. USearch builds its graph in insertion order, and with more than one
  worker that order is nondeterministic, so both ``add`` and ``search`` are pinned to one
  thread here. This repo's contract is same-input-same-output; the cost is build speed on
  a machine with cores to spare, which a one-off index build can afford.

Assumptions: the post does not say what text is embedded beyond "the LLM-generated
    summaries", so :attr:`cybernaut_mini.models.ShardManifest.summary` is embedded on its
    own — not summary plus title. The title is three keywords in this repo, which is a
    lexical signal already carried by the sparse factor; feeding it into the dense factor
    too would correlate the two rankers RRF is meant to keep independent. The post also
    does not give HNSW hyper-parameters, so USearch's own defaults for ``connectivity``
    and the expansion factors are used and surfaced as constructor arguments.

Alternatives rejected: brute-force cosine over every shard centroid, which is what
    ``routing.py`` does today. It is exact and it is what :meth:`exact_neighbours` still
    provides for measurement, but it is O(n_shards) per query and the post's 250,000
    shards is exactly the scale at which that stops being acceptable. Also rejected:
    embedding the shard centroid instead of the summary. A centroid is the mean of the
    documents' vectors, which is cheaper (no encoder call at build time) but is a
    different object from what the post describes, and it is a mean of *document*
    embeddings rather than an embedding of a *description* — it cannot represent a shard
    whose documents straddle two clusters. :func:`summary_vectors` therefore encodes text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from usearch.index import Index

from cybernaut_mini.providers.embeddings import FloatArray, l2_normalize

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cybernaut_mini.models import ShardManifest
    from cybernaut_mini.providers.embeddings import EmbeddingProvider

#: USearch defaults, restated here so the values this repo builds with are visible and
#: are not silently changed by a USearch upgrade.
DEFAULT_CONNECTIVITY = 16
DEFAULT_EXPANSION_ADD = 128
DEFAULT_EXPANSION_SEARCH = 64

_FINGERPRINT_NAME = "hnsw_fingerprint.json"
_GRAPH_NAME = "hnsw_shards.usearch"
_VECTORS_NAME = "hnsw_shards.npy"


class HnswCacheError(ValueError):
    """A persisted HNSW index does not match the index it is being loaded for."""


@dataclass(frozen=True)
class DenseNeighbours:
    """Shard ids best-first with their cosine similarities, aligned element-wise."""

    shard_ids: tuple[int, ...]
    scores: tuple[float, ...]

    def as_dict(self) -> dict[int, float]:
        return dict(zip(self.shard_ids, self.scores, strict=True))


@dataclass(frozen=True)
class RecallReport:
    """How much recall the approximate search gives up against exact cosine.

    ``recall`` is the mean over queries of ``|approx_top_k ∩ exact_top_k| / k``, the
    standard recall@k for approximate nearest neighbour search.
    """

    k: int
    n_queries: int
    n_shards: int
    recall: float
    #: Per-query overlap counts, so a failure says *which* query lost neighbours.
    overlaps: tuple[int, ...]

    def __str__(self) -> str:  # pragma: no cover - human-readable summary only
        return (
            f"recall@{self.k}={self.recall:.4f} over {self.n_queries} queries "
            f"and {self.n_shards} shards"
        )


def summary_vectors(
    manifests: Mapping[int, ShardManifest],
    provider: EmbeddingProvider,
    *,
    shard_ids: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], FloatArray]:
    """Embed every shard's summary, returning the shard ids and an L2-normalised matrix.

    Only manifests are read — no document, token or vector file is touched — so this
    stays cheap on a 1,000,000-document index. A shard whose summary is empty (legal:
    the manifest allows it) still gets a row, so the row order always matches
    ``shard_ids`` and a shard is never silently unreachable.
    """
    ids = tuple(sorted(manifests)) if shard_ids is None else tuple(shard_ids)
    if not ids:
        msg = "cannot build a dense shard index over zero shards"
        raise ValueError(msg)
    texts = [manifests[shard_id].summary for shard_id in ids]
    vectors = np.asarray(provider.embed_documents(texts), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        msg = f"provider returned {vectors.shape!r} for {len(ids)} summaries"
        raise ValueError(msg)
    return ids, l2_normalize(vectors)


class HnswShardIndex:
    """A USearch HNSW graph over shard summary vectors, plus the exact baseline.

    The graph answers :meth:`neighbours` in time that grows with the log of the shard
    count instead of linearly with it, which is the whole reason stage 5 can afford
    250,000 shards. :meth:`exact_neighbours` answers the same question by scanning, and
    exists so :meth:`recall_report` can quantify what the approximation costs.
    """

    def __init__(
        self,
        shard_ids: Sequence[int],
        vectors: FloatArray,
        *,
        connectivity: int = DEFAULT_CONNECTIVITY,
        expansion_add: int = DEFAULT_EXPANSION_ADD,
        expansion_search: int = DEFAULT_EXPANSION_SEARCH,
        _index: Index | None = None,
    ) -> None:
        if len(shard_ids) != vectors.shape[0]:
            msg = f"{len(shard_ids)} shard ids but {vectors.shape[0]} vectors"
            raise ValueError(msg)
        if len(set(shard_ids)) != len(shard_ids):
            msg = "shard ids passed to HnswShardIndex must be unique"
            raise ValueError(msg)
        self.shard_ids: tuple[int, ...] = tuple(int(s) for s in shard_ids)
        self.vectors: FloatArray = np.ascontiguousarray(vectors, dtype=np.float32)
        self.dim: int = int(self.vectors.shape[1])
        self.connectivity = connectivity
        self.expansion_add = expansion_add
        self.expansion_search = expansion_search
        self._keys = np.arange(len(self.shard_ids), dtype=np.uint64)
        self._index: Index = _index if _index is not None else self._build()

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #

    def _new_index(self) -> Index:
        return Index(
            ndim=self.dim,
            metric="cos",
            dtype="f32",
            connectivity=self.connectivity,
            expansion_add=self.expansion_add,
            expansion_search=self.expansion_search,
            multi=False,
        )

    def _build(self) -> Index:
        index = self._new_index()
        # Insertion order is ascending row (== ascending shard id) and threads=1, so the
        # graph — and therefore every approximate result read off it — is reproducible.
        index.add(self._keys, self.vectors, threads=1)
        return index

    @classmethod
    def from_manifests(
        cls,
        manifests: Mapping[int, ShardManifest],
        provider: EmbeddingProvider,
        *,
        shard_ids: Sequence[int] | None = None,
        connectivity: int = DEFAULT_CONNECTIVITY,
        expansion_add: int = DEFAULT_EXPANSION_ADD,
        expansion_search: int = DEFAULT_EXPANSION_SEARCH,
    ) -> HnswShardIndex:
        """Embed the shard summaries with ``provider`` and build the graph over them."""
        ids, vectors = summary_vectors(manifests, provider, shard_ids=shard_ids)
        return cls(
            ids,
            vectors,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
        )

    def __len__(self) -> int:
        return len(self.shard_ids)

    # ------------------------------------------------------------------ #
    # Search                                                              #
    # ------------------------------------------------------------------ #

    def _prepare_query(self, query_vector: FloatArray) -> FloatArray:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            msg = f"query vector has dim {query.shape[0]}, index has dim {self.dim}"
            raise ValueError(msg)
        norm = float(np.linalg.norm(query))
        return query if norm == 0.0 else (query / norm).astype(np.float32)

    def _order(self, rows: Sequence[int], sims: Sequence[float], k: int) -> DenseNeighbours:
        """Sort by descending similarity, ties by ascending shard id, then truncate."""
        pairs = sorted(
            ((self.shard_ids[row], float(sim)) for row, sim in zip(rows, sims, strict=True)),
            key=lambda pair: (-pair[1], pair[0]),
        )[:k]
        return DenseNeighbours(
            shard_ids=tuple(shard_id for shard_id, _ in pairs),
            scores=tuple(score for _, score in pairs),
        )

    def neighbours(self, query_vector: FloatArray, k: int) -> DenseNeighbours:
        """Approximate top-``k`` shards by cosine, via the HNSW graph."""
        if k < 1:
            msg = f"k must be >= 1, got {k}"
            raise ValueError(msg)
        query = self._prepare_query(query_vector)
        matches = self._index.search(query, min(k, len(self)), threads=1)
        rows = [int(key) for key in matches.keys]
        # USearch reports cosine *distance*; recover the similarity the post talks about
        # from the stored vector so the number in the trace is a real cosine.
        sims = [float(self.vectors[row] @ query) for row in rows]
        return self._order(rows, sims, k)

    def exact_neighbours(self, query_vector: FloatArray, k: int) -> DenseNeighbours:
        """Exact top-``k`` shards by cosine, by scanning every shard vector."""
        if k < 1:
            msg = f"k must be >= 1, got {k}"
            raise ValueError(msg)
        query = self._prepare_query(query_vector)
        sims = self.vectors @ query
        k = min(k, len(self))
        # argpartition picks the top-k cheaply; _order does the deterministic tie-break.
        candidate_rows = np.argpartition(-sims, k - 1)[:k]
        return self._order(
            [int(row) for row in candidate_rows],
            [float(sims[row]) for row in candidate_rows],
            k,
        )

    def recall_report(self, query_vectors: FloatArray, k: int) -> RecallReport:
        """Measure recall@k of :meth:`neighbours` against :meth:`exact_neighbours`."""
        queries = np.atleast_2d(np.asarray(query_vectors, dtype=np.float32))
        effective_k = min(k, len(self))
        overlaps: list[int] = []
        for query in queries:
            approx = set(self.neighbours(query, effective_k).shard_ids)
            exact = set(self.exact_neighbours(query, effective_k).shard_ids)
            overlaps.append(len(approx & exact))
        recall = sum(overlaps) / (len(overlaps) * effective_k) if overlaps else 0.0
        return RecallReport(
            k=effective_k,
            n_queries=len(overlaps),
            n_shards=len(self),
            recall=recall,
            overlaps=tuple(overlaps),
        )

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def fingerprint(self, embedding_model: str) -> dict[str, object]:
        """Identity of the index this graph was built for; checked on load."""
        return {
            "embedding_model": embedding_model,
            "dim": self.dim,
            "n_shards": len(self),
            "shard_ids_digest": _shard_ids_digest(self.shard_ids),
            "connectivity": self.connectivity,
            "expansion_add": self.expansion_add,
            "expansion_search": self.expansion_search,
        }

    def save(self, directory: Path, *, embedding_model: str) -> None:
        """Persist the graph, its vectors and a fingerprint into ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._index.save(str(directory / _GRAPH_NAME))
        np.save(directory / _VECTORS_NAME, self.vectors)
        payload = dict(self.fingerprint(embedding_model))
        payload["shard_ids"] = list(self.shard_ids)
        (directory / _FINGERPRINT_NAME).write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, *, embedding_model: str) -> HnswShardIndex:
        """Restore a graph saved by :meth:`save`, or raise :class:`HnswCacheError`."""
        directory = Path(directory)
        fingerprint_path = directory / _FINGERPRINT_NAME
        if not fingerprint_path.exists():
            msg = f"no persisted HNSW shard index in {directory}"
            raise HnswCacheError(msg)
        payload = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if payload.get("embedding_model") != embedding_model:
            msg = (
                f"persisted HNSW shard index was built with embedding model "
                f"{payload.get('embedding_model')!r}, not {embedding_model!r}"
            )
            raise HnswCacheError(msg)
        shard_ids = [int(s) for s in payload["shard_ids"]]
        vectors = np.load(directory / _VECTORS_NAME).astype(np.float32)
        index = Index.restore(str(directory / _GRAPH_NAME))
        if index is None:  # pragma: no cover - corrupt file
            msg = f"could not restore HNSW graph from {directory / _GRAPH_NAME}"
            raise HnswCacheError(msg)
        return cls(
            shard_ids,
            vectors,
            connectivity=int(payload["connectivity"]),
            expansion_add=int(payload["expansion_add"]),
            expansion_search=int(payload["expansion_search"]),
            _index=index,
        )


def _shard_ids_digest(shard_ids: Sequence[int]) -> str:
    joined = ",".join(str(s) for s in shard_ids).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]
