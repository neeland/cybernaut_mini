"""Vanilla Sparse and Entity Sparse Similarity: one inverted sparse matrix, two uses.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 5, "Shard Selection":
    "**Vanilla Sparse Similarity -** This selector computes the sparse similarity between
    the keywords in the question and the keywords in each shard. It's essentially just a
    massive sparse TF-IDF matrix." and "**Entity Sparse Similarity -** If the question
    contains one or more entities, we also compute the sparse similarity between those
    entities and the entities in each shard. Again, it is a massive sparse matrix."
    Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Techniques
----------
- A shards x terms sparse TF-IDF matrix [disclosed] — the post says so twice, in those
  words, and names SciPy in the stage's stack.
- The same matrix type for keywords and for entities [disclosed] — the post describes the
  entity selector as the keyword selector applied to a different vocabulary, so this
  module is one class instantiated twice, not two near-copies.
- Conditional entity factor [disclosed] — "*If* the question contains one or more
  entities". No entities means the factor is not computed and never reaches RRF.
- Column-major (CSC) traversal [inferred] — the post does not say how the matrix is
  multiplied. Iterating the columns of the query's terms touches only shards that
  actually contain one of those terms, so a question of eight keywords never looks at
  most of 250,000 shards. A row-major ``M @ q`` would be simpler and would visit every
  stored non-zero.

Assumptions: shard-side keyword weights are taken as stored.
    :func:`cybernaut_mini.indexing.compute_keywords` already writes ``tf * idf`` into
    :attr:`~cybernaut_mini.models.ShardKeyword.weight`, so re-weighting the rows would
    square the IDF. IDF is therefore applied to the *query* side only and the rows are
    L2-normalised — the classic ``ltc.lnc`` arrangement, and the reason
    :meth:`SparseShardMatrix.score` returns a cosine in [0, 1]. Entity rows carry raw
    counts rather than a weight, so for those the matrix does apply IDF itself
    (``weight_rows=True``); the post does not say which side weights, only that both are
    sparse similarities.

Alternatives rejected: ``sklearn.feature_extraction.text.TfidfVectorizer`` over the shard
    summaries. It is one line and it is well tested, but it would re-derive term
    statistics from summary prose when the index already stores per-shard keyword weights
    computed over the shards' full token streams — strictly less information, and a
    second, silently different notion of what a shard's keywords are. Also rejected:
    scoring every shard and returning a full ranking the way ``routing.py`` does. That
    makes the ranked list dense and RRF simpler, but it reintroduces the O(n_shards) scan
    this stage exists to avoid; unscored shards are simply absent from the list, which is
    exactly the input RRF is designed for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

# SciPy ships no type stubs and this package cannot add ``scipy-stubs`` to the mypy
# overrides in pyproject.toml, so the import is silenced here rather than
# repository-wide. Everything read off the matrix below is re-annotated explicitly.
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cybernaut_mini.models import ShardManifest


@dataclass(frozen=True)
class SparseHits:
    """Shard ids best-first with their sparse cosine scores, aligned element-wise."""

    shard_ids: tuple[int, ...]
    scores: tuple[float, ...]

    def as_dict(self) -> dict[int, float]:
        return dict(zip(self.shard_ids, self.scores, strict=True))


def _idf(document_frequency: int, n_shards: int) -> float:
    """Smoothed IDF, identical in form to :func:`cybernaut_mini.indexing.compute_keywords`."""
    return math.log((1 + n_shards) / (1 + document_frequency)) + 1.0


class SparseShardMatrix:
    """A shards x terms sparse matrix with an inverted-index scorer.

    Rows are L2-normalised, so :meth:`score` against an L2-normalised query vector is a
    cosine similarity. Only the columns of the query's own terms are traversed, so the
    cost of a query is proportional to how many shards mention its terms, not to how many
    shards exist.
    """

    def __init__(
        self,
        shard_terms: Mapping[int, Mapping[str, float]],
        *,
        weight_rows: bool = False,
    ) -> None:
        self.shard_ids: tuple[int, ...] = tuple(sorted(shard_terms))
        self.weight_rows = weight_rows
        n_shards = len(self.shard_ids)

        vocabulary: dict[str, int] = {}
        for shard_id in self.shard_ids:
            for term in sorted(shard_terms[shard_id]):
                if term not in vocabulary:
                    vocabulary[term] = len(vocabulary)
        self.vocabulary: dict[str, int] = vocabulary

        document_frequency = np.zeros(len(vocabulary), dtype=np.int64)
        for shard_id in self.shard_ids:
            for term in shard_terms[shard_id]:
                document_frequency[vocabulary[term]] += 1
        self.document_frequency = document_frequency
        self.idf = np.array(
            [_idf(int(df), n_shards) for df in document_frequency], dtype=np.float64
        )

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for row, shard_id in enumerate(self.shard_ids):
            terms = shard_terms[shard_id]
            for term in sorted(terms):
                column = vocabulary[term]
                value = float(terms[term])
                if weight_rows:
                    value *= float(self.idf[column])
                if value == 0.0:
                    continue
                rows.append(row)
                cols.append(column)
                data.append(value)

        values = np.array(data, dtype=np.float64)
        row_index = np.array(rows, dtype=np.int64)
        col_index = np.array(cols, dtype=np.int64)

        # L2-normalise each row so a dot product against a normalised query is a cosine.
        norms = np.zeros(max(n_shards, 1), dtype=np.float64)
        np.add.at(norms, row_index, values * values)
        norms = np.sqrt(norms)
        safe = np.where(norms[row_index] > 0.0, norms[row_index], 1.0)
        values = values / safe
        self.row_norms = norms

        self._matrix = csc_matrix(
            (values, (row_index, col_index)),
            shape=(max(n_shards, 1), max(len(vocabulary), 1)),
        )
        self._indptr: np.ndarray = self._matrix.indptr
        self._indices: np.ndarray = self._matrix.indices
        self._data: np.ndarray = self._matrix.data

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.shard_ids), len(self.vocabulary))

    @property
    def nnz(self) -> int:
        return int(self._matrix.nnz)

    def density(self) -> float:
        """Fraction of the shards x terms grid that is actually stored."""
        rows, columns = self.shape
        if rows == 0 or columns == 0:
            return 0.0
        return self.nnz / (rows * columns)

    def row(self, shard_id: int) -> dict[str, float]:
        """One shard's normalised row as ``term -> weight``, for inspection and tests."""
        position = self.shard_ids.index(shard_id)
        reverse = {column: term for term, column in self.vocabulary.items()}
        out: dict[str, float] = {}
        for column in range(len(self.vocabulary)):
            start, end = int(self._indptr[column]), int(self._indptr[column + 1])
            for offset in range(start, end):
                if int(self._indices[offset]) == position:
                    out[reverse[column]] = float(self._data[offset])
        return out

    # ------------------------------------------------------------------ #
    # Scoring                                                             #
    # ------------------------------------------------------------------ #

    def query_vector(self, terms: Mapping[str, float]) -> dict[int, float]:
        """IDF-weight and L2-normalise a query's terms, dropping out-of-vocabulary ones.

        Out-of-vocabulary terms are dropped *before* normalisation: a term no shard
        contains contributes nothing to any dot product, so keeping it in the norm would
        shrink every score by an amount that depends on how exotic the question is.
        """
        weighted: dict[int, float] = {}
        for term, weight in terms.items():
            column = self.vocabulary.get(term)
            if column is None or weight == 0.0:
                continue
            weighted[column] = weighted.get(column, 0.0) + float(weight) * float(self.idf[column])
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        if norm == 0.0:
            return {}
        return {column: value / norm for column, value in weighted.items()}

    def score(self, terms: Mapping[str, float], *, top_n: int | None = None) -> SparseHits:
        """Cosine between the query's terms and every shard that shares at least one.

        Shards sharing no term are absent from the result, not present with a zero.
        Ties break by ascending shard id, so the ordering is deterministic.
        """
        query = self.query_vector(terms)
        accumulator: dict[int, float] = {}
        for column, weight in sorted(query.items()):
            start, end = int(self._indptr[column]), int(self._indptr[column + 1])
            for offset in range(start, end):
                row = int(self._indices[offset])
                accumulator[row] = accumulator.get(row, 0.0) + weight * float(self._data[offset])
        ordered = sorted(
            ((self.shard_ids[row], score) for row, score in accumulator.items() if score > 0.0),
            key=lambda pair: (-pair[1], pair[0]),
        )
        if top_n is not None:
            ordered = ordered[:top_n]
        return SparseHits(
            shard_ids=tuple(shard_id for shard_id, _ in ordered),
            scores=tuple(float(score) for _, score in ordered),
        )

    # ------------------------------------------------------------------ #
    # Constructors from an index's manifests                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_keywords(cls, manifests: Mapping[int, ShardManifest]) -> SparseShardMatrix:
        """Build the keyword matrix from stored TF-IDF keyword weights (no re-weighting)."""
        shard_terms: dict[int, dict[str, float]] = {}
        for shard_id, manifest in manifests.items():
            shard_terms[shard_id] = {
                keyword.term: keyword.weight
                for keyword in manifest.keywords
                if keyword.weight != 0.0
            }
        return cls(shard_terms, weight_rows=False)

    @classmethod
    def from_entities(cls, manifests: Mapping[int, ShardManifest]) -> SparseShardMatrix:
        """Build the entity matrix from raw entity counts, which this matrix IDF-weights."""
        shard_terms: dict[int, dict[str, float]] = {}
        for shard_id, manifest in manifests.items():
            shard_terms[shard_id] = {
                entity.text: float(entity.count) for entity in manifest.entities if entity.count > 0
            }
        return cls(shard_terms, weight_rows=True)


def term_frequencies(terms: Sequence[str]) -> dict[str, float]:
    """Count occurrences of each term, the query-side ``tf`` the matrix expects."""
    counts: dict[str, float] = {}
    for term in terms:
        counts[term] = counts.get(term, 0.0) + 1.0
    return counts
