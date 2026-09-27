"""Incremental streaming ingest: route to nearest centroid, append, re-evaluate.

The production system indexes a growing web corpus continuously; this module scales
that to a laptop by replaying a real corpus in simulated "days". New documents are
routed to the nearest *existing* shard centroid (no re-clustering), appended to that
shard, and only the touched shards' artifacts are recomputed. After every increment
the standard evaluation harness re-runs over a fixed judgment set, producing the
nDCG-vs-corpus-size curve that shows whether quality is stable under growth. Each
day's documents also stream through the entity loop's sampled discovery and
importance-weighted flush scheduling, so the loop's X% and flush claims are
exercised under realistic arrival, not one-shot batches.

Blog ref: https://nosible.com/blog/introducing-cybernaut-1-agentic-search-with-mcts —
    "consistently match or outperforms leading search engines, even as we continue
    expanding our web coverage (currently growing at ~20 million webpages per day)";
    and https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "Every new document we index is written to the collection it belongs to and
    every so often those writes are flushed to disk." Local copies:
    ``docs/blog-archive/introducing-cybernaut-1-agentic-search-with-mcts.md``,
    ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions:
    - Incremental mode routes to the nearest existing centroid and *freezes* the
      centroids [inferred]: the posts never describe re-clustering on arrival, and a
      drifting centroid would silently re-route future documents away from the shard
      that holds their neighbours. The full rebuild
      (:func:`cybernaut_mini.sharding.shard_documents`) remains the canonical mode;
      this module's bootstrap calls it once and appends thereafter.
    - "Per-shard append" means only touched shards pay: manifests are cached and
      recomputed solely for shards that received documents since the last build,
      which is the laptop analogue of flushing one collection without rewriting the
      rest of the index.
    - The judgment set is fixed across increments (filtered once to queries whose
      relevant documents appear *somewhere* in the planned stream): the curve's
      x-axis must be corpus size alone, so the query set may not change with it.
      Early days legitimately score lower when their relevant documents have not
      arrived yet — that is the realistic cold-corpus behaviour, not noise.
    - The gap analysis scales the post's 20M pages/day to 5-10k passages per
      simulated day; the ``docs_per_day`` default is 5000, and the committed fixture
      corpus (460 documents) is exercised with proportionally smaller days in tests.
      Same code path, smaller constant.
    - Documents are embedded as ``f"{title}\\n{text}"`` and L2-renormalised before
      routing, matching the build pipeline exactly — routing quality claims are
      void if stream-time vectors differ from build-time vectors.

Alternatives rejected:
    - Re-clustering (full ``shard_documents``) every day and diffing: that measures
      rebuild quality, not incremental-append quality — the entire question is what
      degrades when you *don't* re-cluster.
    - Updating centroids as running means of their members: plausible, but it makes
      the routing time-order-dependent and un-replayable, and neither post describes
      it; frozen centroids keep every simulated run reproducible from (corpus, seed).
    - Writing each increment to disk through ``write_index`` and reloading: the
      byte-level layout is already covered by its own tests; building the
      :class:`~cybernaut_mini.indexing.LoadedIndex` in memory keeps a multi-day
      simulation in seconds without weakening what the curve measures.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from cybernaut_mini.entities.flush import (
    FlushBuffer,
    FlushReport,
    FlushScheduler,
    run_flush_cycle,
    texts_from,
)
from cybernaut_mini.entities.store import EntityStore
from cybernaut_mini.entities.suggest import accumulate
from cybernaut_mini.entities.tagger import (
    DEFAULT_SAMPLE_RATE,
    CollectionTagger,
    DiscoveryTagger,
)
from cybernaut_mini.models import IndexMeta, Judgment, ShardManifest, canonical_dumps

if TYPE_CHECKING:
    import numpy.typing as npt

    from cybernaut_mini.config import AppConfig
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.models import Document
    from cybernaut_mini.providers.embeddings import EmbeddingProvider
    from cybernaut_mini.text import TextProcessor

    FloatArray = npt.NDArray[np.float32]

__all__ = [
    "DEFAULT_DOCS_PER_DAY",
    "DayReport",
    "EntityStreamState",
    "StreamingIndex",
    "dump_curve",
    "route_to_centroids",
    "simulate_stream",
    "usable_judgments",
]

logger = logging.getLogger(__name__)

#: The gap analysis' laptop scaling of the post's ~20M pages/day.
DEFAULT_DOCS_PER_DAY = 5000


def _l2_normalize_rows(matrix: FloatArray) -> FloatArray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


def route_to_centroids(vectors: FloatArray, centroids: FloatArray) -> list[int]:
    """Nearest-centroid shard per row: argmax cosine, ties to the lowest shard id.

    Both operands are L2-normalised, so cosine is the dot product — the same
    geometry :mod:`cybernaut_mini.sharding` clusters in. ``np.argmax`` returns the
    first maximum, which is the lowest shard id.
    """
    if vectors.shape[0] == 0:
        return []
    similarities = vectors @ centroids.T
    return [int(label) for label in np.argmax(similarities, axis=1)]


class StreamingIndex:
    """A shard layout that grows by append instead of re-clustering.

    :meth:`bootstrap` clusters an initial corpus once (the canonical build);
    :meth:`append` routes new documents to their nearest frozen centroid; and
    :meth:`to_loaded_index` materialises a query-ready
    :class:`~cybernaut_mini.indexing.LoadedIndex`, recomputing manifests only for
    shards that changed since the previous build.
    """

    def __init__(
        self,
        *,
        documents: Sequence[Document],
        vectors: FloatArray,
        labels: Sequence[int],
        centroids: FloatArray,
        seed: int,
        embedding_model: str,
        processor: TextProcessor,
    ) -> None:
        if len(documents) != vectors.shape[0] or len(documents) != len(labels):
            msg = "documents, vectors and labels must be the same length"
            raise ValueError(msg)
        self.centroids = centroids
        self.seed = seed
        self.embedding_model = embedding_model
        self._processor = processor
        self._documents: list[Document] = []
        self._rows: list[FloatArray] = []
        self._labels: list[int] = []
        self._doc_tokens: dict[str, list[str]] = {}
        self._manifest_cache: dict[int, ShardManifest] = {}
        self._dirty: set[int] = set(range(centroids.shape[0]))
        self._ingest(documents, vectors, list(labels))

    @classmethod
    def bootstrap(
        cls,
        documents: Sequence[Document],
        vectors: FloatArray,
        *,
        n_shards: int,
        seed: int,
        embedding_model: str,
        processor: TextProcessor,
    ) -> StreamingIndex:
        """One canonical clustered build; every later arrival is an append."""
        from cybernaut_mini.sharding import shard_documents

        normalized = _l2_normalize_rows(vectors)
        result = shard_documents(normalized, n_shards=n_shards, seed=seed)
        return cls(
            documents=documents,
            vectors=normalized,
            labels=result.labels,
            centroids=np.asarray(result.centroids, dtype=np.float32),
            seed=seed,
            embedding_model=embedding_model,
            processor=processor,
        )

    def _ingest(
        self, documents: Sequence[Document], vectors: FloatArray, labels: list[int]
    ) -> None:
        for position, document in enumerate(documents):
            self._documents.append(document)
            self._rows.append(vectors[position])
            self._labels.append(labels[position])
            self._doc_tokens[document.id] = self._processor.content_tokens(
                f"{document.title}\n{document.text}"
            )
            self._dirty.add(labels[position])

    def append(self, documents: Sequence[Document], vectors: FloatArray) -> list[int]:
        """Route each document to its nearest frozen centroid and append it there."""
        normalized = _l2_normalize_rows(vectors)
        labels = route_to_centroids(normalized, self.centroids)
        self._ingest(documents, normalized, labels)
        return labels

    @property
    def corpus_size(self) -> int:
        return len(self._documents)

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(self._labels)

    @property
    def n_shards(self) -> int:
        return int(self.centroids.shape[0])

    def shard_document_ids(self, shard_id: int) -> list[str]:
        return [
            document.id
            for document, label in zip(self._documents, self._labels, strict=True)
            if label == shard_id
        ]

    def dirty_shards(self) -> tuple[int, ...]:
        return tuple(sorted(self._dirty))

    def _build_manifest(
        self,
        shard_id: int,
        doc_ids: list[str],
        doc_by_id: dict[str, Document],
        vectors: FloatArray,
        row_map: dict[str, int],
    ) -> ShardManifest:
        from cybernaut_mini.indexing import (
            _shard_summary,
            _shard_title,
            compute_entities,
            compute_keywords,
            compute_shard_term_graphs,
        )

        shard_tokens = [
            token for doc_id in doc_ids for token in self._doc_tokens.get(doc_id, [])
        ]
        keywords = compute_keywords({shard_id: shard_tokens}, max_keywords=30)[shard_id]
        term_graph = compute_shard_term_graphs(
            {shard_id: doc_ids},
            self._doc_tokens,
            window=5,
            min_edge_count=2,
            priority_terms={shard_id: [keyword.term for keyword in keywords]},
        )[shard_id]
        centroid = self.centroids[shard_id]
        return ShardManifest(
            shard_id=shard_id,
            document_ids=doc_ids,
            centroid=[float(value) for value in centroid],
            title=_shard_title(shard_id, keywords),
            summary=_shard_summary(shard_id, doc_ids, doc_by_id, vectors, row_map, centroid),
            keywords=keywords,
            entities=compute_entities([[] for _ in doc_ids], max_entities=30),
            term_graph=term_graph,
            document_count=len(doc_ids),
            embedding_model=self.embedding_model,
        )

    def to_loaded_index(self) -> LoadedIndex:
        """A query-ready in-memory index; only dirty shards recompute manifests."""
        from cybernaut_mini.indexing import LoadedIndex

        vectors = np.vstack(self._rows).astype(np.float32)
        row_map = {document.id: row for row, document in enumerate(self._documents)}
        doc_by_id = {document.id: document for document in self._documents}
        shard_doc_ids: dict[int, list[str]] = {s: [] for s in range(self.n_shards)}
        for document, label in zip(self._documents, self._labels, strict=True):
            shard_doc_ids[label].append(document.id)
        for shard_id in sorted(self._dirty):
            self._manifest_cache[shard_id] = self._build_manifest(
                shard_id, shard_doc_ids[shard_id], doc_by_id, vectors, row_map
            )
        self._dirty.clear()
        meta = IndexMeta(
            embedding_model=self.embedding_model,
            embedding_dim=int(vectors.shape[1]),
            n_shards=self.n_shards,
            n_documents=len(self._documents),
            seed=self.seed,
        )
        return LoadedIndex(
            meta,
            list(self._documents),
            vectors,
            row_map,
            dict(self._manifest_cache),
            self._doc_tokens,
            by_id=doc_by_id,
        )


@dataclass
class EntityStreamState:
    """The entity loop riding the stream: sampled discovery, buffered writes, flush.

    Collections are the shards documents land in (``shard-<id>``), so flush
    importance follows real write pressure. The per-collection production automata
    are rebuilt lazily from the store and dropped after every flush cycle, exactly
    as flush-time rebuild semantics dictate.
    """

    store: EntityStore
    buffer: FlushBuffer
    scheduler: FlushScheduler = field(default_factory=FlushScheduler)
    discovery: DiscoveryTagger | None = None
    sample_rate: float = DEFAULT_SAMPLE_RATE
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    texts: dict[str, str] = field(default_factory=dict)
    sampled_chunks: int = 0
    promoted: list[str] = field(default_factory=list)
    _taggers: dict[str, CollectionTagger] = field(default_factory=dict)

    def _production(self, collection_id: str) -> CollectionTagger:
        tagger = self._taggers.get(collection_id)
        if tagger is None:
            tagger = CollectionTagger(self.store.active_patterns(collection_id))
            self._taggers[collection_id] = tagger
        return tagger

    def observe(self, collection_id: str, chunk_id: str, text: str) -> None:
        """One arriving chunk: buffer the write; sample X% into discovery."""
        self.texts[chunk_id] = text
        self.buffer.add(collection_id, chunk_id, text)
        if self.discovery is None or self.rng.random() >= self.sample_rate:
            return
        self.sampled_chunks += 1
        update = accumulate(
            self.store,
            collection_id,
            self.discovery.discover(text),
            self._production(collection_id),
        )
        self.promoted.extend(update.promoted)

    def flush(self, budget: int) -> list[FlushReport]:
        """Flush up to *budget* collections by importance; drop cached automata."""
        reports = run_flush_cycle(
            self.store, self.buffer, self.scheduler, texts_from(self.texts), budget=budget
        )
        self._taggers.clear()
        return reports


@dataclass(frozen=True)
class DayReport:
    """One point on the growth curve: corpus size, quality, and loop activity."""

    day: int
    corpus_size: int
    added: int
    touched_shards: tuple[int, ...]
    metrics: dict[str, dict[str, object]]  # mode -> ModeMetrics.as_dict()
    sampled_chunks: int
    flushed_collections: int
    tags_written: int
    seconds: float

    def ndcg_at_10(self, mode: str) -> float:
        value = self.metrics[mode]["ndcg_at_10"]
        if not isinstance(value, int | float):  # pragma: no cover — metrics are numeric
            msg = f"ndcg_at_10 for mode {mode!r} is not numeric: {value!r}"
            raise TypeError(msg)
        return float(value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "corpus_size": self.corpus_size,
            "added": self.added,
            "touched_shards": list(self.touched_shards),
            "metrics": self.metrics,
            "sampled_chunks": self.sampled_chunks,
            "flushed_collections": self.flushed_collections,
            "tags_written": self.tags_written,
            "seconds": self.seconds,
        }


def dump_curve(reports: Sequence[DayReport]) -> str:
    """The nDCG-vs-corpus-size curve as canonical JSON (the repo's artifact rule)."""
    return canonical_dumps([report.as_dict() for report in reports])


def usable_judgments(
    judgments: Sequence[Judgment], planned_doc_ids: Sequence[str]
) -> list[Judgment]:
    """Judgments with at least one positively judged document in the planned stream.

    Filtered once against everything that will *ever* arrive, then held fixed, so
    the curve varies only with corpus size.
    """
    planned = set(planned_doc_ids)
    return [
        judgment
        for judgment in judgments
        if any(
            grade > 0 and doc_id in planned
            for doc_id, grade in judgment.relevant_document_ids.items()
        )
    ]


def _embed_documents(
    documents: Sequence[Document], provider: EmbeddingProvider
) -> FloatArray:
    texts = [f"{document.title}\n{document.text}" for document in documents]
    return _l2_normalize_rows(np.asarray(provider.embed_documents(texts), dtype=np.float32))


def simulate_stream(
    documents: Sequence[Document],
    judgments: Sequence[Judgment],
    *,
    provider: EmbeddingProvider,
    processor: TextProcessor,
    config: AppConfig,
    base_size: int,
    docs_per_day: int = DEFAULT_DOCS_PER_DAY,
    n_shards: int = 8,
    seed: int = 42,
    days: int | None = None,
    modes: tuple[str, ...] = ("hybrid",),
    entity_state: EntityStreamState | None = None,
    shard_beam_n: int = 100,
) -> list[DayReport]:
    """Replay *documents* as streaming days and measure quality at every size.

    Day 0 is the bootstrap build over the first *base_size* documents; each later
    day appends the next *docs_per_day* documents by nearest-centroid routing,
    streams them through the entity loop (when *entity_state* is given) with a
    flush cycle at day end, rebuilds only the touched shards' manifests, and
    re-runs :func:`cybernaut_mini.evals.evaluate` on the fixed judgment set.
    """
    from cybernaut_mini.evals import evaluate

    if base_size < n_shards:
        msg = f"base_size ({base_size}) must be >= n_shards ({n_shards})"
        raise ValueError(msg)
    if docs_per_day < 1:
        msg = f"docs_per_day must be >= 1, got {docs_per_day}"
        raise ValueError(msg)

    fixed_judgments = usable_judgments(judgments, [document.id for document in documents])
    base = documents[:base_size]
    stream = StreamingIndex.bootstrap(
        base,
        _embed_documents(base, provider),
        n_shards=n_shards,
        seed=seed,
        embedding_model=provider.identifier,
        processor=processor,
    )

    def _observe(batch: Sequence[Document], labels: Sequence[int]) -> tuple[int, int, int]:
        if entity_state is None:
            return 0, 0, 0
        before = entity_state.sampled_chunks
        for document, label in zip(batch, labels, strict=True):
            entity_state.observe(
                f"shard-{label}", document.id, f"{document.title}\n{document.text}"
            )
        reports = entity_state.flush(budget=stream.n_shards)
        sampled = entity_state.sampled_chunks - before
        return sampled, sum(r.tags_written for r in reports), len(reports)

    def _measure(
        day: int, added: int, touched: tuple[int, ...], sampled: int, tags: int, flushed: int
    ) -> DayReport:
        started = time.perf_counter()
        index = stream.to_loaded_index()
        results = evaluate(
            index,
            list(fixed_judgments),
            config=config,
            processor=processor,
            provider=provider,
            modes=modes,
            shard_beam_n=shard_beam_n,
        )
        report = DayReport(
            day=day,
            corpus_size=stream.corpus_size,
            added=added,
            touched_shards=touched,
            metrics={metric.mode: metric.as_dict() for metric in results},
            sampled_chunks=sampled,
            flushed_collections=flushed,
            tags_written=tags,
            seconds=time.perf_counter() - started,
        )
        logger.info(
            "day %d: corpus=%d added=%d shards_touched=%d",
            day,
            report.corpus_size,
            added,
            len(touched),
        )
        return report

    reports: list[DayReport] = []
    sampled, tags, flushed = _observe(base, stream.labels)
    reports.append(
        _measure(0, len(base), tuple(range(stream.n_shards)), sampled, tags, flushed)
    )

    remaining = list(documents[base_size:])
    day = 0
    while remaining and (days is None or day < days):
        day += 1
        batch = remaining[:docs_per_day]
        remaining = remaining[docs_per_day:]
        labels = stream.append(batch, _embed_documents(batch, provider))
        sampled, tags, flushed = _observe(batch, labels)
        reports.append(
            _measure(day, len(batch), tuple(sorted(set(labels))), sampled, tags, flushed)
        )
    return reports
