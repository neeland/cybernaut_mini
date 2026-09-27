"""Flush: versioned pattern propagation, stale-only rescans, importance scheduling.

Writes into a collection accumulate in an in-memory buffer; every so often a
collection is flushed. At flush time the collection rebuilds its Aho-Corasick
automaton from its current pattern version and rescans exactly two groups of
chunks: the buffered new ones and the ones whose ``last_tagged_version`` predates
the current patterns. Which collection flushes next is decided by an importance
score over pending writes and recent query traffic.

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "Every new document we index is written to the collection it belongs to and
    every so often those writes are flushed to disk. Larger or most important
    collections are flushed more frequently", and "when the collection is flushed it
    will check for new patterns. If new patterns are found it will tag all untagged
    chunks and update the patterns associated with that collection. This happens in
    seconds." Local copy:
    ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions:
    - Importance is ``w1 * pending_writes + w2 * recent_queries`` [inferred]: the
      post says "larger or most important" without a formula; pending writes measure
      size pressure, recent queries measure importance, and a weighted sum is the
      simplest scheduler that honours both. Defaults ``w1=1.0, w2=0.5`` make a
      write outweigh a query, since only writes create untagged chunks.
    - "Tag all untagged chunks" is implemented as *stale-version* chunks: a chunk
      stamped with pattern version ``v`` is untagged **with respect to** any pattern
      added after ``v``. Rescanning only those chunks — never the whole collection —
      is what keeps a flush in the post's "seconds".
    - A flush re-derives a chunk's full tag set (delete-then-insert) rather than
      appending: patterns are only ever added in this loop, so a rescan against the
      full automaton is a superset of previous tags, and replacing avoids reasoning
      about tag provenance per version.
    - The buffer holds ``(chunk_id, text)`` pairs because the store deliberately
      does not keep chunk text; texts for *stale* chunks come from a caller-supplied
      ``chunk_text`` function reading the repo's read-only shard artifacts.
    - Flushing resets the collection's recent-query counter [inferred]: the counter
      models "traffic since last flush", which is the quantity that should compete
      for the next flush slot.

Alternatives rejected:
    - A wall-clock flush interval per collection: the post explicitly makes flush
      frequency a function of size/importance, not time; a timer would flush idle
      collections and starve hot ones.
    - Persisting the pending buffer in SQLite: durable, but the post's buffer is the
      in-memory write path in front of the on-disk collection; simulating that
      faithfully means the buffer really is memory and a crash loses only unflushed
      writes.
    - Incremental automaton updates instead of a rebuild: ahocorasick-rs automata
      are immutable once built, and construction is linear in total pattern length —
      a rebuild per flush at ~100 patterns per entity is microseconds, not a cost
      worth engineering around.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from cybernaut_mini.entities.store import EntityStore
from cybernaut_mini.entities.tagger import CollectionTagger

__all__ = [
    "DEFAULT_W_PENDING",
    "DEFAULT_W_QUERIES",
    "FlushBuffer",
    "FlushReport",
    "FlushScheduler",
    "PendingChunk",
    "flush_collection",
    "run_flush_cycle",
    "texts_from",
]

#: A pending write outweighs a recent query: writes create untagged chunks.
DEFAULT_W_PENDING = 1.0
DEFAULT_W_QUERIES = 0.5


@dataclass(frozen=True)
class PendingChunk:
    """One buffered write: a chunk that has not reached its collection yet."""

    chunk_id: str
    text: str


@dataclass
class _CollectionState:
    pending: list[PendingChunk] = field(default_factory=list)
    recent_queries: int = 0


class FlushBuffer:
    """The in-memory write buffer and query counter, per collection."""

    def __init__(self) -> None:
        self._state: dict[str, _CollectionState] = {}

    def _get(self, collection_id: str) -> _CollectionState:
        return self._state.setdefault(collection_id, _CollectionState())

    def add(self, collection_id: str, chunk_id: str, text: str) -> None:
        self._get(collection_id).pending.append(PendingChunk(chunk_id=chunk_id, text=text))

    def record_query(self, collection_id: str) -> None:
        self._get(collection_id).recent_queries += 1

    def collections(self) -> list[str]:
        return sorted(self._state)

    def pending_count(self, collection_id: str) -> int:
        return len(self._get(collection_id).pending)

    def recent_queries(self, collection_id: str) -> int:
        return self._get(collection_id).recent_queries

    def drain(self, collection_id: str) -> tuple[PendingChunk, ...]:
        """Remove and return the pending writes; also resets the query counter."""
        state = self._get(collection_id)
        drained = tuple(state.pending)
        state.pending.clear()
        state.recent_queries = 0
        return drained


class FlushScheduler:
    """Importance-weighted flush ordering: ``w1 * pending + w2 * recent_queries``."""

    def __init__(
        self,
        w_pending: float = DEFAULT_W_PENDING,
        w_queries: float = DEFAULT_W_QUERIES,
    ) -> None:
        self.w_pending = w_pending
        self.w_queries = w_queries

    def priority(self, buffer: FlushBuffer, collection_id: str) -> float:
        return self.w_pending * buffer.pending_count(
            collection_id
        ) + self.w_queries * buffer.recent_queries(collection_id)

    def order(self, buffer: FlushBuffer) -> list[str]:
        """Collections by descending priority; ties break lexicographically."""
        return sorted(
            buffer.collections(),
            key=lambda cid: (-self.priority(buffer, cid), cid),
        )

    def next_collection(self, buffer: FlushBuffer) -> str | None:
        """The collection most worth flushing, or None when nothing is pending."""
        for collection_id in self.order(buffer):
            if self.priority(buffer, collection_id) > 0.0:
                return collection_id
        return None


@dataclass(frozen=True)
class FlushReport:
    """What one flush did — including the wall-clock seconds the post brags about."""

    collection_id: str
    version: int
    new_chunks: int
    stale_chunks: int
    chunks_scanned: int
    tags_written: int
    seconds: float


def flush_collection(
    store: EntityStore,
    buffer: FlushBuffer,
    collection_id: str,
    chunk_text: Callable[[str], str],
) -> FlushReport:
    """Flush one collection: rebuild the automaton, rescan new + stale chunks only.

    *chunk_text* maps a chunk id to its text and is consulted only for stale chunks
    (buffered chunks carry their own text). Chunks are stamped with the pattern
    version they were scanned against, so the next flush skips them unless new
    patterns have arrived since.
    """
    started = time.perf_counter()
    version = store.pattern_version(collection_id)
    tagger = CollectionTagger(store.active_patterns(collection_id))

    fresh = buffer.drain(collection_id)
    texts: dict[str, str] = {}
    for chunk in fresh:
        store.register_chunk(collection_id, chunk.chunk_id)
        texts[chunk.chunk_id] = chunk.text
    stale = store.stale_chunks(collection_id)
    fresh_ids = {chunk.chunk_id for chunk in fresh}

    tags_written = 0
    scanned: list[str] = []
    for chunk_id in stale:
        text = texts.get(chunk_id)
        if text is None:
            text = chunk_text(chunk_id)
        matches = tagger.tag(text)
        tags_written += store.write_tags(
            collection_id,
            chunk_id,
            sorted({(match.entity_id, match.pattern) for match in matches}),
        )
        scanned.append(chunk_id)
    store.mark_chunks_tagged(collection_id, scanned, version)
    return FlushReport(
        collection_id=collection_id,
        version=version,
        new_chunks=len(fresh),
        stale_chunks=len([chunk_id for chunk_id in stale if chunk_id not in fresh_ids]),
        chunks_scanned=len(scanned),
        tags_written=tags_written,
        seconds=time.perf_counter() - started,
    )


def run_flush_cycle(
    store: EntityStore,
    buffer: FlushBuffer,
    scheduler: FlushScheduler,
    chunk_text: Callable[[str], str],
    budget: int = 1,
) -> list[FlushReport]:
    """Flush up to *budget* collections, most important first."""
    reports: list[FlushReport] = []
    for _ in range(budget):
        collection_id = scheduler.next_collection(buffer)
        if collection_id is None:
            break
        reports.append(flush_collection(store, buffer, collection_id, chunk_text))
    return reports


def texts_from(mapping: dict[str, str]) -> Callable[[str], str]:
    """A ``chunk_text`` provider over an in-memory mapping (tests, small corpora)."""

    def lookup(chunk_id: str) -> str:
        return mapping.get(chunk_id, "")

    return lookup
