"""SQLite state for the self-organising entity loop: entities, patterns, suggestions, tags.

One small database is the loop's whole memory. The slow discovery tagger writes
*suggestions* into it, the Resolver writes accepted *entities*, distillation writes
versioned *patterns*, and flush-time tagging writes *tags* and per-chunk version
stamps. Every component of :mod:`cybernaut_mini.entities` talks to this store and to
nothing else, so the loop can be stopped, inspected with ``sqlite3``, and resumed.

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "Those suggestions are accumulated in a small database and, once a certain
    threshold is met, the suggestion is sent to the Resolver Agent", and "when the
    collection is flushed it will check for new patterns … and update the patterns
    associated with that collection". Local copy:
    ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions:
    - The suggestions table is exactly the post's accumulator: one row per
      ``(collection_id, surface_norm)`` with a count bumped by SQLite UPSERT and a
      status column tracking the surface through the loop
      (``pending → ready → resolved | rejected``). The post never names the columns;
      these are the minimum that make "accumulate, threshold, send to Resolver"
      expressible [inferred].
    - Patterns are versioned per collection, not globally: the post flushes
      *collections* independently, so "are there new patterns since my last flush?"
      is a per-collection question. A pattern row carries the version at which it was
      added; the active set at version ``v`` is every row with ``version <= v``.
      Adding a batch that contains nothing new does NOT bump the version — otherwise
      every no-op resolution would mark every chunk in the collection stale.
    - Chunks carry ``last_tagged_version`` so a flush can rescan *only* stale chunks.
      The store deliberately does not hold chunk text: the corpus lives in the
      repo's shard artifacts and is read-only here; holding a second copy would let
      the two drift.
    - Entity records are stored as canonical JSON (:func:`~cybernaut_mini.models.
      canonical_dumps`) keyed by an entity id that is the Wikidata QID whenever the
      Resolver found one. QID-keying is what makes cross-collection resolution
      "extremely cache friendly": a second collection suggesting the same company is
      answered by one SELECT.
    - One connection, autocommit-per-method, no threads. The loop is sequential
      (tag → suggest → resolve → distil → flush); concurrency would buy nothing at
      laptop scale and would force WAL/locking decisions the post says nothing about.

Alternatives rejected:
    - A JSONL ledger in the style of :mod:`cybernaut_mini.storage`: append-friendly,
      but the suggestion counter is an UPSERT-shaped workload and replaying a ledger
      to answer "count for this surface" on every discovery batch is quadratic in
      exactly the hot path.
    - One patterns table without versions, plus a "dirty" flag per collection: loses
      the ability to say *which* chunks are stale, so every flush would rescan the
      whole collection — the post's "this happens in seconds" depends on not doing
      that.
    - SQLAlchemy or an ORM: three tables and a dozen statements do not justify a
      dependency the repo does not already carry.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from cybernaut_mini.models import canonical_dumps

__all__ = [
    "SUGGESTION_PENDING",
    "SUGGESTION_READY",
    "SUGGESTION_REJECTED",
    "SUGGESTION_RESOLVED",
    "EntityStore",
    "StoredEntity",
    "Suggestion",
]

#: Suggestion lifecycle. A surface starts pending, becomes ready when its count
#: crosses the promotion threshold, and ends resolved or rejected by the Resolver.
SUGGESTION_PENDING = "pending"
SUGGESTION_READY = "ready"
SUGGESTION_RESOLVED = "resolved"
SUGGESTION_REJECTED = "rejected"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id   TEXT PRIMARY KEY,
    record_json TEXT NOT NULL,
    resolved_by TEXT NOT NULL,
    sources_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS collections (
    collection_id   TEXT PRIMARY KEY,
    pattern_version INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS patterns (
    collection_id TEXT NOT NULL,
    version       INTEGER NOT NULL,
    pattern       TEXT NOT NULL,
    entity_id     TEXT NOT NULL,
    PRIMARY KEY (collection_id, pattern, entity_id)
);
CREATE TABLE IF NOT EXISTS suggestions (
    collection_id TEXT NOT NULL,
    surface_norm  TEXT NOT NULL,
    count         INTEGER NOT NULL DEFAULT 1,
    status        TEXT NOT NULL DEFAULT 'pending',
    PRIMARY KEY (collection_id, surface_norm)
);
CREATE TABLE IF NOT EXISTS chunks (
    collection_id       TEXT NOT NULL,
    chunk_id            TEXT NOT NULL,
    last_tagged_version INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (collection_id, chunk_id)
);
CREATE TABLE IF NOT EXISTS tags (
    collection_id TEXT NOT NULL,
    chunk_id      TEXT NOT NULL,
    entity_id     TEXT NOT NULL,
    pattern       TEXT NOT NULL,
    PRIMARY KEY (collection_id, chunk_id, entity_id, pattern)
);
"""


@dataclass(frozen=True)
class Suggestion:
    """One accumulated discovery surface for one collection."""

    collection_id: str
    surface_norm: str
    count: int
    status: str


@dataclass(frozen=True)
class StoredEntity:
    """A resolved entity as the store returns it."""

    entity_id: str
    record: dict[str, Any]
    resolved_by: str
    sources: tuple[str, ...]


class EntityStore:
    """All persistent state of the entity loop behind one SQLite file.

    ``path`` may be ``":memory:"`` (the test default) or a file path, whose parent
    directories are created on demand.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EntityStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- suggestions --------------------------------------------------------

    def upsert_suggestion(self, collection_id: str, surface_norm: str) -> int:
        """Insert the surface at count 1, or bump its count; return the new count.

        This is the post's "suggestions are accumulated in a small database" in one
        statement. Status is never touched by the bump, so a surface that was already
        promoted (or rejected) keeps its place in the lifecycle while its evidence
        keeps growing.
        """
        row = self._conn.execute(
            """
            INSERT INTO suggestions (collection_id, surface_norm) VALUES (?, ?)
            ON CONFLICT (collection_id, surface_norm) DO UPDATE SET count = count + 1
            RETURNING count
            """,
            (collection_id, surface_norm),
        ).fetchone()
        self._conn.commit()
        return int(row[0])

    def suggestions(
        self, collection_id: str | None = None, status: str | None = None
    ) -> list[Suggestion]:
        """Suggestions, optionally filtered, ordered by descending count then surface."""
        clauses: list[str] = []
        params: list[str] = []
        if collection_id is not None:
            clauses.append("collection_id = ?")
            params.append(collection_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT collection_id, surface_norm, count, status FROM suggestions {where} "
            "ORDER BY count DESC, surface_norm ASC",
            params,
        ).fetchall()
        return [Suggestion(str(r[0]), str(r[1]), int(r[2]), str(r[3])) for r in rows]

    def promote_ready(self, collection_id: str, threshold: int) -> list[str]:
        """Move pending suggestions at ``count >= threshold`` to ready; return them.

        The returned surfaces are the batch to hand to the Resolver. Sorted so the
        best-evidenced surface is resolved first.
        """
        rows = self._conn.execute(
            """
            UPDATE suggestions SET status = ?
            WHERE collection_id = ? AND status = ? AND count >= ?
            RETURNING surface_norm, count
            """,
            (SUGGESTION_READY, collection_id, SUGGESTION_PENDING, threshold),
        ).fetchall()
        self._conn.commit()
        rows.sort(key=lambda r: (-int(r[1]), str(r[0])))
        return [str(r[0]) for r in rows]

    def set_suggestion_status(self, collection_id: str, surface_norm: str, status: str) -> None:
        self._conn.execute(
            "UPDATE suggestions SET status = ? WHERE collection_id = ? AND surface_norm = ?",
            (status, collection_id, surface_norm),
        )
        self._conn.commit()

    # -- entities -----------------------------------------------------------

    def put_entity(
        self,
        entity_id: str,
        record: Mapping[str, Any],
        resolved_by: str,
        sources: Sequence[str],
    ) -> None:
        """Store (or replace) a resolved entity record as canonical JSON."""
        self._conn.execute(
            "INSERT OR REPLACE INTO entities (entity_id, record_json, resolved_by, sources_json) "
            "VALUES (?, ?, ?, ?)",
            (
                entity_id,
                canonical_dumps(dict(record)),
                resolved_by,
                canonical_dumps(sorted(sources)),
            ),
        )
        self._conn.commit()

    def get_entity(self, entity_id: str) -> StoredEntity | None:
        row = self._conn.execute(
            "SELECT entity_id, record_json, resolved_by, sources_json FROM entities "
            "WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
        if row is None:
            return None
        return StoredEntity(
            entity_id=str(row[0]),
            record=dict(json.loads(str(row[1]))),
            resolved_by=str(row[2]),
            sources=tuple(json.loads(str(row[3]))),
        )

    def entity_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT entity_id FROM entities ORDER BY entity_id").fetchall()
        return [str(r[0]) for r in rows]

    # -- patterns -----------------------------------------------------------

    def pattern_version(self, collection_id: str) -> int:
        row = self._conn.execute(
            "SELECT pattern_version FROM collections WHERE collection_id = ?",
            (collection_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def add_patterns(
        self, collection_id: str, entity_id: str, patterns: Iterable[str]
    ) -> int:
        """Add an entity's patterns to a collection; return the collection's version.

        Only genuinely new ``(pattern, entity_id)`` rows bump the version. A
        resolution whose patterns were all known leaves the version — and therefore
        every chunk's staleness — untouched.
        """
        existing = {
            (str(r[0]), str(r[1]))
            for r in self._conn.execute(
                "SELECT pattern, entity_id FROM patterns WHERE collection_id = ?",
                (collection_id,),
            ).fetchall()
        }
        fresh = sorted(
            {p for p in patterns if p and (p, entity_id) not in existing}
        )
        version = self.pattern_version(collection_id)
        if not fresh:
            return version
        version += 1
        self._conn.execute(
            "INSERT INTO collections (collection_id, pattern_version) VALUES (?, ?) "
            "ON CONFLICT (collection_id) DO UPDATE SET pattern_version = ?",
            (collection_id, version, version),
        )
        self._conn.executemany(
            "INSERT INTO patterns (collection_id, version, pattern, entity_id) "
            "VALUES (?, ?, ?, ?)",
            [(collection_id, version, pattern, entity_id) for pattern in fresh],
        )
        self._conn.commit()
        return version

    def active_patterns(self, collection_id: str) -> list[tuple[str, str]]:
        """Every ``(pattern, entity_id)`` active for the collection, sorted."""
        rows = self._conn.execute(
            "SELECT pattern, entity_id FROM patterns WHERE collection_id = ? "
            "ORDER BY pattern, entity_id",
            (collection_id,),
        ).fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    def collections(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT collection_id FROM collections ORDER BY collection_id"
        ).fetchall()
        return [str(r[0]) for r in rows]

    # -- chunks and tags ----------------------------------------------------

    def register_chunk(self, collection_id: str, chunk_id: str) -> None:
        """Record that a chunk exists in a collection, untagged (version 0)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO chunks (collection_id, chunk_id) VALUES (?, ?)",
            (collection_id, chunk_id),
        )
        self._conn.commit()

    def stale_chunks(self, collection_id: str) -> list[str]:
        """Chunks whose last tagging predates the collection's current patterns."""
        version = self.pattern_version(collection_id)
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE collection_id = ? AND last_tagged_version < ? "
            "ORDER BY chunk_id",
            (collection_id, version),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def mark_chunks_tagged(
        self, collection_id: str, chunk_ids: Iterable[str], version: int
    ) -> None:
        self._conn.executemany(
            "UPDATE chunks SET last_tagged_version = ? WHERE collection_id = ? AND chunk_id = ?",
            [(version, collection_id, chunk_id) for chunk_id in chunk_ids],
        )
        self._conn.commit()

    def write_tags(
        self, collection_id: str, chunk_id: str, tags: Iterable[tuple[str, str]]
    ) -> int:
        """Replace a chunk's tags with ``(entity_id, pattern)`` rows; return how many."""
        self._conn.execute(
            "DELETE FROM tags WHERE collection_id = ? AND chunk_id = ?",
            (collection_id, chunk_id),
        )
        rows = [(collection_id, chunk_id, entity_id, pattern) for entity_id, pattern in tags]
        self._conn.executemany(
            "INSERT OR IGNORE INTO tags (collection_id, chunk_id, entity_id, pattern) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def tags_for(
        self, collection_id: str, chunk_id: str | None = None
    ) -> list[tuple[str, str, str]]:
        """``(chunk_id, entity_id, pattern)`` rows for a collection (or one chunk)."""
        if chunk_id is None:
            rows = self._conn.execute(
                "SELECT chunk_id, entity_id, pattern FROM tags WHERE collection_id = ? "
                "ORDER BY chunk_id, entity_id, pattern",
                (collection_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT chunk_id, entity_id, pattern FROM tags "
                "WHERE collection_id = ? AND chunk_id = ? ORDER BY entity_id, pattern",
                (collection_id, chunk_id),
            ).fetchall()
        return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]

    def occupancy_cells(self) -> list[tuple[str, str]]:
        """Distinct ``(entity_id, collection_id)`` cells that hold at least one tag."""
        rows = self._conn.execute(
            "SELECT DISTINCT entity_id, collection_id FROM tags ORDER BY entity_id, collection_id"
        ).fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]
