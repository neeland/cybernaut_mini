"""The WORLD event store: one record per real-world event, point-in-time safe.

Where the SEARCH pillar stores documents, the WORLD pillar stores *events*: a
de-duplicated cluster of syndicated coverage collapsed into one record carrying a
date, a title, an embedding, publisher breadth, and tag layers (entities, tickers,
countries, IPTC-ish topics). Every derived index — GPR, TPU/EPU, risk-on/risk-off —
is a group-by over this table, so the schema here mirrors the record the posts
print, and the build is pure: same clusters and documents in, byte-identical
canonical JSON out.

Blog ref: https://nosible.com/blog/point-in-time-knowledge-graphs-over-named-entities
    — the abridged Microsoft/Activision record whose field names this schema keeps
    verbatim (``event{date,title,country}``, ``signals{sentiment,
    materiality_score}``, ``coverage{total_coverage,total_netlocs}``,
    ``entities{TYPE:{name:count}}``, ``tickers[{name,ticker_eodhd}]``);
    https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world
    — the ``iptc_level_1..3`` and ``ent_gpe`` fields and "one real event is one
    record, no matter how many outlets repeat it";
    https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty
    — "Ours dates each event by the day its coverage peaked." Local copies under
    ``docs/blog-archive/``.

Assumptions:
    - The input contract is WS2's :class:`cybernaut_mini.dedup.EventCluster`: the
      coverage-peak date, apex election, and ``total_netlocs`` breadth are computed
      there and are trusted here, not recomputed.
    - The event embedding is the L2-renormalized centroid of its members' vectors.
      The posts store the apex article's embedding; a centroid is the natural
      stand-in when members were embedded independently, and for a tight
      near-duplicate cluster (cosine >= 0.9) the two are nearly identical.
    - ``materiality_score`` is the laptop proxy ``log1p(total_netlocs) /
      log1p(running_max_netlocs)``, where the running max is *expanding in date
      order* — never the corpus-wide max — because a corpus-wide max computed over
      2019 data would leak the future into a 2017 event's field, breaking the
      point-in-time rule every field of this table must obey. Events built from a
      corpus prefix therefore keep byte-identical records when later data arrives.
    - Tag layers (entities, tickers, country, topics) start empty and are filled by
      the sibling modules' ``tag_events`` passes; ``sentiment`` stays ``None`` until
      the sentiment workstream supplies per-event labels. An untagged event is a
      valid event.
    - Parquet is the analytical store (one row per event, embedding as a list
      column, ``entities`` as a canonical-JSON string column because its keys vary
      per row) and canonical JSONL is the exchange/versioning format; the two
      round-trip through the same pydantic model.

Alternatives rejected:
    - Storing one flat pydantic model instead of the nested ``event``/``signals``/
      ``coverage`` blocks: flatter code, but the posts print the nested record and
      the KG post's appendix indexes into it (``event["entities"][type]``), so
      keeping the shape lets their code run against ours.
    - Materiality normalized by the corpus-wide max (the literal reading of the
      gap-matrix formula): rejected for the foreknowledge leak above; the expanding
      max is the same formula made causal.
    - numpy ``.npy`` sidecar for embeddings (the index layout's idiom): the event
      table is small (one row per *event*, not per document) and keeping the vector
      in-row makes the parquet self-contained for pandas/polars consumers.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from cybernaut_mini.dedup import EventCluster
from cybernaut_mini.models import Document, canonical_dumps
from cybernaut_mini.world.vectors import matryoshka_truncate

__all__ = [
    "EventCoverage",
    "EventHeader",
    "EventSignals",
    "TickerRef",
    "WorldEvent",
    "build_events",
    "embedding_matrix",
    "event_breadths",
    "event_dates",
    "read_events_jsonl",
    "read_events_parquet",
    "write_events_jsonl",
    "write_events_parquet",
]

FloatArray = npt.NDArray[np.float32]


class EventHeader(BaseModel):
    """The ``event{date,title,country}`` block of the published record."""

    model_config = ConfigDict(extra="forbid")

    date: dt.date | None = None
    title: str = Field(min_length=1)
    country: str | None = None


class EventSignals(BaseModel):
    """The ``signals{sentiment,materiality_score}`` block."""

    model_config = ConfigDict(extra="forbid")

    sentiment: str | None = None
    materiality_score: float = Field(default=0.0, ge=0.0, le=1.0)


class EventCoverage(BaseModel):
    """The ``coverage{total_coverage,total_netlocs}`` block."""

    model_config = ConfigDict(extra="forbid")

    total_coverage: int = Field(ge=1)
    total_netlocs: int = Field(ge=0)


class TickerRef(BaseModel):
    """One resolved listing: ``{name, ticker_eodhd}`` exactly as the posts print it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    ticker_eodhd: str = Field(min_length=1)


class WorldEvent(BaseModel):
    """One de-duplicated real-world event with its tag layers."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    event: EventHeader
    signals: EventSignals = Field(default_factory=EventSignals)
    coverage: EventCoverage
    entities: dict[str, dict[str, int]] = Field(default_factory=dict)
    tickers: list[TickerRef] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)
    iptc_level_1: str | None = None
    iptc_level_2: str | None = None
    iptc_level_3: str | None = None
    ent_gpe: list[str] = Field(default_factory=list)
    apex_doc_id: str = Field(min_length=1)
    member_doc_ids: list[str] = Field(min_length=1)

    @property
    def breadth(self) -> int:
        """``breadth(e)`` — the ``total_netlocs`` weight every index uses."""
        return self.coverage.total_netlocs


def _sort_key(cluster: EventCluster) -> tuple[bool, str, str]:
    day = cluster.date.isoformat() if cluster.date else ""
    return (cluster.date is None, day, cluster.event_id)


def build_events(
    clusters: Sequence[EventCluster],
    documents: Sequence[Document],
    embeddings: FloatArray,
    *,
    row_map: Mapping[str, int] | None = None,
) -> list[WorldEvent]:
    """Fold event clusters over a dated corpus into WORLD event records.

    ``embeddings`` rows align with ``documents`` unless ``row_map`` maps document
    id -> row. Output order is (dated first, date, event_id) — the order the
    expanding materiality max requires — and the build is deterministic.
    """
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or (row_map is None and matrix.shape[0] != len(documents)):
        msg = f"embeddings must cover all {len(documents)} documents, got {matrix.shape}"
        raise ValueError(msg)
    by_id = {doc.id: doc for doc in documents}
    rows = row_map if row_map is not None else {doc.id: i for i, doc in enumerate(documents)}

    events: list[WorldEvent] = []
    running_max = 0
    for cluster in sorted(clusters, key=_sort_key):
        members = [by_id[doc_id] for doc_id in cluster.member_doc_ids]
        member_rows = np.asarray([rows[doc.id] for doc in members], dtype=np.int64)
        centroid = matrix[member_rows].mean(axis=0, keepdims=True)
        centroid = matryoshka_truncate(centroid, None)[0]
        apex = by_id[cluster.apex_doc_id]
        running_max = max(running_max, cluster.total_netlocs)
        materiality = (
            math.log1p(cluster.total_netlocs) / math.log1p(running_max)
            if running_max > 0
            else 0.0
        )
        events.append(
            WorldEvent(
                event_id=cluster.event_id,
                event=EventHeader(date=cluster.date, title=apex.title),
                signals=EventSignals(materiality_score=materiality),
                coverage=EventCoverage(
                    total_coverage=len(cluster.member_doc_ids),
                    total_netlocs=cluster.total_netlocs,
                ),
                embedding=[float(v) for v in centroid],
                apex_doc_id=cluster.apex_doc_id,
                member_doc_ids=list(cluster.member_doc_ids),
            )
        )
    return events


def embedding_matrix(events: Sequence[WorldEvent]) -> FloatArray:
    """Row-aligned float32 matrix of event embeddings (renormalized defensively)."""
    if not events:
        return np.zeros((0, 0), dtype=np.float32)
    return matryoshka_truncate(
        np.asarray([event.embedding for event in events], dtype=np.float32), None
    )


def event_dates(events: Sequence[WorldEvent]) -> list[dt.date | None]:
    """Row-aligned event dates (coverage-peak days; ``None`` for undated clusters)."""
    return [event.event.date for event in events]


def event_breadths(events: Sequence[WorldEvent]) -> npt.NDArray[np.float64]:
    """Row-aligned ``breadth(e)`` weights."""
    return np.asarray([event.breadth for event in events], dtype=np.float64)


# ---------------------------------------------------------------------- #
# Persistence                                                            #
# ---------------------------------------------------------------------- #


def write_events_jsonl(path: Path, events: Sequence[WorldEvent]) -> None:
    """One canonical-JSON line per event — byte-identical across runs."""
    lines = [canonical_dumps(event.model_dump(mode="json")) for event in events]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_events_jsonl(path: Path) -> list[WorldEvent]:
    events: list[WorldEvent] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(WorldEvent.model_validate(json.loads(line)))
    return events


def write_events_parquet(path: Path, events: Sequence[WorldEvent]) -> None:
    """Parquet analytical store: one row per event, nested blocks flattened.

    ``entities`` is stored as a canonical-JSON string column (its keys vary per
    row, which arrow structs cannot express); everything else is a native column.
    """
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    records: list[dict[str, Any]] = []
    for event in events:
        dump = event.model_dump(mode="json")
        records.append(
            {
                "event_id": dump["event_id"],
                "date": dump["event"]["date"],
                "title": dump["event"]["title"],
                "country": dump["event"]["country"],
                "sentiment": dump["signals"]["sentiment"],
                "materiality_score": dump["signals"]["materiality_score"],
                "total_coverage": dump["coverage"]["total_coverage"],
                "total_netlocs": dump["coverage"]["total_netlocs"],
                "entities": canonical_dumps(dump["entities"]),
                "tickers": dump["tickers"],
                "embedding": dump["embedding"],
                "iptc_level_1": dump["iptc_level_1"],
                "iptc_level_2": dump["iptc_level_2"],
                "iptc_level_3": dump["iptc_level_3"],
                "ent_gpe": dump["ent_gpe"],
                "apex_doc_id": dump["apex_doc_id"],
                "member_doc_ids": dump["member_doc_ids"],
            }
        )
    schema = pa.schema(
        [
            ("event_id", pa.string()),
            ("date", pa.string()),
            ("title", pa.string()),
            ("country", pa.string()),
            ("sentiment", pa.string()),
            ("materiality_score", pa.float64()),
            ("total_coverage", pa.int64()),
            ("total_netlocs", pa.int64()),
            ("entities", pa.string()),
            (
                "tickers",
                pa.list_(pa.struct([("name", pa.string()), ("ticker_eodhd", pa.string())])),
            ),
            ("embedding", pa.list_(pa.float32())),
            ("iptc_level_1", pa.string()),
            ("iptc_level_2", pa.string()),
            ("iptc_level_3", pa.string()),
            ("ent_gpe", pa.list_(pa.string())),
            ("apex_doc_id", pa.string()),
            ("member_doc_ids", pa.list_(pa.string())),
        ]
    )
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, path)


def read_events_parquet(path: Path) -> list[WorldEvent]:
    import pyarrow.parquet as pq

    events: list[WorldEvent] = []
    for row in pq.read_table(path).to_pylist():
        events.append(
            WorldEvent(
                event_id=row["event_id"],
                event=EventHeader(
                    date=dt.date.fromisoformat(row["date"]) if row["date"] else None,
                    title=row["title"],
                    country=row["country"],
                ),
                signals=EventSignals(
                    sentiment=row["sentiment"],
                    materiality_score=row["materiality_score"],
                ),
                coverage=EventCoverage(
                    total_coverage=row["total_coverage"],
                    total_netlocs=row["total_netlocs"],
                ),
                entities=json.loads(row["entities"]),
                tickers=[TickerRef.model_validate(t) for t in row["tickers"] or []],
                embedding=[float(v) for v in row["embedding"] or []],
                iptc_level_1=row["iptc_level_1"],
                iptc_level_2=row["iptc_level_2"],
                iptc_level_3=row["iptc_level_3"],
                ent_gpe=list(row["ent_gpe"] or []),
                apex_doc_id=row["apex_doc_id"],
                member_doc_ids=list(row["member_doc_ids"] or []),
            )
        )
    return events
