"""Point-in-time knowledge graphs: lift-scored entity co-mentions over the event store.

Nodes are the NER entities the event store already carries; edges join a target
company to the entities its events co-mention, scored by lift against the same
year's background. Every edge is bucketed on its event date and scored with no
cumulative or forward-looking state, so a graph built ``as_of`` any day is byte-
identical however much later data exists — the property the audit and the tests
pin.

Blog ref: https://nosible.com/blog/point-in-time-knowledge-graphs-over-named-entities
    — ``lift = (co / target_events) / (entity_events / total_events)``; "Blackwell
    was co-mentioned with NVIDIA in 69 of NVIDIA's 2,651 events in 2024 and
    appeared in only 75 events in the whole corpus that year, which works out to
    roughly 780 times chance"; "a normalized version of it lets a strong link on
    a rare product rank alongside one on a common name"; "The point-in-time
    property is enforced by bucketing every edge on its document date and scoring
    it against the same year's background, with no cumulative or forward-looking
    state"; the provenance audit ("24 of 24 in the last run, across 955 real
    documents"); the appendix's ``co_mentions_by_year`` and ``lift`` functions,
    ported here over :class:`~cybernaut_mini.world.events.WorldEvent`. Local copy
    under ``docs/blog-archive/``.

Assumptions:
    - ``min_co=3`` and ``min_lift=2.0`` are the laptop stand-ins for World-scale
      edge admission (the post filters ubiquitous near-1.0 pairs; exact cutoffs
      are not published), and both are config knobs, not constants.
    - The default edge score is ``log_lift`` (natural log, so chance is 0 and the
      780x pairing is ~6.66); ``npmi`` — ``log(lift) / -log(co / total_events)``
      — is the normalized variant behind a flag, letting a rare-product edge rank
      alongside a common-name one.
    - Undated events cannot be bucketed on a document date, so they are excluded
      from every count. An ``as_of`` bound keeps events dated on or before it;
      the year buckets are then first-class outputs (one canonical-JSON file per
      year in :func:`write_graph`).
    - Entity canonicalization (suffix stripping, surname folding) happens in
      :mod:`cybernaut_mini.world.ner` before events reach this module; names here
      are read as-is.

Alternatives rejected:
    - Cumulative (all-years-to-date) buckets: the post's matrices are per-year
      views and cumulative state is exactly the "forward-looking state" the PIT
      rule forbids inside a bucket.
    - Scoring against a corpus-wide (all-years) background: a 2016 edge would
      then move when 2024 data arrives, breaking bit-identical replay of past
      buckets — each year is scored against the same year's background only.
    - Storing the graph as one monolithic JSON: per-year files make "appending a
      new year leaves old buckets byte-identical" a filesystem-level diff.
"""

from __future__ import annotations

import collections
import datetime as dt
import math
import random
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from cybernaut_mini.models import canonical_dumps
from cybernaut_mini.world.events import WorldEvent

__all__ = [
    "EgoNetwork",
    "KGEdge",
    "audit_edges",
    "build_graph",
    "co_mentions_by_year",
    "ego_network",
    "lift",
    "write_graph",
]


def co_mentions_by_year(
    events: Sequence[WorldEvent], target_ticker: str, entity_type: str
) -> dict[str, collections.Counter[str]]:
    """Count, per year, the events mentioning both a target company and each entity.

    The appendix function ported over :class:`WorldEvent`: bucketed by the event
    date so the view is point-in-time by construction; undated events abstain.
    """
    per_year: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for event in events:
        if event.event.date is None:
            continue
        tickers = {ref.ticker_eodhd for ref in event.tickers}
        if target_ticker not in tickers:
            continue
        year = event.event.date.isoformat()[:4]
        for name in event.entities.get(entity_type, {}):
            per_year[year][name] += 1
    return dict(per_year)


def lift(co: int, target_events: int, entity_events: int, total_events: int) -> float:
    """Association strength of a co-mention versus chance in one period.

    ``(co / target_events) / (entity_events / total_events)`` — 1.0 is chance,
    higher is a stronger association (the NVIDIA/Blackwell quadruple 69 / 2,651 /
    75 / 2,249,039 works out to roughly 780).
    """
    return (co / target_events) / (entity_events / total_events)


class KGEdge(BaseModel):
    """One admitted target-to-entity edge with its full provenance quadruple."""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(min_length=1)
    entity: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    year: str = Field(min_length=4, max_length=4)
    co: int = Field(ge=1)
    target_events: int = Field(ge=1)
    entity_events: int = Field(ge=1)
    total_events: int = Field(ge=1)
    lift: float
    score: float
    event_ids: list[str] = Field(min_length=1)

    @property
    def share(self) -> float:
        """The entity's share of the target's coverage that year (``co / target``)."""
        return self.co / self.target_events


def _dated(events: Sequence[WorldEvent], as_of: dt.date | None) -> list[WorldEvent]:
    kept = [event for event in events if event.event.date is not None]
    if as_of is not None:
        kept = [
            event
            for event in kept
            if event.event.date is not None and event.event.date <= as_of
        ]
    return kept


def _year(event: WorldEvent) -> str:
    assert event.event.date is not None  # callers pass _dated() output
    return event.event.date.isoformat()[:4]


def _edge_score(lift_value: float, co: int, total: int, score: str) -> float:
    if score == "log_lift":
        return math.log(lift_value)
    if score == "npmi":
        return math.log(lift_value) / -math.log(co / total)
    msg = f"unknown score {score!r}; expected 'log_lift' or 'npmi'"
    raise ValueError(msg)


def build_graph(
    events: Sequence[WorldEvent],
    target_ticker: str,
    entity_type: str,
    *,
    as_of: dt.date | None = None,
    score: str = "log_lift",
    min_co: int = 3,
    min_lift: float = 2.0,
) -> dict[str, list[KGEdge]]:
    """Per-year admitted edges for one (target, entity layer) pair.

    Each year's edges use only events dated in that year (and on or before
    ``as_of`` when given), scored against the same year's background, so past
    buckets are byte-identical however much later data is appended.
    """
    kept = _dated(events, as_of)
    totals: collections.Counter[str] = collections.Counter(_year(event) for event in kept)
    entity_counts: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for event in kept:
        for name in event.entities.get(entity_type, {}):
            entity_counts[_year(event)][name] += 1

    target_rows = [
        event for event in kept if target_ticker in {ref.ticker_eodhd for ref in event.tickers}
    ]
    target_totals: collections.Counter[str] = collections.Counter(
        _year(event) for event in target_rows
    )
    supporting: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for event in target_rows:
        for name in event.entities.get(entity_type, {}):
            supporting[(_year(event), name)].append(event.event_id)

    graph: dict[str, list[KGEdge]] = {}
    for year, counter in sorted(co_mentions_by_year(kept, target_ticker, entity_type).items()):
        edges: list[KGEdge] = []
        for name, co in sorted(counter.items()):
            if co < min_co:
                continue
            value = lift(co, target_totals[year], entity_counts[year][name], totals[year])
            if value < min_lift:
                continue
            edges.append(
                KGEdge(
                    target=target_ticker,
                    entity=name,
                    entity_type=entity_type,
                    year=year,
                    co=co,
                    target_events=target_totals[year],
                    entity_events=entity_counts[year][name],
                    total_events=totals[year],
                    lift=value,
                    score=_edge_score(value, co, totals[year], score),
                    event_ids=sorted(supporting[(year, name)]),
                )
            )
        edges.sort(key=lambda edge: (-edge.score, edge.entity))
        graph[year] = edges
    return graph


class EgoNetwork(BaseModel):
    """One (target, year) view: lift edges to the target plus entity-entity links."""

    model_config = ConfigDict(extra="forbid")

    target: str
    year: str
    nodes: list[KGEdge]
    links: list[tuple[str, str, int]] = Field(default_factory=list)


def ego_network(
    events: Sequence[WorldEvent],
    target_ticker: str,
    entity_types: Sequence[str],
    year: str,
    *,
    as_of: dt.date | None = None,
    score: str = "log_lift",
    min_co: int = 3,
    min_lift: float = 2.0,
    link_min_co: int = 2,
) -> EgoNetwork:
    """The force-directed view: kept edges to the target, and lines joining
    entities that also co-occur with each other (within the target's events).
    """
    nodes: list[KGEdge] = []
    for entity_type in entity_types:
        graph = build_graph(
            events,
            target_ticker,
            entity_type,
            as_of=as_of,
            score=score,
            min_co=min_co,
            min_lift=min_lift,
        )
        nodes.extend(graph.get(year, []))
    names = {(node.entity_type, node.entity) for node in nodes}

    pair_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    for event in _dated(events, as_of):
        if _year(event) != year:
            continue
        if target_ticker not in {ref.ticker_eodhd for ref in event.tickers}:
            continue
        present = sorted(
            entity
            for entity_type in entity_types
            for entity in event.entities.get(entity_type, {})
            if (entity_type, entity) in names
        )
        for i, left in enumerate(present):
            for right in present[i + 1 :]:
                if left != right:
                    pair_counts[(left, right)] += 1
    links = [
        (left, right, count)
        for (left, right), count in sorted(pair_counts.items())
        if count >= link_min_co
    ]
    nodes.sort(key=lambda node: (-node.score, node.entity_type, node.entity))
    return EgoNetwork(target=target_ticker, year=year, nodes=nodes, links=links)


def audit_edges(
    graph: dict[str, list[KGEdge]],
    events: Iterable[WorldEvent],
    *,
    sample_size: int = 24,
    seed: int = 0,
) -> dict[str, object]:
    """The provenance audit: sample edges, pull their supporting events back, and
    assert every one falls inside the year the edge belongs to.

    Returns the "24/24 across N docs" summary the post prints; ``passed`` counts
    edges whose every supporting event exists and is dated in the edge's year.
    """
    by_id = {event.event_id: event for event in events}
    edges = [edge for year_edges in graph.values() for edge in year_edges]
    rng = random.Random(seed)
    sampled = rng.sample(edges, min(sample_size, len(edges)))
    passed = 0
    documents: set[str] = set()
    for edge in sampled:
        ok = True
        for event_id in edge.event_ids:
            event = by_id.get(event_id)
            if event is None or event.event.date is None:
                ok = False
                continue
            if event.event.date.isoformat()[:4] != edge.year:
                ok = False
            documents.update(event.member_doc_ids)
        passed += int(ok)
    summary = f"{passed}/{len(sampled)} across {len(documents)} docs"
    return {
        "sampled": len(sampled),
        "passed": passed,
        "documents": len(documents),
        "summary": summary,
    }


def write_graph(directory: Path, graph: dict[str, list[KGEdge]]) -> list[Path]:
    """One canonical-JSON file per year bucket — byte-identical across runs.

    Appending a later year adds a file; existing year files never change, which
    is the point-in-time property expressed as a filesystem diff.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for year in sorted(graph):
        payload = [edge.model_dump(mode="json") for edge in graph[year]]
        path = directory / f"{year}.json"
        path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")
        written.append(path)
    return written
