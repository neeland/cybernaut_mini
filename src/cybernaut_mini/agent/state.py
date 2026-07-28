"""Immutable search state carried by each tree node."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from cybernaut_mini.agent.actions import Action
from cybernaut_mini.models import MetadataFilter, SearchHit
from cybernaut_mini.routing import RoutingSignals


class Stage(StrEnum):
    EXPLORE = "explore"
    REFINE = "refine"
    EXPLOIT = "exploit"


@dataclass(frozen=True)
class SearchState:
    """Frozen after node creation; derived states are produced via ``evolve``."""

    question: str
    query: str
    stage: Stage
    shard_ids: tuple[int, ...] = ()
    expansions: tuple[str, ...] = ()
    metadata_filter: MetadataFilter | None = None
    lexical_weight: float = 1.0
    dense_weight: float = 1.0
    routing_signals: RoutingSignals | None = None
    hits: tuple[SearchHit, ...] = ()
    evidence_terms: tuple[str, ...] = ()
    parent_action: Action | None = None

    def evolve(self, **changes: Any) -> SearchState:
        return replace(self, **changes)


@dataclass(frozen=True)
class StateSummary:
    """The compact view of search progress handed to query generators."""

    current_query: str
    stage: Stage
    top_shard_ids: tuple[int, ...] = ()
    expansions: tuple[str, ...] = ()
    missing_keywords: tuple[str, ...] = field(default=())
    entities: tuple[str, ...] = ()
    evidence_terms: tuple[str, ...] = ()
