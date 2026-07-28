"""Typed action union for the search agent.

Every state transition is one of these actions. ``action_sort_key`` provides the
deterministic tie-break order (action type name, then normalized payload).
P0 never invents metadata filters: ``RetainFilter`` only carries the user's own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cybernaut_mini.models import MetadataFilter, canonical_dumps


@dataclass(frozen=True)
class RewriteQuery:
    text: str


@dataclass(frozen=True)
class AddExpansions:
    terms: tuple[str, ...]


@dataclass(frozen=True)
class NarrowShards:
    shard_ids: tuple[int, ...]


@dataclass(frozen=True)
class AdjustHybridWeights:
    lexical_weight: float
    dense_weight: float


@dataclass(frozen=True)
class RetainFilter:
    metadata_filter: MetadataFilter


Action = RewriteQuery | AddExpansions | NarrowShards | AdjustHybridWeights | RetainFilter


def action_payload(action: Action) -> dict[str, Any]:
    if isinstance(action, RewriteQuery):
        return {"text": action.text}
    if isinstance(action, AddExpansions):
        return {"terms": list(action.terms)}
    if isinstance(action, NarrowShards):
        return {"shard_ids": list(action.shard_ids)}
    if isinstance(action, AdjustHybridWeights):
        return {
            "lexical_weight": action.lexical_weight,
            "dense_weight": action.dense_weight,
        }
    return {"metadata_filter": action.metadata_filter.model_dump(mode="json")}


def describe_action(action: Action) -> dict[str, Any]:
    """Trace-ready representation: {"type": ..., **payload}."""
    return {"type": type(action).__name__, **action_payload(action)}


def action_sort_key(action: Action) -> tuple[str, str]:
    """Deterministic ordering: action type name, then canonical payload JSON."""
    return (type(action).__name__, canonical_dumps(action_payload(action)))
