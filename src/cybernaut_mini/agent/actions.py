"""Typed action union for the search agent.

Every state transition is one of these actions. ``action_sort_key`` provides the
deterministic tie-break order (action type name, then normalized payload).
P0 never invents metadata filters: ``RetainFilter`` only carries the user's own.

Blog ref: https://nosible.com/blog/introducing-cybernaut-1-agentic-search-with-mcts —
    the agent "tunes every knob" of the retrieval pipeline, including the hybrid
    lexical/dense balance and the rerankers. The post never enumerates its action
    space, so these six actions are the build plan's reading of "tune on the fly"
    restricted to knobs this replica actually has. Local copy:
    ``docs/blog-archive/introducing-cybernaut-1-agentic-search-with-mcts.md``.

Assumptions:
    - The disclosed move list (rewrite/expand/narrow the query, add or drop shards,
      retune weights, change filters) maps onto these six dataclasses. There is no
      "drop shard" action because a branch's shard set is recomputed by routing
      from its query, so narrowing is the only shard-level move the replica can
      express without inventing a shard set the router would not have chosen.
    - Every action is a typed value, never an arbitrary config patch. A model may
      propose a query or a bounded weight multiplier, but it cannot set a field
      the rest of the system does not read.
    - ``RetainFilter`` re-executes a filter the *user* supplied. No action
      manufactures a metadata filter, so the agent can never silently drop
      documents the caller asked for by narrowing on a fabricated predicate.
    - ``AdjustHybridWeights`` carries multipliers on the configured RRF weights,
      not absolute weights, so one action means the same thing under every config
      and a variant can never zero out a retriever by accident.
    - Ordering is ``(type name, canonical payload JSON)``, so UCT ties resolve
      identically on every run — the property that makes a trace replayable.

Alternatives considered:
    - Free-form JSON actions emitted by the LLM: more expressive and closer to
      "the LLM proposes a refinement", but an unrecognised field would either
      crash a search turn or silently no-op, and two runs could not be compared.
      Typed actions make the proposal space auditable.
    - One generic ``SetConfig(patch)`` action: fewer classes to maintain, but it
      collapses distinct moves into a single trace label and lets a proposal
      touch fields that are not search knobs (seed, provider, index path).
    - Deriving the action from a candidate's ``origin`` string instead of passing
      it explicitly: cheaper, but the trace would then report the generator's
      label rather than the mutation actually executed, and the knob variants —
      which carry no meaningful ``origin`` of their own — would lose their action.
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


@dataclass(frozen=True)
class ToggleRerankers:
    """Run the branch with the stage-6 shard rerankers switched on or off."""

    enabled: bool


Action = (
    RewriteQuery
    | AddExpansions
    | NarrowShards
    | AdjustHybridWeights
    | RetainFilter
    | ToggleRerankers
)


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
    if isinstance(action, ToggleRerankers):
        return {"enabled": action.enabled}
    return {"metadata_filter": action.metadata_filter.model_dump(mode="json")}


def describe_action(action: Action) -> dict[str, Any]:
    """Trace-ready representation: {"type": ..., **payload}."""
    return {"type": type(action).__name__, **action_payload(action)}


def action_sort_key(action: Action) -> tuple[str, str]:
    """Deterministic ordering: action type name, then canonical payload JSON."""
    return (type(action).__name__, canonical_dumps(action_payload(action)))
