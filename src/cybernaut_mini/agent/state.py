"""Immutable search state carried by each tree node.

Blog ref: https://nosible.com/blog/introducing-cybernaut-1-agentic-search-with-mcts —
    the build plan reads the agent's "state" as the current search configuration
    (query text, expansion terms, active shards, filters, lexical and semantic
    weight) that each action mutates [inferred]. The post publishes no state
    schema of its own. Local copy:
    ``docs/blog-archive/introducing-cybernaut-1-agentic-search-with-mcts.md``.

Assumptions:
    - ``SearchState`` is frozen. A parent and a child never alias: a transition
      produces a new value via :meth:`SearchState.evolve`, so a node's state
      cannot change after its reward was computed and a trace's node list stays
      truthful.
    - The state carries the retrieved ``hits`` and the ``routing_signals`` that
      produced them, not just the query. Re-running retrieval to reconstruct a
      node would spend budget the search already allocated, and reward values
      would depend on live index state instead of the executed plan.
    - ``Stage`` is a ``StrEnum``, so a stage name is both the trace label and a
      dict key, and the three values are exactly the three search phases.
    - ``StateSummary`` is the only view handed to a query generator. It holds
      derived evidence — expansions, missing shard keywords, entities, title
      terms, top hits with their signal bundles — and deliberately not full hit
      texts, so a generator's prompt cannot grow with corpus size.

Alternatives considered:
    - A mutable state object shared across the tree: cheaper to build (no
      ``replace`` copies), but a child's mutation would retroactively rewrite its
      parent's recorded state, making both the trace and the UCT values lie.
    - Storing only the query and re-retrieving on demand: would make nodes cheap,
      but reward would depend on run order and provider state, and a node could
      not be replayed from the trace without re-running the whole search.
    - Passing the whole ``SearchNode`` to the generator: exposes visit counts and
      values that bias a small model toward whatever it saw last, and couples
      the provider package to the tree implementation.
"""

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
    """The compact view of search progress handed to query generators.

    ``top_hits`` carries the current best hits *with their signal bundles*
    (bm25/dense/rrf per hit) so a model-backed generator can be shown the
    retriever's evidence as a table — see ``providers/signals.py``. Empty before
    the first retrieval and for the heuristic generator's purposes.
    """

    current_query: str
    stage: Stage
    top_shard_ids: tuple[int, ...] = ()
    expansions: tuple[str, ...] = ()
    missing_keywords: tuple[str, ...] = field(default=())
    entities: tuple[str, ...] = ()
    evidence_terms: tuple[str, ...] = ()
    top_hits: tuple[SearchHit, ...] = ()
