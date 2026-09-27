"""Search tree nodes, UCT selection, and backpropagation.

UCT(i) = mean_value(i) + c * sqrt(ln(N_parent + 1) / (N_i + 1))

Selection order: unvisited children first (in action_sort_key order), then highest
UCT with ties broken by action type name and normalized action payload.

Blog ref: https://nosible.com/blog/introducing-cybernaut-1-agentic-search-with-mcts —
    the post describes balancing "exploration, exploitation, and inference cost"
    over a search tree but discloses no reward function, no UCT constant, no tree
    width and no depth. Everything numeric here is the build plan's choice.
    Local copy:
    ``docs/blog-archive/introducing-cybernaut-1-agentic-search-with-mcts.md``.

Assumptions:
    - UCT uses the +1-smoothed form ``Q(i) + c*sqrt(ln(N_parent+1)/(N_i+1))``.
      The unsmoothed ``Q = total/visits`` is 0/0 on a fresh child; the smoothing
      is what makes the first visit well defined, and at the root the exploration
      term for an unvisited child is ``c*sqrt(ln 1 / 1) = 0``.
    - Unvisited children are expanded before any UCT comparison, in
      ``action_sort_key`` order. This modifies textbook UCT: the first sweep of a
      freshly expanded node becomes a deterministic enumeration of its actions
      rather than a value comparison in which every child scores zero.
    - Backpropagation adds the leaf reward unchanged to every ancestor. There is
      no discount factor: the post states none, and discounting would make a
      node's mean value depend on the depth at which it happened to be reached,
      which a fixed 18-call budget does not justify.
    - ``SearchNode`` is mutable and owns its visit counters. Exactly one agent
      writes a tree; traces read the frozen values copied into ``NodeTrace``
      rather than the live node, so a trace cannot change after the run.

Alternatives considered:
    - Full MCTS with rollouts to a terminal state: the textbook way to estimate
      an unexpanded child's value, and what the build guide suggests. Rejected
      because every rollout step is a real retrieval call, so random playouts
      would consume the same 18-call budget the search exists to allocate.
    - PUCT / AlphaZero-style selection with a learned prior: strictly better when
      a policy network is available, but the post discloses no policy model and
      this replica has no network to train one against.
    - Plain best-first (greedy) selection: no exploration term at all, so the
      first candidate's descendants would take the whole budget and the disclosed
      wide-to-narrow shape would never happen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from cybernaut_mini.agent.actions import action_sort_key
from cybernaut_mini.agent.state import SearchState


@dataclass
class SearchNode:
    id: int
    state: SearchState
    parent: SearchNode | None = None
    children: list[SearchNode] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    reward: float | None = None
    reward_components: dict[str, float] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0

    def uct_score(self, exploration_constant: float) -> float:
        parent_visits = self.parent.visits if self.parent is not None else 0
        exploration = math.sqrt(math.log(parent_visits + 1) / (self.visits + 1))
        return self.mean_value + exploration_constant * exploration

    def add_child(self, node_id: int, state: SearchState) -> SearchNode:
        child = SearchNode(id=node_id, state=state, parent=self)
        self.children.append(child)
        return child

    def backpropagate(self, value: float) -> None:
        """Propagate the leaf reward unchanged to this node and every ancestor."""
        node: SearchNode | None = self
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    def path_from_root(self) -> list[SearchNode]:
        path: list[SearchNode] = []
        node: SearchNode | None = self
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path


def _tie_key(node: SearchNode) -> tuple[str, str]:
    action = node.state.parent_action
    if action is None:
        return ("", "")
    return action_sort_key(action)


def select_child(children: list[SearchNode], exploration_constant: float) -> SearchNode:
    """UCT child selection: unvisited first, then max UCT, deterministic tie-breaks."""
    if not children:
        msg = "select_child requires at least one child"
        raise ValueError(msg)
    unvisited = [child for child in children if child.visits == 0]
    if unvisited:
        return min(unvisited, key=_tie_key)
    # max() keeps the first maximal element, so pre-sorting by the tie key makes
    # equal UCT scores resolve to the smallest (action type, payload) pair.
    ordered = sorted(children, key=_tie_key)
    return max(ordered, key=lambda child: child.uct_score(exploration_constant))
