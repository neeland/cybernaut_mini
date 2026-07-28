from __future__ import annotations

import pytest

from cybernaut_mini.agent.actions import (
    AddExpansions,
    AdjustHybridWeights,
    NarrowShards,
    RewriteQuery,
    action_sort_key,
    describe_action,
)
from cybernaut_mini.agent.node import SearchNode, select_child
from cybernaut_mini.agent.policy import compute_reward
from cybernaut_mini.agent.state import SearchState, Stage
from cybernaut_mini.models import Document, JudgeScore, SearchHit


def make_state(query: str = "q", action: object | None = None) -> SearchState:
    return SearchState(
        question="q",
        query=query,
        stage=Stage.EXPLORE,
        parent_action=action,  # type: ignore[arg-type]
    )


def make_hit(doc_id: str, *, dense: float | None, bm25: float | None, rank: int) -> SearchHit:
    return SearchHit(
        document=Document(id=doc_id, title=f"Title {doc_id}", text="body text"),
        rank=rank,
        score=1.0,
        shard_id=0,
        dense_score=dense,
        bm25_score=bm25,
        query_variant="q",
        snippet="body text",
    )


# ----------------------------- reward -------------------------------- #


def test_reward_empty_hits_is_zero() -> None:
    reward, components = compute_reward(
        JudgeScore(relevance=1.0, coverage=1.0, redundancy=0.0, reason="x"), []
    )
    assert reward == 0.0
    assert components["reward"] == 0.0


def test_reward_matches_weighted_formula() -> None:
    hits = [
        make_hit("a", dense=1.0, bm25=2.0, rank=1),
        make_hit("b", dense=0.0, bm25=0.0, rank=2),
    ]
    judge = JudgeScore(relevance=1.0, coverage=1.0, redundancy=1.0, reason="x")
    reward, components = compute_reward(judge, hits)
    # dense/lexical normalized means over [1,0]/[2,0] = 0.5 each.
    expected = 0.45 * 1.0 + 0.20 * 1.0 + 0.20 * 0.5 + 0.10 * 0.5 - 0.05 * 1.0
    assert reward == pytest.approx(expected)
    assert components["dense"] == pytest.approx(0.5)
    assert components["lexical"] == pytest.approx(0.5)


def test_reward_is_clamped_to_unit_interval() -> None:
    hits = [make_hit("a", dense=1.0, bm25=1.0, rank=1)]
    judge = JudgeScore(relevance=1.0, coverage=1.0, redundancy=0.0, reason="x")
    reward, _ = compute_reward(judge, hits)
    assert 0.0 <= reward <= 1.0
    # Max achievable is 0.95 by design (single constant hit -> normalized mean 1.0).
    assert reward == pytest.approx(0.95)


# ------------------------------ node --------------------------------- #


def test_backpropagation_reaches_every_ancestor() -> None:
    root = SearchNode(id=0, state=make_state())
    child = root.add_child(1, make_state("c", RewriteQuery("c")))
    grandchild = child.add_child(2, make_state("g", RewriteQuery("g")))
    grandchild.backpropagate(0.5)
    assert grandchild.visits == child.visits == root.visits == 1
    assert root.total_value == pytest.approx(0.5)
    assert child.mean_value == pytest.approx(0.5)


def test_select_child_prefers_unvisited_in_tiebreak_order() -> None:
    root = SearchNode(id=0, state=make_state())
    root.visits = 5
    a = root.add_child(1, make_state("a", RewriteQuery("zzz")))
    b = root.add_child(2, make_state("b", AddExpansions(("term",))))
    a.visits, a.total_value = 3, 3.0
    # b is unvisited -> selected regardless of a's value.
    assert select_child(root.children, 1.2) is b


def test_select_child_unvisited_tiebreak_by_action_key() -> None:
    root = SearchNode(id=0, state=make_state())
    first = root.add_child(1, make_state("a", AddExpansions(("a",))))
    root.add_child(2, make_state("b", RewriteQuery("b")))
    # AddExpansions sorts before RewriteQuery by type name.
    assert select_child(root.children, 1.2) is first


def test_select_child_uses_uct_when_all_visited() -> None:
    root = SearchNode(id=0, state=make_state())
    root.visits = 10
    a = root.add_child(1, make_state("a", RewriteQuery("a")))
    b = root.add_child(2, make_state("b", RewriteQuery("b")))
    a.visits, a.total_value = 1, 1.0
    b.visits, b.total_value = 5, 1.0
    # a has higher mean value and fewer visits -> higher UCT.
    assert select_child(root.children, 1.2) is a


def test_uct_tie_breaks_deterministically() -> None:
    root = SearchNode(id=0, state=make_state())
    root.visits = 4
    a = root.add_child(1, make_state("a", RewriteQuery("b")))
    b = root.add_child(2, make_state("b", AddExpansions(("a",))))
    a.visits, a.total_value = 2, 1.0
    b.visits, b.total_value = 2, 1.0
    # Identical UCT -> smaller action_sort_key wins (AddExpansions < RewriteQuery).
    assert select_child(root.children, 1.2) is b


# ----------------------------- actions ------------------------------- #


def test_action_sort_key_orders_by_type_then_payload() -> None:
    keys = sorted(
        [
            action_sort_key(RewriteQuery("b")),
            action_sort_key(AddExpansions(("a",))),
            action_sort_key(AdjustHybridWeights(1.0, 1.0)),
            action_sort_key(NarrowShards((1, 2))),
        ]
    )
    assert [k[0] for k in keys] == [
        "AddExpansions",
        "AdjustHybridWeights",
        "NarrowShards",
        "RewriteQuery",
    ]


def test_describe_action_is_trace_ready() -> None:
    assert describe_action(RewriteQuery("hi")) == {"type": "RewriteQuery", "text": "hi"}
    assert describe_action(NarrowShards((2, 1))) == {
        "type": "NarrowShards",
        "shard_ids": [2, 1],
    }
