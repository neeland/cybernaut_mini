from __future__ import annotations

import pytest

from cybernaut_mini.rrf import RankedList, rrf_fuse


def test_rrf_scores_match_formula() -> None:
    fused = rrf_fuse(
        [
            RankedList(name="dense", weight=1.0, ids=("a", "b")),
            RankedList(name="lexical", weight=1.0, ids=("b", "a")),
        ],
        k=60,
    )
    by_id = {item.id: item for item in fused}
    assert by_id["a"].score == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["b"].score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_weights_scale_contributions() -> None:
    fused = rrf_fuse(
        [
            RankedList(name="dense", weight=1.0, ids=("a",)),
            RankedList(name="entity", weight=0.5, ids=("a",)),
        ],
        k=60,
    )
    item = fused[0]
    assert item.contributions["dense"] == pytest.approx(1 / 61)
    assert item.contributions["entity"] == pytest.approx(0.5 / 61)
    assert item.score == pytest.approx(1.5 / 61)


def test_rrf_tie_breaks_by_ascending_id() -> None:
    fused = rrf_fuse(
        [
            RankedList(name="dense", weight=1.0, ids=("z", "a")),
            RankedList(name="lexical", weight=1.0, ids=("a", "z")),
        ]
    )
    assert [item.id for item in fused] == ["a", "z"]


def test_rrf_omitted_ranker_absent_from_contributions() -> None:
    fused = rrf_fuse([RankedList(name="dense", weight=1.0, ids=("a",))])
    assert "entity" not in fused[0].contributions
    assert fused[0].ranks == {"dense": 1}


def test_rrf_exposes_ranks_and_ranker_scores() -> None:
    fused = rrf_fuse(
        [RankedList(name="dense", weight=1.0, ids=("a", "b"), scores=(0.9, 0.5))]
    )
    assert fused[0].ranks["dense"] == 1
    assert fused[0].ranker_scores["dense"] == pytest.approx(0.9)
    assert fused[1].ranks["dense"] == 2


def test_rrf_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        rrf_fuse([], k=0)
