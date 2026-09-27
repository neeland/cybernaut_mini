"""Greedy iterative-addition + bootstrap tests on hand-constructed numeric
label matrices (pure math), pinning the post's ±1 band and trace semantics."""

from __future__ import annotations

import numpy as np
import pytest

from cybernaut_mini.config import ConfigError
from cybernaut_mini.sentiment.ensemble import (
    bootstrap_stability,
    ensemble_accuracy,
    greedy_forward_selection,
    sign_threshold,
)

# Gold plus two 5/6-accurate columns that are perfect TOGETHER: A errs +1 on the
# last row, B errs -1 on the last row, so the summed ensemble cancels to gold.
GOLD = [1, -1, 0, 1, -1, 0]
MATRIX: dict[str, list[int]] = {
    "A": [1, -1, 0, 1, -1, 1],
    "B": [1, -1, 0, 1, -1, -1],
    "noise": [0, 1, 0, -1, 1, 0],  # 2/6 vs gold (rows 2 and 5 only)
}


def test_sign_threshold_band_is_inclusive_at_one() -> None:
    scores = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    assert sign_threshold(scores).tolist() == [-1, -1, 0, 0, 0, 1, 1]


def test_single_member_ensemble_is_the_member_itself() -> None:
    columns = {name: np.asarray(values, dtype=np.int64) for name, values in MATRIX.items()}
    gold = np.asarray(GOLD, dtype=np.int64)
    assert ensemble_accuracy(columns, ["A"], gold) == pytest.approx(5 / 6)
    assert ensemble_accuracy(columns, ["noise"], gold) == pytest.approx(2 / 6)


def test_greedy_selection_finds_the_complementary_pair_with_trace() -> None:
    result = greedy_forward_selection(MATRIX, GOLD)
    assert result.members == ["A", "B"]  # tie at 5/6 breaks alphabetically
    assert result.accuracy == 1.0
    assert [step.member for step in result.steps] == ["A", "B"]
    assert result.steps[0].accuracy == pytest.approx(5 / 6)
    assert result.steps[0].gain == pytest.approx(5 / 6)
    assert result.steps[1].accuracy == 1.0
    assert result.steps[1].gain == pytest.approx(1 / 6)


def test_greedy_selection_stops_when_no_model_is_additive() -> None:
    """A perfect single column: nothing can be additive after it."""
    matrix = {"perfect": list(GOLD), "noise": MATRIX["noise"]}
    result = greedy_forward_selection(matrix, GOLD)
    assert result.members == ["perfect"]
    assert result.accuracy == 1.0
    assert len(result.steps) == 1


def test_greedy_selection_respects_exclusions() -> None:
    result = greedy_forward_selection(MATRIX, GOLD, exclude=["A"])
    assert "A" not in result.members
    with pytest.raises(ConfigError):
        greedy_forward_selection(MATRIX, GOLD, exclude=["A", "B", "noise"])


def test_greedy_selection_validates_shapes() -> None:
    with pytest.raises(ConfigError):
        greedy_forward_selection({"A": [1, 0]}, GOLD)
    with pytest.raises(ConfigError):
        greedy_forward_selection({}, GOLD)
    with pytest.raises(ConfigError):
        greedy_forward_selection(MATRIX, [])


def test_greedy_selection_is_deterministic() -> None:
    first = greedy_forward_selection(MATRIX, GOLD)
    second = greedy_forward_selection(MATRIX, GOLD)
    assert first == second


# ── bootstrap stability ──────────────────────────────────────────────────────


def test_bootstrap_tallies_winner_shares_that_sum_to_one() -> None:
    shares = bootstrap_stability(MATRIX, GOLD, n_runs=200, subsample=0.75, seed=42)
    assert shares  # at least one winning ensemble
    total = sum(share for _, share in shares)
    assert total == pytest.approx(1.0)
    # Descending by share; members are sorted tuples.
    values = [share for _, share in shares]
    assert values == sorted(values, reverse=True)
    for members, _ in shares:
        assert list(members) == sorted(members)


def test_bootstrap_prefers_the_true_winning_pair() -> None:
    shares = bootstrap_stability(MATRIX, GOLD, n_runs=200, subsample=0.75, seed=42)
    winners = dict(shares)
    assert winners.get(("A", "B"), 0.0) >= max(
        share for members, share in shares if members != ("A", "B")
    )


def test_bootstrap_is_seeded() -> None:
    a = bootstrap_stability(MATRIX, GOLD, n_runs=50, seed=7)
    b = bootstrap_stability(MATRIX, GOLD, n_runs=50, seed=7)
    assert a == b


def test_bootstrap_validates_parameters() -> None:
    with pytest.raises(ConfigError):
        bootstrap_stability(MATRIX, GOLD, subsample=0.0)
    with pytest.raises(ConfigError):
        bootstrap_stability(MATRIX, GOLD, subsample=1.5)
    with pytest.raises(ConfigError):
        bootstrap_stability(MATRIX, GOLD, n_runs=0)
