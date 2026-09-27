"""Benchmark harness tests: agreement/kappa arithmetic on hand-computed numeric
matrices (pure math), plus the timing and cost harnesses over real labelers."""

from __future__ import annotations

import math

import pytest

from cybernaut_mini.config import ConfigError
from cybernaut_mini.sentiment.benchmark import (
    agreement,
    agreement_matrix,
    cohens_kappa,
    hypothetical_cost,
    kappa_with_reference,
    speed_ratio,
    time_labelers,
)
from cybernaut_mini.sentiment.labelers import TextBlobLabeler, VaderLabeler

# Hand-checkable numeric label columns (pure-math fixture, not corpus data).
MATRIX: dict[str, list[int]] = {
    "ref": [1, 1, 0, -1],
    "clone": [1, 1, 0, -1],  # agreement 1.0 with ref
    "half": [1, 0, 0, 1],  # agreement 0.5 with ref
}


def test_agreement_fraction() -> None:
    assert agreement(MATRIX["ref"], MATRIX["clone"]) == 1.0
    assert agreement(MATRIX["ref"], MATRIX["half"]) == 0.5


def test_agreement_rejects_misaligned_columns() -> None:
    with pytest.raises(ConfigError):
        agreement([1, 0], [1])
    with pytest.raises(ConfigError):
        agreement([], [])


def test_agreement_matrix_sorted_by_reference() -> None:
    result = agreement_matrix(MATRIX, "ref")
    assert result.order == ["ref", "clone", "half"]
    assert result.value("ref", "ref") == 1.0
    assert result.value("ref", "half") == 0.5
    assert result.value("half", "clone") == 0.5
    row = result.reference_row()
    assert row["clone"] == 1.0 and row["half"] == 0.5


def test_agreement_matrix_injects_random_baseline_row() -> None:
    result = agreement_matrix(MATRIX, "ref", random_seed=42)
    assert "Random" in result.order
    assert 0.0 <= result.value("ref", "Random") <= 1.0
    # Seeded: identical across calls.
    again = agreement_matrix(MATRIX, "ref", random_seed=42)
    assert result.values == again.values


def test_agreement_matrix_unknown_reference_raises() -> None:
    with pytest.raises(ConfigError):
        agreement_matrix(MATRIX, "nope")


def test_cohens_kappa_perfect_and_chance() -> None:
    assert cohens_kappa([1, 0, -1, 1], [1, 0, -1, 1]) == 1.0
    # A constant rater: observed 0.5 equals chance 0.5 → kappa 0.
    assert cohens_kappa([1, 0, 1, 0], [1, 1, 1, 1]) == pytest.approx(0.0)
    # Textbook example: po=0.75, pe=0.5 → kappa 0.5.
    assert cohens_kappa([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(0.5)


def test_cohens_kappa_degenerate_identical_constants() -> None:
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) == 1.0


def test_kappa_with_reference_sorted_descending() -> None:
    scores = kappa_with_reference(MATRIX, "ref")
    assert list(scores) == ["clone", "half"]
    assert scores["clone"] == 1.0
    assert scores["half"] < scores["clone"]


# ── timing harness (real labelers over real-shaped text) ─────────────────────


def test_timing_harness_and_speed_ratio() -> None:
    stories = ["Central bank holds interest rates steady amid mixed inflation signals."] * 8
    labelers = [TextBlobLabeler(0.10), VaderLabeler(0.10)]
    timings = time_labelers(stories, labelers)
    assert set(timings) == {"TextBlob-0.10", "VADER-0.10"}
    for result in timings.values():
        assert result.seconds > 0
        assert result.n_stories == 8
        assert result.stories_per_second > 0
    ratio = speed_ratio(timings, "VADER-0.10", "TextBlob-0.10")
    assert ratio == pytest.approx(
        timings["TextBlob-0.10"].seconds / timings["VADER-0.10"].seconds
    )
    with pytest.raises(ConfigError):
        speed_ratio(timings, "VADER-0.10", "FinBERT")


# ── hypothetical cost extrapolation ──────────────────────────────────────────


def test_hypothetical_cost_extrapolates_to_ten_million_stories() -> None:
    report = hypothetical_cost(
        prompt_tokens_per_story=300.0,
        completion_tokens_per_story=2.0,
        usd_per_million_prompt_tokens=30.0,
        usd_per_million_completion_tokens=60.0,
    )
    per_story = (300 * 30 + 2 * 60) / 1_000_000
    assert report["cost_per_story"] == pytest.approx(per_story)
    assert report["total_cost"] == pytest.approx(per_story * 10_000_000)
    assert report["n_stories"] == 10_000_000
    assert math.isfinite(report["total_cost"])


def test_hypothetical_cost_rejects_bad_inputs() -> None:
    with pytest.raises(ConfigError):
        hypothetical_cost(
            prompt_tokens_per_story=-1.0,
            completion_tokens_per_story=0.0,
            usd_per_million_prompt_tokens=1.0,
            usd_per_million_completion_tokens=1.0,
        )
