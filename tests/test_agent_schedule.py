"""Tests for the config-driven agent stage schedule and its annealing numbers.

The annealed schedule is the post's wide-shallow-to-narrow-deep progression
stated as numbers — branching decays, per-node depth grows, generator
temperature anneals — and ``schedule_from_config`` is the seam a config file
reaches it through. These tests pin the numbers and every rejection path, so a
typo in a config fails loudly at load rather than silently running the default.

Every test runs offline. No API key, no model download, no network.
"""

from __future__ import annotations

import pytest

from cybernaut_mini.agent.schedule import (
    ANNEALED_SCHEDULE,
    DEFAULT_SCHEDULE,
    StagePolicy,
    schedule_from_config,
)
from cybernaut_mini.agent.state import Stage


def test_default_schedule_keeps_the_legacy_candidate_budget() -> None:
    assert [DEFAULT_SCHEDULE[stage].candidates for stage in Stage] == [5, 9, 4]
    assert sum(policy.candidates for policy in DEFAULT_SCHEDULE.values()) == 18


def test_annealed_schedule_narrows_branching() -> None:
    """k_children decays 4 -> 2 -> 1 (Explore branches from the root only)."""
    assert ANNEALED_SCHEDULE[Stage.EXPLORE].candidates == 4
    assert ANNEALED_SCHEDULE[Stage.EXPLORE].per_survivor is None
    assert ANNEALED_SCHEDULE[Stage.REFINE].per_survivor == 2
    assert ANNEALED_SCHEDULE[Stage.EXPLOIT].per_survivor == 1


def test_annealed_schedule_deepens_retrieval() -> None:
    """top_k grows 10 -> 20 -> 40."""
    assert [ANNEALED_SCHEDULE[stage].hits_per_shard for stage in Stage] == [10, 20, 40]


def test_both_schedules_anneal_temperature() -> None:
    """Generator temperature anneals 1.0 -> 0.6 -> 0.3."""
    for schedule in (DEFAULT_SCHEDULE, ANNEALED_SCHEDULE):
        assert [schedule[stage].temperature for stage in Stage] == [1.0, 0.6, 0.3]


def test_refine_declares_the_hybrid_weight_grid_and_rerank_toggle() -> None:
    """The {0.5, 1.0, 1.5} lexical grid (dense held at 1.0; 1.0 is the parent)."""
    for schedule in (DEFAULT_SCHEDULE, ANNEALED_SCHEDULE):
        refine = schedule[Stage.REFINE]
        assert refine.weight_variants == ((0.5, 1.0), (1.5, 1.0))
        assert refine.try_rerank_off is True


def test_explore_judges_wide_and_cheap() -> None:
    for schedule in (DEFAULT_SCHEDULE, ANNEALED_SCHEDULE):
        assert schedule[Stage.EXPLORE].use_model_judge is False
        assert schedule[Stage.REFINE].use_model_judge is True
        assert schedule[Stage.EXPLOIT].use_model_judge is True


# ------------------------ schedule_from_config ------------------------ #


def test_none_and_default_name_return_the_default_schedule() -> None:
    assert schedule_from_config(None) is DEFAULT_SCHEDULE
    assert schedule_from_config("default") is DEFAULT_SCHEDULE
    assert schedule_from_config("  Annealed ") is ANNEALED_SCHEDULE


def test_unknown_schedule_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown stage schedule"):
        schedule_from_config("exponential")


def test_mapping_overrides_one_stage_and_keeps_the_rest() -> None:
    schedule = schedule_from_config({"refine": {"hits_per_shard": 20, "temperature": 0.5}})
    assert schedule[Stage.REFINE].hits_per_shard == 20
    assert schedule[Stage.REFINE].temperature == 0.5
    # Untouched fields and stages keep the default values.
    assert schedule[Stage.REFINE].candidates == DEFAULT_SCHEDULE[Stage.REFINE].candidates
    assert schedule[Stage.EXPLORE] is DEFAULT_SCHEDULE[Stage.EXPLORE]
    assert schedule[Stage.EXPLOIT] is DEFAULT_SCHEDULE[Stage.EXPLOIT]


def test_weight_variants_are_coerced_to_float_pairs() -> None:
    schedule = schedule_from_config({"refine": {"weight_variants": [[1, 2], (0.5, 1)]}})
    assert schedule[Stage.REFINE].weight_variants == ((1.0, 2.0), (0.5, 1.0))


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown agent stage"):
        schedule_from_config({"warmup": {"candidates": 1}})


def test_unknown_policy_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown stage-policy field"):
        schedule_from_config({"refine": {"branching": 2}})


def test_non_mapping_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        schedule_from_config({"refine": 3})


@pytest.mark.parametrize("bad", ["0.5,1.0", [[0.5]], [[0.5, 1.0, 2.0]], [0.5]])
def test_malformed_weight_variants_are_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="weight_variants"):
        schedule_from_config({"refine": {"weight_variants": bad}})


def test_stage_policy_is_frozen() -> None:
    policy = StagePolicy(candidates=1, shards=1, hits_per_shard=1, survivors=1)
    with pytest.raises(AttributeError):
        policy.candidates = 2  # type: ignore[misc]
