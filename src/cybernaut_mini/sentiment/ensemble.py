"""Greedy iterative-addition ensembles with bootstrap stability testing.

The teacher in the ensemble-and-distil recipe is not a model: it is the row-sum
of a handful of labeler columns, chosen greedily. This module owns the choosing
— forward selection maximizing three-class accuracy of the sign-thresholded sum
against a gold column — and the 1,000-run bootstrap that turns one selection
into a probability distribution over ensembles.

Blog ref: https://nosible.com/blog/ensemble-and-distil — "One trick I like to
    use when building ensembles is iterative addition. This is a greedy
    procedure that starts off with the best model and then iteratively includes
    the most 'additive' model until no more models are additive"; "we threshold
    it such that any aggregates greater than 1 are 'Positive'. Any aggregates
    less than -1 are 'Negative'. And anything in between is 'Neutral'"; "Across
    1,000 simulations this ensemble was the best 51% of the time". Local copy:
    ``docs/blog-archive/ensemble-and-distil.md``.

Assumptions:
    - The threshold band is INCLUSIVE at ±1, following the post's appendix code
      (``teacher_llm_classes[teacher_llm <= -1] = -1`` / ``>= 1``) rather than
      its prose ("greater than 1"); the code is what produced the published
      numbers, so the code wins.
    - A single member's ensemble is the member itself: labels are in {-1, 0, 1},
      so sign-threshold(sum of one column) is the identity, which is why the
      greedy loop can seed with "the best model" and still use one accuracy
      function throughout.
    - Ties in the greedy argmax break by column name, so a run is deterministic
      given the matrix — the bootstrap then owes ALL its variance to row
      subsampling, which is the quantity the post is measuring.
    - The post does not state its simulation subsample size; the gap matrix
      fixes 75% row subsamples without replacement, and the fraction is a
      parameter so the sensitivity is one call away.
    - Gold is any integer column in {-1, 0, 1}: hand labels when you have them,
      a best-LLM column at scale (the post's own proxy-gold move). The
      selection code cannot tell the difference, and should not.

Alternatives rejected:
    - Exhaustive subset search: 2^n subsets is feasible for n=18 columns but
      the post's procedure IS greedy forward selection — the trace of per-step
      gains ("Then add GPT-3.5 (+2.80% boost)") is part of the published
      output, and an exhaustive argmax has no such trace.
    - Weighted member sums: the post's ensemble is deliberately unweighted; the
      distillation student is where continuous weights live.
    - Bootstrap WITH replacement: closer to the classical bootstrap, but
      duplicated rows would let a single story vote twice in the accuracy,
      which the post's "over a sample of your data" phrasing does not suggest.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cybernaut_mini.config import ConfigError

__all__ = [
    "SelectionResult",
    "SelectionStep",
    "bootstrap_stability",
    "ensemble_accuracy",
    "greedy_forward_selection",
    "sign_threshold",
]

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

#: A label column as callers hold it: a plain list or an int numpy array.
Labels = Sequence[int] | IntArray


def sign_threshold(scores: npt.ArrayLike) -> IntArray:
    """Sum-of-labels → three classes with the post's inclusive ±1 band."""
    values = np.asarray(scores, dtype=np.float64)
    classes = np.zeros(len(values), dtype=np.int64)
    classes[values <= -1] = -1
    classes[values >= 1] = 1
    return classes


def _validate(
    matrix: Mapping[str, Labels], gold: Labels
) -> tuple[dict[str, IntArray], IntArray]:
    gold_arr = np.asarray(gold, dtype=np.int64)
    if gold_arr.size == 0:
        msg = "gold column is empty"
        raise ConfigError(msg)
    if not matrix:
        msg = "label matrix has no columns"
        raise ConfigError(msg)
    columns: dict[str, IntArray] = {}
    for name in sorted(matrix):
        column = np.asarray(matrix[name], dtype=np.int64)
        if column.shape != gold_arr.shape:
            msg = f"column {name!r} has {column.size} rows, gold has {gold_arr.size}"
            raise ConfigError(msg)
        columns[name] = column
    return columns, gold_arr


def ensemble_accuracy(
    columns: Mapping[str, IntArray], members: Sequence[str], gold: IntArray
) -> float:
    """Accuracy of sign-threshold(sum of member label columns) against gold."""
    if not members:
        msg = "an ensemble needs at least one member"
        raise ConfigError(msg)
    total = np.sum([columns[name] for name in members], axis=0)
    return float(np.mean(sign_threshold(total) == gold))


@dataclass(frozen=True)
class SelectionStep:
    """One greedy step: the member added, the new accuracy, the gain over before."""

    member: str
    accuracy: float
    gain: float

    def as_dict(self) -> dict[str, object]:
        return {"member": self.member, "accuracy": self.accuracy, "gain": self.gain}


@dataclass(frozen=True)
class SelectionResult:
    """The winning ensemble plus the per-step trace the post publishes."""

    members: list[str]
    accuracy: float
    steps: list[SelectionStep]

    def as_dict(self) -> dict[str, object]:
        return {
            "members": self.members,
            "accuracy": self.accuracy,
            "steps": [step.as_dict() for step in self.steps],
        }


def greedy_forward_selection(
    matrix: Mapping[str, Labels],
    gold: Labels,
    *,
    exclude: Sequence[str] = (),
) -> SelectionResult:
    """Iterative addition: seed with the best column, add while additive.

    *exclude* removes columns that must not be members — the gold column itself
    when it lives inside the matrix, or the Random baseline.
    """
    columns, gold_arr = _validate(matrix, gold)
    for name in exclude:
        columns.pop(name, None)
    if not columns:
        msg = "no candidate columns left after exclusions"
        raise ConfigError(msg)

    candidates = sorted(columns)
    members: list[str] = []
    steps: list[SelectionStep] = []
    accuracy = 0.0
    while candidates:
        scored = [
            (ensemble_accuracy(columns, [*members, name], gold_arr), name)
            for name in candidates
        ]
        # Argmax on accuracy; ties break to the alphabetically first column so a
        # run is a pure function of (matrix, gold).
        best_accuracy, best_name = min(scored, key=lambda item: (-item[0], item[1]))
        gain = best_accuracy - accuracy
        if members and gain <= 0:
            break
        members.append(best_name)
        candidates.remove(best_name)
        steps.append(SelectionStep(member=best_name, accuracy=best_accuracy, gain=gain))
        accuracy = best_accuracy
    return SelectionResult(members=members, accuracy=accuracy, steps=steps)


def bootstrap_stability(
    matrix: Mapping[str, Labels],
    gold: Labels,
    *,
    n_runs: int = 1000,
    subsample: float = 0.75,
    seed: int = 42,
    exclude: Sequence[str] = (),
) -> list[tuple[tuple[str, ...], float]]:
    """Rerun the greedy selection on row subsamples; tally which ensemble wins.

    Returns ``(sorted_member_tuple, win_share)`` pairs, descending by share —
    the post's "this ensemble was the best 51% of the time" chart. Rows are
    drawn without replacement at *subsample* of the data per run.
    """
    if not 0.0 < subsample <= 1.0:
        msg = f"subsample must be in (0, 1], got {subsample}"
        raise ConfigError(msg)
    if n_runs <= 0:
        msg = f"n_runs must be positive, got {n_runs}"
        raise ConfigError(msg)
    columns, gold_arr = _validate(matrix, gold)
    n_rows = gold_arr.size
    n_keep = max(1, round(subsample * n_rows))
    rng = np.random.default_rng(seed)

    wins: Counter[tuple[str, ...]] = Counter()
    for _ in range(n_runs):
        rows = rng.choice(n_rows, size=n_keep, replace=False)
        sub_matrix = {name: column[rows] for name, column in columns.items()}
        result = greedy_forward_selection(sub_matrix, gold_arr[rows], exclude=exclude)
        wins[tuple(sorted(result.members))] += 1

    return sorted(
        ((members, count / n_runs) for members, count in wins.items()),
        key=lambda item: (-item[1], item[0]),
    )
