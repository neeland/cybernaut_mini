"""Pairwise agreement, kappa, timing and hypothetical-cost harness for labelers.

Everything here is pure computation over an in-memory label matrix
(``{column_name: [labels...]}``); nothing touches the network. The outputs are
the post's three artifacts: the agreement matrix sorted by a reference column,
the model timing table behind the "339x faster" finding, and the
cost-per-10M-stories extrapolation bar chart's underlying numbers.

Blog ref: https://nosible.com/blog/news-sentiment-showdown-who-checks-vibes-best —
    "The matrix below shows the percentage of times that each model agreed with
    every other model across all three classes. The rows and columns have been
    ordered by how much the model agreed with the hand-labeled human results";
    "VADER ... is 339x faster than FinBERT"; "the estimated total cost incurred
    if we used that model to label 10 million news stories". Local copy:
    ``docs/blog-archive/news-sentiment-showdown-who-checks-vibes-best.md``.

Assumptions:
    - Agreement is reported as a fraction in [0, 1]; the post's matrices show
      percentages, which is a rendering choice, and fractions compose directly
      with Cohen's kappa and the ensemble accuracy in :mod:`.ensemble`.
    - The reference column is a parameter, because the post itself uses two:
      human gold for the 250-story matrix and Text-Bison (the best model, i.e.
      proxy gold) for the 10,368-story matrix. Same function, two calls.
    - Cohen's kappa is the repo's extension over the post's raw agreement — the
      gap matrix asks for it explicitly — computed from the standard
      ``(p_o - p_e) / (1 - p_e)`` with marginal chance agreement; degenerate
      marginals (both raters constant and identical) return 1.0 agreement / 0.0
      kappa-denominator, resolved as kappa 1.0 when observed agreement is total
      and 0.0 otherwise, which is the scipy/sklearn convention.
    - Timing measures :func:`time.perf_counter` around a single ``label()``
      batch. The post's 339x is a throughput ratio at their scale; at laptop
      scale the RATIO is the reproducible object, not the absolute seconds.
    - Cost extrapolation is hypothetical by construction: token counts are
      estimated from the prompt (chars/4 unless measured counts are supplied)
      and prices are caller-supplied USD per million tokens, because published
      API prices move and hard-coding them would rot.

Alternatives rejected:
    - ``sklearn.metrics.cohen_kappa_score``: one import for ten lines of
      arithmetic, and the pure-numpy version documents the marginal-chance
      formula the tests pin down.
    - A pandas DataFrame as the matrix container: the repo keeps ``src/``
      pandas-free (no pandas-stubs under strict mypy); a dict of columns plus an
      explicit row/column order carries the same information and serialises
      through ``canonical_dumps`` untouched.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cybernaut_mini.config import ConfigError
from cybernaut_mini.sentiment.labelers import Labeler, RandomLabeler

__all__ = [
    "AgreementMatrix",
    "TimingResult",
    "agreement",
    "agreement_matrix",
    "cohens_kappa",
    "hypothetical_cost",
    "kappa_with_reference",
    "speed_ratio",
    "time_labelers",
]


IntArray = npt.NDArray[np.int64]

#: A label column as callers hold it: a plain list or an int numpy array.
Labels = Sequence[int] | IntArray


def _column(matrix: Mapping[str, Labels], name: str) -> IntArray:
    try:
        return np.asarray(matrix[name], dtype=np.int64)
    except KeyError as exc:
        msg = f"column {name!r} not in label matrix (have: {sorted(matrix)})"
        raise ConfigError(msg) from exc


def agreement(a: Labels, b: Labels) -> float:
    """Fraction of rows on which two label columns assign the same class."""
    left = np.asarray(a, dtype=np.int64)
    right = np.asarray(b, dtype=np.int64)
    if left.shape != right.shape or left.size == 0:
        msg = f"agreement needs two equal-length non-empty columns, got {left.shape}/{right.shape}"
        raise ConfigError(msg)
    return float(np.mean(left == right))


def cohens_kappa(a: Labels, b: Labels) -> float:
    """Cohen's kappa: agreement corrected for marginal chance agreement."""
    left = np.asarray(a, dtype=np.int64)
    right = np.asarray(b, dtype=np.int64)
    observed = agreement(left, right)
    classes = np.union1d(left, right)
    expected = float(
        sum(np.mean(left == cls) * np.mean(right == cls) for cls in classes)
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return float((observed - expected) / (1.0 - expected))


@dataclass(frozen=True)
class AgreementMatrix:
    """A pairwise agreement matrix with rows/columns sorted by the reference.

    ``order[0]`` is the reference column itself; the rest descend by agreement
    with it — the post's presentation, so the top-left corner reads as the
    leaderboard.
    """

    reference: str
    order: list[str]
    values: list[list[float]]

    def value(self, row: str, col: str) -> float:
        return self.values[self.order.index(row)][self.order.index(col)]

    def reference_row(self) -> dict[str, float]:
        """Agreement of every column with the reference, in sorted order."""
        row = self.values[0]
        return {name: row[index] for index, name in enumerate(self.order)}

    def as_dict(self) -> dict[str, object]:
        """A ``canonical_dumps``-ready payload."""
        return {"reference": self.reference, "order": self.order, "values": self.values}


def agreement_matrix(
    matrix: Mapping[str, Labels],
    reference: str,
    *,
    random_seed: int | None = None,
) -> AgreementMatrix:
    """All-pairs agreement, sorted by agreement with *reference*.

    When *random_seed* is given and no ``Random`` column exists, the uniform
    random baseline is generated and included, reproducing the post's Random
    row without requiring the caller to have run :class:`RandomLabeler`.
    """
    columns: dict[str, IntArray] = {name: _column(matrix, name) for name in matrix}
    if reference not in columns:
        msg = f"reference column {reference!r} not in label matrix"
        raise ConfigError(msg)
    if random_seed is not None and "Random" not in columns:
        n_rows = len(next(iter(columns.values())))
        random_column = RandomLabeler(random_seed).label([""] * n_rows)
        columns["Random"] = np.asarray(random_column, dtype=np.int64)

    ref = columns[reference]
    others = sorted(
        (name for name in columns if name != reference),
        key=lambda name: (-agreement(ref, columns[name]), name),
    )
    order = [reference, *others]
    values = [
        [agreement(columns[row], columns[col]) for col in order] for row in order
    ]
    return AgreementMatrix(reference=reference, order=order, values=values)


def kappa_with_reference(
    matrix: Mapping[str, Labels], reference: str
) -> dict[str, float]:
    """Cohen's kappa of every column against *reference*, sorted descending."""
    ref = _column(matrix, reference)
    scores = {
        name: cohens_kappa(ref, _column(matrix, name))
        for name in matrix
        if name != reference
    }
    return dict(sorted(scores.items(), key=lambda item: (-item[1], item[0])))


@dataclass(frozen=True)
class TimingResult:
    """Wall-clock cost of one labeler over one batch."""

    seconds: float
    n_stories: int

    @property
    def stories_per_second(self) -> float:
        return self.n_stories / self.seconds if self.seconds > 0 else float("inf")

    def as_dict(self) -> dict[str, float]:
        return {
            "seconds": self.seconds,
            "n_stories": float(self.n_stories),
            "stories_per_second": self.stories_per_second,
        }


def time_labelers(
    stories: Sequence[str], labelers: Sequence[Labeler]
) -> dict[str, TimingResult]:
    """Time each labeler over the same batch — the VADER-vs-FinBERT speed table."""
    timings: dict[str, TimingResult] = {}
    for labeler in labelers:
        start = time.perf_counter()
        labeler.label(stories)
        elapsed = time.perf_counter() - start
        timings[labeler.name] = TimingResult(seconds=elapsed, n_stories=len(stories))
    return timings


def speed_ratio(timings: Mapping[str, TimingResult], fast: str, slow: str) -> float:
    """How many times faster *fast* is than *slow* (the post's "339x" statistic)."""
    if fast not in timings or slow not in timings:
        msg = f"speed_ratio needs both {fast!r} and {slow!r} in the timing table"
        raise ConfigError(msg)
    return timings[slow].seconds / timings[fast].seconds


def hypothetical_cost(
    *,
    prompt_tokens_per_story: float,
    completion_tokens_per_story: float,
    usd_per_million_prompt_tokens: float,
    usd_per_million_completion_tokens: float,
    n_stories: int = 10_000_000,
) -> dict[str, float]:
    """Extrapolate hosted-LLM labeling cost to *n_stories* (default 10M).

    Returns ``cost_per_story`` and ``total_cost`` in USD — the two bars in the
    post's cost chart ("the estimated total cost incurred if we used that model
    to label 10 million news stories").
    """
    if min(prompt_tokens_per_story, completion_tokens_per_story) < 0 or n_stories <= 0:
        msg = "token counts must be >= 0 and n_stories > 0"
        raise ConfigError(msg)
    cost_per_story = (
        prompt_tokens_per_story * usd_per_million_prompt_tokens
        + completion_tokens_per_story * usd_per_million_completion_tokens
    ) / 1_000_000.0
    return {
        "cost_per_story": cost_per_story,
        "total_cost": cost_per_story * n_stories,
        "n_stories": float(n_stories),
    }
