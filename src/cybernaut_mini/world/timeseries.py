"""Dated series arithmetic for the WORLD indices: aligned, trailing, and causal.

Every WORLD index is "a number per day (or month), built only from information
available on that day". This module is the one place that discipline is enforced:
an immutable :class:`TimeSeries` (strictly increasing dates, one value each) plus
the handful of operations the posts' formulas need — daily/monthly aggregation,
trailing-window means for the corpus-growth denominators, first differences,
Pearson on levels and changes, and the rebase/rescale chart norms.

Blog ref: https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world
    — "the trailing 12-month average of total monthly breadth" (the ``B(m)``
    denominator) and "we rescale each series to average 100 over 2020 to 2024";
    https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty
    — "trailing 12-month average of total daily breadth" and "all correlations are
    computed on the common window ... on jointly non-null rows after rebasing each
    series to its own mean". Local copies under ``docs/blog-archive/``.

Assumptions:
    - Series are represented as (sorted unique dates, float64 values), not as a
      pandas index. The posts' math is a dozen group-bys and rolling means; numpy
      covers all of it, and the repo's strict-mypy gate has no pandas stubs (and
      ``pyproject.toml`` is frozen for this workstream), so pandas would force
      type-ignores through every index module for no analytical gain.
    - Trailing windows are *inclusive of the current point* — ``rolling(12).mean()``
      semantics — because that is what the posts' pandas one-liners compute. The
      strictly-before variant needed by the asymmetric smoother's trigger statistics
      lives in :mod:`cybernaut_mini.world.smoother`, where the post says "with no
      look-ahead".
    - Date-aware trailing means (:meth:`TimeSeries.trailing_mean_days`) average over
      the *observed* points inside the window. Calendar days with no events count
      as zero only if the caller says so via :meth:`TimeSeries.fill_daily` first;
      at NOSIBLE scale there is no empty day, at fixture scale there are many, and
      silently averaging in invented zeros would change the denominator's meaning.
    - Pearson needs two aligned points with nonzero variance; anything less returns
      ``nan`` rather than raising, because validation code sweeps many windows and
      a degenerate window is an answer ("no evidence"), not a bug.
    - Monthly keys are the first day of the month, quarterly keys the first day of
      the quarter, so a resampled series is still a plain :class:`TimeSeries` and
      every operation composes.

Alternatives rejected:
    - pandas ``Series``/``DataFrame`` throughout: the posts' own idiom, but see
      above — the typing cost lands on every downstream module. The notebooks (out
      of scope here) remain free to convert via :meth:`TimeSeries.to_mapping`.
    - A dense daily array from corpus start to end as the canonical form: makes
      rolling windows trivial but bakes the zero-fill decision into the type;
      keeping observed points canonical and making the fill explicit preserves
      the distinction between "no events" and "measured zero".
    - Mutable accumulation (append-style builders): the indices are pure functions
      over a frozen event table; immutable series make byte-identical reruns the
      default instead of a discipline.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "TimeSeries",
    "daily_ratio",
    "daily_sum",
]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TimeSeries:
    """Immutable dated series: strictly increasing dates, one float per date."""

    dates: tuple[dt.date, ...]
    values: FloatArray

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=np.float64)
        object.__setattr__(self, "values", array)
        if array.ndim != 1 or array.shape[0] != len(self.dates):
            msg = f"values shape {array.shape} does not match {len(self.dates)} dates"
            raise ValueError(msg)
        if any(b <= a for a, b in zip(self.dates, self.dates[1:], strict=False)):
            msg = "dates must be strictly increasing"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.dates)

    @classmethod
    def from_mapping(cls, mapping: Mapping[dt.date, float]) -> TimeSeries:
        dates = tuple(sorted(mapping))
        return cls(dates, np.asarray([mapping[d] for d in dates], dtype=np.float64))

    def to_mapping(self) -> dict[dt.date, float]:
        return {d: float(v) for d, v in zip(self.dates, self.values, strict=True)}

    # ------------------------------------------------------------------ #
    # Alignment and correlation                                          #
    # ------------------------------------------------------------------ #

    def align(self, other: TimeSeries) -> tuple[FloatArray, FloatArray]:
        """Values of both series on their common dates (jointly non-null rows)."""
        common = sorted(set(self.dates) & set(other.dates))
        mine, theirs = self.to_mapping(), other.to_mapping()
        left = np.asarray([mine[d] for d in common], dtype=np.float64)
        right = np.asarray([theirs[d] for d in common], dtype=np.float64)
        return left, right

    def pearson(self, other: TimeSeries) -> float:
        """Pearson correlation on common dates; ``nan`` when degenerate."""
        left, right = self.align(other)
        if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
            return math.nan
        return float(np.corrcoef(left, right)[0, 1])

    # ------------------------------------------------------------------ #
    # Causal transforms                                                  #
    # ------------------------------------------------------------------ #

    def diff(self) -> TimeSeries:
        """First differences (changes), dated at the later point of each pair."""
        return TimeSeries(self.dates[1:], np.diff(self.values))

    def trailing_mean(self, window: int, min_periods: int | None = None) -> TimeSeries:
        """Positional trailing mean over the last ``window`` points, current included.

        ``rolling(window, min_periods).mean()`` semantics: points with fewer than
        ``min_periods`` observations in the window are dropped from the result.
        """
        if window <= 0:
            msg = f"window must be positive, got {window}"
            raise ValueError(msg)
        minimum = window if min_periods is None else min_periods
        dates: list[dt.date] = []
        out: list[float] = []
        cumsum = np.concatenate([[0.0], np.cumsum(self.values)])
        for i in range(len(self)):
            start = max(0, i + 1 - window)
            count = i + 1 - start
            if count < minimum:
                continue
            dates.append(self.dates[i])
            out.append(float((cumsum[i + 1] - cumsum[start]) / count))
        return TimeSeries(tuple(dates), np.asarray(out, dtype=np.float64))

    def trailing_mean_days(self, window_days: int, min_days: int = 1) -> TimeSeries:
        """Date-aware trailing mean over observed points within ``window_days``.

        The window is ``(date - window_days, date]``. Only observed points are
        averaged; call :meth:`fill_daily` first to make missing days count as zero.
        """
        if window_days <= 0:
            msg = f"window_days must be positive, got {window_days}"
            raise ValueError(msg)
        ordinals = np.asarray([d.toordinal() for d in self.dates], dtype=np.int64)
        cumsum = np.concatenate([[0.0], np.cumsum(self.values)])
        starts = np.searchsorted(ordinals, ordinals - window_days, side="right")
        dates: list[dt.date] = []
        out: list[float] = []
        for i in range(len(self)):
            count = i + 1 - int(starts[i])
            if count < min_days:
                continue
            dates.append(self.dates[i])
            out.append(float((cumsum[i + 1] - cumsum[starts[i]]) / count))
        return TimeSeries(tuple(dates), np.asarray(out, dtype=np.float64))

    def fill_daily(self, fill: float = 0.0) -> TimeSeries:
        """Expand to a continuous daily range from first to last date."""
        if not self.dates:
            return self
        mapping = self.to_mapping()
        start, end = self.dates[0], self.dates[-1]
        days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
        return TimeSeries(
            tuple(days),
            np.asarray([mapping.get(d, fill) for d in days], dtype=np.float64),
        )

    # ------------------------------------------------------------------ #
    # Resampling                                                         #
    # ------------------------------------------------------------------ #

    def _resample(self, key: Callable[[dt.date], dt.date], how: str) -> TimeSeries:
        buckets: dict[dt.date, list[float]] = {}
        for date, value in zip(self.dates, self.values, strict=True):
            buckets.setdefault(key(date), []).append(float(value))
        if how == "sum":
            reduced = {k: float(sum(v)) for k, v in buckets.items()}
        elif how == "mean":
            reduced = {k: float(sum(v) / len(v)) for k, v in buckets.items()}
        else:  # pragma: no cover - guarded by the callers
            msg = f"unknown resample mode {how!r}"
            raise ValueError(msg)
        return TimeSeries.from_mapping(reduced)

    def resample_monthly(self, how: str = "sum") -> TimeSeries:
        """Aggregate to months, keyed by the first day of each month."""
        return self._resample(lambda d: d.replace(day=1), how)

    def resample_quarterly(self, how: str = "sum") -> TimeSeries:
        """Aggregate to quarters, keyed by the first day of each quarter."""
        return self._resample(lambda d: d.replace(month=3 * ((d.month - 1) // 3) + 1, day=1), how)

    # ------------------------------------------------------------------ #
    # Chart norms                                                        #
    # ------------------------------------------------------------------ #

    def _window_mean(self, window: tuple[dt.date, dt.date] | None) -> float:
        if window is None:
            selected = self.values
        else:
            start, end = window
            mask = np.asarray([start <= d <= end for d in self.dates], dtype=bool)
            selected = self.values[mask]
        if selected.size == 0:
            return math.nan
        return float(np.mean(selected))

    def rebase_to_own_mean(self, window: tuple[dt.date, dt.date] | None = None) -> TimeSeries:
        """Divide by the series' own mean (over ``window`` when given)."""
        mean = self._window_mean(window)
        if not math.isfinite(mean) or mean == 0.0:
            return TimeSeries(self.dates, np.full(len(self), math.nan))
        return TimeSeries(self.dates, self.values / mean)

    def rescale_to_mean_100(self, window: tuple[dt.date, dt.date] | None = None) -> TimeSeries:
        """Rescale so the series averages 100 over ``window`` (the posts' chart norm)."""
        rebased = self.rebase_to_own_mean(window)
        return TimeSeries(rebased.dates, rebased.values * 100.0)

    def zscore(self) -> TimeSeries:
        """Full-series z-score (the per-region chart norm; not causal — charts only)."""
        std = float(np.std(self.values))
        if std == 0.0:
            return TimeSeries(self.dates, np.zeros(len(self)))
        return TimeSeries(self.dates, (self.values - float(np.mean(self.values))) / std)

    def divide(self, denominator: TimeSeries) -> TimeSeries:
        """Element-wise ratio on common dates; zero denominators drop the date."""
        common = [
            d
            for d in self.dates
            if d in set(denominator.dates)
        ]
        num, den = self.to_mapping(), denominator.to_mapping()
        kept = [d for d in common if den[d] != 0.0]
        return TimeSeries(
            tuple(kept),
            np.asarray([num[d] / den[d] for d in kept], dtype=np.float64),
        )


def daily_sum(
    dates: Sequence[dt.date | None],
    weights: Iterable[float],
    mask: Sequence[bool] | npt.NDArray[np.bool_] | None = None,
) -> TimeSeries:
    """Sum ``weights`` per calendar day; undated rows abstain. ``mask`` selects rows."""
    totals: dict[dt.date, float] = {}
    for index, (date, weight) in enumerate(zip(dates, weights, strict=True)):
        if date is None:
            continue
        if mask is not None and not bool(mask[index]):
            continue
        totals[date] = totals.get(date, 0.0) + float(weight)
    return TimeSeries.from_mapping(totals)


def daily_ratio(numerator: TimeSeries, denominator: TimeSeries) -> TimeSeries:
    """Convenience alias for :meth:`TimeSeries.divide` reading as the posts' fractions."""
    return numerator.divide(denominator)
