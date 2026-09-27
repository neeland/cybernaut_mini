"""The state-dependent asymmetric EWMA smoother for net-polarity series.

A raw daily net-polarity line is noisy. The posts overlay a smoother that reacts
instantly to shocks toward uncertainty and relaxes slowly otherwise: when today's
change is a sharp uptick — more than ``k`` standard deviations above the trailing
year of daily changes, computed strictly before today — the EWMA switches to a
fast 3-day half-life; on calm days and on moves toward resolution it runs a slow
30-day half-life.

Blog ref: https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty
    — "When polarity jumps sharply toward uncertainty, 1.5 standard deviations
    above its trailing year with no look-ahead, the smoother switches to a fast
    3-day half-life and picks up the shock at once. On calm days, and on moves
    toward resolution, it runs a slow 30-day half-life. The asymmetry is there to
    capture the spikes quickly." Local copy:
    ``docs/blog-archive/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty.md``.

Assumptions:
    - "Trailing year with no look-ahead" means the mean and standard deviation of
      the daily *changes* over the 365 calendar days strictly before the current
      point (pandas ``diff().shift(1).rolling('365D')`` semantics). Including
      today's change in its own trigger statistics would be a one-day look-ahead.
    - Half-life ``h`` maps to ``alpha = 1 - 0.5**(1/h)`` — the standard EWMA
      half-life parameterization.
    - Until the trigger window holds at least ``min_history`` changes (default
      30), the smoother stays in the slow regime: with no distribution to compare
      against, "sharp" is undefined and the conservative regime is the honest one.
    - "Toward uncertainty" is the positive direction of the series (net polarity
      is scored -1 resolved .. +1 uncertain), so only positive upticks can trip
      the fast regime — that *is* the asymmetry.

Alternatives rejected:
    - A symmetric two-sided trigger (|change| > k sigma): explicitly not what the
      post describes; downward resolution moves are meant to decay slowly.
    - Z-scoring against trailing *levels* instead of changes: a persistent high
      plateau would keep the fast regime latched on; the post triggers on jumps.
    - Applying the fast alpha for a fixed cooldown window after a trigger: extra
      state and a parameter the post does not have — the regime is re-decided
      from the trigger statistic at every step.
"""

from __future__ import annotations

import numpy as np

from cybernaut_mini.world.timeseries import TimeSeries

__all__ = [
    "asymmetric_ewma",
    "half_life_alpha",
]


def half_life_alpha(half_life: float) -> float:
    """EWMA alpha for a half-life in observations: ``1 - 0.5**(1/h)``."""
    if half_life <= 0:
        msg = f"half-life must be positive, got {half_life}"
        raise ValueError(msg)
    return float(1.0 - 0.5 ** (1.0 / half_life))


def asymmetric_ewma(
    series: TimeSeries,
    *,
    fast_half_life: float = 3.0,
    slow_half_life: float = 30.0,
    trailing_days: int = 365,
    sigma_threshold: float = 1.5,
    min_history: int = 30,
) -> TimeSeries:
    """State-dependent EWMA: fast on sharp upticks toward uncertainty, slow otherwise.

    For each point after the first, the day's change is compared against the mean
    and standard deviation of the changes in the ``trailing_days`` window strictly
    before it (no look-ahead). ``change > mean + sigma_threshold * std`` selects
    the fast half-life for that step; everything else — calm days, downticks, and
    the warm-up before ``min_history`` changes exist — uses the slow half-life.
    """
    if len(series) == 0:
        return series
    alpha_fast = half_life_alpha(fast_half_life)
    alpha_slow = half_life_alpha(slow_half_life)

    values = series.values
    ordinals = np.asarray([d.toordinal() for d in series.dates], dtype=np.int64)
    changes = np.diff(values)
    change_days = ordinals[1:]  # change i is dated at the later point of its pair

    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        # Changes strictly before this point, within the trailing calendar window.
        lo = int(np.searchsorted(change_days, ordinals[i] - trailing_days, side="right"))
        window = changes[lo : i - 1]
        alpha = alpha_slow
        if window.size >= min_history:
            mean = float(np.mean(window))
            std = float(np.std(window))
            # With a perfectly flat history (std 0) any genuine uptick is
            # infinitely many sigmas above the mean, so the comparison is kept
            # with std = 0 rather than special-cased away.
            if float(changes[i - 1]) > mean + sigma_threshold * std:
                alpha = alpha_fast
        smoothed[i] = smoothed[i - 1] + alpha * (values[i] - smoothed[i - 1])
    return TimeSeries(series.dates, smoothed)
