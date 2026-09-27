"""The NOSIBLE-TPU/EPU family: uncertainty-weighted attention over the anchor engine.

TPU is the daily share of publisher attention on trade-policy events, tilted by
the ``w_unc`` uncertainty weight; EPU is the same recipe monthly, restricted to
US-attributed events on both lines to match the published index's own
normalization; categories add the 0.25 secondary gate; and the signed
net-polarity series plus the explain-month diagnostic make every reading
inspectable event by event.

Blog ref: https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty
    — ``NOSIBLE-TPU(t) = sum of breadth(e) * w_unc(e) over relevant events on day
    t / trailing 12-month average of total daily breadth``; ``NOSIBLE-EPU(m) =
    sum of breadth(e) * w_unc(e) over relevant US events in month t / trailing
    12-month average of total US-attributed monthly breadth`` ("This index is
    US-only on both lines ... We scale US attention by US attention"); "A
    national-security or healthcare event must also sit close enough (0.25) to
    that category's own sentence pairs"; ``net_polarity(t) = sum of breadth(e) *
    polarity(e) over relevant events on day t / sum of breadth(e) over relevant
    events on day t``; and the lever ablation ("The cyan line drops the shutdown
    lever ... The whole COVID gap comes down to that one lever"). Local copy:
    ``docs/blog-archive/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty.md``.

Assumptions:
    - Anchor scores arrive precomputed (:func:`cybernaut_mini.world.anchors.score_events`)
      and row-aligned with the events; a leave-one-lever-out rerun re-embeds a
      smaller pair list and rescores — seconds over a cached event matrix — which
      is why the ablation is a parameter (``levers=``) upstream, not logic here.
    - "Trailing 12-month average of total daily breadth" is the trailing-365-day
      mean over a zero-filled daily calendar with a 30-day warm-up (same
      convention as the GPR module); the monthly EPU denominator is the trailing
      12-month mean of monthly US breadth published from 6 months.
    - US attribution = the event's resolved main country is ``US`` — the tagged
      primary country, exactly the post's rule. On CC-News this attribution layer
      is the replica's weakest link (heuristic NER feeding a strict resolver),
      which the EPU docstring and README say out loud rather than hide.
    - ``explain_month`` returns a plain dict (top events by contribution, the
      near-misses just below the relevance floor, optional residual vs a
      benchmark) so the CLI wiring can print it without this module importing
      any CLI machinery.

Alternatives rejected:
    - Restricting only the numerator to US events: the post is explicit that both
      lines are US-only ("the way the published index scales US articles by US
      articles"); a global denominator would re-introduce corpus-mix drift.
    - A category gate on the category pairs alone (no headline relevance AND):
      the paper's category series are subsets of the headline index, and the post
      words the rule as "must *also* sit close enough".
    - Storing per-event contributions on the events: contributions depend on the
      anchor set and thresholds in use; they belong to a scoring run, not to the
      point-in-time event record.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from cybernaut_mini.world import countries
from cybernaut_mini.world.anchors import AnchorScores
from cybernaut_mini.world.events import WorldEvent, event_breadths, event_dates
from cybernaut_mini.world.timeseries import TimeSeries, daily_sum

__all__ = [
    "epu_monthly",
    "explain_month",
    "net_polarity",
    "tpu_daily",
    "us_attributed_mask",
]

BoolArray = npt.NDArray[np.bool_]

_WINDOW_DAYS = 365
_MIN_DAYS = 30
_WINDOW_MONTHS = 12
_MIN_MONTHS = 6


def us_attributed_mask(events: Sequence[WorldEvent]) -> BoolArray:
    """US-attributed = the event's tagged primary country resolves to ``US``."""
    return np.asarray(
        [
            (countries.resolve(event.event.country) == "US")
            if event.event.country
            else False
            for event in events
        ],
        dtype=bool,
    )


def tpu_daily(
    events: Sequence[WorldEvent],
    scores: AnchorScores,
    *,
    floor: float | None = None,
    relevant: BoolArray | None = None,
) -> TimeSeries:
    """``NOSIBLE-TPU(t)``: breadth * w_unc over relevant events / trailing breadth.

    ``relevant`` defaults to ``scores.relevance >= floor`` (pass the anchor set's
    ``relevance_floor``); supply a mask directly to compose extra gates.
    """
    if relevant is None:
        if floor is None:
            msg = "pass either floor= or a precomputed relevant= mask"
            raise ValueError(msg)
        relevant = scores.relevant_mask(floor)
    dates = event_dates(events)
    breadths = event_breadths(events)
    numerator = daily_sum(dates, breadths * scores.w_unc, mask=relevant)
    baseline = (
        daily_sum(dates, breadths).fill_daily(0.0).trailing_mean_days(_WINDOW_DAYS, _MIN_DAYS)
    )
    return numerator.divide(baseline)


def epu_monthly(
    events: Sequence[WorldEvent],
    scores: AnchorScores,
    *,
    floor: float,
    us_mask: BoolArray | None = None,
    category: BoolArray | None = None,
) -> TimeSeries:
    """``NOSIBLE-EPU(m)``: US-over-US, numerator and denominator on the same filter.

    ``category`` (optional) ANDs the 0.25 secondary gate from
    :func:`cybernaut_mini.world.anchors.category_mask` into the numerator — the
    national-security / healthcare sub-indices.
    """
    us = us_attributed_mask(events) if us_mask is None else np.asarray(us_mask, dtype=bool)
    included = np.logical_and(scores.relevant_mask(floor), us)
    if category is not None:
        included = np.logical_and(included, np.asarray(category, dtype=bool))
    dates = event_dates(events)
    breadths = event_breadths(events)
    numerator = daily_sum(dates, breadths * scores.w_unc, mask=included).resample_monthly("sum")
    denominator = (
        daily_sum(dates, breadths, mask=us)
        .resample_monthly("sum")
        .trailing_mean(_WINDOW_MONTHS, min_periods=_MIN_MONTHS)
    )
    return numerator.divide(denominator)


def net_polarity(
    events: Sequence[WorldEvent],
    scores: AnchorScores,
    *,
    floor: float,
) -> TimeSeries:
    """``net_polarity(t)``: breadth-weighted mean polarity of the day's relevant events.

    -1 fully resolved .. +1 fully uncertain; overlay
    :func:`cybernaut_mini.world.smoother.asymmetric_ewma` for the published chart.
    """
    relevant = scores.relevant_mask(floor)
    dates = event_dates(events)
    breadths = event_breadths(events)
    numerator = daily_sum(dates, breadths * scores.polarity, mask=relevant)
    denominator = daily_sum(dates, breadths, mask=relevant)
    return numerator.divide(denominator)


def _in_month(date: dt.date | None, month: dt.date) -> bool:
    return date is not None and date.year == month.year and date.month == month.month


def explain_month(
    events: Sequence[WorldEvent],
    scores: AnchorScores,
    month: dt.date,
    *,
    floor: float,
    top_n: int = 20,
    near_miss_band: float = 0.05,
    replica: TimeSeries | None = None,
    benchmark: TimeSeries | None = None,
) -> dict[str, Any]:
    """The decisions-vs-conditions diagnostic, made inspectable.

    Returns the month's top ``top_n`` counted events by ``breadth * w_unc``
    contribution, the top ``top_n`` events sitting just below the relevance floor
    (within ``near_miss_band``) — the conditions coverage a decisions index
    excludes — and, when both series are given, the replica-vs-benchmark residual
    on their mean-rebased levels for that month.
    """
    month = month.replace(day=1)
    breadths = event_breadths(events)
    rows: list[dict[str, Any]] = [
        {
            "event_id": event.event_id,
            "date": event.event.date.isoformat() if event.event.date else None,
            "title": event.event.title,
            "breadth": int(event.breadth),
            "relevance": float(scores.relevance[index]),
            "nearest_anchor": scores.nearest_anchor[index],
            "polarity": float(scores.polarity[index]),
            "w_unc": float(scores.w_unc[index]),
            "contribution": float(breadths[index] * scores.w_unc[index]),
        }
        for index, event in enumerate(events)
        if _in_month(event.event.date, month)
    ]
    counted = sorted(
        (row for row in rows if float(row["relevance"]) >= floor),
        key=lambda row: (-float(row["contribution"]), str(row["event_id"])),
    )
    near_misses = sorted(
        (row for row in rows if floor - near_miss_band <= float(row["relevance"]) < floor),
        key=lambda row: (-float(row["relevance"]), str(row["event_id"])),
    )
    residual: float | None = None
    if replica is not None and benchmark is not None:
        left = replica.rebase_to_own_mean().to_mapping().get(month)
        right = benchmark.rebase_to_own_mean().to_mapping().get(month)
        if left is not None and right is not None:
            residual = float(left - right)
    return {
        "month": month.isoformat(),
        "relevance_floor": float(floor),
        "residual_vs_benchmark": residual,
        "top_events": counted[:top_n],
        "just_below_floor": near_misses[:top_n],
        "counted_events": len(counted),
        "events_in_month": len(rows),
    }
