"""The NOSIBLE-GPR family: attention shares over the event table, one group-by each.

The geopolitical risk index is the share of publisher attention spent on
geopolitical events. This module computes every published cut as pure pandas-free
group-bys over the event store: the global daily share (raw and detrended), the
per-country and bilateral-pair monthly indices over one shared global denominator
``B(m)``, the trade-coercion OR-patch, and the relevance-weighted Oil-GPR AND-gate
with its producer-region breakdown.

Blog ref: https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world
    — ``NOSIBLE-GPR(t) = sum of breadth(e) over geopolitical events on day t /
    sum of breadth(e) over all events on day t``; "The second divides by a
    trailing 12-month average of total attention"; ``B(m)`` "is one global
    denominator shared by every country: the trailing 12-month average of total
    monthly breadth across all events. We divide by this global figure, not by a
    country's own coverage, because in a crisis a country's own coverage spikes
    too and would cancel the signal"; the pair index "self-join each event's
    attribution set into unordered pairs"; the trade patch ``geopolitical(e) OR
    trade(e) >= 0.40`` ("Nothing else changed ... which is why the conflict
    countries do not move"); and ``Oil-GPR(m) = sum of relevance(e) x breadth(e)
    over events ... geopolitical AND relevance >= floor / B(m)``. Local copy:
    ``docs/blog-archive/rebuilding-the-geopolitical-risk-index-from-nosible-world.md``.

Assumptions:
    - ``B(m)`` is ``total monthly breadth -> trailing_mean(12, min_periods=6)``:
      twelve months inclusive of the current one, published once at least six
      months exist — early months of a corpus have no meaningful baseline. It is
      computed over *all* events once and shared by every country, pair, and oil
      series; :func:`per_country_denominator` exists only as the ablation showing
      why a per-country denominator self-cancels crisis spikes.
    - The daily detrended denominator is the trailing-365-day mean of total daily
      breadth over a zero-filled calendar (a day with no events is a measured
      zero of attention, not a missing observation) with a 30-day minimum warm-up.
    - Masks and scores arrive row-aligned from :mod:`cybernaut_mini.world.topics`
      and :mod:`cybernaut_mini.world.anchors`; this module never re-derives them,
      so an ablation rerun swaps one mask and touches nothing else.
    - Producer regions are a committed mapping of alpha-2 codes; an oil event
      lands in a region when its attribution set intersects it. The mapping is a
      laptop stand-in for the paper's region tags.

Alternatives rejected:
    - Per-country denominators (a country's own trailing coverage): the post's
      own anti-example — kept available as the ablation, never the default.
    - Deduplicating pair rows by weighting each pair by 1/len(pairs): the post
      self-joins attribution sets; an event naming three countries genuinely
      contributes full breadth to each of its three pairs.
    - Precomputing a dense country x month matrix: dict-of-TimeSeries keeps the
      fixture-scale output inspectable and the big-corpus path identical.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from cybernaut_mini.world import countries
from cybernaut_mini.world.events import WorldEvent, event_breadths, event_dates
from cybernaut_mini.world.timeseries import TimeSeries, daily_sum

__all__ = [
    "OIL_PRODUCER_REGIONS",
    "gpr_country_monthly",
    "gpr_daily",
    "gpr_pairs_monthly",
    "monthly_breadth_denominator",
    "oil_gpr",
    "oil_gpr_by_region",
    "per_country_denominator",
    "trade_patched_mask",
]

BoolArray = npt.NDArray[np.bool_]
FloatVec = npt.NDArray[np.float64]

#: Producer-region mapping for the Oil-GPR breakdown (alpha-2 codes).
OIL_PRODUCER_REGIONS: dict[str, tuple[str, ...]] = {
    "middle_east": ("SA", "IR", "IQ", "AE", "KW", "QA", "OM", "BH", "YE", "SY", "IL"),
    "north_america": ("US", "CA", "MX"),
    "russia_caspian": ("RU", "KZ", "AZ", "TM", "UZ"),
    "africa": ("NG", "DZ", "LY", "AO", "EG", "GA", "GH", "SD", "SS"),
    "latin_america": ("VE", "BR", "CO", "EC", "AR", "GY", "TT"),
    "asia_pacific": ("CN", "IN", "ID", "MY", "AU", "VN", "BN"),
    "europe": ("NO", "GB", "NL", "DK", "RO"),
}

#: Trailing-12-month window (months) and its publication minimum for ``B(m)``.
_B_WINDOW_MONTHS = 12
_B_MIN_MONTHS = 6

#: Daily detrend window (calendar days) and warm-up for the daily denominator.
_DAILY_WINDOW_DAYS = 365
_DAILY_MIN_DAYS = 30


def monthly_breadth_denominator(events: Sequence[WorldEvent]) -> TimeSeries:
    """``B(m)``: trailing 12-month average of total monthly breadth, all events.

    One global series shared by every country, pair, and oil index — never
    per-country (see :func:`per_country_denominator` for the ablation).
    """
    monthly = daily_sum(event_dates(events), event_breadths(events)).resample_monthly("sum")
    return monthly.trailing_mean(_B_WINDOW_MONTHS, min_periods=_B_MIN_MONTHS)


def per_country_denominator(events: Sequence[WorldEvent], country: str, *, cap: int = 15
                            ) -> TimeSeries:
    """The rejected per-country denominator, kept as the self-cancellation ablation.

    Dividing a country's geopolitical breadth by the same country's total breadth
    cancels the crisis spike (its own coverage spikes too); the post ships this
    comparison to justify the global ``B(m)``.
    """
    member = np.asarray(
        [country in attribution for attribution in countries.attributions(events, cap=cap)],
        dtype=bool,
    )
    monthly = daily_sum(
        event_dates(events), event_breadths(events), mask=member
    ).resample_monthly("sum")
    return monthly.trailing_mean(_B_WINDOW_MONTHS, min_periods=_B_MIN_MONTHS)


def gpr_daily(
    events: Sequence[WorldEvent],
    geopolitical: BoolArray,
    *,
    detrend: bool = True,
) -> TimeSeries:
    """The global daily index: geopolitical attention share, detrended by default.

    ``detrend=False`` divides by total attention the same day (the raw share);
    ``detrend=True`` divides by the trailing-365-day mean of total daily breadth,
    the version the post carries "for every country, every pair, and oil" because
    a growing corpus sags the raw share.
    """
    dates = event_dates(events)
    breadths = event_breadths(events)
    numerator = daily_sum(dates, breadths, mask=geopolitical)
    total = daily_sum(dates, breadths)
    if not detrend:
        return numerator.divide(total)
    baseline = total.fill_daily(0.0).trailing_mean_days(_DAILY_WINDOW_DAYS, _DAILY_MIN_DAYS)
    return numerator.divide(baseline)


def _monthly_masked(
    events: Sequence[WorldEvent], mask: BoolArray, weights: FloatVec | None = None
) -> TimeSeries:
    values = event_breadths(events) if weights is None else weights
    return daily_sum(event_dates(events), values, mask=mask).resample_monthly("sum")


def gpr_country_monthly(
    events: Sequence[WorldEvent],
    included: BoolArray,
    *,
    cap: int = 15,
    denominator: TimeSeries | None = None,
) -> dict[str, TimeSeries]:
    """Per-country monthly index: explode attribution, group by (country, month), / B(m).

    ``included`` is the inclusion mask (plain ``geopolitical_mask``, or the
    trade-patched OR mask). ``denominator`` defaults to the shared global
    :func:`monthly_breadth_denominator`.
    """
    baseline = denominator if denominator is not None else monthly_breadth_denominator(events)
    attribution_sets = countries.attributions(events, cap=cap)
    per_country: dict[str, TimeSeries] = {}
    codes = sorted({code for attribution in attribution_sets for code in attribution})
    for code in codes:
        member = np.asarray(
            [code in attribution for attribution in attribution_sets], dtype=bool
        )
        numerator = _monthly_masked(events, np.logical_and(included, member))
        series = numerator.divide(baseline)
        if len(series):
            per_country[code] = series
    return per_country


def gpr_pairs_monthly(
    events: Sequence[WorldEvent],
    included: BoolArray,
    *,
    cap: int = 15,
    denominator: TimeSeries | None = None,
) -> dict[tuple[str, str], TimeSeries]:
    """Bilateral index: every unordered country pair an event names, over B(m)."""
    baseline = denominator if denominator is not None else monthly_breadth_denominator(events)
    attribution_sets = countries.attributions(events, cap=cap)
    pair_rows: dict[tuple[str, str], list[int]] = {}
    for index, attribution in enumerate(attribution_sets):
        if not included[index]:
            continue
        for pair in itertools.combinations(sorted(attribution), 2):
            pair_rows.setdefault(pair, []).append(index)
    per_pair: dict[tuple[str, str], TimeSeries] = {}
    for pair, row_indices in sorted(pair_rows.items()):
        member = np.zeros(len(events), dtype=bool)
        member[np.asarray(row_indices, dtype=np.int64)] = True
        series = _monthly_masked(events, member).divide(baseline)
        if len(series):
            per_pair[pair] = series
    return per_pair


def trade_patched_mask(
    geopolitical: BoolArray, trade_scores: FloatVec, *, floor: float = 0.40
) -> BoolArray:
    """The OR-patch: ``geopolitical(e) OR trade(e) >= floor``.

    Everything else about the country/pair indices stays identical, which is why
    conflict-country series are bit-identical before and after — the invariance
    the tests assert.
    """
    return np.asarray(
        np.logical_or(geopolitical, np.asarray(trade_scores, dtype=np.float64) >= floor),
        dtype=bool,
    )


def oil_gpr(
    events: Sequence[WorldEvent],
    geopolitical: BoolArray,
    oil_relevance: FloatVec,
    *,
    floor: float = 0.30,
    denominator: TimeSeries | None = None,
) -> TimeSeries:
    """``Oil-GPR(m)``: relevance-weighted breadth over the geopolitical AND-gate, / B(m).

    Keeps events that are both geopolitical and at/above the oil-supply relevance
    floor, weights each by ``relevance(e) * breadth(e)``, and divides by the same
    global ``B(m)``.
    """
    relevance = np.asarray(oil_relevance, dtype=np.float64)
    mask = np.asarray(np.logical_and(geopolitical, relevance >= floor), dtype=bool)
    weights = relevance * event_breadths(events)
    baseline = denominator if denominator is not None else monthly_breadth_denominator(events)
    return _monthly_masked(events, mask, weights).divide(baseline)


def oil_gpr_by_region(
    events: Sequence[WorldEvent],
    geopolitical: BoolArray,
    oil_relevance: FloatVec,
    *,
    floor: float = 0.30,
    regions: Mapping[str, tuple[str, ...]] | None = None,
    cap: int = 15,
    denominator: TimeSeries | None = None,
) -> dict[str, TimeSeries]:
    """Per-producer-region Oil-GPR: attribute each surviving event to its regions."""
    region_map = dict(regions) if regions is not None else OIL_PRODUCER_REGIONS
    relevance = np.asarray(oil_relevance, dtype=np.float64)
    survives = np.asarray(np.logical_and(geopolitical, relevance >= floor), dtype=bool)
    weights = relevance * event_breadths(events)
    baseline = denominator if denominator is not None else monthly_breadth_denominator(events)
    attribution_sets = countries.attributions(events, cap=cap)
    out: dict[str, TimeSeries] = {}
    for region, codes in region_map.items():
        code_set = frozenset(codes)
        member = np.asarray(
            [bool(code_set.intersection(attribution)) for attribution in attribution_sets],
            dtype=bool,
        )
        series = _monthly_masked(events, np.logical_and(survives, member), weights)
        divided = series.divide(baseline)
        if len(divided):
            out[region] = divided
    return out
