"""GPR and TPU/EPU index formulas over the real fixture event table.

Every formula is re-derived independently in the test (manual dict group-bys over
the same real events) and compared against the module's output — the masks and
scores that select events are numeric arrays chosen by the test (pure math), the
events themselves are the real fixture table. The design invariants the posts
insist on are pinned: ``B(m)`` is global (the per-country ablation self-cancels),
the trade OR-patch leaves untouched-country series bit-identical, Oil-GPR is an
AND-gate weighted by relevance x breadth, EPU is US-over-US on both lines, and
net polarity is the breadth-weighted mean.

Blog ref: https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world
    and https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty
    — the exact fraction definitions quoted in the modules under test. Local
    copies under ``docs/blog-archive/``.

Assumptions: the fixture slice spans four dated days across two months, so
monthly denominators use explicit test-supplied baselines (``min_periods`` at
publication scale would blank a two-month window — an answer, not a bug); the
daily formulas are asserted with the raw same-day denominator where the
arithmetic is exactly checkable.

Alternatives rejected: fabricating a longer synthetic event history (forbidden);
snapshotting index values (opaque — re-derivation is the proof).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from cybernaut_mini.world import countries
from cybernaut_mini.world.anchors import AnchorScores
from cybernaut_mini.world.events import event_breadths, event_dates
from cybernaut_mini.world.indices import gpr, policy
from cybernaut_mini.world.timeseries import daily_sum
from world_helpers import world_events


def _scores(relevance: np.ndarray, polarity: np.ndarray) -> AnchorScores:
    return AnchorScores(
        relevance=relevance.astype(np.float64),
        nearest_anchor=("",) * len(relevance),
        polarity=polarity.astype(np.float64),
        w_unc=(1.0 + polarity.astype(np.float64)) / 2.0,
        best_pair=("",) * len(relevance),
    )


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ------------------------------------------------------------------ #
# GPR family                                                         #
# ------------------------------------------------------------------ #


def test_gpr_daily_raw_share_matches_manual_group_by(rng: np.random.Generator) -> None:
    events = list(world_events())
    mask = rng.random(len(events)) < 0.3
    series = gpr.gpr_daily(events, mask, detrend=False)
    expected: dict[dt.date, list[float]] = {}
    for event, included in zip(events, mask, strict=True):
        if event.event.date is None:
            continue
        num, den = expected.setdefault(event.event.date, [0.0, 0.0])
        expected[event.event.date] = [num + (event.breadth if included else 0.0),
                                      den + event.breadth]
    for date, value in series.to_mapping().items():
        num, den = expected[date]
        assert value == pytest.approx(num / den)


def test_gpr_detrended_uses_trailing_denominator(rng: np.random.Generator) -> None:
    events = list(world_events())
    mask = np.ones(len(events), dtype=bool)
    detrended = gpr.gpr_daily(events, mask, detrend=True)
    raw = gpr.gpr_daily(events, mask, detrend=False)
    # With everything included the raw share is identically 1; the detrended
    # series is the day's breadth over its own trailing-year mean, which is not.
    assert np.allclose(raw.values, 1.0)
    assert len(detrended) > 0
    assert not np.allclose(detrended.values, 1.0)


def test_country_index_explodes_attribution_over_global_denominator() -> None:
    events = list(world_events())
    included = np.ones(len(events), dtype=bool)
    monthly_total = daily_sum(event_dates(events), event_breadths(events)).resample_monthly("sum")
    baseline = monthly_total.trailing_mean(12, min_periods=1)
    per_country = gpr.gpr_country_monthly(events, included, denominator=baseline)
    assert per_country, "the real slice attributes events to countries"
    attribution_sets = countries.attributions(events)
    for code, series in per_country.items():
        expected: dict[dt.date, float] = {}
        for event, attribution in zip(events, attribution_sets, strict=True):
            if code in attribution and event.event.date is not None:
                month = event.event.date.replace(day=1)
                expected[month] = expected.get(month, 0.0) + event.breadth
        base = baseline.to_mapping()
        for month, value in series.to_mapping().items():
            assert value == pytest.approx(expected[month] / base[month])


def test_pair_index_is_unordered_combinations() -> None:
    events = list(world_events())
    included = np.ones(len(events), dtype=bool)
    baseline = (
        daily_sum(event_dates(events), event_breadths(events))
        .resample_monthly("sum")
        .trailing_mean(12, min_periods=1)
    )
    pairs = gpr.gpr_pairs_monthly(events, included, denominator=baseline)
    assert pairs, "some real event names at least two countries"
    for (left, right), series in pairs.items():
        assert left < right  # unordered, stored sorted
        assert len(series) > 0
    # A pair can never out-sum its members' country series.
    per_country = gpr.gpr_country_monthly(events, included, denominator=baseline)
    for (left, right), series in pairs.items():
        for month, value in series.to_mapping().items():
            assert value <= per_country[left].to_mapping()[month] + 1e-9
            assert value <= per_country[right].to_mapping()[month] + 1e-9


def test_trade_patch_is_or_gate_and_leaves_conflict_series_bit_identical(
    rng: np.random.Generator,
) -> None:
    events = list(world_events())
    geo = rng.random(len(events)) < 0.25
    trade_scores = rng.random(len(events)) * 0.39  # everything below the 0.40 floor
    patched = gpr.trade_patched_mask(geo, trade_scores, floor=0.40)
    assert np.array_equal(patched, geo)  # no event clears the floor: nothing changes
    trade_scores[0] = 0.95
    patched = gpr.trade_patched_mask(geo, trade_scores, floor=0.40)
    assert patched[0]
    assert np.array_equal(patched[1:], geo[1:])
    # Invariance: countries untouched by the newly-admitted events keep
    # bit-identical series (the post: "the conflict countries do not move").
    baseline = (
        daily_sum(event_dates(events), event_breadths(events))
        .resample_monthly("sum")
        .trailing_mean(12, min_periods=1)
    )
    before = gpr.gpr_country_monthly(events, geo, denominator=baseline)
    after = gpr.gpr_country_monthly(events, patched, denominator=baseline)
    admitted = countries.attribution(events[0])
    for code, series in before.items():
        if code in admitted:
            continue
        assert np.array_equal(series.values, after[code].values)
        assert series.dates == after[code].dates


def test_oil_gpr_is_relevance_weighted_and_gate(rng: np.random.Generator) -> None:
    events = list(world_events())
    geo = rng.random(len(events)) < 0.5
    relevance = rng.random(len(events))
    baseline = (
        daily_sum(event_dates(events), event_breadths(events))
        .resample_monthly("sum")
        .trailing_mean(12, min_periods=1)
    )
    series = gpr.oil_gpr(events, geo, relevance, floor=0.30, denominator=baseline)
    expected: dict[dt.date, float] = {}
    for index, event in enumerate(events):
        if event.event.date is None or not geo[index] or relevance[index] < 0.30:
            continue
        month = event.event.date.replace(day=1)
        expected[month] = expected.get(month, 0.0) + relevance[index] * event.breadth
    base = baseline.to_mapping()
    assert series.to_mapping() == pytest.approx(
        {month: value / base[month] for month, value in expected.items()}
    )


def test_per_country_denominator_self_cancels() -> None:
    """The post's ablation: dividing a country by itself flattens its spike."""
    events = list(world_events())
    per_country = countries.attributions(events)
    code = max(
        {c for a in per_country for c in a},
        key=lambda c: sum(1 for a in per_country if c in a),
    )
    member = np.asarray([code in a for a in per_country], dtype=bool)
    own = gpr.per_country_denominator(events, code)
    # Numerator: ALL of the country's events counted as geopolitical.
    numerator = daily_sum(event_dates(events), event_breadths(events), mask=member)
    monthly = numerator.resample_monthly("sum")
    own_index = monthly.divide(own.trailing_mean(1, min_periods=1))
    # When a country's whole coverage is the "crisis", own-denominator = 1 flat:
    # the spike has cancelled itself.
    assert np.allclose(own_index.values, 1.0)


def test_oil_regions_partition_uses_attribution_intersection(
    rng: np.random.Generator,
) -> None:
    events = list(world_events())
    geo = np.ones(len(events), dtype=bool)
    relevance = np.ones(len(events), dtype=np.float64)
    baseline = (
        daily_sum(event_dates(events), event_breadths(events))
        .resample_monthly("sum")
        .trailing_mean(12, min_periods=1)
    )
    regions = {"us_only": ("US",), "cn_only": ("CN",)}
    by_region = gpr.oil_gpr_by_region(
        events, geo, relevance, floor=0.0, regions=regions, denominator=baseline
    )
    attribution_sets = countries.attributions(events)
    for region, codes in regions.items():
        rows = [
            event
            for event, attribution in zip(events, attribution_sets, strict=True)
            if set(codes) & set(attribution) and event.event.date is not None
        ]
        if not rows:
            assert region not in by_region
            continue
        expected: dict[dt.date, float] = {}
        for event in rows:
            month = event.event.date.replace(day=1)  # type: ignore[union-attr]
            expected[month] = expected.get(month, 0.0) + event.breadth
        base = baseline.to_mapping()
        assert by_region[region].to_mapping() == pytest.approx(
            {month: value / base[month] for month, value in expected.items()}
        )


# ------------------------------------------------------------------ #
# TPU / EPU family                                                   #
# ------------------------------------------------------------------ #


def test_tpu_daily_numerator_is_breadth_times_w_unc(rng: np.random.Generator) -> None:
    events = list(world_events())
    scores = _scores(rng.random(len(events)), rng.random(len(events)) * 2.0 - 1.0)
    series = policy.tpu_daily(events, scores, floor=0.35)
    breadths = event_breadths(events)
    numerator = daily_sum(
        event_dates(events), breadths * scores.w_unc, mask=scores.relevant_mask(0.35)
    )
    baseline = (
        daily_sum(event_dates(events), breadths).fill_daily(0.0).trailing_mean_days(365, 30)
    )
    assert series.to_mapping() == pytest.approx(numerator.divide(baseline).to_mapping())
    with pytest.raises(ValueError):
        policy.tpu_daily(events, scores)  # neither floor nor mask


def test_epu_is_us_over_us_on_both_lines(rng: np.random.Generator) -> None:
    events = list(world_events())
    scores = _scores(np.ones(len(events)), np.zeros(len(events)))  # all relevant, neutral
    us = policy.us_attributed_mask(events)
    assert us.any(), "the real slice carries US-attributed events"
    series = policy.epu_monthly(events, scores, floor=0.5)
    breadths = event_breadths(events)
    us_monthly = daily_sum(event_dates(events), breadths, mask=us).resample_monthly("sum")
    numerator = daily_sum(
        event_dates(events), breadths * 0.5, mask=us
    ).resample_monthly("sum")
    expected = numerator.divide(us_monthly.trailing_mean(12, min_periods=6))
    assert series.to_mapping() == pytest.approx(expected.to_mapping())
    # A non-US-relevant event never enters: zeroing non-US rows changes nothing.
    masked = _scores(np.where(us, 1.0, 0.0), np.zeros(len(events)))
    assert policy.epu_monthly(events, masked, floor=0.5).to_mapping() == pytest.approx(
        series.to_mapping()
    )


def test_category_gate_ands_into_the_numerator(rng: np.random.Generator) -> None:
    events = list(world_events())
    scores = _scores(np.ones(len(events)), np.zeros(len(events)))
    nothing = policy.epu_monthly(
        events, scores, floor=0.5, category=np.zeros(len(events), dtype=bool)
    )
    assert all(value == 0.0 for value in nothing.to_mapping().values())
    everything = policy.epu_monthly(
        events, scores, floor=0.5, category=np.ones(len(events), dtype=bool)
    )
    assert everything.to_mapping() == pytest.approx(
        policy.epu_monthly(events, scores, floor=0.5).to_mapping()
    )


def test_net_polarity_is_breadth_weighted_mean(rng: np.random.Generator) -> None:
    events = list(world_events())
    polarity = rng.random(len(events)) * 2.0 - 1.0
    scores = _scores(np.ones(len(events)), polarity)
    series = policy.net_polarity(events, scores, floor=0.5)
    breadths = event_breadths(events)
    for date, value in series.to_mapping().items():
        rows = [
            index
            for index, event in enumerate(events)
            if event.event.date == date
        ]
        weighted = sum(breadths[i] * polarity[i] for i in rows)
        total = sum(breadths[i] for i in rows)
        assert value == pytest.approx(weighted / total)
        assert -1.0 <= value <= 1.0


def test_explain_month_reports_contributions_and_near_misses() -> None:
    events = list(world_events())
    n_rows = len(events)
    relevance = np.linspace(0.0, 1.0, n_rows)
    scores = _scores(relevance, np.zeros(n_rows))
    month = next(e.event.date for e in events if e.event.date is not None)
    report = policy.explain_month(events, scores, month, floor=0.5, near_miss_band=0.5)
    assert report["month"] == month.replace(day=1).isoformat()
    assert report["counted_events"] == len(report["top_events"]) or (
        report["counted_events"] > 20 and len(report["top_events"]) == 20
    )
    contributions = [row["contribution"] for row in report["top_events"]]
    assert contributions == sorted(contributions, reverse=True)
    for row in report["top_events"]:
        assert row["relevance"] >= 0.5
        assert row["contribution"] == pytest.approx(row["breadth"] * row["w_unc"])
    for row in report["just_below_floor"]:
        assert 0.0 <= row["relevance"] < 0.5
    assert report["events_in_month"] >= report["counted_events"]
    assert report["residual_vs_benchmark"] is None
