"""Validation harness: correlation protocol, episode ranks, spikes, offline gating.

The correlation and rank machinery is pure math, tested on numeric series with
known relationships (a series correlates 1.0 with itself and with any positive
rescaling of itself — which is exactly why the harness always prints the
published-vs-published bar next to a replica score). The published-series loader
is tested for its *gating*: with no cache and no opt-in env var it must raise a
ConfigError naming both, and it must parse a cached CSV without any network.

Blog ref: https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty
    — the correlation protocol, the pairwise agreement table, the 60th-percentile
    episode rule, and the tariff-rate lead check. Local copy under
    ``docs/blog-archive/``.

Assumptions: the cached-CSV test writes a tiny CSV whose *format* mirrors the
reference files (a date column and a value column); the numbers are arbitrary
numeric test values for a parser, not a fabricated benchmark — no published
values are invented or asserted.

Alternatives rejected: recording a real benchmark CSV as a committed fixture
(license and staleness — the cache directory pattern exists for exactly this).
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest

from cybernaut_mini.config import ConfigError
from cybernaut_mini.world import validate
from cybernaut_mini.world.timeseries import TimeSeries


def _monthly(values: list[float], start: dt.date = dt.date(2018, 1, 1)) -> TimeSeries:
    dates = []
    year, month = start.year, start.month
    for _ in values:
        dates.append(dt.date(year, month, 1))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return TimeSeries(tuple(dates), np.asarray(values, dtype=np.float64))


# ------------------------------------------------------------------ #
# Correlation protocol                                               #
# ------------------------------------------------------------------ #


def test_correlation_report_perfect_agreement_survives_rescaling() -> None:
    series = _monthly([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 2.0, 7.0, 8.0, 1.0, 9.0, 3.0])
    rescaled = TimeSeries(series.dates, series.values * 40.0)  # different units
    report = validate.correlation_report(series, rescaled, pair="self~self", bar="tpu~bbd")
    assert report.monthly_levels == pytest.approx(1.0)
    assert report.monthly_changes == pytest.approx(1.0)
    assert report.quarterly_levels == pytest.approx(1.0)
    assert report.quarterly_changes == pytest.approx(1.0)
    assert report.published_bar == validate.PUBLISHED_AGREEMENT["tpu~bbd"]


def test_correlation_on_jointly_nonnull_rows_only() -> None:
    left = _monthly([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    right = _monthly([2.0, 4.0, 6.0], start=dt.date(2018, 4, 1))  # overlaps months 4-6
    report = validate.correlation_report(left, right)
    assert report.monthly_levels == pytest.approx(1.0)  # aligned on the common window
    no_overlap = _monthly([1.0, 2.0], start=dt.date(2030, 1, 1))
    degenerate = validate.correlation_report(left, no_overlap)
    assert math.isnan(degenerate.monthly_levels)  # "no evidence", not an exception


def test_published_agreement_bar_carries_the_posts_numbers() -> None:
    monthly_levels, monthly_changes, quarterly_levels, quarterly_changes = (
        validate.PUBLISHED_AGREEMENT["tpu~bbd"]
    )
    assert (monthly_levels, monthly_changes) == (0.96, 0.70)
    assert (quarterly_levels, quarterly_changes) == (0.98, 0.93)
    assert validate.PUBLISHED_AGREEMENT["epu_10paper~epu_newsbank"][:2] == (0.92, 0.65)
    assert validate.PUBLISHED_AGREEMENT["gpr~ai_gpr"][:2] == (0.90, 0.79)
    text = validate.format_report(
        [validate.correlation_report(_monthly([1, 2, 3, 4]), _monthly([1, 2, 3, 4]),
                                     pair="x", bar="tpu~bbd")]
    )
    assert "0.96/0.70" in text  # the bar is ALWAYS printed next to the replica score


# ------------------------------------------------------------------ #
# Episodes, ranks, spikes                                            #
# ------------------------------------------------------------------ #


def test_episode_config_loads_the_in_window_trade_war_rounds() -> None:
    episodes = validate.load_episodes()
    assert episodes["section_232_steel_and_aluminium"] == dt.date(2018, 3, 1)
    assert episodes["china_section_301_round_one"] == dt.date(2018, 6, 1)
    assert episodes["august_2019_escalation"] == dt.date(2019, 8, 1)
    assert all(day.day == 1 for day in episodes.values())


def test_episode_rank_alignment_and_sixty_percentile_rule() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    series = _monthly(values)
    episodes = {
        "top_month": dt.date(2018, 10, 1),  # rank 10/10
        "median_month": dt.date(2018, 5, 1),  # rank 5/10
        "outside_window": dt.date(2030, 1, 1),
    }
    alignment = validate.episode_rank_alignment(series, episodes)
    assert alignment["top_month"] == pytest.approx(1.0)
    assert alignment["median_month"] == pytest.approx(0.5)
    assert math.isnan(alignment["outside_window"])
    verdict = validate.passes_rank_rule(alignment, percentile=0.60)
    assert verdict == {"top_month": True, "median_month": False}  # nan excluded, not failed


def test_spike_checklist_local_maxima_within_one_month() -> None:
    series = _monthly([1.0, 5.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0])
    named = {
        "on_the_peak": dt.date(2018, 2, 1),
        "one_month_off": dt.date(2018, 5, 1),  # peak is 2018-06
        "in_the_trough": dt.date(2018, 4, 1),
    }
    result = validate.spike_checklist(series, named, window_months=1)
    assert result["on_the_peak"]
    assert result["one_month_off"]
    assert not result["in_the_trough"]


def test_tariff_rate_lead_pairs_level_with_next_quarter_change() -> None:
    # Index level in quarter q equals the tariff-rate change in quarter q+1 by
    # construction, so the lead correlation must be exactly 1.
    index = _monthly([1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0])
    quarters = index.resample_quarterly("mean")  # levels [1, 3, 2, 5] on 2018 Q1-Q4
    # tariff_rate whose change in quarter q+1 equals the index level in q:
    # cumulative sums 0, 1, 4, 6, 11 over 2018Q1 .. 2019Q1.
    tariff_rate = TimeSeries(
        (*quarters.dates, dt.date(2019, 1, 1)),
        np.concatenate([[0.0], np.cumsum(quarters.values)]),
    )
    lead = validate.tariff_rate_lead(index, tariff_rate)
    assert lead == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# Offline gating of the published-series cache                       #
# ------------------------------------------------------------------ #


def test_loader_raises_without_cache_or_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(validate.DOWNLOADS_ENV, raising=False)
    with pytest.raises(ConfigError) as excinfo:
        validate.load_published_series(
            "tpu_monthly", date_column="date", value_column="value", cache_dir=tmp_path
        )
    message = str(excinfo.value)
    assert validate.DOWNLOADS_ENV in message
    assert "tpu_monthly" in message


def test_loader_rejects_unknown_series(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        validate.load_published_series(
            "not-a-series", date_column="date", value_column="value", cache_dir=tmp_path
        )


def test_loader_parses_a_cached_csv_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(validate.DOWNLOADS_ENV, raising=False)
    (tmp_path / "tpu_monthly.csv").write_text(
        "date,value\n2018-01-01,3.5\n2018-02-01,4.5\nbad-row,\n", encoding="utf-8"
    )
    series = validate.load_published_series(
        "tpu_monthly", date_column="date", value_column="value", cache_dir=tmp_path
    )
    assert series.to_mapping() == {
        dt.date(2018, 1, 1): 3.5,
        dt.date(2018, 2, 1): 4.5,
    }


@pytest.mark.skipif(
    not (validate.REFERENCE_DIR / "tpu_monthly.csv").exists(),
    reason="published TPU series not cached under data/01_raw/reference/ (opt-in download)",
)
def test_real_published_tpu_series_when_cached() -> None:  # pragma: no cover - opt-in
    series = validate.load_published_series(
        "tpu_monthly", date_column="date", value_column="value"
    )
    assert len(series) > 100
