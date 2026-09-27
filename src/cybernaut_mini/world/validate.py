"""Benchmark validation harness: correlations, episode ranks, spikes, tariff lead.

A replica index is only as good as its agreement with the published series, and
the honest yardstick is how well the published series agree with *each other*.
This module implements the posts' validation protocol: align on jointly non-null
rows, rebase each series to its own mean, Pearson on levels AND first differences
at monthly and quarterly frequency, always printed next to the
published-vs-published bar; the episode rank-alignment table with the
60th-percentile rule; the spike checklist (local maxima within +/-1 month of
named events); and the tariff-rate lead test. Published CSVs are opt-in downloads
cached under ``data/01_raw/reference/`` so every test path stays offline.

Blog ref: https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty
    — "all correlations are computed on the common window ... on jointly non-null
    rows after rebasing each series to its own mean"; the pairwise table ("TPU ~
    BBD (the bar) 0.96 / 0.70 ... 0.98 / 0.93"); "Every one of the twelve
    episodes we flag lands above the 60th percentile in all three"; "Each index's
    level lines up with the next quarter's change in the tariff rate at 0.61 for
    TPU, 0.59 for BBD and 0.66 for NOSIBLE"; sources appendix (matteoiacoviello.com,
    policyuncertainty.com, BEA via DBnomics).
    https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world
    — "the raw daily share at 0.90 on the levels and 0.79 on the stricter
    month-to-month changes, and the detrended version at 0.89 and 0.75". Local
    copies under ``docs/blog-archive/``.

Assumptions:
    - Rebasing to the mean leaves Pearson unchanged mathematically; the harness
      rebases anyway because the posts publish rebased series and the residuals
      (:func:`cybernaut_mini.world.indices.policy.explain_month`) need a common
      scale. Correlations are computed on the aligned common window only.
    - Episode percentile ranks use ``scipy.stats.rankdata`` over the series'
      monthly values; the pytest thresholds are laptop-softened (the fixture
      corpus is a 460-document slice, not 14.9M events) but the *rule* — every
      flagged episode above the 60th percentile — is the published one.
    - Published-series downloads are gated behind
      ``CYBERNAUT_MINI_WORLD_DOWNLOADS=1``: absent the gate and the cache, the
      loader raises :class:`~cybernaut_mini.config.ConfigError` naming both, and
      tests skip. Cached files are plain CSVs keyed by series name.
    - The published-vs-published agreement numbers are committed constants: they
      are the posts' printed values, kept next to every replica score so a report
      can never show a replica correlation without its yardstick.

Alternatives rejected:
    - Fetching benchmarks at test time when the network happens to be up:
      nondeterministic CI and a violation of the offline-default rule.
    - Spearman for the headline agreement: the posts publish Pearson on levels
      and changes; ranks appear only in the episode table, where rankdata is
      exactly what "ranked out of 135 months" means.
    - Shipping published index values as committed fixtures: their licenses vary
      and the values update monthly; a cache directory the user populates is
      reproducible without redistribution.
"""

from __future__ import annotations

import datetime as dt
import os
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from cybernaut_mini.config import ConfigError
from cybernaut_mini.world.timeseries import TimeSeries

__all__ = [
    "DOWNLOADS_ENV",
    "PUBLISHED_AGREEMENT",
    "PUBLISHED_SOURCES",
    "REFERENCE_DIR",
    "CorrelationReport",
    "correlation_report",
    "episode_rank_alignment",
    "format_report",
    "load_episodes",
    "load_published_series",
    "passes_rank_rule",
    "spike_checklist",
    "tariff_rate_lead",
]

DOWNLOADS_ENV = "CYBERNAUT_MINI_WORLD_DOWNLOADS"
REFERENCE_DIR = Path("data/01_raw/reference")
DEFAULT_EPISODES_PATH = Path("configs/world/episodes.yaml")

#: The published-vs-published agreement bar, printed next to every replica score.
#: pair -> (monthly levels, monthly changes, quarterly levels, quarterly changes);
#: ``nan`` where the posts do not print a number.
PUBLISHED_AGREEMENT: dict[str, tuple[float, float, float, float]] = {
    "tpu~bbd": (0.96, 0.70, 0.98, 0.93),
    "epu_10paper~epu_newsbank": (0.92, 0.65, float("nan"), float("nan")),
    "gpr~ai_gpr": (0.90, 0.79, float("nan"), float("nan")),
}

#: Opt-in download sources for the free published series (cached as CSV).
PUBLISHED_SOURCES: dict[str, str] = {
    "tpu_monthly": "https://www.matteoiacoviello.com/tpu_files/tpu_web_latest.csv",
    "gpr_monthly": "https://www.matteoiacoviello.com/gpr_files/gpr_web_latest.csv",
    "epu_monthly": (
        "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.csv"
    ),
    "epu_categorical": (
        "https://www.policyuncertainty.com/media/Categorical_EPU_Data.csv"
    ),
    "tariff_rate_quarterly": (
        "https://api.db.nomics.world/v22/series/BEA/NIPA-T40205"
        "?observations=1&format=csv"
    ),
}


# ---------------------------------------------------------------------- #
# Correlation protocol                                                   #
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class CorrelationReport:
    """Replica-vs-published agreement at both frequencies, next to its yardstick."""

    pair: str
    monthly_levels: float
    monthly_changes: float
    quarterly_levels: float
    quarterly_changes: float
    published_bar: tuple[float, float, float, float] | None

    def rows(self) -> list[tuple[str, float]]:
        return [
            ("monthly levels", self.monthly_levels),
            ("monthly changes", self.monthly_changes),
            ("quarterly levels", self.quarterly_levels),
            ("quarterly changes", self.quarterly_changes),
        ]


def _freq_pair(replica: TimeSeries, published: TimeSeries, how: str) -> tuple[float, float]:
    if how == "monthly":
        left, right = replica.resample_monthly("mean"), published.resample_monthly("mean")
    else:
        left, right = replica.resample_quarterly("mean"), published.resample_quarterly("mean")
    left, right = left.rebase_to_own_mean(), right.rebase_to_own_mean()
    return left.pearson(right), left.diff().pearson(right.diff())


def correlation_report(
    replica: TimeSeries,
    published: TimeSeries,
    *,
    pair: str = "replica~published",
    bar: str | None = None,
) -> CorrelationReport:
    """Pearson on levels AND changes, monthly and quarterly, on jointly non-null rows.

    ``bar`` names a :data:`PUBLISHED_AGREEMENT` row to carry alongside — the
    published-vs-published agreement the replica score must be judged against.
    """
    monthly_levels, monthly_changes = _freq_pair(replica, published, "monthly")
    quarterly_levels, quarterly_changes = _freq_pair(replica, published, "quarterly")
    return CorrelationReport(
        pair=pair,
        monthly_levels=monthly_levels,
        monthly_changes=monthly_changes,
        quarterly_levels=quarterly_levels,
        quarterly_changes=quarterly_changes,
        published_bar=PUBLISHED_AGREEMENT.get(bar) if bar else None,
    )


# ---------------------------------------------------------------------- #
# Episode rank alignment                                                 #
# ---------------------------------------------------------------------- #


def load_episodes(path: Path = DEFAULT_EPISODES_PATH) -> dict[str, dt.date]:
    """Named episodes -> month (first day) from ``configs/world/episodes.yaml``."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    episodes = payload.get("episodes")
    if not isinstance(episodes, dict) or not episodes:
        msg = f"{path} must contain a non-empty 'episodes' mapping"
        raise ConfigError(msg)
    out: dict[str, dt.date] = {}
    for name, month in episodes.items():
        parsed = month if isinstance(month, dt.date) else dt.date.fromisoformat(f"{month}-01")
        out[str(name)] = parsed.replace(day=1)
    return out


def episode_rank_alignment(
    series: TimeSeries, episodes: Mapping[str, dt.date]
) -> dict[str, float]:
    """Percentile rank (0..1) of each episode month within the monthly series.

    ``rankdata`` over the monthly values — "ranked out of 135 months". Episodes
    outside the series' window get ``nan`` (an answer, not an error: the fixture
    corpus covers a narrower window than the posts').
    """
    from scipy.stats import rankdata  # type: ignore[import-untyped]

    monthly = series.resample_monthly("mean")
    if len(monthly) == 0:
        return {name: float("nan") for name in episodes}
    percentiles = rankdata(monthly.values) / len(monthly)
    by_month = dict(zip(monthly.dates, percentiles, strict=True))
    return {
        name: float(by_month.get(month.replace(day=1), float("nan")))
        for name, month in episodes.items()
    }


def passes_rank_rule(
    alignment: Mapping[str, float], *, percentile: float = 0.60
) -> dict[str, bool]:
    """The published rule: every flagged episode lands above the 60th percentile.

    Episodes with ``nan`` ranks (outside the window) are excluded, not failed.
    """
    return {
        name: bool(rank > percentile)
        for name, rank in alignment.items()
        if not np.isnan(rank)
    }


def spike_checklist(
    series: TimeSeries, named_events: Mapping[str, dt.date], *, window_months: int = 1
) -> dict[str, bool]:
    """Does the monthly series have a local maximum within +/-``window_months``?

    The GPR sanity check: every named real-world spike should show up as a local
    peak near its month. A month is a local maximum when it is strictly greater
    than both neighbours (boundary months compare against their one neighbour) —
    a flat plateau is not a spike.
    """
    monthly = series.resample_monthly("mean")
    values = monthly.values
    is_peak = np.zeros(len(monthly), dtype=bool)
    for index in range(len(monthly)):
        left_ok = index == 0 or values[index] > values[index - 1]
        right_ok = index == len(monthly) - 1 or values[index] > values[index + 1]
        is_peak[index] = left_ok and right_ok
    month_index = {date: index for index, date in enumerate(monthly.dates)}
    out: dict[str, bool] = {}
    for name, when in named_events.items():
        target = when.replace(day=1)
        hits = []
        for offset in range(-window_months, window_months + 1):
            month = target.month - 1 + offset
            candidate = dt.date(target.year + month // 12, month % 12 + 1, 1)
            position = month_index.get(candidate)
            if position is not None:
                hits.append(bool(is_peak[position]))
        out[name] = any(hits)
    return out


# ---------------------------------------------------------------------- #
# Tariff-rate lead test                                                  #
# ---------------------------------------------------------------------- #


def tariff_rate_lead(index_quarterly: TimeSeries, tariff_rate: TimeSeries) -> float:
    """Correlation of the index's level with the NEXT quarter's tariff-rate change.

    The post's realized-policy check: "Each index's level lines up with the next
    quarter's change in the tariff rate at 0.61 for TPU, 0.59 for BBD and 0.66
    for NOSIBLE." ``tariff_rate(q)`` is customs duties / goods imports (BEA NIPA
    via DBnomics).
    """
    changes = tariff_rate.resample_quarterly("mean").diff()
    # Lead: pair the index level in quarter q with the change dated q+1 by
    # shifting the change series back one quarter.
    shifted_dates = []
    for date in changes.dates:
        month = date.month - 4
        shifted_dates.append(dt.date(date.year + month // 12, month % 12 + 1, 1))
    led = TimeSeries(tuple(shifted_dates), changes.values)
    return index_quarterly.resample_quarterly("mean").pearson(led)


# ---------------------------------------------------------------------- #
# Opt-in published-series cache                                          #
# ---------------------------------------------------------------------- #


def _parse_series_csv(text: str, date_column: str, value_column: str) -> TimeSeries:
    import csv
    import io

    mapping: dict[dt.date, float] = {}
    for row in csv.DictReader(io.StringIO(text)):
        raw_date = (row.get(date_column) or "").strip()
        raw_value = (row.get(value_column) or "").strip()
        if not raw_date or not raw_value:
            continue
        try:
            date = dt.date.fromisoformat(raw_date[:10])
            value = float(raw_value)
        except ValueError:
            continue
        mapping[date.replace(day=1)] = value
    if not mapping:
        msg = f"no ({date_column!r}, {value_column!r}) rows parsed from the reference CSV"
        raise ConfigError(msg)
    return TimeSeries.from_mapping(mapping)


def load_published_series(
    name: str,
    *,
    date_column: str,
    value_column: str,
    cache_dir: Path = REFERENCE_DIR,
    sources: Mapping[str, str] | None = None,
) -> TimeSeries:
    """Load a published benchmark series from the cache, downloading only if allowed.

    The cache file is ``cache_dir / f"{name}.csv"``. When it is absent, the
    download runs only with ``CYBERNAUT_MINI_WORLD_DOWNLOADS=1`` in the
    environment; otherwise a :class:`ConfigError` names the gate and the path, so
    offline runs (and tests) fail fast or skip.
    """
    table = dict(sources) if sources is not None else PUBLISHED_SOURCES
    cache_path = cache_dir / f"{name}.csv"
    if not cache_path.exists():
        if name not in table:
            msg = f"unknown published series {name!r}; available: {sorted(table)}"
            raise ConfigError(msg)
        if not os.environ.get(DOWNLOADS_ENV, "").strip():
            msg = (
                f"published series {name!r} is not cached at {cache_path} and downloads "
                f"are opt-in; set {DOWNLOADS_ENV}=1 to fetch it"
            )
            raise ConfigError(msg)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(table[name], timeout=60) as response:
            cache_path.write_bytes(response.read())
    return _parse_series_csv(
        cache_path.read_text(encoding="utf-8", errors="replace"), date_column, value_column
    )


def format_report(reports: Sequence[CorrelationReport]) -> str:
    """Plain-text table: every replica score printed beside its published bar."""
    lines = ["pair | monthly lvl/chg | quarterly lvl/chg | published bar (m-lvl/m-chg)"]
    for report in reports:
        bar = report.published_bar
        bar_text = f"{bar[0]:.2f}/{bar[1]:.2f}" if bar is not None else "-"
        lines.append(
            f"{report.pair} | {report.monthly_levels:.2f}/{report.monthly_changes:.2f} | "
            f"{report.quarterly_levels:.2f}/{report.quarterly_changes:.2f} | {bar_text}"
        )
    return "\n".join(lines)
