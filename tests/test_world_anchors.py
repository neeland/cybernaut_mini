"""Anchor engine + asymmetric smoother: verbatim configs, pair-routing math, causality.

Two layers of evidence. The YAML layer pins the published artifacts: the 17
stress anchors, the trade 3+2 (five sentences), the EPU 60-sentence appendix
(ten levers / 22 pairs plus two 4-pair category blocks), the 3 oil anchors and
the single trade-coercion phrase, with the posts' thresholds carried as data.
The math layer checks the formulas on synthetic unit vectors, where every cosine
is chosen by construction: max-cosine relevance with argmax labels, best-pair
tanh polarity routing, ``w_unc`` bounds, category masks, separation diagnostics,
and the smoother's asymmetry and no-look-ahead property.

Blog ref: https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty
    — the formulas and the appendix counts;
    https://nosible.com/blog/turning-news-into-a-risk-on-risk-off-equity-signal —
    "We define market stress as 17 short concepts". Local copies under
    ``docs/blog-archive/``.

Assumptions: synthetic orthonormal anchor vectors are pure math (allowed), not
fabricated corpus data — no invented documents or events appear anywhere here.

Alternatives rejected: semantic routing assertions on hash-embedder cosines (a
hash space has no semantics to assert); snapshotting full score arrays (opaque —
the formula identities are the contract).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from cybernaut_mini.config import ConfigError
from cybernaut_mini.world import anchors
from cybernaut_mini.world.events import embedding_matrix
from cybernaut_mini.world.smoother import asymmetric_ewma, half_life_alpha
from cybernaut_mini.world.timeseries import TimeSeries
from world_helpers import frozen_embedder, world_events

# ------------------------------------------------------------------ #
# The verbatim YAML anchor sets                                      #
# ------------------------------------------------------------------ #


def test_stress_set_carries_exactly_17_anchors() -> None:
    stress = anchors.load_anchor_set("stress")
    assert len(stress.anchors) == 17
    assert stress.sentence_count() == 17
    assert stress.relevance_floor == 0.30


def test_trade_set_is_five_sentences_with_published_thresholds() -> None:
    trade = anchors.load_anchor_set("trade")
    assert len(trade.anchors) == 3  # tariffs, agreements, disputes
    assert sum(len(pairs) for pairs in trade.levers.values()) == 1  # one matched pair
    assert trade.sentence_count() == 5  # "The whole index is five sentences"
    assert trade.relevance_floor == 0.35
    assert trade.polarity_temperature == 0.1


def test_epu_set_is_the_sixty_sentence_appendix() -> None:
    epu = anchors.load_anchor_set("epu")
    assert len(epu.levers) == 10  # ten levers
    assert sum(len(pairs) for pairs in epu.levers.values()) == 22  # 22 matched pairs
    assert set(epu.categories) == {"national_security", "healthcare"}
    assert all(len(pairs) == 4 for pairs in epu.categories.values())
    assert epu.sentence_count() == 60
    assert not epu.anchors  # relevance reads the lever sentences themselves
    assert epu.category_floor == 0.25
    assert "shutting_down_the_economy" in epu.levers  # the COVID-gap ablation lever


def test_oil_and_trade_coercion_sets() -> None:
    oil = anchors.load_anchor_set("oil")
    assert len(oil.anchors) == 3
    assert oil.relevance_floor == 0.30
    patch = anchors.load_anchor_set("trade_coercion")
    assert len(patch.anchors) == 1
    assert patch.relevance_floor == 0.40


def test_unknown_set_and_unknown_lever_raise_config_errors() -> None:
    with pytest.raises(ConfigError):
        anchors.load_anchor_set("no-such-set")
    with pytest.raises(ConfigError):
        anchors.load_anchor_set("epu").lever_pairs(include=["no-such-lever"])


# ------------------------------------------------------------------ #
# Scoring math on synthetic unit vectors                             #
# ------------------------------------------------------------------ #


def _embedded_synthetic() -> anchors.EmbeddedAnchorSet:
    """Two levers whose pair vectors are orthonormal basis vectors (pure math)."""
    config = anchors.AnchorSet(
        name="synthetic",
        source="unit-test geometry, not a corpus",
        thresholds={"relevance_floor": 0.5, "polarity_temperature": 0.1},
        levers={
            "alpha": [anchors.AnchorPair(uncertain="u-alpha", certain="c-alpha")],
            "beta": [anchors.AnchorPair(uncertain="u-beta", certain="c-beta")],
        },
    )
    eye = np.eye(4, dtype=np.float32)
    return anchors.EmbeddedAnchorSet(
        config=config,
        relevance_labels=("alpha", "beta"),
        relevance_matrix=np.zeros((0, 4), dtype=np.float32),
        pair_labels=("alpha", "beta"),
        pair_uncertain=eye[[0, 2]],  # U_alpha = e0, U_beta = e2
        pair_certain=eye[[1, 3]],  # C_alpha = e1, C_beta = e3
    )


def test_best_pair_routing_and_tanh_polarity() -> None:
    embedded = _embedded_synthetic()
    events = np.asarray(
        [
            [0.8, 0.2, 0.1, 0.0],  # alpha-uncertain leaning
            [0.1, 0.9, 0.0, 0.0],  # alpha-certain leaning
            [0.0, 0.0, 0.3, 0.7],  # beta-certain leaning
        ],
        dtype=np.float32,
    )
    unit = (events / np.linalg.norm(events, axis=1, keepdims=True)).astype(np.float64)
    scores = anchors.score_events(events, embedded)
    assert scores.best_pair == ("alpha", "alpha", "beta")
    # polarity = tanh((U_bp - C_bp) / 0.1) on the routed pair, exactly.
    expected = np.tanh((unit[:, [0, 0, 2]].diagonal() - unit[:, [1, 1, 3]].diagonal()) / 0.1)
    assert np.allclose(scores.polarity, expected, atol=1e-6)
    assert np.array_equal(scores.w_unc, (1.0 + scores.polarity) / 2.0)  # exact identity
    assert scores.polarity[0] > 0 > scores.polarity[1]
    assert scores.polarity[2] < 0


def test_epu_shape_relevance_is_best_pair_better_framing() -> None:
    embedded = _embedded_synthetic()
    events = np.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)  # exactly C_alpha
    scores = anchors.score_events(events, embedded)
    # A lever's score is the highest of its two framings: here cos(C_alpha)=1.
    assert scores.relevance[0] == pytest.approx(1.0)
    assert scores.nearest_anchor == ("alpha",)
    assert scores.relevant_mask(0.5).tolist() == [True]
    assert scores.relevant_mask(1.1).tolist() == [False]


def test_flat_anchor_relevance_reads_the_anchors_block() -> None:
    config = anchors.AnchorSet(
        name="flat",
        source="unit-test geometry",
        anchors={"one": "s1", "two": "s2"},
    )
    matrix = np.eye(3, dtype=np.float32)
    embedded = anchors.EmbeddedAnchorSet(
        config=config,
        relevance_labels=("one", "two"),
        relevance_matrix=matrix[[0, 1]],
        pair_labels=(),
        pair_uncertain=np.zeros((0, 3), dtype=np.float32),
        pair_certain=np.zeros((0, 3), dtype=np.float32),
    )
    scores = anchors.score_events(matrix[[1, 2]], embedded)
    assert scores.relevance == pytest.approx([1.0, 0.0])
    assert scores.nearest_anchor[0] == "two"
    # No pairs: neutral polarity, w_unc = 0.5.
    assert scores.polarity == pytest.approx([0.0, 0.0])
    assert scores.w_unc == pytest.approx([0.5, 0.5])


def test_category_mask_is_max_cosine_over_pair_sentences() -> None:
    matrix = np.eye(3, dtype=np.float32)
    mask = anchors.category_mask(matrix, (matrix[[0]], matrix[[1]]), floor=0.9)
    assert mask.tolist() == [True, True, False]


def test_polarity_separation_and_floor_sweep() -> None:
    embedded = _embedded_synthetic()
    events = np.asarray(
        [[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]], dtype=np.float32
    )
    scores = anchors.score_events(events, embedded)
    relevant = np.asarray([True, True, False], dtype=bool)
    stats = anchors.polarity_separation(scores, relevant)
    assert stats["on_topic_mean_w_unc"] == pytest.approx(float(np.mean(scores.w_unc[:2])))
    assert stats["off_topic_mean_w_unc"] == pytest.approx(0.5)
    assert 0.0 <= stats["share_above_0.6"] <= 1.0
    sweep = anchors.sweep_relevance_floor(scores, floors=(0.0, 2.0))
    assert sweep[0.0] == 1.0 and sweep[2.0] == 0.0


def test_leave_one_lever_out_shrinks_the_pair_list() -> None:
    epu = anchors.load_anchor_set("epu")
    embedder = frozen_embedder()
    keep = [name for name in epu.levers if name != "shutting_down_the_economy"]
    full = anchors.EmbeddedAnchorSet.embed(epu, embedder)
    ablated = anchors.EmbeddedAnchorSet.embed(epu, embedder, levers=keep)
    assert len(full.pair_labels) == 22
    assert len(ablated.pair_labels) == 20  # the shutdown lever carries 2 pairs
    assert "shutting_down_the_economy" not in ablated.pair_labels


def test_scoring_real_events_is_deterministic_and_bounded() -> None:
    matrix = embedding_matrix(list(world_events()))
    embedded = anchors.EmbeddedAnchorSet.embed(anchors.load_anchor_set("trade"), frozen_embedder())
    first = anchors.score_events(matrix, embedded)
    second = anchors.score_events(matrix, embedded)
    assert np.array_equal(first.relevance, second.relevance)
    assert np.array_equal(first.polarity, second.polarity)
    assert np.all(np.abs(first.relevance) <= 1.0 + 1e-6)
    assert np.all((first.w_unc >= 0.0) & (first.w_unc <= 1.0))
    assert first.best_pair == second.best_pair


# ------------------------------------------------------------------ #
# Asymmetric EWMA smoother                                           #
# ------------------------------------------------------------------ #


def _flat_then_spike(n_flat: int = 120, spike: float = 5.0) -> TimeSeries:
    start = dt.date(2020, 1, 1)
    dates = tuple(start + dt.timedelta(days=i) for i in range(n_flat + 1))
    values = np.zeros(n_flat + 1)
    values[-1] = spike
    return TimeSeries(dates, values)


def test_half_life_alpha_formula() -> None:
    assert half_life_alpha(1.0) == pytest.approx(0.5)
    assert half_life_alpha(3.0) == pytest.approx(1.0 - 0.5 ** (1.0 / 3.0))
    with pytest.raises(ValueError):
        half_life_alpha(0.0)


def test_upward_spike_uses_fast_regime_downward_uses_slow() -> None:
    up = _flat_then_spike(spike=5.0)
    down = TimeSeries(up.dates, -up.values)
    fast = asymmetric_ewma(up)
    slow = asymmetric_ewma(down)
    alpha_fast, alpha_slow = half_life_alpha(3.0), half_life_alpha(30.0)
    assert fast.values[-1] == pytest.approx(alpha_fast * 5.0)
    assert slow.values[-1] == pytest.approx(alpha_slow * -5.0)
    # The asymmetry: the uptick is picked up ~8x harder than the equal downtick.
    assert abs(fast.values[-1]) > 5 * abs(slow.values[-1])


def test_smoother_has_no_look_ahead() -> None:
    series = _flat_then_spike()
    truncated = TimeSeries(series.dates[:-1], series.values[:-1])
    full = asymmetric_ewma(series)
    prefix = asymmetric_ewma(truncated)
    assert np.array_equal(full.values[:-1], prefix.values)


def test_warmup_stays_in_slow_regime() -> None:
    start = dt.date(2020, 1, 1)
    dates = tuple(start + dt.timedelta(days=i) for i in range(10))
    values = np.zeros(10)
    values[-1] = 100.0  # a huge jump, but with < min_history changes before it
    smoothed = asymmetric_ewma(TimeSeries(dates, values))
    assert smoothed.values[-1] == pytest.approx(half_life_alpha(30.0) * 100.0)
