"""sentiment_label / sentiment_distil pipeline tests: DAG shape, plus an
offline end-to-end run through the node functions on real fixture headlines
(the proxy-gold path, so no judgments are invented)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cybernaut_mini.config import ConfigError
from cybernaut_mini.models import canonical_dumps
from cybernaut_mini.pipelines.sentiment_distil.nodes import (
    bootstrap_ensemble,
    distil_students,
    select_ensemble,
)
from cybernaut_mini.pipelines.sentiment_label.nodes import (
    benchmark_labels,
    estimate_costs,
    extract_gold_labels,
    label_stories,
    prepare_stories,
    time_pool,
)
from cybernaut_mini.sentiment.data import load_fixture_stories

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DOCUMENTS = REPO_ROOT / "data" / "01_raw" / "fixtures" / "documents.jsonl"

LABELER_PARAMS: dict[str, object] = {
    "textblob_thresholds": [0.10, 0.30],
    "vader_thresholds": [0.10, 0.20],
    "random_seed": 42,
    "timing_rows": 8,
}
BENCHMARK_PARAMS: dict[str, object] = {
    "reference": "VADER-0.10",
    "random_seed": 42,
    "n_stories": 10_000_000,
    "cost_models": [
        {
            "name": "hosted-frontier",
            "prompt_tokens_per_story": 330,
            "completion_tokens_per_story": 2,
            "usd_per_million_prompt_tokens": 10.0,
            "usd_per_million_completion_tokens": 30.0,
        }
    ],
}
ENSEMBLE_PARAMS: dict[str, object] = {
    "proxy_column": "VADER-0.10",
    "exclude": ["Random"],
    "bootstrap_runs": 25,
    "bootstrap_subsample": 0.75,
    "bootstrap_seed": 42,
}
DISTIL_PARAMS: dict[str, object] = {
    "embedding": {"provider": "hash", "dim": 32},
    "test_size": 0.25,
    "random_state": 42,
    "baseline_columns": ["VADER-0.20"],
}


@pytest.fixture(scope="module")
def rows() -> list[dict[str, object]]:
    stories = load_fixture_stories(FIXTURE_DOCUMENTS, max_rows=32)
    return [{"text": story} for story in stories]


def test_pipelines_build_with_unique_node_names() -> None:
    from cybernaut_mini.pipelines.sentiment_distil import (
        create_pipeline as create_distil,
    )
    from cybernaut_mini.pipelines.sentiment_label import (
        create_pipeline as create_label,
    )

    label_pipeline = create_label()
    distil_pipeline = create_distil()
    assert len(label_pipeline.nodes) == 6
    assert len(distil_pipeline.nodes) == 3
    names = [node.name for node in label_pipeline.nodes + distil_pipeline.nodes]
    assert len(names) == len(set(names))
    # The distil pipeline consumes the label pipeline's outputs by name.
    label_outputs = label_pipeline.all_outputs()
    assert {"sentiment_stories", "sentiment_label_matrix"} <= label_outputs
    assert {"sentiment_stories", "sentiment_label_matrix"} <= distil_pipeline.all_inputs()


def test_offline_end_to_end_through_the_node_functions(
    rows: list[dict[str, object]],
) -> None:
    stories = prepare_stories(rows, dict(LABELER_PARAMS))
    assert len(stories) == 32

    matrix = label_stories(stories, dict(LABELER_PARAMS))
    assert sorted(matrix) == [
        "Random",
        "TextBlob-0.10",
        "TextBlob-0.30",
        "VADER-0.10",
        "VADER-0.20",
    ]

    report = benchmark_labels(matrix, [], dict(BENCHMARK_PARAMS))
    agreement = report["agreement"]
    assert agreement["reference"] == "VADER-0.10"
    assert agreement["order"][0] == "VADER-0.10"
    assert set(report["kappa"]) == set(matrix) - {"VADER-0.10"}

    timing = time_pool(stories, dict(LABELER_PARAMS))
    assert set(timing) == set(matrix)

    costs = estimate_costs(dict(BENCHMARK_PARAMS))
    assert costs["models"]["hosted-frontier"]["total_cost"] > 0

    trace = select_ensemble(matrix, [], dict(ENSEMBLE_PARAMS))
    assert trace["members"]
    assert "VADER-0.10" not in trace["members"]  # the proxy cannot vote for itself
    assert "Random" not in trace["members"]
    assert trace["steps"][0]["gain"] > 0

    stability = bootstrap_ensemble(matrix, [], dict(ENSEMBLE_PARAMS))
    total = sum(entry["win_share"] for entry in stability["ensembles"])
    assert total == pytest.approx(1.0)

    results = distil_students(stories, matrix, trace, dict(DISTIL_PARAMS))
    assert "VADER-0.20" in results
    student_rows = [name for name in results if name.startswith("OLS(")]
    assert len(student_rows) == 1
    row = results[student_rows[0]]
    assert set(row) == {"Parameters", "Runtime", "Dimensions", "Accuracy"}
    assert row["Dimensions"] == 32.0

    # Every report is canonical-JSON serialisable except NaN metadata fields,
    # which the canonical writer rejects by design — strip them first the way
    # the CSV writer renders them empty.
    canonical_dumps({"agreement": agreement, "trace": trace, "stability": stability})


def test_extract_gold_labels_maps_dataset_label_strings() -> None:
    schema_rows: list[dict[str, object]] = [
        {"text": "row", "label": "positive"},
        {"text": "row", "label": "Neutral"},
        {"text": "row", "label": "NEGATIVE"},
        {"text": "row", "label": 1},
    ]
    labels = extract_gold_labels(schema_rows, {})
    assert labels == [1, 0, -1, 1]
    with pytest.raises(ConfigError, match="unknown gold label"):
        extract_gold_labels([{"text": "row", "label": "meh"}], {})
    with pytest.raises(ConfigError, match="missing gold label"):
        extract_gold_labels([{"text": "row"}], {})


def test_select_ensemble_requires_gold_or_proxy(rows: list[dict[str, object]]) -> None:
    stories = prepare_stories(rows, dict(LABELER_PARAMS))
    matrix = label_stories(stories, dict(LABELER_PARAMS))
    with pytest.raises(ConfigError, match="proxy_column"):
        select_ensemble(matrix, [], {"exclude": ["Random"]})
    with pytest.raises(ConfigError, match="not in label matrix"):
        select_ensemble(matrix, [], {"proxy_column": "GPT-4-Original"})


def test_prepare_stories_rejects_empty(rows: list[dict[str, object]]) -> None:
    with pytest.raises(ConfigError):
        prepare_stories([{"text": ""}], {})
