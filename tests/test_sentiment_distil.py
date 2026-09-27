"""Distillation tests: the verbatim sentence constructor and split parameters,
the row-sum teacher, OLS students on hash embeddings of real fixture headlines,
and the exact four-column Results.csv shape."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cybernaut_mini.config import ConfigError
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.sentiment.data import load_fixture_stories
from cybernaut_mini.sentiment.distil import (
    RESULTS_COLUMNS,
    ProviderSurveyEncoder,
    build_sentences,
    distil_student,
    run_survey,
    teacher_classes,
    teacher_scores,
    write_results_csv,
)
from cybernaut_mini.sentiment.labelers import TextBlobLabeler, VaderLabeler, build_label_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DOCUMENTS = REPO_ROOT / "data" / "01_raw" / "fixtures" / "documents.jsonl"


@pytest.fixture(scope="module")
def stories() -> list[str]:
    return load_fixture_stories(FIXTURE_DOCUMENTS, max_rows=32)


@pytest.fixture(scope="module")
def matrix(stories: list[str]) -> dict[str, list[int]]:
    pool = [TextBlobLabeler(0.10), VaderLabeler(0.10), VaderLabeler(0.20)]
    return build_label_matrix(stories, pool)


def test_build_sentences_uses_the_posts_exact_constructor() -> None:
    sentences = build_sentences(["Headline one"], ["Description one"])
    assert sentences == ["Headline one. Description one"]
    with pytest.raises(ConfigError):
        build_sentences(["a"], [])


def test_teacher_is_unweighted_row_sum_with_band(matrix: dict[str, list[int]]) -> None:
    members = ["TextBlob-0.10", "VADER-0.10"]
    scores = teacher_scores(matrix, members)
    expected = np.asarray(matrix["TextBlob-0.10"], dtype=np.float64) + np.asarray(
        matrix["VADER-0.10"], dtype=np.float64
    )
    assert np.array_equal(scores, expected)
    classes = teacher_classes(scores)
    assert set(classes.tolist()) <= {-1, 0, 1}
    # Inclusive band: a row summing to exactly ±1 is classified, not neutral.
    assert teacher_classes(np.asarray([1.0, -1.0, 0.5])).tolist() == [1, -1, 0]


def test_teacher_requires_known_members(matrix: dict[str, list[int]]) -> None:
    with pytest.raises(ConfigError):
        teacher_scores(matrix, [])
    with pytest.raises(ConfigError):
        teacher_scores(matrix, ["GPT-4-Original"])


def test_distil_student_runs_on_hash_embeddings_and_is_deterministic(
    stories: list[str], matrix: dict[str, list[int]]
) -> None:
    embedder = HashEmbedder(dim=64)
    embeddings = np.asarray(embedder.embed_documents(stories), dtype=np.float64)
    teacher = teacher_scores(matrix, ["TextBlob-0.10", "VADER-0.10"])
    first = distil_student(embeddings, teacher)
    second = distil_student(embeddings, teacher)
    assert first == second  # random_state=42 pins the split
    assert 0.0 <= first.accuracy <= 1.0
    assert first.n_test == 8  # 32 rows * test_size 0.25
    assert first.n_train == 24


def test_distil_student_validates_alignment(stories: list[str]) -> None:
    with pytest.raises(ConfigError):
        distil_student(np.zeros((4, 8)), np.zeros(5))
    with pytest.raises(ConfigError):
        distil_student(np.zeros((4, 8)), np.zeros(4))  # too few rows to split


# ── survey harness ───────────────────────────────────────────────────────────


def test_run_survey_emits_students_and_nan_injected_baselines(
    stories: list[str], matrix: dict[str, list[int]]
) -> None:
    encoders = [
        ProviderSurveyEncoder(HashEmbedder(dim=64)),
        ProviderSurveyEncoder(HashEmbedder(dim=32)),
    ]
    results = run_survey(
        stories,
        matrix,
        ["TextBlob-0.10", "VADER-0.10"],
        encoders,
        baseline_columns=["VADER-0.20"],
    )
    assert set(results) == {"VADER-0.20", "OLS(hash-64)", "OLS(hash-32)"}
    baseline = results["VADER-0.20"]
    assert math.isnan(baseline.parameters)
    assert math.isnan(baseline.runtime)
    assert math.isnan(baseline.dimensions)
    assert 0.0 <= baseline.accuracy <= 1.0
    student = results["OLS(hash-64)"]
    assert student.dimensions == 64.0
    assert student.runtime > 0.0
    assert math.isnan(student.parameters)  # static provider: no torch parameters
    assert 0.0 <= student.accuracy <= 1.0


def test_run_survey_validates_inputs(
    stories: list[str], matrix: dict[str, list[int]]
) -> None:
    with pytest.raises(ConfigError):
        run_survey(stories, matrix, ["TextBlob-0.10"], [])
    with pytest.raises(ConfigError):
        run_survey(
            stories,
            matrix,
            ["TextBlob-0.10"],
            [ProviderSurveyEncoder(HashEmbedder(dim=16))],
            baseline_columns=["missing-column"],
        )


def test_write_results_csv_matches_the_appendix_shape(
    tmp_path: Path, stories: list[str], matrix: dict[str, list[int]]
) -> None:
    results = run_survey(
        stories,
        matrix,
        ["TextBlob-0.10", "VADER-0.10"],
        [ProviderSurveyEncoder(HashEmbedder(dim=16))],
        baseline_columns=["VADER-0.20"],
    )
    path = tmp_path / "Results.csv"
    write_results_csv(results, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "," + ",".join(RESULTS_COLUMNS)
    assert RESULTS_COLUMNS == ("Parameters", "Runtime", "Dimensions", "Accuracy")
    # Baseline row: NaN metadata rendered empty, accuracy present.
    baseline_line = next(line for line in lines if line.startswith("VADER-0.20,"))
    _name, parameters, runtime, dimensions, accuracy = baseline_line.split(",")
    assert (parameters, runtime, dimensions) == ("", "", "")
    assert float(accuracy) >= 0.0
    student_line = next(line for line in lines if line.startswith("OLS(hash-16),"))
    assert student_line.split(",")[3] == "16.0"
