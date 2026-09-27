"""Distil the ensemble teacher into OLS students on frozen sentence embeddings.

The teacher is the unweighted row-sum of the winning ensemble members (a
continuous score in ``[-n, n]``); each student is a plain
``sklearn.linear_model.LinearRegression`` fitted on frozen embeddings of
``f"{Headline}. {Description}"`` to predict that score, thresholded back to
three classes with the same ±1 band. The survey harness fits one student per
embedding checkpoint and emits the post's exact four-column ``Results.csv``.

Blog ref: https://nosible.com/blog/ensemble-and-distil — the appendix's 198-line
    script: ``sentences.append(f"{document['Headline']}. {document['Description']}")``,
    ``train_test_split(in_df, out_df, test_size=0.25, random_state=42)``,
    ``LinearRegression().fit(x_train.values, y_train["Teacher"].values)``,
    predictions banded at ±1, accuracy measured against the thresholded teacher
    on the test split, and ``results_df.to_csv("Results.csv")`` with columns
    Parameters/Runtime/Dimensions/Accuracy where the pre-existing labeler
    baselines are injected with NaN metadata when ``model_ix == 0``. Local
    copy: ``docs/blog-archive/ensemble-and-distil.md``.

Assumptions:
    - The ``". "`` sentence constructor and the absence of E5 ``query:`` /
      ``passage:`` prefixes are kept verbatim — the post embeds raw sentences
      even for E5 checkpoints. That is an ablatable quirk, not an accident to
      fix silently; :func:`build_sentences` is the single place to change it.
    - "Accuracy" is agreement with the TEACHER's thresholded classes on the
      held-out quarter, not with human gold — the post's students are graded
      against the teacher ("out-of-sample accuracy versus the Teacher model"),
      and the labeler baseline rows are graded the same way so the table is
      one comparable column.
    - The survey encoder is a protocol (name, parameter count, dim, encode) so
      the repo's offline embedding providers can stand in for the
      sentence-transformers checkpoints in tests; parameter count is ``nan``
      when the encoder cannot report one, exactly like the baselines' NaN rows.
    - ``Results.csv`` mirrors ``pandas.DataFrame.to_csv``: an unnamed index
      column, the four headers, and NaN written as the empty string, so the
      file diffs cleanly against one produced by the blog's own script.
    - The default checkpoint list is the CPU-friendly slice of the post's 35
      (bge-micro-v2 ... multilingual-e5-small), because the survey's point —
      the accuracy-vs-size frontier — survives the truncation while a laptop
      run stays in minutes.

Alternatives rejected:
    - Grading students against gold instead of the teacher: better science,
      wrong replica — it would silently change every Accuracy number the post
      reports, and gold is only 250 rows in the source anyway.
    - LogisticRegression on the three classes: a classifier would dodge the
      threshold step, but the post's students regress the continuous teacher
      score precisely so the ±1 band is shared between teacher and student.
    - Caching embeddings inside this module: the Kedro pipeline owns caching
      (the repo's embed_cache pattern); the library stays a pure function of
      (sentences, encoder).
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from cybernaut_mini.config import ConfigError
from cybernaut_mini.providers.embeddings import EmbeddingProvider
from cybernaut_mini.sentiment.ensemble import IntArray, Labels, sign_threshold

__all__ = [
    "DEFAULT_SURVEY_CHECKPOINTS",
    "ProviderSurveyEncoder",
    "SentenceTransformerSurveyEncoder",
    "StudentEvaluation",
    "SurveyEncoder",
    "SurveyRow",
    "build_sentences",
    "distil_student",
    "run_survey",
    "teacher_classes",
    "teacher_scores",
    "write_results_csv",
]

FloatArray = npt.NDArray[np.float64]

#: The CPU-friendly slice of the post's 35-checkpoint survey, smallest first,
#: chosen per the gap matrix (all run on MPS/CPU in minutes at laptop scale).
DEFAULT_SURVEY_CHECKPOINTS: tuple[str, ...] = (
    "TaylorAI/bge-micro-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "TaylorAI/gte-tiny",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
    "intfloat/e5-small-unsupervised",
    "intfloat/e5-base-v2",
    "sentence-transformers/sentence-t5-base",
    "intfloat/multilingual-e5-small",
)

#: The Results.csv header, verbatim from the appendix's experiment dict.
RESULTS_COLUMNS: tuple[str, ...] = ("Parameters", "Runtime", "Dimensions", "Accuracy")


def build_sentences(headlines: Sequence[str], descriptions: Sequence[str]) -> list[str]:
    """The post's exact sentence constructor: ``f"{Headline}. {Description}"``."""
    if len(headlines) != len(descriptions):
        msg = f"{len(headlines)} headlines vs {len(descriptions)} descriptions"
        raise ConfigError(msg)
    return [
        f"{headline}. {description}"
        for headline, description in zip(headlines, descriptions, strict=True)
    ]


def teacher_scores(matrix: Mapping[str, Labels], members: Sequence[str]) -> FloatArray:
    """The teacher: the UNWEIGHTED row-sum of the winning members' labels."""
    if not members:
        msg = "teacher needs at least one ensemble member"
        raise ConfigError(msg)
    missing = [name for name in members if name not in matrix]
    if missing:
        msg = f"teacher members missing from label matrix: {missing}"
        raise ConfigError(msg)
    columns = [np.asarray(matrix[name], dtype=np.float64) for name in members]
    return np.asarray(np.sum(columns, axis=0), dtype=np.float64)


def teacher_classes(scores: FloatArray) -> IntArray:
    """Teacher classes: the same inclusive ±1 band the students are graded with."""
    return sign_threshold(scores)


@dataclass(frozen=True)
class StudentEvaluation:
    """One fitted student's held-out agreement with the teacher."""

    accuracy: float
    n_train: int
    n_test: int

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "n_train": float(self.n_train),
            "n_test": float(self.n_test),
        }


def distil_student(
    embeddings: FloatArray,
    teacher: FloatArray,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> StudentEvaluation:
    """Fit ``LinearRegression`` on frozen embeddings; grade thresholded predictions.

    ``test_size=0.25, random_state=42`` are the post's exact split parameters.
    """
    features = np.asarray(embeddings, dtype=np.float64)
    targets = np.asarray(teacher, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != targets.shape[0]:
        msg = f"embeddings {features.shape} do not align with teacher {targets.shape}"
        raise ConfigError(msg)
    if features.shape[0] < 8:
        msg = f"need at least 8 rows to split and fit, got {features.shape[0]}"
        raise ConfigError(msg)

    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=test_size, random_state=random_state
    )
    student = LinearRegression()
    student.fit(x_train, y_train)
    predictions = np.asarray(student.predict(x_test), dtype=np.float64).flatten()
    predicted_classes = sign_threshold(predictions)
    target_classes = sign_threshold(np.asarray(y_test, dtype=np.float64))
    accuracy = float(np.mean(predicted_classes == target_classes))
    return StudentEvaluation(accuracy=accuracy, n_train=len(x_train), n_test=len(x_test))


class SurveyEncoder(Protocol):
    """What the survey needs from an embedding checkpoint."""

    @property
    def name(self) -> str: ...

    @property
    def parameter_count(self) -> float: ...

    def encode(self, sentences: Sequence[str]) -> FloatArray: ...


class ProviderSurveyEncoder:
    """Adapt any repo :class:`EmbeddingProvider` into a survey encoder.

    This is how the hash embedder (offline tests) and the model2vec default
    reuse the survey harness; parameter counts are ``nan`` because static
    providers have no torch parameters to count.
    """

    def __init__(self, provider: EmbeddingProvider, *, name: str | None = None) -> None:
        self._provider = provider
        self._name = name or provider.identifier

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameter_count(self) -> float:
        return float("nan")

    def encode(self, sentences: Sequence[str]) -> FloatArray:
        return np.asarray(
            self._provider.embed_documents(list(sentences)), dtype=np.float64
        )


class SentenceTransformerSurveyEncoder:
    """A sentence-transformers checkpoint, loaded the way the appendix loads it.

    ``trust_remote_code=True`` matches the post's loader; construction downloads
    weights, so this encoder is opt-in (tests gate it behind
    ``CYBERNAUT_MINI_SENTIMENT_DOWNLOADS``). Sentences are encoded RAW — no E5
    prefixes — to stay faithful to the source script.
    """

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            msg = (
                "the survey encoder needs the optional 'st' extra "
                "(sentence-transformers + torch); install it with `uv sync --extra st`."
            )
            raise ConfigError(msg) from exc

        self._name = model_name
        self._model = SentenceTransformer(
            model_name_or_path=model_name, revision=revision, trust_remote_code=True
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameter_count(self) -> float:
        return float(sum(p.numel() for p in self._model.parameters()))

    def encode(self, sentences: Sequence[str]) -> FloatArray:
        vectors = self._model.encode(list(sentences), show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float64)


@dataclass(frozen=True)
class SurveyRow:
    """One Results.csv row: the appendix's four-key experiment dict."""

    parameters: float
    runtime: float
    dimensions: float
    accuracy: float

    def as_dict(self) -> dict[str, float]:
        return {
            "Parameters": self.parameters,
            "Runtime": self.runtime,
            "Dimensions": self.dimensions,
            "Accuracy": self.accuracy,
        }


def run_survey(
    sentences: Sequence[str],
    matrix: Mapping[str, Labels],
    members: Sequence[str],
    encoders: Sequence[SurveyEncoder],
    *,
    baseline_columns: Sequence[str] = (),
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, SurveyRow]:
    """The four-column survey over embedding checkpoints, baselines injected once.

    For each encoder: embed, time it, fit an OLS student, grade against the
    thresholded teacher on the shared split. On the FIRST encoder only
    (``model_ix == 0`` in the appendix), every *baseline_column* — a raw
    labeler column — is graded on the same test rows and stored with NaN
    Parameters/Runtime/Dimensions. Students are keyed ``OLS({basename})``.
    """
    if not encoders:
        msg = "run_survey needs at least one encoder"
        raise ConfigError(msg)
    teacher = teacher_scores(matrix, members)
    if len(sentences) != teacher.shape[0]:
        msg = f"{len(sentences)} sentences vs {teacher.shape[0]} label rows"
        raise ConfigError(msg)

    results: dict[str, SurveyRow] = {}
    nan = float("nan")
    for model_ix, encoder in enumerate(encoders):
        start = time.perf_counter()
        embeddings = encoder.encode(sentences)
        runtime = time.perf_counter() - start

        if model_ix == 0:
            # NaN-injection of the labeler baselines, graded on the same split.
            row_index = np.arange(teacher.shape[0])
            _, test_rows = train_test_split(
                row_index, test_size=test_size, random_state=random_state
            )
            target_classes = sign_threshold(teacher[test_rows])
            for column in baseline_columns:
                if column not in matrix:
                    msg = f"baseline column {column!r} not in label matrix"
                    raise ConfigError(msg)
                labels = np.asarray(matrix[column], dtype=np.int64)[test_rows]
                accuracy = float(np.mean(labels == target_classes))
                results[column] = SurveyRow(
                    parameters=nan, runtime=nan, dimensions=nan, accuracy=accuracy
                )

        evaluation = distil_student(
            embeddings, teacher, test_size=test_size, random_state=random_state
        )
        student_name = f"OLS({encoder.name.split('/')[-1]})"
        results[student_name] = SurveyRow(
            parameters=encoder.parameter_count,
            runtime=runtime,
            dimensions=float(embeddings.shape[1]),
            accuracy=evaluation.accuracy,
        )
    return results


def write_results_csv(results: Mapping[str, SurveyRow], path: Path) -> None:
    """Write the survey as the post's ``Results.csv`` (pandas ``to_csv`` shape)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", *RESULTS_COLUMNS])
        for name, row in results.items():
            rendered = [
                "" if math.isnan(value) else repr(value)
                for value in (row.parameters, row.runtime, row.dimensions, row.accuracy)
            ]
            writer.writerow([name, *rendered])
