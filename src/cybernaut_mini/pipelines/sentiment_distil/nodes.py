"""sentiment_distil pipeline nodes: matrix + gold → ensemble → teacher → students.

The dynamic-feeling parts of the recipe (greedy selection, bootstrap) are still
deterministic functions of (matrix, gold, seed), so they sit comfortably in a
static DAG; each node returns a canonical-JSON-ready dict.

Blog ref: https://nosible.com/blog/ensemble-and-distil — the five-step pattern
    "Curate → Label → Ensemble → Distil → Scale"; this pipeline is steps three
    and four over a frozen label matrix from ``sentiment_label``. Local copy:
    ``docs/blog-archive/ensemble-and-distil.md``.

Assumptions:
    - The offline default embeds with the repo's hash provider (a real,
      deterministic provider — see ``providers/embeddings.py``); the
      sentence-transformers survey over the post's checkpoints is opt-in via
      ``survey_checkpoints`` because each checkpoint is a download.
    - Gold may be empty (unlabeled corpora): the ensemble then selects against
      the configured proxy column, the post's own move at 10k scale.
    - The bootstrap runs on the same exclusions as the selection, so the
      winner tally is a distribution over exactly the ensembles the selection
      could have produced.

Alternatives rejected:
    - Emitting the fitted sklearn students as artifacts: a pickled
      LinearRegression is neither canonical JSON nor needed — the survey's
      published object is the Results table, and refitting is milliseconds.
"""

from __future__ import annotations

from typing import Any

from cybernaut_mini.config import ConfigError, EmbeddingConfig
from cybernaut_mini.providers.embeddings import create_embedding_provider
from cybernaut_mini.sentiment.distil import (
    ProviderSurveyEncoder,
    SentenceTransformerSurveyEncoder,
    SurveyEncoder,
    run_survey,
)
from cybernaut_mini.sentiment.ensemble import bootstrap_stability, greedy_forward_selection


def _resolve_gold(
    matrix: dict[str, list[int]], gold: list[int], ensemble_params: dict[str, Any]
) -> tuple[list[int], list[str]]:
    """Gold rows when available, else the configured proxy column; plus exclusions."""
    exclude = [str(name) for name in ensemble_params.get("exclude", ["Random"])]
    n_rows = len(next(iter(matrix.values())))
    if gold and len(gold) == n_rows:
        return [int(v) for v in gold], exclude
    proxy = ensemble_params.get("proxy_column")
    if proxy is None:
        msg = (
            "no gold labels and no proxy_column configured; the ensemble needs a "
            "reference to select against"
        )
        raise ConfigError(msg)
    proxy_name = str(proxy)
    if proxy_name not in matrix:
        msg = f"proxy_column {proxy_name!r} not in label matrix"
        raise ConfigError(msg)
    return list(matrix[proxy_name]), [*exclude, proxy_name]


def select_ensemble(
    matrix: dict[str, list[int]], gold: list[int], ensemble_params: dict[str, Any]
) -> dict[str, Any]:
    """Greedy iterative addition with the per-step trace the post publishes."""
    reference, exclude = _resolve_gold(matrix, gold, ensemble_params)
    result = greedy_forward_selection(matrix, reference, exclude=exclude)
    return result.as_dict()


def bootstrap_ensemble(
    matrix: dict[str, list[int]], gold: list[int], ensemble_params: dict[str, Any]
) -> dict[str, Any]:
    """1,000-run (configurable) bootstrap on 75% subsamples; winner tally."""
    reference, exclude = _resolve_gold(matrix, gold, ensemble_params)
    shares = bootstrap_stability(
        matrix,
        reference,
        n_runs=int(ensemble_params.get("bootstrap_runs", 1000)),
        subsample=float(ensemble_params.get("bootstrap_subsample", 0.75)),
        seed=int(ensemble_params.get("bootstrap_seed", 42)),
        exclude=exclude,
    )
    return {
        "ensembles": [
            {"members": list(members), "win_share": share} for members, share in shares
        ]
    }


def _build_encoders(distil_params: dict[str, Any]) -> list[SurveyEncoder]:
    checkpoints = [str(name) for name in distil_params.get("survey_checkpoints", [])]
    if checkpoints:
        return [SentenceTransformerSurveyEncoder(name) for name in checkpoints]
    embedding = EmbeddingConfig.model_validate(distil_params.get("embedding", {}))
    provider = create_embedding_provider(embedding)
    return [ProviderSurveyEncoder(provider)]


def distil_students(
    stories: list[str],
    matrix: dict[str, list[int]],
    ensemble_trace: dict[str, Any],
    distil_params: dict[str, Any],
) -> dict[str, Any]:
    """Fit one OLS student per encoder against the winning ensemble's teacher.

    Returns the Results.csv-shaped table (Parameters/Runtime/Dimensions/
    Accuracy per row) with the labeler baselines NaN-injected on the first
    encoder, exactly as the appendix script does at ``model_ix == 0``.
    """
    members = [str(name) for name in ensemble_trace["members"]]
    encoders = _build_encoders(distil_params)
    baselines = [str(name) for name in distil_params.get("baseline_columns", sorted(matrix))]
    results = run_survey(
        stories,
        matrix,
        members,
        encoders,
        baseline_columns=[name for name in baselines if name in matrix],
        test_size=float(distil_params.get("test_size", 0.25)),
        random_state=int(distil_params.get("random_state", 42)),
    )
    return {name: row.as_dict() for name, row in results.items()}
