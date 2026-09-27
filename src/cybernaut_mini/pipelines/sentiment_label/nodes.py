"""sentiment_label pipeline nodes: stories → label matrix → reports.

Each node is a pure function over plain rows/dicts so the DAG stays replayable
and every output serialises through the canonical JSON writer unchanged. The
labeler pool itself is constructed from parameters inside the labeling node —
labelers are not data, so they never cross a node boundary.

Blog ref: https://nosible.com/blog/news-sentiment-showdown-who-checks-vibes-best —
    the post's method as a static DAG: build the representative dataset, label
    it with every model, publish the agreement matrices, the timing table, and
    the 10M-story cost bars. Local copy:
    ``docs/blog-archive/news-sentiment-showdown-who-checks-vibes-best.md``.

Assumptions:
    - Input rows are the cached NOSIBLE/financial-sentiment JSONL (``text`` +
      string ``label``) or any row list with those keys; the gold column is
      emitted separately so the benchmark can use it as reference and the
      ensemble as its target without the matrix carrying a magic column.
    - The timing node re-runs the pool over a bounded slice (``timing_rows``)
      rather than reusing the labeling node's wall clock, because Kedro node
      runtimes include I/O and would poison the model-vs-model ratio.
    - The default reference for the agreement matrix is the gold column when
      present, else the configured proxy column — the post's two matrices are
      exactly these two cases.

Alternatives rejected:
    - One mega-node returning every report: cheaper to wire but the reports
      have different cadences (costs change when prices change, agreement only
      when labels do), and separate nodes let Kedro skip what is unchanged.
    - Passing constructed labelers between nodes: not serialisable, and a
      labeler carrying a loaded FinBERT would defeat catalog caching.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cybernaut_mini.config import ConfigError
from cybernaut_mini.sentiment.benchmark import (
    agreement_matrix,
    hypothetical_cost,
    kappa_with_reference,
    time_labelers,
)
from cybernaut_mini.sentiment.data import LABEL_TO_INT
from cybernaut_mini.sentiment.labelers import build_label_matrix, build_pool


def prepare_stories(
    rows: list[dict[str, Any]], labeler_params: dict[str, Any]
) -> list[str]:
    """Extract the story text column ('text' by default) from raw rows."""
    field = str(labeler_params.get("text_field", "text"))
    max_rows: int | None = labeler_params.get("max_rows")
    stories = [str(row[field]) for row in rows[: max_rows or len(rows)] if row.get(field)]
    if not stories:
        msg = f"no stories found under field {field!r}"
        raise ConfigError(msg)
    return stories


def extract_gold_labels(
    rows: list[dict[str, Any]], labeler_params: dict[str, Any]
) -> list[int]:
    """Gold labels as integers; string labels map positive/neutral/negative → 1/0/-1."""
    field = str(labeler_params.get("label_field", "label"))
    max_rows: int | None = labeler_params.get("max_rows")
    labels: list[int] = []
    for row in rows[: max_rows or len(rows)]:
        if not row.get(labeler_params.get("text_field", "text")):
            continue
        value = row.get(field)
        if isinstance(value, str):
            key = value.strip().casefold()
            if key not in LABEL_TO_INT:
                msg = f"unknown gold label {value!r}"
                raise ConfigError(msg)
            labels.append(LABEL_TO_INT[key])
        elif value is None:
            msg = f"row missing gold label field {field!r}"
            raise ConfigError(msg)
        else:
            labels.append(int(value))
    return labels


def label_stories(
    stories: list[str], labeler_params: dict[str, Any]
) -> dict[str, list[int]]:
    """Run the configured labeler pool: one column per labeler."""
    pool = build_pool(labeler_params)
    return build_label_matrix(stories, pool)


def benchmark_labels(
    matrix: dict[str, list[int]],
    gold: list[int],
    benchmark_params: dict[str, Any],
) -> dict[str, Any]:
    """Agreement matrix (sorted by reference), kappa table, Random baseline row."""
    full: dict[str, list[int]] = dict(matrix)
    reference = "gold"
    if gold and len(gold) == len(next(iter(matrix.values()))):
        full["gold"] = [int(v) for v in gold]
    else:
        reference = str(benchmark_params.get("reference", next(iter(matrix))))
    seed = benchmark_params.get("random_seed", 42)
    result = agreement_matrix(
        full, reference, random_seed=None if seed is None else int(seed)
    )
    return {
        "agreement": result.as_dict(),
        "kappa": kappa_with_reference(full, reference),
    }


def time_pool(
    stories: list[str], labeler_params: dict[str, Any]
) -> dict[str, dict[str, float]]:
    """Wall-clock the pool over a bounded slice — the VADER-vs-FinBERT table."""
    n_rows = int(labeler_params.get("timing_rows", min(len(stories), 64)))
    pool = build_pool(labeler_params)
    timings = time_labelers(stories[:n_rows], pool)
    return {name: result.as_dict() for name, result in timings.items()}


def estimate_costs(benchmark_params: dict[str, Any]) -> dict[str, Any]:
    """Hypothetical hosted-LLM labeling cost per configured price model."""
    n_stories = int(benchmark_params.get("n_stories", 10_000_000))
    reports: dict[str, Any] = {}
    for spec in benchmark_params.get("cost_models", []):
        model: Mapping[str, Any] = spec
        reports[str(model["name"])] = hypothetical_cost(
            prompt_tokens_per_story=float(model["prompt_tokens_per_story"]),
            completion_tokens_per_story=float(model["completion_tokens_per_story"]),
            usd_per_million_prompt_tokens=float(model["usd_per_million_prompt_tokens"]),
            usd_per_million_completion_tokens=float(
                model["usd_per_million_completion_tokens"]
            ),
            n_stories=n_stories,
        )
    return {"n_stories": n_stories, "models": reports}
