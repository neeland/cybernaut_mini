"""Evaluation pipeline definition.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — the evaluation
    harness for the eight-stage pipeline: the post's per-shard "Evals" idea taken
    to whole-corpus modes [inferred]. Local copy:
    ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - The DAG is three nodes and one direction: validate -> evaluate -> report.
      Validation is its own node so a bad qrels file fails before any retrieval
      runs, and the report is its own node so the metrics list stays the
      persisted artifact and the summary can be regenerated from it.
    - Every tunable arrives as a Kedro parameter (``params:embedding``,
      ``params:rrf``, ``params:agent``, ``params:seed``, ``params:offline``,
      ``params:shard_beam_n``), so a benchmark run is described entirely by
      config plus the two catalog paths and ``kedro viz`` shows the real inputs.
    - ``judgments_list`` and ``metrics_list`` are plain JSON-compatible lists
      because the catalog boundary is JSON; the nodes convert to pydantic models
      and back on either side of the dataset.
    - ``shard_beam_n`` defaults to 100, matching ``DEFAULT_CANDIDATE_DEPTH`` in
      ``query/s5_select`` — the router width the system actually runs with — so
      the reported ``shard_recall@N`` describes this replica rather than an
      arbitrarily wide router.

Alternatives considered:
    - A single node calling ``evals.evaluate`` directly: the CLI already does
      this and it is fewer moving parts. Rejected for the Kedro path because the
      point of the pipeline is a replayable, viz-able artifact in which the qrels
      and the index are catalog entries rather than CLI arguments.
    - Running the four modes as four parallel nodes: tempting for wall-clock, but
      all four share one routing call for the static modes, so four nodes would
      either recompute it or need a fifth dataset to carry the shared result.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from cybernaut_mini.pipelines.evaluation.nodes import (
    evaluate_node,
    report_node,
    validate_judgments,
)


def create_pipeline() -> Pipeline:
    """Return the evaluation pipeline."""
    return pipeline(
        [
            node(
                func=validate_judgments,
                inputs="judgments",
                outputs="judgments_list",
                name="validate_judgments",
            ),
            node(
                func=evaluate_node,
                inputs=[
                    "shard_index",
                    "judgments_list",
                    "params:embedding",
                    "params:rrf",
                    "params:agent",
                    "params:seed",
                    "params:offline",
                    "params:shard_beam_n",
                ],
                outputs="metrics_list",
                name="evaluate_node",
            ),
            node(
                func=report_node,
                inputs=["metrics_list"],
                outputs="eval_report",
                name="report_node",
            ),
        ]
    )
