"""Evaluation pipeline definition."""

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
