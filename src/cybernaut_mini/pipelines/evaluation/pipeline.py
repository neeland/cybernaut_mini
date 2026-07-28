"""Evaluation pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from cybernaut_mini.pipelines.evaluation.nodes import (
    evaluate_node,
    load_index_node,
    load_judgments_node,
    report_node,
)


def create_pipeline() -> Pipeline:
    """Return the evaluation pipeline."""
    return pipeline(
        [
            node(
                func=load_index_node,
                inputs=["params:index_path"],
                outputs="validated_index_path",
                name="load_index_node",
            ),
            node(
                func=load_judgments_node,
                inputs=["params:judgments_path"],
                outputs="judgments_list",
                name="load_judgments_node",
            ),
            node(
                func=evaluate_node,
                inputs=[
                    "validated_index_path",
                    "judgments_list",
                    "params:embedding",
                    "params:rrf",
                    "params:agent",
                    "params:seed",
                    "params:offline",
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
