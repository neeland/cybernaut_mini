"""Kedro pipeline registry: build and evaluation DAGs.

The search/agent side is request-time and dynamically branching, so it is a plain
library behind the Typer CLI, not a pipeline.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline

from cybernaut_mini.pipelines.evaluation import create_pipeline as create_evaluation
from cybernaut_mini.pipelines.index_build import create_pipeline as create_index_build


def register_pipelines() -> dict[str, Pipeline]:
    index_build = create_index_build()
    evaluation = create_evaluation()
    pipelines: dict[str, Pipeline] = {
        "index_build": index_build,
        "evaluation": evaluation,
        "__default__": index_build,
    }
    return pipelines
