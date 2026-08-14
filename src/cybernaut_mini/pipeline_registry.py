"""Kedro pipeline registry: build and evaluation DAGs.

The search/agent side is request-time and dynamically branching, so it is a plain
library behind the Typer CLI, not a pipeline.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline

from cybernaut_mini.pipelines.corpus_ingest import create_pipeline as create_corpus_ingest
from cybernaut_mini.pipelines.corpus_ingest.pipeline import create_judgment_pipeline
from cybernaut_mini.pipelines.evaluation import create_pipeline as create_evaluation
from cybernaut_mini.pipelines.index_build import create_pipeline as create_index_build


def register_pipelines() -> dict[str, Pipeline]:
    corpus_ingest = create_corpus_ingest()
    judgment_ingest = create_judgment_pipeline()
    index_build = create_index_build()
    evaluation = create_evaluation()
    pipelines: dict[str, Pipeline] = {
        "corpus_ingest": corpus_ingest,
        "judgment_ingest": judgment_ingest,
        "index_build": index_build,
        "evaluation": evaluation,
        # Acquisition then build: `kedro run --pipeline production` takes a pinned
        # source all the way to a queryable index, including validated judgments.
        "production": corpus_ingest + judgment_ingest + index_build,
        "__default__": index_build,
    }
    return pipelines
