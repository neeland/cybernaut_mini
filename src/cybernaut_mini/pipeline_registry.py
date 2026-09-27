"""Kedro pipeline registry: build and evaluation DAGs.

The search/agent side is request-time and dynamically branching, so it is a plain
library behind the Typer CLI, not a pipeline.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — the build-time half
    of the architecture diagram is a DAG (corpus -> documents -> shards -> index)
    and the query-time half is the eight-stage request path. This module registers
    the former and deliberately leaves the latter in the library. Local copy:
    ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - The registry is the complete list of DAGs Kedro can run. ``production`` is
      not a new pipeline but the composition
      ``corpus_ingest + judgment_ingest + index_build``, so acquisition and build
      stay separately runnable and separately debuggable.
    - ``__default__`` is ``index_build``: a bare ``kedro run`` builds an index
      rather than re-fetching a corpus, because fetching is the slow, networked,
      once-only step.
    - Routing, retrieval and agent search are not registered. They branch at
      request time on prior results and share one budget counter across stages,
      which Kedro's static-DAG catalog model cannot express; ``cli.py`` exposes
      them as Typer commands, and ``kedro viz`` therefore shows only the build and
      evaluation DAGs.
    - The ``sentiment_label`` and ``sentiment_distil`` packages exist and are not
      registered here; they run through their node functions directly until their
      catalog entries and parameter blocks land (see their READMEs).

Alternatives considered:
    - Registering an ``agent`` pipeline that wraps one search request: makes the
      agent visible in ``kedro viz``, but a one-node DAG over a question string is
      a dashboard for a library call, and it would imply the search is replayable
      from catalog datasets when it is not.
    - Registering ``evaluation`` inside ``production``: a green build would then
      imply a benchmark pass, and re-evaluating an existing index would require
      rebuilding it first.
    - Shipping only the flat pipelines without ``production``: callers would have
      to remember the three-pipeline composition, and the named composition is
      what keeps an end-to-end run honest about what it includes.
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
