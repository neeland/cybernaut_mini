"""sentiment_distil pipeline definition.

Blog ref: https://nosible.com/blog/ensemble-and-distil — steps three and four of
    "Curate → Label → Ensemble → Distil → Scale", run over the frozen label
    matrix that ``sentiment_label`` produced. Local copy:
    ``docs/blog-archive/ensemble-and-distil.md``.

Assumptions:
    - NOT registered in ``pipeline_registry.py``; the integration pass owns
      registration and the catalog entries below.
    - Consumes ``sentiment_stories`` / ``sentiment_label_matrix`` /
      ``sentiment_gold`` by name, so composing ``sentiment_label +
      sentiment_distil`` into one run needs no glue nodes.

Alternatives rejected:
    - Re-labeling inside this pipeline: distillation sweeps (encoders, seeds,
      thresholds) must be free to re-run without a single labeler call — the
      matrix is the frozen interface between the two posts.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from cybernaut_mini.pipelines.sentiment_distil.nodes import (
    bootstrap_ensemble,
    distil_students,
    select_ensemble,
)


def create_pipeline() -> Pipeline:
    """Return the sentiment_distil pipeline.

    DAG shape::

        sentiment_label_matrix ─┬─ select_ensemble ──→ sentiment_ensemble_trace ─┐
        sentiment_gold ─────────┼─ bootstrap ───────→ sentiment_bootstrap_report │
        sentiment_stories ──────┴────────────────────── distil_students ←────────┘
                                                              │
                                                  sentiment_distil_results

    Catalog entries required (integration step):
        - ``sentiment_ensemble_trace``, ``sentiment_bootstrap_report``,
          ``sentiment_distil_results`` — ``data/08_reporting`` (JSON).

    Parameters required: ``sentiment_ensemble``, ``sentiment_distil``
    (see ``configs/sentiment/labelers.yaml`` for the reference block).
    """
    return pipeline(
        [
            node(
                func=select_ensemble,
                inputs=[
                    "sentiment_label_matrix",
                    "sentiment_gold",
                    "params:sentiment_ensemble",
                ],
                outputs="sentiment_ensemble_trace",
                name="select_sentiment_ensemble",
            ),
            node(
                func=bootstrap_ensemble,
                inputs=[
                    "sentiment_label_matrix",
                    "sentiment_gold",
                    "params:sentiment_ensemble",
                ],
                outputs="sentiment_bootstrap_report",
                name="bootstrap_sentiment_ensemble",
            ),
            node(
                func=distil_students,
                inputs=[
                    "sentiment_stories",
                    "sentiment_label_matrix",
                    "sentiment_ensemble_trace",
                    "params:sentiment_distil",
                ],
                outputs="sentiment_distil_results",
                name="distil_sentiment_students",
            ),
        ]
    )
