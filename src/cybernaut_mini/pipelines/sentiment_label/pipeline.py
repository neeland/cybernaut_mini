"""sentiment_label pipeline definition.

A static DAG per the repo's Kedro-for-static / library-for-dynamic split: the
labeling survey has a fixed shape (stories → label matrix → three reports), so
it belongs in a pipeline, while anything iterative (active learning, resolver
loops) stays library code.

Blog ref: https://nosible.com/blog/news-sentiment-showdown-who-checks-vibes-best —
    the post's end-to-end method: curate a representative dataset, label it
    with a pool of models under one shared prompt, publish agreement, timing
    and cost. Local copy:
    ``docs/blog-archive/news-sentiment-showdown-who-checks-vibes-best.md``.

Assumptions:
    - NOT registered in ``pipeline_registry.py`` — the integration pass owns
      registration, the catalog entries, and the parameter block (see the
      README's catalog section).
    - The label matrix is the expensive artifact (it may include model
      downloads and LLM calls), so it is its own catalog output; the reports
      re-derive from it for free.

Alternatives rejected:
    - Folding this into ``sentiment_distil``: the two posts are separable and
      the distil pipeline should be re-runnable over a frozen label matrix
      without re-labeling anything.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from cybernaut_mini.pipelines.sentiment_label.nodes import (
    benchmark_labels,
    estimate_costs,
    extract_gold_labels,
    label_stories,
    prepare_stories,
    time_pool,
)


def create_pipeline() -> Pipeline:
    """Return the sentiment_label pipeline.

    DAG shape::

        sentiment_rows ─ prepare_stories ─→ sentiment_stories ─┐
        sentiment_rows ─ extract_gold ───→ sentiment_gold ──┐  ├─ label ─→ matrix
                                                            └──┴─ benchmark / timing
                                                       params ── costs

    Catalog entries required (integration step):
        - ``sentiment_rows`` — JSONL rows (the cached NOSIBLE/financial-sentiment
          snapshot under ``data/01_raw/financial_sentiment/``).
        - ``sentiment_stories``, ``sentiment_gold`` — ``data/02_intermediate``.
        - ``sentiment_label_matrix`` — ``data/03_primary`` (JSON).
        - ``sentiment_agreement_report``, ``sentiment_timing_report``,
          ``sentiment_cost_report`` — ``data/08_reporting`` (JSON).

    Parameters required: ``sentiment_labelers``, ``sentiment_benchmark``
    (see ``configs/sentiment/labelers.yaml`` for the reference block).
    """
    return pipeline(
        [
            node(
                func=prepare_stories,
                inputs=["sentiment_rows", "params:sentiment_labelers"],
                outputs="sentiment_stories",
                name="prepare_sentiment_stories",
            ),
            node(
                func=extract_gold_labels,
                inputs=["sentiment_rows", "params:sentiment_labelers"],
                outputs="sentiment_gold",
                name="extract_sentiment_gold",
            ),
            node(
                func=label_stories,
                inputs=["sentiment_stories", "params:sentiment_labelers"],
                outputs="sentiment_label_matrix",
                name="label_sentiment_stories",
            ),
            node(
                func=benchmark_labels,
                inputs=[
                    "sentiment_label_matrix",
                    "sentiment_gold",
                    "params:sentiment_benchmark",
                ],
                outputs="sentiment_agreement_report",
                name="benchmark_sentiment_labels",
            ),
            node(
                func=time_pool,
                inputs=["sentiment_stories", "params:sentiment_labelers"],
                outputs="sentiment_timing_report",
                name="time_sentiment_pool",
            ),
            node(
                func=estimate_costs,
                inputs="params:sentiment_benchmark",
                outputs="sentiment_cost_report",
                name="estimate_sentiment_costs",
            ),
        ]
    )
