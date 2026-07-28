"""corpus_ingest pipeline definition.

Acquisition is a separate pipeline from ``index_build`` on purpose: fetching a
corpus is a slow, networked, rate-limited step that should run once, while an index
gets rebuilt many times over the same snapshot as configuration is tuned.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — supplies the
    "Documents" node at the head of the build-time flow. Local copy:
    ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - Acquisition and build have different cadences. Splitting them means a config
      sweep costs zero network calls.
    - ``corpus_ingest + index_build`` composes into the ``production`` pipeline, so
      the split costs nothing when a single end-to-end run is what you want.

Alternatives considered:
    - One pipeline with fetch as its first node: a single command, no composition
      needed. Rejected because every index rebuild would then re-hit the Hub, which
      is slow, rate-limited, and pointless when the snapshot has not changed.
    - A standalone ``scripts/fetch_corpus.py``: the smallest possible change.
      Rejected because it puts acquisition outside the catalog, which is exactly the
      arrangement this pipeline exists to remove — no lineage, no environment
      switching, no versioning.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from cybernaut_mini.pipelines.corpus_ingest.nodes import (
    normalize_corpus,
    select_documents,
    snapshot_raw_corpus,
)


def create_pipeline() -> Pipeline:
    """Return the corpus_ingest pipeline (raw -> intermediate -> primary)."""
    return pipeline(
        [
            node(
                func=snapshot_raw_corpus,
                inputs="raw_corpus_source",
                outputs="raw_corpus",
                name="snapshot_raw_corpus",
            ),
            node(
                func=normalize_corpus,
                inputs=["raw_corpus", "params:corpus"],
                outputs="normalized_documents",
                name="normalize_corpus",
            ),
            node(
                func=select_documents,
                inputs=["normalized_documents", "params:corpus_selection"],
                outputs="documents",
                name="select_documents",
            ),
        ]
    )
