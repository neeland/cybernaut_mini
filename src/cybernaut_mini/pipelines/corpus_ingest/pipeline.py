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
    build_miracl_judgments,
    merge_documents,
    normalize_corpus,
    select_documents,
    snapshot_raw_corpus,
    validate_judgments,
)


def create_judgment_pipeline() -> Pipeline:
    """Return the judgment_ingest pipeline: MIRACL source → judgments.

    Runs independently of ``corpus_ingest``. The ``validate_judgments`` node
    depends on ``documents`` (produced by the corpus branch), so it must run
    after that branch completes. In practice: run ``judgment_ingest`` only when
    ``documents`` already exists (post corpus_ingest), or as part of ``production``
    where Kedro resolves the dependency automatically.
    """
    return pipeline(
        [
            node(
                func=build_miracl_judgments,
                inputs="miracl_en_dev_source",
                outputs="miracl_judgments_unvalidated",
                name="build_miracl_judgments",
            ),
            node(
                func=validate_judgments,
                inputs=["miracl_judgments_unvalidated", "documents"],
                outputs="judgments",
                name="validate_judgments",
            ),
        ]
    )


def create_pipeline() -> Pipeline:
    """Return the corpus_ingest pipeline: two source branches → merge → select.

    DAG shape::

        raw_ccnews_source  → snapshot_raw_ccnews  → normalize_ccnews  ─┐
                                                                         ├─ merge_documents → select_documents → documents
        raw_miracl_source  → snapshot_raw_miracl  → normalize_miracl  ─┘

    Why two branches:
        CC-News provides the bulk English corpus (~190k rows after language
        filter). MIRACL corpus provides the 7,921 unique passages referenced in
        the MIRACL en-dev qrels. Keeping them separate lets each source use its
        own ``CorpusSourceConfig`` (different column names, id prefixes, field
        maps) without any conditional logic inside the nodes.

    Fidelity guarantee:
        ``merge_documents`` places MIRACL passages after CC-News rows, so
        ``select_documents`` sees them before the ``max_documents`` cap is
        applied. Provided ``max_documents`` ≥ total MIRACL passages (~8k), every
        qrels-positive document survives into the primary layer.

    Catalog entries required (``conf/base/catalog.yml``):
        - ``raw_ccnews_source`` — :class:`HuggingFaceDataset` (streaming, filter en)
        - ``raw_ccnews`` — :class:`kedro_datasets.json.JSONDataset`
        - ``normalized_ccnews`` — :class:`kedro_datasets.json.JSONDataset`
        - ``raw_miracl_source`` — :class:`HfFileDataset` (JSONL.gz via HfFileSystem)
        - ``raw_miracl`` — :class:`kedro_datasets.json.JSONDataset`
        - ``normalized_miracl`` — :class:`kedro_datasets.json.JSONDataset`

    Parameters required (``conf/base/parameters.yml``):
        - ``corpus_ccnews`` — :class:`CorpusSourceConfig` for CC-News
        - ``corpus_miracl`` — :class:`CorpusSourceConfig` for MIRACL corpus
        - ``corpus_selection`` — unchanged (language filter + max_documents cap)
    """
    return pipeline(
        [
            # ── CC-News branch ─────────────────────────────────────────────
            node(
                func=snapshot_raw_corpus,
                inputs="raw_ccnews_source",
                outputs="raw_ccnews",
                name="snapshot_raw_ccnews",
            ),
            node(
                func=normalize_corpus,
                inputs=["raw_ccnews", "params:corpus_ccnews"],
                outputs="normalized_ccnews",
                name="normalize_ccnews",
            ),
            # ── MIRACL corpus branch ────────────────────────────────────────
            node(
                func=snapshot_raw_corpus,
                inputs="raw_miracl_source",
                outputs="raw_miracl",
                name="snapshot_raw_miracl",
            ),
            node(
                func=normalize_corpus,
                inputs=["raw_miracl", "params:corpus_miracl"],
                outputs="normalized_miracl",
                name="normalize_miracl",
            ),
            # ── Merge + select ──────────────────────────────────────────────
            node(
                func=merge_documents,
                inputs=["normalized_ccnews", "normalized_miracl"],
                outputs="merged_documents",
                name="merge_documents",
            ),
            node(
                func=select_documents,
                inputs=["merged_documents", "params:corpus_selection"],
                outputs="documents",
                name="select_documents",
            ),
        ]
    )
