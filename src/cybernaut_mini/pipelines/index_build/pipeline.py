"""index_build pipeline definition.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — this is the build
    half of the architecture diagram: documents in, a sharded and embedded index
    out. It produces the artifacts stages 5-8 read at request time (centroids and
    shard summaries for stage 5, keywords and entities for the sparse selectors,
    per-shard term graphs for stage 7). Local copy:
    ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - Six pure nodes over plain Python values (dicts, lists, floats) — never
      paths, never numpy arrays. ``ShardIndexDataset.save`` performs the canonical
      write when the ``shard_index`` dataset is persisted, and that write builds
      the per-shard artifacts (phrase bloom, zstd dictionary, vocabulary) — not a
      node.
    - ``process_text`` and ``embed_documents`` both branch off ``raw_documents``
      and are independent, while ``shard`` depends only on ``vectors_list``. The
      DAG says so explicitly instead of serialising the branches by accident.
    - ``ingest_documents`` validates the Document schema and the
      no-duplicate-ids rule, so a malformed corpus fails here rather than part-way
      through a multi-hour build.
    - The embedding step is crash-resumable through its own on-disk chunk cache
      (5,000 documents per chunk, keyed on model id + revision). That cache under
      ``data/06_models/embed_cache`` is the single place this pipeline accepts a
      file-system side effect; no node opens a corpus or index path.
    - ``__default__`` in :mod:`cybernaut_mini.pipeline_registry` is this pipeline,
      so a bare ``kedro run`` builds an index.

Alternatives considered:
    - One monolithic build node: fewer datasets to name, but it would hide the
      four expensive, separately cacheable steps from ``kedro viz`` and force a
      crash resume to restart the entire build.
    - Letting nodes write the index directly: removes the payload dict and the
      ``ShardIndexDataset`` indirection, but then the index location becomes a
      code constant instead of a catalog entry, and the prod environment could
      not point a build at a different artifact directory without a code change.
    - Computing one corpus-wide term graph and copying it into every manifest: see
      ``nodes.py`` for the measurements. Rejected on manifest size, retained heap,
      and the stage-7 requirement that a shard's synonyms be unambiguous within
      that shard.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from cybernaut_mini.pipelines.index_build.nodes import (
    build_index_payload,
    build_manifests,
    embed_documents,
    ingest_documents,
    process_text,
    shard,
)


def create_pipeline() -> Pipeline:
    """Return the index_build pipeline."""
    return pipeline(
        [
            node(
                func=ingest_documents,
                inputs=["documents"],
                outputs="raw_documents",
                name="ingest_documents",
            ),
            node(
                func=process_text,
                inputs=["raw_documents"],
                outputs="text_result",
                name="process_text",
            ),
            node(
                func=embed_documents,
                inputs=["raw_documents", "params:embedding", "params:offline"],
                outputs="vectors_list",
                name="embed_documents",
            ),
            node(
                func=shard,
                inputs=["vectors_list", "params:index", "params:seed"],
                outputs="shard_result",
                name="shard",
            ),
            node(
                func=build_manifests,
                inputs=[
                    "raw_documents",
                    "vectors_list",
                    "shard_result",
                    "text_result",
                    "params:embedding",
                    "params:index",
                ],
                outputs="manifests_list",
                name="build_manifests",
            ),
            node(
                func=build_index_payload,
                inputs=[
                    "raw_documents",
                    "vectors_list",
                    "manifests_list",
                    "text_result",
                    "params:embedding",
                    "params:seed",
                ],
                outputs="shard_index",
                name="build_index_payload",
            ),
        ]
    )
