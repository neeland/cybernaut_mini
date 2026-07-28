"""index_build pipeline definition."""

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
