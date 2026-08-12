"""Pure-function Kedro nodes for the index_build pipeline.

Each function accepts and returns plain Python objects (dicts, lists, str) so
Kedro can pass them through its in-memory dataset layer without custom pickling.
Heavy objects (numpy arrays, pydantic models) are converted at node boundaries.

No node performs file I/O: the corpus arrives from the catalog's ``documents``
dataset and the finished index leaves as a payload dict that
:class:`~cybernaut_mini.datasets.ShardIndexDataset` writes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cybernaut_mini.config import EmbeddingConfig, IndexConfig
from cybernaut_mini.indexing import (
    _shard_summary,
    _shard_title,
    compute_entities,
    compute_keywords,
    compute_term_graph,
)
from cybernaut_mini.ingest import validate_records
from cybernaut_mini.models import (
    Document,
    IndexMeta,
    ShardManifest,
)
from cybernaut_mini.providers.embeddings import create_embedding_provider
from cybernaut_mini.sharding import shard_documents
from cybernaut_mini.text import TextProcessor


def ingest_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate the catalog-supplied corpus before anything expensive runs.

    The catalog has already parsed the JSONL; this enforces the Document schema
    and the no-duplicate-ids rule, so a malformed corpus fails before the
    embedding step rather than part-way through it.

    Returns a list of model dicts (model_dump output) for downstream nodes.
    """
    if not documents:
        msg = "corpus is empty; nothing to index"
        raise ValueError(msg)
    docs = validate_records(list(enumerate(documents, start=1)), label="record")
    return [doc.model_dump(mode="json") for doc in docs]


def process_text(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Tokenize and extract entities for each document.

    Uses ``TextProcessor(use_spacy=None)`` which auto-detects spaCy availability.

    Returns a dict with keys:
    - ``"tokens"``: mapping doc_id -> list of content tokens
    - ``"entities"``: mapping doc_id -> list of entity strings
    """
    processor = TextProcessor(use_spacy=None)
    tokens: dict[str, list[str]] = {}
    entities: dict[str, list[str]] = {}
    for raw in documents:
        doc = Document.model_validate(raw)
        combined = f"{doc.title}\n{doc.text}"
        tokens[doc.id] = processor.content_tokens(combined)
        entities[doc.id] = processor.entities(combined)
    return {"tokens": tokens, "entities": entities}


def embed_documents(
    documents: list[dict[str, Any]],
    embedding_params: dict[str, Any],
    offline: bool,
) -> list[list[float]]:
    """Embed each document as 'title\\ntext' using the configured provider.

    Returns a list of float lists (row i = documents[i]) so Kedro can
    serialize the result; the node downstream converts back to numpy.
    """
    config = EmbeddingConfig.model_validate(embedding_params)
    provider = create_embedding_provider(config, offline=offline)
    texts = []
    for raw in documents:
        doc = Document.model_validate(raw)
        texts.append(f"{doc.title}\n{doc.text}")
    matrix = provider.embed_documents(texts)
    result: list[list[float]] = matrix.tolist()
    return result


def shard(
    vectors_list: list[list[float]],
    index_params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Run MiniBatchKMeans sharding with repair and balancing.

    Returns ``{"labels": [...], "centroids": [[...], ...]}``.
    """
    vectors = np.array(vectors_list, dtype=np.float32)
    index_config = IndexConfig.model_validate(index_params)
    result = shard_documents(vectors, n_shards=index_config.n_shards, seed=seed)
    return {
        "labels": result.labels,
        "centroids": result.centroids.tolist(),
    }


def build_manifests(
    documents: list[dict[str, Any]],
    vectors_list: list[list[float]],
    shard_result: dict[str, Any],
    text_result: dict[str, Any],
    embedding_params: dict[str, Any],
    index_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute per-shard features and produce ShardManifest dicts."""
    docs = [Document.model_validate(raw) for raw in documents]
    doc_by_id = {doc.id: doc for doc in docs}
    vectors = np.array(vectors_list, dtype=np.float32)
    row_map = {doc.id: idx for idx, doc in enumerate(docs)}

    labels: list[int] = shard_result["labels"]
    centroids_list: list[list[float]] = shard_result["centroids"]
    centroids = np.array(centroids_list, dtype=np.float32)

    all_tokens: dict[str, list[str]] = text_result["tokens"]
    all_entities: dict[str, list[str]] = text_result["entities"]

    embedding_config = EmbeddingConfig.model_validate(embedding_params)
    index_config = IndexConfig.model_validate(index_params)
    n_shards = index_config.n_shards

    # Group documents by shard.
    shard_doc_ids: dict[int, list[str]] = {s: [] for s in range(n_shards)}
    for idx, doc in enumerate(docs):
        shard_doc_ids[labels[idx]].append(doc.id)

    # Per-shard token lists (concatenated content tokens).
    shard_tokens: dict[int, list[str]] = {
        s: [tok for doc_id in ids for tok in all_tokens.get(doc_id, [])]
        for s, ids in shard_doc_ids.items()
    }

    # Compute keywords.
    keywords_by_shard = compute_keywords(shard_tokens, index_config.max_keywords)

    # Compute term graph (over all docs).
    all_doc_tokens_list = [all_tokens.get(doc.id, []) for doc in docs]
    global_term_graph = compute_term_graph(
        all_doc_tokens_list,
        window=index_config.cooccurrence_window,
        min_edge_count=index_config.min_edge_count,
    )

    embedding_model = embedding_config.identifier()

    manifests: list[dict[str, Any]] = []
    for shard_id in range(n_shards):
        doc_ids = shard_doc_ids[shard_id]
        centroid = centroids[shard_id]

        # Per-shard entity aggregation.
        shard_entity_lists = [all_entities.get(doc_id, []) for doc_id in doc_ids]
        entities = compute_entities(shard_entity_lists, index_config.max_entities)

        keywords = keywords_by_shard[shard_id]

        # Shard summary: titles of 3 closest docs to centroid.
        summary = _shard_summary(shard_id, doc_ids, doc_by_id, vectors, row_map, centroid)
        title = _shard_title(shard_id, keywords)

        manifest = ShardManifest(
            shard_id=shard_id,
            document_ids=doc_ids,
            centroid=centroid.tolist(),
            title=title,
            summary=summary,
            keywords=keywords,
            entities=entities,
            term_graph=global_term_graph,
            document_count=len(doc_ids),
            embedding_model=embedding_model,
        )
        manifests.append(manifest.model_dump(mode="json"))

    return manifests


def build_index_payload(
    documents: list[dict[str, Any]],
    vectors_list: list[list[float]],
    manifests_list: list[dict[str, Any]],
    text_result: dict[str, Any],
    embedding_params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Assemble the complete index as a plain payload for the catalog to persist.

    Returns the dict consumed by :class:`~cybernaut_mini.datasets.ShardIndexDataset`,
    whose ``save`` performs the canonical write. The node itself touches no path,
    so the index location is a catalog concern and swapping it per environment
    needs no code change.
    """
    vectors = np.array(vectors_list, dtype=np.float32)

    embedding_config = EmbeddingConfig.model_validate(embedding_params)
    embedding_model = embedding_config.identifier()

    meta = IndexMeta(
        embedding_model=embedding_model,
        embedding_revision=(
            embedding_config.revision if embedding_config.provider != "hash" else None
        ),
        embedding_dim=vectors.shape[1],
        n_shards=len(manifests_list),
        n_documents=len(documents),
        seed=seed,
    )

    return {
        "meta": meta.model_dump(mode="json"),
        "documents": documents,
        "vectors": vectors_list,
        "manifests": manifests_list,
        "doc_tokens": text_result["tokens"],
    }
