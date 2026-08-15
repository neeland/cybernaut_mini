"""Pure-function Kedro nodes for the index_build pipeline.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 7 says the synonym
    graph belongs to the shard, not the corpus [disclosed]: "we use the synonym graphs
    in each shard to probabilistically expand our search terms. Because each shard is
    lexically and semantically coherent, the synonyms in each shard are unambiguous.
    Put simply, 'gene' in shard 11,343 only has the genetic meaning. We don't need to
    worry that Gene is also a name and, in some other shards, would co-occur a lot
    with 'Willy Wonka'." Local copy: docs/blog-archive/the-road-to-cybernaut-1.md

Each function accepts and returns plain Python objects (dicts, lists, str) so
Kedro can pass them through its in-memory dataset layer without custom pickling.
Heavy objects (numpy arrays, pydantic models) are converted at node boundaries.

No node performs file I/O: the corpus arrives from the catalog's ``documents``
dataset and the finished index leaves as a payload dict that
:class:`~cybernaut_mini.datasets.ShardIndexDataset` writes.

Assumptions: this node does not compute the term graph itself. It delegates to
    :func:`cybernaut_mini.indexing.compute_shard_term_graphs`, which is also what any
    direct :func:`~cybernaut_mini.indexing.write_index` caller uses, so the CLI build
    and this pipeline cannot drift on the one decision that makes stage 7 work. The
    per-shard graph cap (terms and neighbours) is the library's, not this node's —
    :class:`~cybernaut_mini.config.IndexConfig` exposes ``cooccurrence_window`` and
    ``min_edge_count`` but no cap, so the documented defaults apply [inferred].

Alternatives rejected: computing one graph over the whole corpus and assigning it to
    every manifest, which is what this node used to do. Measured on a 2,000-document /
    32-shard build at production parameters: 4,072 KB of manifest per shard and
    301.1 MB of retained heap at load, against 80.1 KB and 8.1 MB once the graph is
    per shard and capped — and it hands shard 11,343 the Willy Wonka sense of "gene".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from cybernaut_mini.config import EmbeddingConfig, IndexConfig
from cybernaut_mini.indexing import (
    _shard_summary,
    _shard_title,
    compute_entities,
    compute_keywords,
    compute_shard_term_graphs,
)
from cybernaut_mini.ingest import validate_records
from cybernaut_mini.models import (
    Document,
    IndexMeta,
    ShardManifest,
)
from cybernaut_mini.providers.embeddings import (
    SentenceTransformersEmbedder,
    create_embedding_provider,
)
from cybernaut_mini.sharding import shard_documents
from cybernaut_mini.text import TextProcessor

#: Number of documents encoded per cache chunk.  At 200k docs this produces
#: 40 chunks of 5k each; a crash loses at most one chunk's worth of work.
EMBED_CHUNK_SIZE = 5_000


def _embed_cache_dir(config: EmbeddingConfig) -> Path:
    """Stable directory path for the per-build embedding chunk cache.

    Keyed on ``(identifier, revision)`` so a config change (different model,
    different revision, provider switch) automatically invalidates the cache
    without manual cleanup.
    """
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", config.identifier())
    rev = (config.revision or "unpinned")[:12]
    return Path("data") / "06_models" / "embed_cache" / f"{safe_id}-{rev}"


def _embed_with_cache(
    texts: list[str],
    provider: Any,
    config: EmbeddingConfig,
    *,
    chunk_size: int = EMBED_CHUNK_SIZE,
    cache_dir: Path | None = None,
) -> list[list[float]]:
    """Encode ``texts`` in chunks, caching each chunk to disk.

    Crash-resumable: completed chunks are loaded from ``.npy`` cache files and
    only missing or incomplete chunks are re-encoded.  A resume after a crash
    at chunk *k* re-encodes from chunk *k* onward, not from the beginning.

    Cache structure::

        data/06_models/embed_cache/<model-id>-<rev12>/
            manifest.json          # {chunk_idx: doc_count}
            chunk_0000.npy
            chunk_0001.npy
            ...

    The ``manifest.json`` records how many docs each completed chunk holds.
    A chunk is considered valid when its ``.npy`` file exists **and** the
    manifest entry matches ``len(chunk_texts)``.  A partial/corrupt ``.npy``
    is detected on the next run and overwritten.

    Parameters
    ----------
    texts:
        Flat list of ``"title\\ntext"`` strings, one per document.
    provider:
        Any embedding provider (hash, model2vec, sentence_transformers).
    config:
        The EmbeddingConfig that produced *provider*; used only to compute the
        cache key via :func:`_embed_cache_dir`.
    chunk_size:
        Documents per chunk.  Default is :data:`EMBED_CHUNK_SIZE` (5 000).
    cache_dir:
        Override the cache directory; useful for tests that inject a
        ``tmp_path`` so they never touch ``data/``.
    """
    import logging

    _log = logging.getLogger(__name__)

    resolved_cache = cache_dir if cache_dir is not None else _embed_cache_dir(config)
    resolved_cache.mkdir(parents=True, exist_ok=True)
    manifest_path = resolved_cache / "manifest.json"

    existing: dict[str, int] = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass  # corrupt manifest — treat all chunks as missing

    n_chunks = max(1, (len(texts) + chunk_size - 1) // chunk_size)
    all_vecs: list[np.ndarray] = []
    manifest: dict[str, int] = {}

    for i in range(n_chunks):
        chunk = texts[i * chunk_size : (i + 1) * chunk_size]
        key = str(i)
        chunk_path = resolved_cache / f"chunk_{i:04d}.npy"

        if chunk_path.exists() and existing.get(key) == len(chunk):
            _log.info(
                "embed cache hit  chunk=%04d/%04d  docs=%d",
                i, n_chunks - 1, len(chunk),
            )
            vecs = np.load(chunk_path).astype(np.float32)
        else:
            _log.info(
                "embed encoding   chunk=%04d/%04d  docs=%d",
                i, n_chunks - 1, len(chunk),
            )
            vecs = provider.embed_documents(chunk)
            # Atomic: write to sidecar then rename so a crash mid-write
            # never leaves a truncated .npy at the real path.
            tmp = chunk_path.parent / (chunk_path.stem + ".tmp.npy")
            np.save(tmp, vecs)
            tmp.replace(chunk_path)

        manifest[key] = len(chunk)
        all_vecs.append(vecs)

    # Atomically update the manifest so it always reflects completed chunks.
    tmp_m = manifest_path.with_suffix(".tmp")
    tmp_m.write_text(json.dumps(manifest), encoding="utf-8")
    tmp_m.replace(manifest_path)

    return np.concatenate(all_vecs, axis=0).tolist()


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

    Logs the resolved device and emits a warning when a sentence-transformers
    build silently falls back to CPU on Apple Silicon (MPS available but not
    selected) — the most common cause of a 10-20× throughput regression that
    otherwise appears only in wall-clock time after a full build.
    """
    import logging
    import platform

    _log = logging.getLogger(__name__)

    config = EmbeddingConfig.model_validate(embedding_params)
    provider = create_embedding_provider(config, offline=offline)

    _log.info("embedder=%s  dim=%d  batch_size=%s", provider.identifier, provider.dim,
              getattr(provider, "_batch_size", "n/a"))

    if isinstance(provider, SentenceTransformersEmbedder):
        from cybernaut_mini import accel

        fingerprint = accel.device_fingerprint()
        _log.info("device=%s  fingerprint=%s", provider.device, fingerprint)
        if provider.device == "cpu" and platform.machine() == "arm64":
            _log.warning(
                "ST embedder resolved to CPU on Apple Silicon — MPS is available. "
                "Run via scripts/with-accel.sh or set PYTORCH_ENABLE_MPS_FALLBACK=1 "
                "to activate it. Expect ~10-20× slower embedding than MPS."
            )

    texts = []
    for raw in documents:
        doc = Document.model_validate(raw)
        texts.append(f"{doc.title}\n{doc.text}")

    # Chunked encode with on-disk cache: a crash at chunk k loses at most
    # chunk_size docs of work; the next run resumes from chunk k+1.
    return _embed_with_cache(texts, provider, config)


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

    # Term graphs, one per shard, from that shard's documents only. A single graph
    # over the whole corpus — which this node used to compute and assign to every
    # manifest — is what stage 7 explicitly does not want ("'gene' in shard 11,343
    # only has the genetic meaning"), and it makes each manifest O(corpus). The
    # shard's TF-IDF keywords are pinned so the term cap cannot evict the terms that
    # made this shard distinctive.
    term_graph_by_shard = compute_shard_term_graphs(
        shard_doc_ids,
        all_tokens,
        window=index_config.cooccurrence_window,
        min_edge_count=index_config.min_edge_count,
        priority_terms={
            shard_id: [kw.term for kw in keywords]
            for shard_id, keywords in keywords_by_shard.items()
        },
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
            term_graph=term_graph_by_shard[shard_id],
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
