"""Feature computation and index persistence/loading.

Produces keyword, entity, and term-graph features for each shard, then writes
the full index to disk in a canonical, byte-for-byte deterministic format.
The ``_VALID`` sentinel is written last so a partial write is never mistaken for
a complete index.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt

from cybernaut_mini.models import (
    ARTIFACT_VERSION,
    Document,
    IndexMeta,
    ShardEntity,
    ShardKeyword,
    ShardManifest,
    canonical_dumps,
)

FloatArray = npt.NDArray[np.float32]


class IndexLoadError(ValueError):
    """Raised when a LoadedIndex cannot be read from disk."""


# ------------------------------------------------------------------ #
# Feature computation                                                 #
# ------------------------------------------------------------------ #


def compute_keywords(
    shard_tokens: dict[int, list[str]],
    max_keywords: int,
) -> dict[int, list[ShardKeyword]]:
    """Compute TF-IDF keywords per shard.

    Each shard's concatenated token list is treated as a single document.
    idf = log((1 + n_shards) / (1 + shards_containing_term)) + 1.
    Returns top ``max_keywords`` per shard sorted by (-weight, term).
    """
    n_shards = len(shard_tokens)

    # Document frequency: how many shards contain each term.
    df: Counter[str] = Counter()
    for tokens in shard_tokens.values():
        for term in set(tokens):
            df[term] += 1

    result: dict[int, list[ShardKeyword]] = {}
    for shard_id, tokens in shard_tokens.items():
        total = len(tokens)
        if total == 0:
            result[shard_id] = []
            continue
        tf_counts: Counter[str] = Counter(tokens)
        keywords: list[ShardKeyword] = []
        for term, count in tf_counts.items():
            tf = count / total
            idf = math.log((1 + n_shards) / (1 + df[term])) + 1
            weight = tf * idf
            keywords.append(ShardKeyword(term=term, weight=weight))
        keywords.sort(key=lambda kw: (-kw.weight, kw.term))
        result[shard_id] = keywords[:max_keywords]
    return result


def compute_term_graph(
    doc_tokens: list[list[str]],
    window: int,
    min_edge_count: int,
) -> dict[str, dict[str, float]]:
    """Build a co-occurrence term graph from sliding windows.

    For each document's token list, slide a window of size ``window``,
    counting unordered co-occurrence pairs of distinct terms (each ordered pair
    at each window position is counted once). Edges with count < ``min_edge_count``
    are dropped. Edge weight = count / sum(counts of all surviving outgoing edges),
    normalized per direction so outgoing weights sum to 1.
    Nodes with no surviving edges are omitted.
    """
    # Count directed co-occurrences (a, b) where a != b within each window.
    edge_counts: Counter[tuple[str, str]] = Counter()

    for tokens in doc_tokens:
        n = len(tokens)
        for start in range(n):
            window_tokens = tokens[start : start + window]
            unique_in_window = list(dict.fromkeys(window_tokens))  # deduplicate, preserve order
            for i in range(len(unique_in_window)):
                for j in range(len(unique_in_window)):
                    if i != j:
                        edge_counts[(unique_in_window[i], unique_in_window[j])] += 1

    # Drop edges below threshold.
    surviving: dict[tuple[str, str], int] = {
        edge: cnt for edge, cnt in edge_counts.items() if cnt >= min_edge_count
    }

    # Compute outgoing weight sums per node.
    node_out_totals: Counter[str] = Counter()
    for (src, _dst), cnt in surviving.items():
        node_out_totals[src] += cnt

    graph: dict[str, dict[str, float]] = {}
    for (src, dst), cnt in surviving.items():
        total = node_out_totals[src]
        weight = cnt / total if total > 0 else 0.0
        graph.setdefault(src, {})[dst] = weight

    return graph


def compute_entities(
    doc_entities: list[list[str]],
    max_entities: int,
) -> list[ShardEntity]:
    """Aggregate entity counts and return top ``max_entities`` by (-count, text)."""
    counts: Counter[str] = Counter()
    for entities in doc_entities:
        for entity in entities:
            counts[entity] += 1
    top = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:max_entities]
    return [ShardEntity(text=text, count=count) for text, count in top]


# ------------------------------------------------------------------ #
# Shard summary helpers                                               #
# ------------------------------------------------------------------ #


def _shard_summary(
    shard_id: int,
    document_ids: list[str],
    doc_by_id: dict[str, Document],
    vectors: FloatArray,
    row_map: dict[str, int],
    centroid: FloatArray,
) -> str:
    """Titles of the 3 documents closest to the shard centroid (cosine, tie -> doc id)."""
    if not document_ids:
        return ""
    scored = []
    for doc_id in document_ids:
        row = row_map[doc_id]
        sim = float(vectors[row] @ centroid)
        scored.append((sim, doc_id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    top3 = scored[:3]
    return "; ".join(doc_by_id[doc_id].title for _sim, doc_id in top3)


def _shard_title(shard_id: int, keywords: list[ShardKeyword]) -> str:
    """Top 3 keyword terms joined with ', ', or 'shard-<id>' if no keywords."""
    top3 = [kw.term for kw in keywords[:3]]
    if not top3:
        return f"shard-{shard_id}"
    return ", ".join(top3)


# ------------------------------------------------------------------ #
# Index write                                                         #
# ------------------------------------------------------------------ #


def write_index(
    index_path: Path,
    *,
    meta: IndexMeta,
    documents: list[Document],
    vectors: FloatArray,
    manifests: list[ShardManifest],
    doc_tokens: dict[str, list[str]],
) -> None:
    """Write a complete index to ``index_path``.

    Creates the directory tree, writes all artifacts in canonical JSON / npy
    format, and writes ``_VALID`` last so an interrupted write is never treated
    as complete.
    """
    index_path.mkdir(parents=True, exist_ok=True)

    # documents.jsonl
    docs_lines = "".join(
        canonical_dumps(doc.model_dump(mode="json")) + "\n" for doc in documents
    )
    (index_path / "documents.jsonl").write_text(docs_lines, encoding="utf-8")

    # embeddings.npy
    np.save(index_path / "embeddings.npy", vectors.astype(np.float32))

    # row_map.json  {doc_id -> row index}
    row_map = {doc.id: idx for idx, doc in enumerate(documents)}
    (index_path / "row_map.json").write_text(
        canonical_dumps(row_map) + "\n", encoding="utf-8"
    )

    # tokens.jsonl
    tokens_lines = "".join(
        canonical_dumps({"id": doc.id, "tokens": doc_tokens.get(doc.id, [])}) + "\n"
        for doc in documents
    )
    (index_path / "tokens.jsonl").write_text(tokens_lines, encoding="utf-8")

    # shards/shard_<id:03d>.json
    shards_dir = index_path / "shards"
    shards_dir.mkdir(exist_ok=True)
    for manifest in manifests:
        shard_file = shards_dir / f"shard_{manifest.shard_id:03d}.json"
        shard_file.write_text(
            canonical_dumps(manifest.model_dump(mode="json")) + "\n", encoding="utf-8"
        )

    # index_meta.json
    (index_path / "index_meta.json").write_text(
        canonical_dumps(meta.model_dump(mode="json")) + "\n", encoding="utf-8"
    )

    # _VALID — written last
    (index_path / "_VALID").write_text(ARTIFACT_VERSION, encoding="utf-8")


# ------------------------------------------------------------------ #
# Index loading                                                       #
# ------------------------------------------------------------------ #


class LoadedIndex:
    """A fully-loaded, query-ready index."""

    def __init__(
        self,
        meta: IndexMeta,
        documents: list[Document],
        vectors: FloatArray,
        row_map: dict[str, int],
        manifests: dict[int, ShardManifest],
        doc_tokens: dict[str, list[str]],
    ) -> None:
        self.meta = meta
        self.documents = documents
        self.by_id: dict[str, Document] = {doc.id: doc for doc in documents}
        self.vectors = vectors
        self.row_map = row_map
        self.manifests = manifests
        self.doc_tokens = doc_tokens

        # Build per-shard BM25 index at load time.
        from rank_bm25 import BM25Okapi

        self.bm25: dict[int, BM25Okapi] = {}
        for shard_id, manifest in manifests.items():
            shard_corpus = [
                doc_tokens.get(doc_id, []) for doc_id in manifest.document_ids
            ]
            self.bm25[shard_id] = BM25Okapi(shard_corpus)

    @classmethod
    def load(cls, index_path: Path) -> LoadedIndex:
        """Load a previously written index from ``index_path``.

        Raises :class:`IndexLoadError` if ``_VALID`` is missing or any
        required artifact is absent.
        """
        valid_marker = index_path / "_VALID"
        if not valid_marker.exists():
            msg = f"index at {index_path} is missing _VALID marker (incomplete or corrupt)"
            raise IndexLoadError(msg)

        required = [
            "documents.jsonl",
            "embeddings.npy",
            "row_map.json",
            "tokens.jsonl",
            "index_meta.json",
        ]
        for name in required:
            if not (index_path / name).exists():
                msg = f"index at {index_path} is missing required artifact: {name}"
                raise IndexLoadError(msg)

        # Load meta first to know n_shards.
        meta_raw = json.loads((index_path / "index_meta.json").read_text(encoding="utf-8"))
        meta = IndexMeta.model_validate(meta_raw)

        # Load documents.
        documents: list[Document] = []
        with (index_path / "documents.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    documents.append(Document.model_validate(json.loads(line)))

        # Load vectors.
        vectors: FloatArray = np.load(index_path / "embeddings.npy").astype(np.float32)

        # Load row_map.
        row_map: dict[str, int] = json.loads(
            (index_path / "row_map.json").read_text(encoding="utf-8")
        )

        # Load tokens.
        doc_tokens: dict[str, list[str]] = {}
        with (index_path / "tokens.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    doc_tokens[record["id"]] = record["tokens"]

        # Load shard manifests.
        shards_dir = index_path / "shards"
        manifests: dict[int, ShardManifest] = {}
        for shard_id in range(meta.n_shards):
            shard_file = shards_dir / f"shard_{shard_id:03d}.json"
            if not shard_file.exists():
                msg = f"index at {index_path} is missing shard file: {shard_file.name}"
                raise IndexLoadError(msg)
            raw = json.loads(shard_file.read_text(encoding="utf-8"))
            manifests[shard_id] = ShardManifest.model_validate(raw)

        return cls(
            meta=meta,
            documents=documents,
            vectors=vectors,
            row_map=row_map,
            manifests=manifests,
            doc_tokens=doc_tokens,
        )
