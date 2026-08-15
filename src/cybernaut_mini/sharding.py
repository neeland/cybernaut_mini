"""K-means document sharding with empty-cluster repair and size balancing.

Vectors must be L2-normalized float32 arrays; cosine similarity is then just the
dot product. All repair and balancing steps are deterministic given the same seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass
class ShardingResult:
    labels: list[int]
    centroids: npt.NDArray[np.float32]


def _l2_normalize_rows(matrix: FloatArray) -> FloatArray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


def _compute_centroid(vectors: FloatArray, indices: list[int]) -> FloatArray:
    """L2-normalized mean of selected rows."""
    subset = vectors[indices]
    mean = subset.mean(axis=0).astype(np.float32)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        return mean
    return (mean / norm).astype(np.float32)


def shard_documents(
    vectors: FloatArray,
    n_shards: int,
    seed: int,
) -> ShardingResult:
    """Cluster L2-normalized document vectors into shards.

    Parameters
    ----------
    vectors:
        (n_docs, dim) float32 array, each row L2-normalized.
    n_shards:
        Number of target shards. Must satisfy 1 <= n_shards <= n_docs.
    seed:
        Random state for MiniBatchKMeans.

    Returns
    -------
    ShardingResult with integer labels (one per document) and centroids array
    (n_shards, dim).
    """
    # Deferred: sklearn pulls in scipy + pandas (~3s), and this module is imported
    # by every Kedro session via the pipeline registry — only a build should pay.
    from sklearn.cluster import MiniBatchKMeans

    n_docs, _dim = vectors.shape

    if n_shards < 1:
        msg = f"n_shards must be >= 1, got {n_shards}"
        raise ValueError(msg)
    if n_shards > n_docs:
        msg = f"n_shards ({n_shards}) > n_docs ({n_docs})"
        raise ValueError(msg)

    # batch_size=1024 is the sklearn default and 2× faster than 256 at 200k×384
    # (5.4 s → 2.7 s on Apple M-series; see .omc/research/kedro-optimisation.md).
    # The sklearn recommendation is batch_size ≥ n_clusters × 3 (768 at 256 shards),
    # so 256 was also below the convergence-quality floor — 1024 fixes both.
    kmeans = MiniBatchKMeans(
        n_clusters=n_shards,
        random_state=seed,
        n_init=3,
        batch_size=1024,
    )
    raw_labels: list[int] = kmeans.fit_predict(vectors).tolist()

    # Build mutable shard -> member list mapping.
    shard_members: dict[int, list[int]] = {s: [] for s in range(n_shards)}
    for doc_idx, label in enumerate(raw_labels):
        shard_members[label].append(doc_idx)

    # Recompute centroids from members (not from kmeans.cluster_centers_).
    centroids: dict[int, FloatArray] = {}
    for shard_id, members in shard_members.items():
        if members:
            centroids[shard_id] = _compute_centroid(vectors, members)
        else:
            centroids[shard_id] = np.zeros(vectors.shape[1], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Empty-cluster repair (deterministic, ascending empty shard id).     #
    # ------------------------------------------------------------------ #
    for empty_shard in sorted(s for s, m in shard_members.items() if not m):
        # Find the largest shard; tie -> lowest shard id.
        donor_shard = max(
            (s for s, m in shard_members.items() if len(m) > 1),
            key=lambda s: (len(shard_members[s]), -s),
            default=None,
        )
        if donor_shard is None:
            # Cannot donate without emptying another shard; skip.
            continue

        donor_members = shard_members[donor_shard]
        donor_centroid = centroids[donor_shard]

        # Member with LOWEST cosine similarity to donor centroid; tie -> lowest row index.
        similarities = vectors[donor_members] @ donor_centroid
        min_sim = float(similarities.min())
        candidates = [
            doc_idx
            for local_pos, doc_idx in enumerate(donor_members)
            if float(similarities[local_pos]) == min_sim
        ]
        moved_doc = min(candidates)

        donor_members.remove(moved_doc)
        shard_members[empty_shard] = [moved_doc]

        # Recompute centroids for the two affected shards.
        centroids[donor_shard] = _compute_centroid(vectors, donor_members)
        centroids[empty_shard] = _compute_centroid(vectors, [moved_doc])

    # ------------------------------------------------------------------ #
    # Balancing: while max_size > 1.5 * mean_size, move one member.      #
    # ------------------------------------------------------------------ #
    mean_size = n_docs / n_shards
    threshold = 1.5 * mean_size

    for _iteration in range(n_docs):
        sizes = {s: len(m) for s, m in shard_members.items()}
        max_size = max(sizes.values())
        if max_size <= threshold:
            break

        # Largest shard; tie -> lowest shard id.
        oversized = min(
            (s for s, sz in sizes.items() if sz == max_size),
            key=lambda s: s,
        )
        oversized_members = shard_members[oversized]
        oversized_centroid = centroids[oversized]

        # Below-mean shards.
        below_mean = [s for s, sz in sizes.items() if sz < mean_size]
        if not below_mean:
            break

        # Member with lowest cosine similarity to oversized centroid; tie -> lowest row index.
        sims = vectors[oversized_members] @ oversized_centroid
        min_sim_val = float(sims.min())
        move_candidates = [
            doc_idx
            for local_pos, doc_idx in enumerate(oversized_members)
            if float(sims[local_pos]) == min_sim_val
        ]
        moved_doc = min(move_candidates)

        # Nearest below-mean shard by cosine similarity of moved_doc's vector to centroid.
        moved_vec = vectors[moved_doc]
        target_shard = max(
            below_mean,
            key=lambda s: float(moved_vec @ centroids[s]),
        )

        oversized_members.remove(moved_doc)
        shard_members[target_shard].append(moved_doc)

        # Recompute centroids for both affected shards.
        centroids[oversized] = _compute_centroid(vectors, oversized_members)
        centroids[target_shard] = _compute_centroid(vectors, shard_members[target_shard])

    # Recompute all centroids after balancing.
    for shard_id, members in shard_members.items():
        if members:
            centroids[shard_id] = _compute_centroid(vectors, members)

    # Postcondition: every shard non-empty.
    for shard_id, members in shard_members.items():
        if not members:
            msg = f"shard {shard_id} is empty after repair — this should not happen"
            raise AssertionError(msg)

    # Reconstruct labels array in document order.
    labels = [0] * n_docs
    for shard_id, members in shard_members.items():
        for doc_idx in members:
            labels[doc_idx] = shard_id

    centroids_array = np.stack(
        [centroids[s] for s in range(n_shards)], axis=0
    ).astype(np.float32)

    return ShardingResult(labels=labels, centroids=centroids_array)
