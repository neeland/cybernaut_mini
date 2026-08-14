"""Reranker 3 of 4: Personalized PageRank over a nearest-neighbour graph of shards.

The other three rerankers judge each shard on its own. This one judges a shard by the
company it keeps: build a kNN graph over the selected shards' centroids, start a random
surfer on the shards stage 5 liked most, and let it walk. A shard that is nobody's
neighbour keeps only the mass it was given and sinks; a shard that sits between several
strong shards collects mass from all of them and rises, even if stage 5 ranked it
mid-pack. That is the mechanism the post's worked example needs — the genuinely
relevant shard is not the one stage 5 put first, but it is adjacent to the ones that
matter.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 6, Page (Re)Ranker:
    "it is also possible to use a nearest neighbor graph and the Personalized PageRank
    algorithm to produce a probability mass over your selected shards". Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions (the post gives no parameters at all; all four are [inferred]):
    - **Graph scope.** Nodes are the *selected* shards only, not the whole index. The
      post says "a probability mass over your selected shards", and a graph over all
      1,000 shards would leak mass to shards stage 5 excluded, which stage 6 cannot
      return anyway. Cost is O(n^2) in the ~100 selected shards, which is a 100x100
      dot product.
    - **k = 8** (:data:`DEFAULT_KNN_K`, capped at n-1). Below ~4 the graph fragments
      into disconnected pairs and the walk never leaves its restart set; above ~16 over
      100 nodes every shard is everyone's neighbour and the mass flattens towards the
      restart distribution. 8 keeps a topical neighbourhood connected without joining
      the neighbourhoods to each other.
    - **alpha = 0.85** (:data:`DEFAULT_ALPHA`), the standard PageRank damping factor,
      i.e. a 15% chance per step of teleporting back to the stage-5 head. This is the
      knob that trades "trust stage 5" against "trust the graph"; at 0.85 a shard needs
      real neighbourhood support to overtake a shard ranked above it.
    - **Restart distribution** is Zipf over the stage-5 order: shard at rank r gets
      mass proportional to 1/r, normalised. Rank 1 gets ~3x rank 3 and ~10x rank 10,
      which matches the shape of the post's complaint (the first few selected shards
      were the bad ones, but the ordering still carries information). Shards absent
      from the prior get zero restart mass and can only be reached through the graph;
      with no prior at all the restart is uniform and this degenerates to plain
      PageRank over the kNN graph.

Alternatives rejected: a directed graph with edges pointing from a shard to its k
    nearest neighbours. Rejected because kNN is not symmetric — a hub shard is in
    everyone's neighbour list and would absorb mass it never returns, which is exactly
    the generic-shard failure mode reranking is supposed to fix. The union graph is
    undirected, so mass flows both ways.
Alternatives rejected: cosine similarity as-is for edge weights, including negatives.
    Rejected: a negative weight is meaningless as a transition probability and
    networkx would happily produce a negative "probability mass" from it. Non-positive
    similarities simply do not create an edge.
Alternatives rejected: seeding the restart with stage-5's *scores* rather than its
    ranks. Rejected because the scores are a fused RRF total whose scale depends on how
    many rankers voted, while the ranks are what RRF itself consumes one step later.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx  # type: ignore[import-untyped]
import numpy as np

from cybernaut_mini.query.s6_rerank.base import RerankQuery, ShardCandidate

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_KNN_K",
    "PageRankReranker",
    "build_shard_knn_graph",
    "restart_distribution",
]

#: Neighbours per shard in the kNN graph.
DEFAULT_KNN_K = 8

#: PageRank damping factor: 15% of steps teleport back to the restart distribution.
DEFAULT_ALPHA = 0.85

#: Similarities are rounded before becoming edge weights so that the graph — and
#: therefore the power iteration — is bit-identical across runs and platforms.
_WEIGHT_DECIMALS = 9

#: Power-iteration limits. The tolerance is far below the differences that change an
#: ordering, so the result is converged rather than iteration-count-dependent.
_MAX_ITER = 200
_TOLERANCE = 1e-12


def _unit_rows(candidates: Sequence[ShardCandidate]) -> dict[int, np.ndarray]:
    """L2-normalised centroids, keyed by shard id; shards without one are omitted.

    Centroids of a dimension other than the majority's are omitted too: mixing
    dimensions means the caller merged two indexes, and a truncated dot product would
    be a plausible-looking wrong answer.
    """
    vectors = {
        candidate.shard_id: np.asarray(candidate.centroid, dtype=np.float64)
        for candidate in candidates
        if candidate.centroid
    }
    if not vectors:
        return {}
    dims = sorted({vector.shape[0] for vector in vectors.values()})
    if len(dims) > 1:
        majority = max(dims, key=lambda dim: sum(v.shape[0] == dim for v in vectors.values()))
        vectors = {sid: v for sid, v in vectors.items() if v.shape[0] == majority}

    unit: dict[int, np.ndarray] = {}
    for shard_id, vector in vectors.items():
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            unit[shard_id] = vector / norm
    return unit


def build_shard_knn_graph(
    candidates: Sequence[ShardCandidate], *, k: int = DEFAULT_KNN_K
) -> nx.Graph:
    """Undirected kNN graph over shard centroids, weighted by positive cosine.

    Every candidate becomes a node, including one with no usable centroid: an isolated
    node is a shard the walk can only reach by teleporting, which is the correct
    treatment for a shard whose position is unknown.
    """
    if k < 1:
        msg = f"knn k must be >= 1, got {k}"
        raise ValueError(msg)

    graph = nx.Graph()
    # Sorted insertion, so the node order that networkx builds its matrix from is a
    # property of the shard ids and not of the caller's iteration order.
    shard_ids = sorted({candidate.shard_id for candidate in candidates})
    graph.add_nodes_from(shard_ids)

    unit = _unit_rows(candidates)
    usable = sorted(unit)
    if len(usable) < 2:
        return graph

    matrix = np.vstack([unit[shard_id] for shard_id in usable])
    similarity = matrix @ matrix.T
    neighbours = min(k, len(usable) - 1)

    edges: dict[tuple[int, int], float] = {}
    for row, shard_id in enumerate(usable):
        # Rank by (-similarity, shard id): np.lexsort keys are applied last-first.
        order = np.lexsort((np.asarray(usable), -similarity[row]))
        taken = 0
        for column in order:
            other = usable[int(column)]
            if other == shard_id:
                continue
            weight = round(float(similarity[row, int(column)]), _WEIGHT_DECIMALS)
            if weight <= 0.0:
                break  # sorted descending: everything after this is non-positive too
            edge = (min(shard_id, other), max(shard_id, other))
            edges[edge] = max(edges.get(edge, 0.0), weight)
            taken += 1
            if taken == neighbours:
                break

    for (left, right), weight in sorted(edges.items()):
        graph.add_edge(left, right, weight=weight)
    return graph


def restart_distribution(
    shard_ids: Sequence[int], prior: Sequence[int]
) -> dict[int, float]:
    """Normalised Zipf mass over ``prior``, restricted to ``shard_ids``.

    Falls back to a uniform distribution when the prior order says nothing about any of
    these shards, which turns Personalized PageRank into plain PageRank rather than
    into a division by zero.
    """
    known = set(shard_ids)
    weights = {
        shard_id: 1.0 / rank
        for rank, shard_id in enumerate(prior, start=1)
        if shard_id in known
    }
    total = sum(weights.values())
    if total <= 0.0:
        if not known:
            return {}
        uniform = 1.0 / len(known)
        return {shard_id: uniform for shard_id in sorted(known)}
    return {shard_id: weight / total for shard_id, weight in sorted(weights.items())}


@dataclass(frozen=True)
class PageRankReranker:
    """Scores each shard by its stationary probability under a personalized walk."""

    name: str = "pagerank"
    k: int = DEFAULT_KNN_K
    alpha: float = DEFAULT_ALPHA

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            msg = f"alpha must be in (0, 1), got {self.alpha!r}"
            raise ValueError(msg)

    def score(
        self, query: RerankQuery, candidates: Sequence[ShardCandidate]
    ) -> dict[int, float]:
        """Probability mass per shard; the values sum to 1.0 over the candidates."""
        shard_ids = sorted({candidate.shard_id for candidate in candidates})
        if not shard_ids:
            return {}
        if len(shard_ids) == 1:
            return {shard_ids[0]: 1.0}

        graph = build_shard_knn_graph(candidates, k=self.k)
        personalization = restart_distribution(shard_ids, query.prior)
        ranked: dict[int, float] = nx.pagerank(
            graph,
            alpha=self.alpha,
            personalization=personalization,
            weight="weight",
            max_iter=_MAX_ITER,
            tol=_TOLERANCE,
            dangling=personalization,
        )
        return {shard_id: float(ranked.get(shard_id, 0.0)) for shard_id in shard_ids}
