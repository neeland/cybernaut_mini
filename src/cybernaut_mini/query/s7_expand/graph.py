"""Per-shard synonym graphs and the transition distribution read off them.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 7, "Shard-based
    Question Expansion": "we use the synonym graphs in each shard to probabilistically
    expand our search terms. Because each shard is lexically and semantically coherent,
    the synonyms in each shard are unambiguous. Put simply, 'gene' in shard 11,343 only
    has the genetic meaning." Stack: NumPy. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions: the post says "synonym graph" but never says what an edge weight means, so
    this module reads ``ShardManifest.term_graph`` — a weighted co-occurrence graph, the
    only synonym structure the index actually builds — and turns each node's outgoing
    edges into an L1-normalised distribution over neighbours [inferred]. Normalising per
    ``(shard, term)`` is what makes "probabilistically" literally true: a term's
    neighbours form a probability distribution, so mass can be summed across the routed
    shards and compared between candidates that came from graphs of wildly different
    densities. Without it a shard whose edges happen to be counted rather than scored
    would drown out every other shard.

    The one-shard-one-sense claim [disclosed] is a property of the *graphs*, not of this
    code: this module only guarantees that a term's neighbours are read from the graph of
    the shard being consulted and are never pooled before weighting. A build that copies
    one global graph into every manifest destroys the claim and this module cannot detect
    that; :func:`shard_graphs_from_index` is deliberately per-shard so that it holds the
    moment the graphs do.

Alternatives rejected:
    - Raw edge weights, unnormalised (what ``expansion.expand_query`` does today). Cheap,
      but the score of a candidate then depends on how many neighbours its source term
      happens to have, which is a property of the shard's vocabulary size rather than of
      the candidate.
    - Symmetrising or transitively closing the graph (two-hop expansion). Two hops is
      exactly where "gene" reaches "Willy Wonka" even inside a coherent shard, and the
      post is explicit that this stage must not overwhelm the original words.
    - ``networkx`` for the walk. It is a dependency already, but a one-hop neighbour
      lookup over a ``dict[str, dict[str, float]]`` is the whole algorithm; building a
      ``DiGraph`` per query per shard would cost more than the walk.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cybernaut_mini.indexing import LoadedIndex

#: A shard's synonym graph: source term -> neighbour -> edge weight.
TermGraph = Mapping[str, Mapping[str, float]]

__all__ = [
    "ShardGraph",
    "TermGraph",
    "neighbour_distribution",
    "rank_weights",
    "shard_graphs_from_index",
]


@dataclass(frozen=True, slots=True)
class ShardGraph:
    """One routed shard's synonym graph together with its contribution weight.

    Attributes:
        shard_id: the shard this graph belongs to.
        weight: how much this shard's opinion counts, typically its stage-6 rank
            weight. Must be finite and non-negative; ``0.0`` mutes the shard.
        graph: the shard's term graph, exactly as carried by its manifest.
    """

    shard_id: int
    weight: float
    graph: TermGraph

    def __post_init__(self) -> None:
        if not (self.weight >= 0.0) or self.weight == float("inf"):
            msg = f"shard weight must be finite and non-negative, got {self.weight!r}"
            raise ValueError(msg)


def neighbour_distribution(graph: TermGraph, term: str) -> dict[str, float]:
    """The L1-normalised outgoing edges of ``term``, or ``{}`` when it has none.

    Non-positive edge weights are dropped rather than clamped: a zero or negative
    co-occurrence weight is an absent edge, and keeping it would put mass on a term the
    graph is actively saying is unrelated.

    >>> neighbour_distribution({"gene": {"crispr": 3.0, "genome": 1.0}}, "gene")
    {'crispr': 0.75, 'genome': 0.25}
    >>> neighbour_distribution({"gene": {"crispr": 3.0}}, "yeast")
    {}
    """
    edges = graph.get(term)
    if not edges:
        return {}
    positive = {n: float(w) for n, w in edges.items() if float(w) > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        return {}
    return {n: w / total for n, w in positive.items()}


def rank_weights(shard_ids: Sequence[int], *, k: float = 1.0) -> dict[int, float]:
    """Rank-decayed contribution weights, ``k / (k + rank)`` with ``rank`` 0-based.

    "Higher-ranked shards contribute more" needs a curve, and this is the same
    reciprocal-rank shape the pipeline already uses for fusion (``rrf.rrf_fuse``), so
    stages 5/6 and 7 decay at the same rate rather than at two invented ones. ``k``
    controls flatness: large ``k`` approaches uniform, ``k -> 0`` keeps only rank 0.

    ``shard_ids`` is taken in rank order, best first. Duplicates keep their best rank.

    >>> rank_weights([7, 3, 9])
    {7: 1.0, 3: 0.5, 9: 0.3333333333333333}
    """
    if k <= 0.0:
        msg = f"k must be positive, got {k!r}"
        raise ValueError(msg)
    weights: dict[int, float] = {}
    for rank, shard_id in enumerate(shard_ids):
        if shard_id in weights:
            continue
        weights[shard_id] = k / (k + rank)
    return weights


def shard_graphs_from_index(
    index: LoadedIndex,
    shard_ids: Sequence[int],
    *,
    shard_weights: Mapping[int, float] | None = None,
) -> list[ShardGraph]:
    """Collect the term graphs of ``shard_ids`` from ``index``, in the order given.

    Reads ``index.manifests[shard_id].term_graph`` only — manifests are the resident part
    of a :class:`~cybernaut_mini.indexing.LoadedIndex`, so this touches no sidecar, no
    document and no embedding, and stays O(len(shard_ids)) rather than O(n_shards).

    Shard ids the index does not know are skipped rather than raised on: routing and
    expansion can be pointed at different indexes in a notebook, and losing a shard's
    synonyms is a degradation, not a corruption. Missing weights default to ``1.0``.
    """
    weights = shard_weights or {}
    graphs: list[ShardGraph] = []
    seen: set[int] = set()
    for shard_id in shard_ids:
        if shard_id in seen:
            continue
        manifest = index.manifests.get(shard_id)
        if manifest is None:
            continue
        seen.add(shard_id)
        graphs.append(
            ShardGraph(
                shard_id=shard_id,
                weight=float(weights.get(shard_id, 1.0)),
                graph=manifest.term_graph,
            )
        )
    return graphs
