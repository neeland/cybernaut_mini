"""Weighted Reciprocal Rank Fusion with inspectable per-ranker contributions.

RRF(item) = sum over rankers r of w_r / (k + rank_r(item)), with 1-based ranks.
Ties in the fused score break by ascending item ID so ordering is deterministic.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — the post fuses ranking
    factors with reciprocal rank fusion at stage 5 (shard selection), stage 6 (shard
    reranking) and stage 8 (retrieval), calling RRF "a simple but powerful ensembling
    method". Local copy: ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - Ranks are 1-based and ``k`` defaults to 60, the constant the post's linked RRF
      explainer uses; every caller in this repo relies on that default.
    - Weight is per ``RankedList``, not global, so the same item can be weighted
      differently by the selector and the reranker.
    - Ties break by ascending item id, so a fused ordering is reproducible for a given
      input regardless of dict iteration order.
    - A ranker's optional ``scores`` are carried through verbatim for inspection and
      never alter the fused score; only rank position contributes.
    - Repeated ids within one ranked list are not deduplicated: the caller owns the
      list's well-formedness, and silently collapsing it would hide an upstream bug.

Alternatives considered:
    - Score normalisation (CombSUM / min-max) before summation: rejected because RRF's
      whole point is robustness to incomparable score scales, which the post names.
    - ``ranx`` or a ``pytrec_eval`` fusion helper: rejected as a dependency for a
      ~15-line function whose ``contributions`` dict is what the trace actually needs.
    - Sorting by fused score only and letting ties fall where they may: rejected
      because the trace and the eval harness both compare orderings across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RankedList:
    """One ranker's output, best first. ``scores`` (optional) align with ``ids``."""

    name: str
    weight: float
    ids: tuple[str, ...]
    scores: tuple[float, ...] = ()


@dataclass
class FusedItem:
    id: str
    score: float
    contributions: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    ranker_scores: dict[str, float] = field(default_factory=dict)


def rrf_fuse(lists: list[RankedList], k: int = 60) -> list[FusedItem]:
    if k < 1:
        msg = f"rrf k must be >= 1, got {k}"
        raise ValueError(msg)
    items: dict[str, FusedItem] = {}
    for ranked in lists:
        for position, item_id in enumerate(ranked.ids):
            rank = position + 1
            fused = items.setdefault(item_id, FusedItem(id=item_id, score=0.0))
            contribution = ranked.weight / (k + rank)
            fused.score += contribution
            fused.contributions[ranked.name] = contribution
            fused.ranks[ranked.name] = rank
            if ranked.scores:
                fused.ranker_scores[ranked.name] = ranked.scores[position]
    return sorted(items.values(), key=lambda item: (-item.score, item.id))
