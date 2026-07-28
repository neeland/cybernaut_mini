"""Weighted Reciprocal Rank Fusion with inspectable per-ranker contributions.

RRF(item) = sum over rankers r of w_r / (k + rank_r(item)), with 1-based ranks.
Ties in the fused score break by ascending item ID so ordering is deterministic.
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
