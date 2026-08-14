"""What an intent match *does* to a fused result: the reranking rule for step 5.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 8, step 5:
    "Then do a full-text search over top results for intents." The post says the
    pass happens and says nothing about its effect on the ranking [disclosed]. It
    also uses RRF at stages 5 and 6, so fusing a third ranked list is the house
    idiom rather than a new mechanism. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions — the reading of step 5 [inferred]:
    An intent match RERANKS. It is not a filter and it is not an unbounded score
    bonus. Three properties of the post force that reading:

    1. The pass runs over *top results* only. Something applied to the already
       retrieved head of the ranking cannot add recall, so it cannot be the thing
       that decides membership — it can only reorder what fusion already found.
    2. Filtering on intents would be actively harmful. Intents are exact
       high-precision phrases; a document that answers the question in its own
       words matches none of them, and dropping it would undo the semantic leg of
       the very fusion that just ran. The post's own worked-example snippet matches
       "gene editing" only as "an editing process called CRISPR".
    3. A raw additive bonus is not comparable with a fused score. RRF scores live on
       a ``w / (k + rank)`` scale; adding a TF-IDF-derived number to that is
       arithmetic on two different units and the weight would need retuning
       whenever ``k`` changes.

    So the intent evidence becomes a third :class:`~cybernaut_mini.rrf.RankedList`
    fused with the existing order at the same ``k``. The consequences are worth
    stating because they are the design: a match can promote a document by a bounded
    amount (with ``k=60`` and ``intent_weight=0.5``, a top-intent document at fused
    rank 100 scores 0.0145 against 0.0164 for an intent-less document at fused rank
    1, so it climbs but does not leapfrog the head), no document is ever removed,
    and a document nobody matched keeps its exact relative order among the others.

    When nothing matched at all, the fused ranking is returned untouched — same
    order, same scores, same objects — so the refinement pass is a no-op rather than
    a rescaling.

Alternatives rejected:
    - ``score * (1 + weight * evidence)``: simple and monotone, but it makes the
      boost proportional to the fused score, which is largest exactly where the
      ranking is already confident and least needs help.
    - Sorting by ``(intent_matched, fused_score)``: the strongest possible reading
      of "high-precision phrase", and the one that fails hardest — a single common
      intent then outranks every semantic match, which is the behaviour hybrid
      search exists to avoid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cybernaut_mini.query.s8_retrieve.intent_scan import IntentScan
from cybernaut_mini.rrf import RankedList, rrf_fuse

if TYPE_CHECKING:
    from cybernaut_mini.config import RRFConfig
    from cybernaut_mini.query.s8_retrieve.mapreduce import FusedCandidate

__all__ = [
    "DEFAULT_INTENT_WEIGHT",
    "RefinedCandidate",
    "refine_with_intents",
]

#: Weight of the intent list in the refinement fusion. Matches
#: :attr:`cybernaut_mini.config.RRFConfig.entity_weight`, the project's existing
#: default for a corroborating (as opposed to primary) ranker.
DEFAULT_INTENT_WEIGHT = 0.5


@dataclass(frozen=True)
class RefinedCandidate:
    """A fused candidate after the intent pass, with its before/after ranks."""

    candidate: FusedCandidate
    score: float
    fused_rank: int
    intent_rank: int | None = None
    scan: IntentScan | None = None
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.candidate.doc_id

    @property
    def matched(self) -> bool:
        return self.scan is not None and bool(self.scan)


def refine_with_intents(
    candidates: Sequence[FusedCandidate],
    scans: Mapping[str, IntentScan],
    *,
    rrf_config: RRFConfig,
    intent_weight: float = DEFAULT_INTENT_WEIGHT,
) -> list[RefinedCandidate]:
    """Rerank ``candidates`` by fusing their order with the intent evidence order.

    ``scans`` need only cover the refinement window — documents without a scan, or
    with a scan that matched nothing, simply do not appear in the intent list.
    Ordering is deterministic: the intent list breaks evidence ties on the document
    id, and RRF breaks score ties on the document id.

    Raises :class:`ValueError` for a negative ``intent_weight``.
    """
    if intent_weight < 0.0:
        msg = f"intent_weight must be >= 0, got {intent_weight}"
        raise ValueError(msg)

    fused_rank = {candidate.doc_id: rank for rank, candidate in enumerate(candidates, start=1)}

    matched = [
        scans[candidate.doc_id]
        for candidate in candidates
        if candidate.doc_id in scans and scans[candidate.doc_id].evidence > 0.0
    ]
    matched.sort(key=lambda scan: (-scan.evidence, scan.doc_id))

    if not matched or intent_weight == 0.0:
        # Nothing to say: hand back the fusion untouched, scores included.
        return [
            RefinedCandidate(
                candidate=candidate,
                score=candidate.score,
                fused_rank=rank,
                intent_rank=None,
                scan=scans.get(candidate.doc_id),
                contributions=dict(candidate.contributions),
            )
            for rank, candidate in enumerate(candidates, start=1)
        ]

    intent_rank = {scan.doc_id: rank for rank, scan in enumerate(matched, start=1)}
    by_id = {candidate.doc_id: candidate for candidate in candidates}

    fused = rrf_fuse(
        [
            RankedList(
                name="fused",
                weight=1.0,
                ids=tuple(candidate.doc_id for candidate in candidates),
                scores=tuple(candidate.score for candidate in candidates),
            ),
            RankedList(
                name="intents",
                weight=intent_weight,
                ids=tuple(scan.doc_id for scan in matched),
                scores=tuple(scan.evidence for scan in matched),
            ),
        ],
        k=rrf_config.k,
    )

    refined: list[RefinedCandidate] = []
    for item in fused:
        candidate = by_id[item.id]
        contributions = dict(candidate.contributions)
        contributions.update(item.contributions)
        refined.append(
            RefinedCandidate(
                candidate=candidate,
                score=item.score,
                fused_rank=fused_rank[item.id],
                intent_rank=intent_rank.get(item.id),
                scan=scans.get(item.id),
                contributions=contributions,
            )
        )
    return refined
