"""Turning the shards stage 5 selected into stage-6 candidates, one shard at a time.

This is the only module in the stage that touches a :class:`~cybernaut_mini.indexing.
LoadedIndex`, and it touches it narrowly: for each *selected* shard it reads that
shard's manifest (already resident) and calls
:meth:`~cybernaut_mini.indexing.LoadedIndex.shard_artifacts`, which pulls the sidecar
through the index's bounded LRU. Nothing here iterates ``index.manifests``, materialises
a document, or asks for a token stream, so the cost is O(selected shards), not O(index).

This is also the module that makes :mod:`cybernaut_mini.shard_artifacts` load-bearing:
before stage 6, per-shard bloom filters and Zstandard dictionaries were built, written
and read by nothing.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 6 reranks "the
    selected shards" (the ~100 stage 5 routed to), and the artifacts it reranks with —
    "Each shard has a bloom filter ...", "Each shard has a trained Zstandard text
    compression dictionary ..." — are listed in the post as contents of a shard. Local
    copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions: [inferred] the reranked set is exactly the set stage 5 selected, in stage
    5's order, and a shard id stage 5 never selected is not reconsidered here.
    Reranking is a reordering, not a second retrieval.
Assumptions: [inferred] a shard whose artifact sidecar is missing is a candidate like
    any other, carrying ``None`` for the artifacts it lacks. An index built with
    ``build_artifacts=False`` therefore still reranks — on the neural and page
    rerankers — instead of raising.

Alternatives rejected: :meth:`~cybernaut_mini.indexing.LoadedIndex.manifest_with_artifacts`,
    which returns the manifest with the artifact fields filled in. Rejected because it
    re-serialises the artifacts to their JSON payload form (base64 bloom bits and all)
    only for this module to parse them straight back into a
    :class:`~cybernaut_mini.shard_artifacts.PhraseBloom`; ``shard_artifacts()`` hands
    over the live objects.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from cybernaut_mini.query.s6_rerank.base import RerankQuery, ShardCandidate

if TYPE_CHECKING:
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.query.s3_intents import SearchIntent

__all__ = ["build_rerank_query", "candidates_from_index"]


def candidates_from_index(index: LoadedIndex, shard_ids: Iterable[int]) -> list[ShardCandidate]:
    """Build one :class:`ShardCandidate` per selected shard, preserving the given order.

    Unknown shard ids are skipped rather than raising: stage 5 and stage 6 can be
    pointed at different indexes only by mistake, but a mistake that drops a shard is
    recoverable where one that aborts the query is not. Duplicate ids collapse to the
    first occurrence, because a duplicated candidate would be counted twice by every
    ranked list feeding RRF.
    """
    seen: set[int] = set()
    candidates: list[ShardCandidate] = []
    for shard_id in shard_ids:
        if shard_id in seen or shard_id not in index.manifests:
            continue
        seen.add(shard_id)
        manifest = index.manifests[shard_id]
        artifacts = index.shard_artifacts(shard_id)
        candidates.append(
            ShardCandidate(
                shard_id=shard_id,
                title=manifest.title,
                summary=manifest.summary,
                centroid=tuple(manifest.centroid),
                phrase_bloom=None if artifacts is None else artifacts.phrase_bloom,
                zstd_dictionary=None if artifacts is None else artifacts.zstd_dictionary,
            )
        )
    return candidates


def build_rerank_query(
    text: str,
    *,
    intents: Sequence[SearchIntent] = (),
    prior: Sequence[int] = (),
) -> RerankQuery:
    """Assemble the query side of stage 6 from stage 3's intents and stage 5's order.

    A thin constructor, but it is the documented seam between the stages: ``intents``
    is whatever :func:`cybernaut_mini.query.s3_intents.predict_intents` returned for
    this question, and ``prior`` is the shard order stage 5 produced.
    """
    return RerankQuery(text=text, intents=tuple(intents), prior=tuple(prior))
