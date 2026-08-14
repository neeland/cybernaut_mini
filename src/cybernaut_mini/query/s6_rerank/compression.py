"""Reranker 2 of 4: which shard's trained dictionary compresses the question best?

Compression as similarity: a Zstandard dictionary trained on a shard's text is a
catalogue of that shard's repeated substrings, so a question written in the shard's
vocabulary finds matches in it and shrinks, while an off-topic question finds none and
pays a few bytes for the privilege. The sign alone separates the two cases.

The heavy lifting — training, level pinning, the gain-against-no-dictionary definition
of the score — lives in :func:`cybernaut_mini.shard_artifacts.compression_score`. This
module is the stage-6 adapter around it.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 6, Compression
    Reranker: "Each shard has a trained Zstandard text compression dictionary. The
    shard that can compress the input question the best is more likely to contain the
    most similar content." Tech stack for the stage names Zstandard. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions: [inferred] "the input question" is the raw question text, not the question
    plus its stage-3 intents or stage-7 expansions. Intents are substrings of the
    question, so appending them would compress trivially well against any dictionary
    that already helped with the question and inflate the spread without adding
    evidence.
Assumptions: [inferred] the score is compared across shards for one question, never
    across questions. ``compression_score`` returns a fractional gain against the same
    text compressed with no dictionary, which is exactly what makes the per-shard
    numbers comparable to each other at fixed question length.
Assumptions: [inferred] a shard with no dictionary scores 0.0, which is also the score
    of a dictionary that bought nothing. zstd refuses to train on a shard of a few
    short documents, and that is a legitimate shard, not a build failure.

Alternatives rejected: the ``zlib``-against-a-three-title summary approximation in
    ``routing.py``, which compresses ``summary + question`` and subtracts. Rejected:
    it trains on nothing and measures how well a 200-character summary predicts the
    question, which is a far weaker signal than a dictionary trained on the whole
    shard — and the post says "trained ... dictionary", not "concatenate and compress".
Alternatives rejected: normalised compression distance, NCD(q, shard) computed from
    concatenated compressed sizes. Rejected because it needs the shard's text at query
    time; the trained dictionary is exactly the artifact that makes this affordable at
    query time for ~100 shards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cybernaut_mini.query.s6_rerank.base import RerankQuery, ShardCandidate
from cybernaut_mini.shard_artifacts import DEFAULT_COMPRESSION_LEVEL, compression_score

__all__ = ["CompressionReranker"]


@dataclass(frozen=True)
class CompressionReranker:
    """Scores each shard by how much its dictionary shrinks the question."""

    name: str = "compression"
    #: Must match the level the dictionaries were built for; see
    #: :data:`cybernaut_mini.shard_artifacts.DEFAULT_COMPRESSION_LEVEL`.
    level: int = DEFAULT_COMPRESSION_LEVEL

    def score(
        self, query: RerankQuery, candidates: Sequence[ShardCandidate]
    ) -> dict[int, float]:
        """Fractional compression gain per shard, in ``[-1, 1]``; higher is better."""
        return {
            candidate.shard_id: compression_score(
                candidate.zstd_dictionary, query.text, level=self.level
            )
            for candidate in candidates
        }
