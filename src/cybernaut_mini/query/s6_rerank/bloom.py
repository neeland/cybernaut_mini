"""Reranker 1 of 4: has this shard ever seen the user's search intents?

The cheapest of the four and the only one that can say *no* with authority. A bloom
filter never misses a phrase it holds, so a 0.0 here means the shard genuinely contains
none of the probed strings (the errors all run the other way: a non-zero score can be a
false positive, at the filter's configured 1% rate).

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 6, Bloom Filter
    Reranker: "Each shard has a bloom filter that keeps track of important phrases it
    contains. If a selected shard hasn't seen any of the user's search intents, then
    that shard should get downranked." Tech stack for the stage names rBloom, which is
    what :mod:`cybernaut_mini.shard_artifacts` builds these filters with. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions: [inferred] "hasn't seen any" is graded rather than binary. The post only
    specifies the zero case, so this scores the *fraction* of probes the shard may
    contain: 0.0 is the disclosed downrank signal and the values above it order the
    survivors by how much of the question the shard covers.
Assumptions: [inferred] both the whole intent phrase and its member tokens are probed,
    mixed by :attr:`BloomReranker.phrase_weight` (0.5 by default). Stage 3 emits
    multi-token phrases such as "safer gene-editing medicine", while a shard's filter
    is built from its keywords, entity strings and document titles
    (:func:`cybernaut_mini.indexing._shard_phrases`); phrase-only probing therefore
    scores every shard 0.0 on a typical question and the reranker abstains. A phrase
    hit is the stronger evidence, hence half the weight on far fewer probes.
Assumptions: [inferred] a shard with no filter at all (an index built without
    artifacts) scores 0.0 — the same "no evidence" value as a shard that has seen
    nothing. The distinction is recoverable from :attr:`ShardCandidate.phrase_bloom`
    being ``None``, and when *no* candidate has a filter every score is equal and the
    fusion drops this reranker as abstaining rather than letting it vote.

Alternatives rejected: querying the shard's :class:`~cybernaut_mini.shard_artifacts.
    Vocabulary` instead, which answers exactly rather than probabilistically. Rejected
    for this stage: the vocabulary is the single largest artifact a shard has (up to
    65,536 terms) and stage 6 runs over ~100 shards per query, so the filter's ~1.2 KB
    is the whole point. The vocabulary is the right tool one stage later, per shard.
Alternatives rejected: weighting each intent by its stage-3 TF-IDF score. Rejected as
    unjustifiable from the post, and because a bloom hit is one bit — weighting a bit
    by the query-side importance of the phrase that produced it invents precision the
    filter does not have.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cybernaut_mini.query.s6_rerank.base import (
    RerankQuery,
    ShardCandidate,
    intent_phrases,
    intent_tokens,
)
from cybernaut_mini.shard_artifacts import bloom_contains_any

__all__ = ["DEFAULT_PHRASE_WEIGHT", "BloomReranker"]

#: Share of the score carried by whole-phrase hits; the rest goes to token hits.
DEFAULT_PHRASE_WEIGHT = 0.5


@dataclass(frozen=True)
class BloomReranker:
    """Scores each shard by the fraction of the query's intents its filter may hold."""

    name: str = "bloom"
    phrase_weight: float = DEFAULT_PHRASE_WEIGHT

    def __post_init__(self) -> None:
        if not 0.0 <= self.phrase_weight <= 1.0:
            msg = f"phrase_weight must be in [0, 1], got {self.phrase_weight!r}"
            raise ValueError(msg)

    def score(
        self, query: RerankQuery, candidates: Sequence[ShardCandidate]
    ) -> dict[int, float]:
        """Bloom coverage in ``[0, 1]`` per shard; 0.0 means "has seen none of it"."""
        phrases = intent_phrases(query.intents)
        tokens = intent_tokens(query.intents)
        token_weight = 1.0 - self.phrase_weight

        scores: dict[int, float] = {}
        for candidate in candidates:
            phrase_bloom = candidate.phrase_bloom
            if phrase_bloom is None:
                scores[candidate.shard_id] = 0.0
                continue
            covered = bloom_contains_any(phrase_bloom, phrases) * self.phrase_weight
            covered += bloom_contains_any(phrase_bloom, tokens) * token_weight
            scores[candidate.shard_id] = covered
        return scores
