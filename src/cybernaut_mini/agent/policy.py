"""Reward computation for executed search states.

reward = 0.45*relevance + 0.20*coverage + 0.20*S_dense + 0.10*S_lexical
         + 0.05*(1 - redundancy)

S_dense / S_lexical are the means of the min-max-normalized dense / BM25 scores of
the top five hits. Redundancy enters as its complement (diversity), so the weights
sum to 1.0 and a perfect result set — fully relevant, fully covered, top-ranked in
both retrievers, zero duplication — scores exactly 1.0. Empty results score 0.

Blog ref: https://nosible.com/blog/introducing-cybernaut-1-agentic-search-with-mcts —
    the agent balances "exploration, exploitation, and inference cost", and the
    build guide reads that as ``reward = relevance - lambda * (LLM calls)``
    [inferred]. The quality weights below are this replica's stand-in for the
    undisclosed reward function. Local copy:
    ``docs/blog-archive/introducing-cybernaut-1-agentic-search-with-mcts.md``.

Assumptions:
    - The reward is a weighted *quality* score: relevance 0.45, coverage 0.20,
      dense 0.20, lexical 0.10, diversity 0.05. The weights sum to 1.0, so a
      perfect non-redundant result set scores exactly 1.0 and rewards are
      comparable across questions — which UCT requires, since it averages
      rewards from branches explored at different depths.
    - Inference cost is not a reward term. It is enforced as a hard cap
      (``max_retrieval_calls``, default 18), so no weight trades quality against a
      quantity the budget already bounds and every weight stays a positive
      contribution.
    - ``relevance`` and ``coverage`` are the judge's values verbatim. The two
      retriever terms are means of min-max-normalized scores over the top five
      hits, so they measure relative rank quality *within one result set* rather
      than an absolute score comparable across queries.
    - A constant non-zero score list normalizes to 1.0. BM25 and cosine are not
      on a common scale, and min-max is the cheapest way to stop whichever
      retriever happens to emit larger numbers from dominating its 0.20/0.10.
    - ``JudgeScore.redundancy`` enters as ``1 - redundancy``; the raw value stays
      in the components dict so a trace can show both. An empty result set scores
      0.0 on every component, including ``reward``.

Alternatives considered:
    - The build guide's cost-penalized reward ``relevance - lambda*llm_calls``:
      rejected because the default providers are heuristics with zero model
      calls, so the penalty is identically zero and lambda is untunable; the
      retrieval budget prices the one resource the agent can actually exhaust.
    - Judging with nDCG against the committed qrels: the most honest relevance
      signal in this repo, but qrels exist only for the fixture queries, so a
      live question's reward would be undefined and the agent could only run
      inside the evaluation pipeline.
    - Letting the judge emit the whole reward: fewer moving parts, but the
      retriever's own confidence would then be invisible to the search, and the
      post's agent is described as reading the ranking signals directly.
"""

from __future__ import annotations

from cybernaut_mini.models import JudgeScore, SearchHit

REWARD_WEIGHTS = {
    "relevance": 0.45,
    "coverage": 0.20,
    "dense": 0.20,
    "lexical": 0.10,
    "diversity": 0.05,
}


def _normalized_mean(values: list[float]) -> float:
    """Mean of min-max-normalized values; a constant list maps to 1.0 when positive."""
    if not values:
        return 0.0
    low, high = min(values), max(values)
    if high == low:
        return 1.0 if high > 0 else 0.0
    normalized = [(value - low) / (high - low) for value in values]
    return sum(normalized) / len(normalized)


def compute_reward(
    judge_score: JudgeScore, hits: list[SearchHit]
) -> tuple[float, dict[str, float]]:
    """Return (clamped reward, components dict for the trace)."""
    if not hits:
        components = {name: 0.0 for name in REWARD_WEIGHTS}
        components["redundancy"] = 0.0
        components["reward"] = 0.0
        return 0.0, components

    top5 = hits[:5]
    dense = _normalized_mean([h.dense_score for h in top5 if h.dense_score is not None])
    lexical = _normalized_mean([h.bm25_score for h in top5 if h.bm25_score is not None])

    # Raw redundancy stays in the components for the trace; the weighted term is
    # its complement so every weight is a positive contribution.
    components = {
        "relevance": judge_score.relevance,
        "coverage": judge_score.coverage,
        "dense": dense,
        "lexical": lexical,
        "diversity": 1.0 - judge_score.redundancy,
        "redundancy": judge_score.redundancy,
    }
    raw = sum(weight * components[name] for name, weight in REWARD_WEIGHTS.items())
    reward = min(1.0, max(0.0, raw))
    components["reward"] = reward
    return reward, components
