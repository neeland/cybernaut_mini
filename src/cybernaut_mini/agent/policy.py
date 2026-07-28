"""Reward computation for executed search states.

reward = 0.45*relevance + 0.20*coverage + 0.20*S_dense + 0.10*S_lexical - 0.05*redundancy

S_dense / S_lexical are the means of the min-max-normalized dense / BM25 scores of
the top five hits. The weights intentionally sum to 0.95 before the redundancy
penalty, so 1.0 is unreachable by design. Empty results score exactly 0.
"""

from __future__ import annotations

from cybernaut_mini.models import JudgeScore, SearchHit

REWARD_WEIGHTS = {
    "relevance": 0.45,
    "coverage": 0.20,
    "dense": 0.20,
    "lexical": 0.10,
    "redundancy": -0.05,
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
        components["reward"] = 0.0
        return 0.0, components

    top5 = hits[:5]
    dense = _normalized_mean([h.dense_score for h in top5 if h.dense_score is not None])
    lexical = _normalized_mean([h.bm25_score for h in top5 if h.bm25_score is not None])

    components = {
        "relevance": judge_score.relevance,
        "coverage": judge_score.coverage,
        "dense": dense,
        "lexical": lexical,
        "redundancy": judge_score.redundancy,
    }
    raw = sum(REWARD_WEIGHTS[name] * value for name, value in components.items())
    reward = min(1.0, max(0.0, raw))
    components["reward"] = reward
    return reward, components
