"""Structured agent trace — a product artifact, not debug logging.

Captures every candidate, action, routing/retrieval signal, reward breakdown, and
node statistic so a reader can replay the agent's decisions. No chain-of-thought is
stored; judge reasons are short structured strings. Timing fields are informational
and are the only part that varies between otherwise-identical runs.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from cybernaut_mini.models import canonical_dumps
from cybernaut_mini.routing import RoutingSignals


class StopReason(StrEnum):
    NO_RESULTS = "no_results"
    DUPLICATE_CANDIDATES = "duplicate_candidates"
    BUDGET_EXHAUSTED = "budget_exhausted"


def routing_to_dict(signals: RoutingSignals | None) -> dict[str, object] | None:
    if signals is None:
        return None
    return {
        "dense": {str(k): v for k, v in signals.dense.items()},
        "sparse": {str(k): v for k, v in signals.sparse.items()},
        "entity": (
            None if signals.entity is None else {str(k): v for k, v in signals.entity.items()}
        ),
        "rerank_intent": (
            None
            if signals.rerank_intent is None
            else {str(k): v for k, v in signals.rerank_intent.items()}
        ),
        "rerank_compression": (
            None
            if signals.rerank_compression is None
            else {str(k): v for k, v in signals.rerank_compression.items()}
        ),
        "fused": [{"shard_id": int(item.id), "score": item.score} for item in signals.fused],
    }


class NodeTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: int
    parent_id: int | None
    stage: str
    action: dict[str, object] | None
    query: str
    shard_ids: list[int]
    expansions: list[str]
    lexical_weight: float
    dense_weight: float
    routing: dict[str, object] | None
    hit_ids: list[str]
    reward: float | None
    reward_components: dict[str, float]
    visits: int
    cumulative_value: float
    mean_value: float
    uct: float
    judge_reason: str | None


class StageTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    seconds: float


class AgentTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    question: str
    normalized_tokens: list[str]
    seed: int
    config: dict[str, object]
    nodes: list[NodeTrace]
    selected_path: list[int]
    final_query: str
    final_expansions: list[str]
    final_shard_ids: list[int]
    stop_reason: str | None
    stage_timings: list[StageTiming]
    embedding_calls: int
    retrieval_calls: int
    llm_calls: int

    def decision_fingerprint(self) -> str:
        """Canonical JSON of everything except informational timing fields."""
        payload = self.model_dump(mode="json")
        payload.pop("stage_timings")
        return canonical_dumps(payload)


def make_trace_id(question: str, seed: int, config: dict[str, object]) -> str:
    material = canonical_dumps({"question": question, "seed": seed, "config": config})
    return hashlib.blake2b(material.encode("utf-8"), digest_size=8).hexdigest()
