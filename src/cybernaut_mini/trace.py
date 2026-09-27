"""Structured agent trace — a product artifact, not debug logging.

Captures every candidate, action, routing/retrieval signal, reward breakdown, and
node statistic so a reader can replay the agent's decisions. No chain-of-thought is
stored; judge reasons are short structured strings. Timing fields are informational
and are the only part that varies between otherwise-identical runs.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — the trace records the
    stage 5-8 signals (routing, shard selection, expansion) that the post describes,
    plus the per-node statistics the Cybernaut-1 agentic follow-up needs to explain a
    search. Local copy: ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - The trace is a product artifact, not debug logging: every candidate node, action,
      routing/retrieval signal and reward component is persisted, so a reader can
      replay a run without any other store.
    - No chain-of-thought is stored. ``judge_reason`` is a short structured string
      such as "heuristic: rel=0.36 cov=1.00 red=0.02 over 5 hits", never a model
      transcript.
    - ``decision_fingerprint()`` drops ``stage_timings`` before canonical JSON. Timing
      is the only field allowed to differ between two otherwise-identical runs, so the
      fingerprint is what a reproducibility test compares.
    - ``trace_id`` is a blake2b digest of ``(question, seed, config)``, so the same
      request under the same config gets the same id without a timestamp or a counter.
    - Every model sets ``extra="forbid"``: a trace field renamed in one place fails
      loudly instead of being silently dropped from the artifact.

Alternatives considered:
    - Structured logging or OpenTelemetry spans: rejected because both are
      observational, lossy and non-deterministic; the trace must be a file a reader can
      diff and a test can fingerprint.
    - Persisting the full prompt and raw model output per node: rejected on
      chain-of-thought and size grounds; the judge's structured reason is enough to
      audit a decision.
    - Letting ``stage_timings`` participate in the fingerprint: rejected because it
      makes every run look different and would force reproducibility tests to
      special-case a field instead of comparing one canonical document.
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
