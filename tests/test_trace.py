from __future__ import annotations

import json

from cybernaut_mini.agent.search import run_agent_search
from cybernaut_mini.config import AppConfig, EmbeddingConfig
from cybernaut_mini.indexing import LoadedIndex
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.text import TextProcessor
from cybernaut_mini.trace import AgentTrace, make_trace_id


def _run(index: LoadedIndex):
    return run_agent_search(
        index,
        "solar panel efficiency record",
        config=AppConfig(embedding=EmbeddingConfig(provider="hash", dim=64)),
        processor=TextProcessor(use_spacy=False),
        provider=HashEmbedder(dim=64),
    )


def test_trace_round_trips_through_json(built_index: LoadedIndex) -> None:
    result, _ = _run(built_index)
    payload = result.trace.model_dump_json()
    restored = AgentTrace.model_validate_json(payload)
    assert restored.decision_fingerprint() == result.trace.decision_fingerprint()


def test_trace_id_is_deterministic() -> None:
    config = {"seed": 42, "agent": {"judge": "heuristic"}}
    assert make_trace_id("q", 42, config) == make_trace_id("q", 42, config)
    assert make_trace_id("q", 42, config) != make_trace_id("q", 43, config)


def test_decision_fingerprint_ignores_timing(built_index: LoadedIndex) -> None:
    result, _ = _run(built_index)
    trace = result.trace
    mutated = trace.model_copy(deep=True)
    mutated.stage_timings[0].seconds += 123.0
    assert mutated.decision_fingerprint() == trace.decision_fingerprint()


def test_trace_reports_call_counts_and_tokens(built_index: LoadedIndex) -> None:
    result, _ = _run(built_index)
    trace = result.trace
    assert trace.retrieval_calls == sum(1 for _ in trace.nodes)
    assert trace.embedding_calls > 0
    assert trace.normalized_tokens  # non-empty content tokens
    assert len(trace.stage_timings) == 3


def test_judge_reason_capped(built_index: LoadedIndex) -> None:
    result, _ = _run(built_index)
    for node in result.trace.nodes:
        if node.judge_reason is not None:
            assert len(node.judge_reason) <= 240


def test_trace_config_snapshot_is_json_serializable(built_index: LoadedIndex) -> None:
    result, _ = _run(built_index)
    # The whole trace must serialize with the canonical json encoder.
    json.loads(result.trace.model_dump_json())
