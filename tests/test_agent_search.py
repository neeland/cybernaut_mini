from __future__ import annotations

import pytest

from cybernaut_mini.agent.search import run_agent_search
from cybernaut_mini.config import AppConfig, EmbeddingConfig
from cybernaut_mini.indexing import LoadedIndex
from cybernaut_mini.models import MetadataFilter
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.text import TextProcessor

QUESTION = "gene editing immune response"


def agent_config() -> AppConfig:
    return AppConfig(embedding=EmbeddingConfig(provider="hash", dim=64))


def run(index: LoadedIndex, question: str = QUESTION, **kwargs: object):
    return run_agent_search(
        index,
        question,
        config=agent_config(),
        processor=TextProcessor(use_spacy=False),
        provider=HashEmbedder(dim=64),
        **kwargs,  # type: ignore[arg-type]
    )


def test_trace_contains_all_three_stages(built_index: LoadedIndex) -> None:
    result, _ = run(built_index)
    stages = {node.stage for node in result.trace.nodes}
    assert stages == {"explore", "refine", "exploit"}


def test_results_and_trace_are_deterministic(built_index: LoadedIndex) -> None:
    first, _ = run(built_index)
    second, _ = run(built_index)
    assert first.trace.decision_fingerprint() == second.trace.decision_fingerprint()
    assert [h.document.id for h in first.hits] == [h.document.id for h in second.hits]


def test_retrieval_budget_is_never_exceeded(built_index: LoadedIndex) -> None:
    result, _ = run(built_index)
    assert result.trace.retrieval_calls <= 18


def test_budget_of_one_stops_after_first_call(built_index: LoadedIndex) -> None:
    config = agent_config()
    config.agent.max_retrieval_calls = 1
    result, _ = run_agent_search(
        built_index,
        QUESTION,
        config=config,
        processor=TextProcessor(use_spacy=False),
        provider=HashEmbedder(dim=64),
    )
    assert result.trace.retrieval_calls == 1
    assert result.trace.stop_reason == "budget_exhausted"


def test_heuristic_llm_calls_are_zero(built_index: LoadedIndex) -> None:
    result, _ = run(built_index)
    assert result.trace.llm_calls == 0


def test_terminal_node_hits_are_reproducible(built_index: LoadedIndex) -> None:
    result, agent = run(built_index)
    assert result.hits  # sanity
    replayed = agent.replay(result.plan, result.metadata_filter)
    assert [h.document.id for h in replayed[: len(result.hits)]] == [
        h.document.id for h in result.hits
    ]
    assert [h.score for h in replayed[: len(result.hits)]] == pytest.approx(
        [h.score for h in result.hits]
    )


def test_no_results_when_filter_excludes_everything(built_index: LoadedIndex) -> None:
    impossible = MetadataFilter(metadata_equals={"category": "nonexistent-category"})
    result, _ = run_agent_search(
        built_index,
        QUESTION,
        config=agent_config(),
        processor=TextProcessor(use_spacy=False),
        provider=HashEmbedder(dim=64),
        metadata_filter=impossible,
    )
    assert result.hits == []
    assert result.trace.stop_reason == "no_results"


def test_selected_path_starts_at_root(built_index: LoadedIndex) -> None:
    result, _ = run(built_index)
    assert result.trace.selected_path[0] == 0
    assert len(result.trace.selected_path) >= 2


def test_trace_records_reward_and_routing_signals(built_index: LoadedIndex) -> None:
    result, _ = run(built_index)
    node = result.trace.nodes[0]
    assert set(node.reward_components) >= {
        "relevance",
        "coverage",
        "dense",
        "lexical",
        "redundancy",
    }
    assert node.routing is not None
    assert "dense" in node.routing and "fused" in node.routing
    # Regex processor extracts no entities -> entity ranker omitted, not zero-filled.
    assert node.routing["entity"] is None
