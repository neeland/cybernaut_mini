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
        "diversity",
        "redundancy",
    }
    assert node.routing is not None
    assert "dense" in node.routing and "fused" in node.routing
    # Regex processor extracts no entities -> entity ranker omitted, not zero-filled.
    assert node.routing["entity"] is None


# ---------------- agent completion: knobs, annealing, accounting ---------------- #


def test_trace_emits_hybrid_weight_and_reranker_actions(built_index: LoadedIndex) -> None:
    """The refine stage tries the {0.5, 1.0, 1.5} lexical grid and a rerank-off
    branch, so AdjustHybridWeights/ToggleRerankers are real actions in the trace,
    not dead classes."""
    result, _ = run(built_index)
    actions = [node.action for node in result.trace.nodes if node.action]
    types = {action["type"] for action in actions}
    assert "AdjustHybridWeights" in types
    assert "ToggleRerankers" in types
    weight_nodes = [
        node
        for node in result.trace.nodes
        if node.action and node.action["type"] == "AdjustHybridWeights"
    ]
    assert {node.lexical_weight for node in weight_nodes} <= {0.5, 1.5}
    assert all(node.dense_weight == 1.0 for node in weight_nodes)


def test_phase_accounting_is_recorded_per_stage(built_index: LoadedIndex) -> None:
    result, _ = run(built_index)
    accounting = result.phase_accounting
    assert set(accounting) == {"explore", "refine", "exploit"}
    for stage, values in accounting.items():
        assert {"retrieval_calls", "embedding_calls", "llm_calls", "nodes"} <= set(values), stage
    # The same numbers ride in the replayable trace.
    assert result.trace.config["phase_accounting"] == accounting
    total = sum(values["retrieval_calls"] for values in accounting.values())
    assert total == result.trace.retrieval_calls
    # Temperature anneals wide-to-narrow across the stages.
    assert (
        accounting["explore"]["temperature"]
        > accounting["refine"]["temperature"]
        > accounting["exploit"]["temperature"]
    )


def test_annealed_schedule_runs_and_stays_deterministic(built_index: LoadedIndex) -> None:
    from cybernaut_mini.agent.schedule import ANNEALED_SCHEDULE
    from cybernaut_mini.agent.search import SearchAgent
    from cybernaut_mini.providers.judge import HeuristicJudge
    from cybernaut_mini.providers.query_generator import HeuristicQueryGenerator

    processor = TextProcessor(use_spacy=False)

    def run_annealed():
        agent = SearchAgent(
            built_index,
            processor=processor,
            provider=HashEmbedder(dim=64),
            generator=HeuristicQueryGenerator(processor),
            judge=HeuristicJudge(processor),
            rrf_config=agent_config().rrf,
            stage_schedule=ANNEALED_SCHEDULE,
        )
        return agent.run(QUESTION, None, {})

    first = run_annealed()
    second = run_annealed()
    assert first.hits
    assert first.trace.retrieval_calls <= 18
    assert first.trace.decision_fingerprint() == second.trace.decision_fingerprint()
    types = {node.action["type"] for node in first.trace.nodes if node.action}
    assert "AdjustHybridWeights" in types


def test_config_driven_schedule_reaches_the_agent(built_index: LoadedIndex) -> None:
    """A per-stage override mapping changes what the agent actually executes:
    with the refine knob variants switched off, no AdjustHybridWeights or
    ToggleRerankers node may appear."""
    from cybernaut_mini.agent.schedule import schedule_from_config
    from cybernaut_mini.agent.search import SearchAgent
    from cybernaut_mini.providers.judge import HeuristicJudge
    from cybernaut_mini.providers.query_generator import HeuristicQueryGenerator

    processor = TextProcessor(use_spacy=False)
    schedule = schedule_from_config(
        {"refine": {"weight_variants": [], "try_rerank_off": False}}
    )
    agent = SearchAgent(
        built_index,
        processor=processor,
        provider=HashEmbedder(dim=64),
        generator=HeuristicQueryGenerator(processor),
        judge=HeuristicJudge(processor),
        rrf_config=agent_config().rrf,
        stage_schedule=schedule,
    )
    result = agent.run(QUESTION, None, {})
    types = {node.action["type"] for node in result.trace.nodes if node.action}
    assert "AdjustHybridWeights" not in types
    assert "ToggleRerankers" not in types
