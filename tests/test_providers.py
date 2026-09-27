from __future__ import annotations

import numpy as np
import pytest

from cybernaut_mini.agent.state import Stage, StateSummary
from cybernaut_mini.config import AgentConfig, AppConfig, ConfigError
from cybernaut_mini.models import Document, SearchHit
from cybernaut_mini.providers.judge import CrossEncoderJudge, HeuristicJudge, create_judge
from cybernaut_mini.providers.query_generator import (
    HeuristicQueryGenerator,
    LLMQueryGenerator,
    create_query_generator,
    parse_generated_queries,
)
from cybernaut_mini.text import TextProcessor


def summary(**kwargs: object) -> StateSummary:
    base: dict[str, object] = {"current_query": "gene therapy trial", "stage": Stage.EXPLORE}
    base.update(kwargs)
    return StateSummary(**base)  # type: ignore[arg-type]


def make_hit(doc_id: str, title: str, text: str, rank: int) -> SearchHit:
    return SearchHit(
        document=Document(id=doc_id, title=title, text=text),
        rank=rank,
        score=1.0,
        shard_id=0,
        dense_score=0.5,
        bm25_score=1.0,
        query_variant="q",
        snippet=text[:100],
    )


def test_generator_dedups_and_caps(text_processor: TextProcessor) -> None:
    gen = HeuristicQueryGenerator(text_processor)
    candidates = gen.generate("gene therapy trial", summary(), n=5)
    texts = [c.text for c in candidates]
    assert len(texts) == len(set((c.text, tuple(c.expansions)) for c in candidates))
    assert len(candidates) <= 5
    assert candidates[0].origin == "original"


def test_generator_is_deterministic(text_processor: TextProcessor) -> None:
    gen = HeuristicQueryGenerator(text_processor)
    state = summary(current_query="solar panel efficiency")
    first = gen.generate("solar panel efficiency", state, 5)
    second = gen.generate("solar panel efficiency", state, 5)
    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]


def test_generator_includes_expansion_and_keyword_variants(text_processor: TextProcessor) -> None:
    gen = HeuristicQueryGenerator(text_processor)
    candidates = gen.generate(
        "gene therapy",
        summary(current_query="gene therapy", expansions=("dna",), missing_keywords=("crispr",)),
        n=5,
    )
    origins = {c.origin for c in candidates}
    assert "expanded" in origins
    assert "keyword" in origins


def test_judge_scores_within_unit_interval(text_processor: TextProcessor) -> None:
    judge = HeuristicJudge(text_processor)
    hits = [
        make_hit("a", "Gene therapy trial", "gene therapy trial results were durable", 1),
        make_hit("b", "Solar panels", "solar panels reach efficiency record", 2),
    ]
    score = judge.score("gene therapy results", "gene therapy trial", hits)
    for value in (score.relevance, score.coverage, score.redundancy):
        assert 0.0 <= value <= 1.0
    assert len(score.reason) <= 240


def test_judge_empty_hits_is_zero(text_processor: TextProcessor) -> None:
    judge = HeuristicJudge(text_processor)
    score = judge.score("q", "q", [])
    assert score.relevance == score.coverage == score.redundancy == 0.0


def test_judge_redundancy_high_for_identical_titles(text_processor: TextProcessor) -> None:
    judge = HeuristicJudge(text_processor)
    hits = [
        make_hit("a", "gene therapy trial", "body one", 1),
        make_hit("b", "gene therapy trial", "body two", 2),
    ]
    score = judge.score("gene therapy", "gene therapy", hits)
    assert score.redundancy == 1.0


def test_judge_is_deterministic(text_processor: TextProcessor) -> None:
    judge = HeuristicJudge(text_processor)
    hits = [make_hit("a", "gene therapy", "gene therapy trial", 1)]
    assert judge.score("gene", "gene therapy", hits) == judge.score("gene", "gene therapy", hits)


# ---------------------- model-backed providers ----------------------- #


def test_cross_encoder_judge_calibrates_logits(text_processor: TextProcessor) -> None:
    # Raw logits (outside [0,1]) must be sigmoid-mapped; mean -> relevance, max -> coverage.
    judge = CrossEncoderJudge(
        text_processor,
        model_name="stub/reranker",
        predict_fn=lambda pairs: np.array([4.0, -4.0][: len(pairs)], dtype=np.float32),
    )
    hits = [
        make_hit("a", "Gene therapy trial", "durable response", 1),
        make_hit("b", "Solar panels", "efficiency record", 2),
    ]
    score = judge.score("gene therapy results", "gene therapy", hits)
    # sigmoid(4) ~= 0.982, sigmoid(-4) ~= 0.018 -> mean 0.5, max 0.982.
    assert score.relevance == pytest.approx(0.5, abs=0.01)
    assert score.coverage == pytest.approx(0.982, abs=0.01)


def test_cross_encoder_judge_scores_and_counts_calls(text_processor: TextProcessor) -> None:
    judge = CrossEncoderJudge(
        text_processor,
        model_name="stub/reranker",
        predict_fn=lambda pairs: np.array([0.9, 0.3][: len(pairs)], dtype=np.float32),
    )
    hits = [
        make_hit("a", "Gene therapy trial", "durable response", 1),
        make_hit("b", "Solar panels", "efficiency record", 2),
    ]
    score = judge.score("gene therapy results", "gene therapy", hits)
    assert score.relevance == pytest.approx(0.6)
    assert score.coverage == pytest.approx(0.9)
    assert score.coverage >= score.relevance
    assert score.reason.startswith("cross-encoder(reranker)")
    assert judge.calls == 1
    assert judge.score("q", "q", []).relevance == 0.0
    assert judge.calls == 1  # empty hits never invoke the model


def test_cross_encoder_judge_semantic_redundancy(text_processor: TextProcessor) -> None:
    class IdenticalEmbedder:
        identifier = "stub"
        dim = 2

        def embed_documents(self, texts: list[str]) -> np.ndarray:
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

        def embed_queries(self, texts: list[str]) -> np.ndarray:
            return self.embed_documents(texts)

    judge = CrossEncoderJudge(
        text_processor,
        model_name="stub/reranker",
        embedder=IdenticalEmbedder(),
        predict_fn=lambda pairs: np.full(len(pairs), 0.5, dtype=np.float32),
    )
    hits = [
        make_hit("a", "one thing", "body", 1),
        make_hit("b", "entirely other", "body", 2),
    ]
    # Titles share no tokens, but the embedder says the docs are identical.
    assert judge.score("q", "q", hits).redundancy == pytest.approx(1.0)


def test_parse_generated_queries_strips_noise() -> None:
    raw = (
        "<think>internal musing</think>\n"
        "1. gene therapy durability\n"
        '- "gene therapy trial results"\n'
        "gene therapy durability\n"  # duplicate by lexical form
        "\n"
        "Sure! Here are some queries you could try for this question: " + "x" * 120
    )
    assert parse_generated_queries(raw) == [
        "gene therapy durability",
        "gene therapy trial results",
    ]


def test_llm_generator_uses_model_output(text_processor: TextProcessor) -> None:
    gen = LLMQueryGenerator(
        text_processor,
        model_name="stub/lm",
        chat_fn=lambda system, user: "gene therapy durability\nlong-term gene therapy outcomes",
    )
    candidates = gen.generate("gene therapy trial", summary(), n=4)
    assert candidates[0].origin == "original"
    llm_texts = [c.text for c in candidates if c.origin == "llm"]
    assert llm_texts == ["gene therapy durability", "long-term gene therapy outcomes"]
    assert gen.calls == 1


def test_llm_generator_falls_back_to_heuristic_on_failure(
    text_processor: TextProcessor,
) -> None:
    def broken(system: str, user: str) -> str:
        raise RuntimeError("model exploded")

    gen = LLMQueryGenerator(text_processor, model_name="stub/lm", chat_fn=broken)
    candidates = gen.generate("gene therapy trial", summary(), n=5)
    heuristic = HeuristicQueryGenerator(text_processor).generate(
        "gene therapy trial", summary(), n=5
    )
    assert [c.text for c in candidates] == [c.text for c in heuristic]


def test_factories_default_to_heuristics(text_processor: TextProcessor) -> None:
    config = AgentConfig()
    assert isinstance(create_judge(config, text_processor), HeuristicJudge)
    assert isinstance(create_query_generator(config, text_processor), HeuristicQueryGenerator)


def test_offline_rejects_model_backed_agent_providers() -> None:
    config = AppConfig.model_validate(
        {"embedding": {"provider": "hash"}, "agent": {"judge": "cross_encoder"}}
    )
    with pytest.raises(ConfigError, match="offline mode: agent judge"):
        config.require_offline_compatible()


# -------------------- per-hit signal bundle rendering -------------------- #


def signal_hit(doc_id: str, rank: int, *, rerank: bool = False) -> SearchHit:
    hit = make_hit(doc_id, f"Title {doc_id}", f"Body text for {doc_id}", rank)
    contributions = {"lexical": 0.016, "dense": 0.015}
    if rerank:
        contributions["intent"] = 0.008
    return hit.model_copy(
        update={"bm25_rank": rank, "dense_rank": rank, "rrf_contributions": contributions}
    )


def test_signal_table_renders_one_line_per_hit_with_header() -> None:
    from cybernaut_mini.providers.signals import render_signal_table

    table = render_signal_table([signal_hit("a", 1, rerank=True), signal_hit("b", 2)])
    lines = table.splitlines()
    assert lines[0] == "rank | doc | bm25 | dense | rerank | rrf"
    assert len(lines) == 3
    assert lines[1].startswith("1 | a | 1.000@1 | 0.500@1 | 0.008 | 1.000")
    # A hit with no non-lexical/dense ranker shows "-" in the rerank column.
    assert " - | " in lines[2]


def test_signal_table_caps_rows_and_handles_no_hits() -> None:
    from cybernaut_mini.providers.signals import render_signal_table

    hits = [signal_hit(f"d{i}", i + 1) for i in range(8)]
    assert len(render_signal_table(hits).splitlines()) == 6  # header + 5 rows
    assert render_signal_table([]) == ""


def test_signal_summary_has_a_fixed_shape() -> None:
    from cybernaut_mini.providers.signals import signal_summary

    stats = signal_summary([signal_hit("a", 1), signal_hit("b", 2)])
    assert stats == {
        "bm25_max": 1.0,
        "bm25_mean": 1.0,
        "dense_max": 0.5,
        "dense_mean": 0.5,
        "rrf_max": 1.0,
        "rrf_mean": 1.0,
        "n_hits": 2.0,
    }
    empty = signal_summary([])
    assert empty["n_hits"] == 0.0
    assert set(empty) == set(stats)


def test_generator_prompt_embeds_the_signal_table() -> None:
    from cybernaut_mini.providers.query_generator import build_generator_prompt

    with_hits = build_generator_prompt(
        "gene therapy results",
        summary(top_hits=(signal_hit("a", 1), signal_hit("b", 2))),
        n=3,
    )
    assert "Ranking signals for the current top hits:" in with_hits
    assert "rank | doc | bm25 | dense | rerank | rrf" in with_hits
    without_hits = build_generator_prompt("gene therapy results", summary(), n=3)
    assert "Ranking signals" not in without_hits
