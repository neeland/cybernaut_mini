from __future__ import annotations

from cybernaut_mini.agent.state import Stage, StateSummary
from cybernaut_mini.models import Document, SearchHit
from cybernaut_mini.providers.judge import HeuristicJudge
from cybernaut_mini.providers.query_generator import HeuristicQueryGenerator
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
