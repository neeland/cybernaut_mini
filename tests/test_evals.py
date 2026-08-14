"""Tests for evals.py metric functions and the evaluate() runner.

Uses the session-scoped built_index fixture (HashEmbedder dim=64,
TextProcessor(use_spacy=False)) — no network, no spaCy.
"""

from __future__ import annotations

import pytest

from cybernaut_mini.evals import (
    ModeMetrics,
    dcg,
    evaluate,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    shard_recall_at_n,
)
from cybernaut_mini.models import Judgment

# ------------------------------------------------------------------ #
# Unit tests for pure metric functions                                #
# ------------------------------------------------------------------ #


class TestDcg:
    def test_empty(self) -> None:
        assert dcg([]) == 0.0

    def test_single_relevant(self) -> None:
        # rank 1: 1 / log2(2) = 1.0
        assert abs(dcg([1.0]) - 1.0) < 1e-9

    def test_two_items(self) -> None:
        import math

        expected = 2.0 / math.log2(2) + 1.0 / math.log2(3)
        assert abs(dcg([2.0, 1.0]) - expected) < 1e-9

    def test_zero_gain(self) -> None:
        assert dcg([0.0, 0.0]) == 0.0


class TestRecallAtK:
    def test_all_relevant_found(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=5) == 1.0

    def test_none_found(self) -> None:
        assert recall_at_k(["x", "y"], {"a", "b"}, k=5) == 0.0

    def test_partial(self) -> None:
        # 1 of 2 relevant found → 0.5
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=5) == 0.5

    def test_k_cutoff(self) -> None:
        # "b" is at position 3 (index 2), k=2 should miss it
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=2) == 0.5

    def test_empty_relevant(self) -> None:
        assert recall_at_k(["a", "b"], set(), k=5) == 0.0


class TestMrrAtK:
    def test_first_hit_rank_1(self) -> None:
        assert mrr_at_k(["a", "b", "c"], {"a"}, k=10) == 1.0

    def test_first_hit_rank_3(self) -> None:
        assert abs(mrr_at_k(["x", "y", "a"], {"a"}, k=10) - 1 / 3) < 1e-9

    def test_no_hit_within_k(self) -> None:
        assert mrr_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_hit_just_at_k(self) -> None:
        assert abs(mrr_at_k(["x", "y", "a"], {"a"}, k=3) - 1 / 3) < 1e-9

    def test_hit_beyond_k_ignored(self) -> None:
        # "a" is at rank 4, k=3 → 0
        assert mrr_at_k(["x", "y", "z", "a"], {"a"}, k=3) == 0.0

    def test_empty_relevant(self) -> None:
        assert mrr_at_k(["a", "b"], set(), k=5) == 0.0


class TestNdcgAtK:
    def test_perfect_ranking(self) -> None:
        # Perfect: grade-2 doc first, grade-1 doc second → nDCG == 1.0
        result = ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, k=10)
        assert abs(result - 1.0) < 1e-9

    def test_reversed_ranking(self) -> None:
        # Worst case for 2 docs: lower-grade first
        result = ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, k=10)
        assert 0.0 < result < 1.0

    def test_no_relevant_docs_in_relevance(self) -> None:
        assert ndcg_at_k(["a", "b"], {}, k=10) == 0.0

    def test_zero_grade_only(self) -> None:
        # All grades are 0 → ideal DCG == 0, return 0
        assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, k=10) == 0.0

    def test_not_retrieved(self) -> None:
        # Relevant doc not in ranked list → actual DCG = 0 → nDCG < 1
        result = ndcg_at_k(["x"], {"a": 2}, k=10)
        assert result == 0.0

    def test_k_truncation(self) -> None:
        # grade-2 doc at rank 11 (beyond k=10) should not contribute
        ranked = ["x"] * 10 + ["a"]
        result = ndcg_at_k(ranked, {"a": 2}, k=10)
        assert result == 0.0

    def test_first_hit_rank_1(self) -> None:
        # Single relevant doc retrieved at rank 1 → nDCG = 1.0
        result = ndcg_at_k(["a"], {"a": 1}, k=10)
        assert abs(result - 1.0) < 1e-9


# ------------------------------------------------------------------ #
# shard_recall_at_n                                                   #
# ------------------------------------------------------------------ #

# Minimal doc → shard map: three docs across two shards.
_DOC_TO_SHARD: dict[str, int] = {
    "doc-a": 0,
    "doc-b": 0,
    "doc-c": 1,
}


class TestShardRecallAtN:
    def test_perfect_selection(self) -> None:
        """All relevant docs' shards are selected → 1.0."""
        assert shard_recall_at_n([0, 1], {"doc-a", "doc-c"}, _DOC_TO_SHARD) == pytest.approx(1.0)

    def test_zero_overlap(self) -> None:
        """No relevant doc's shard is selected → 0.0."""
        # shard 1 contains doc-c only; relevant docs are in shard 0
        assert shard_recall_at_n([1], {"doc-a", "doc-b"}, _DOC_TO_SHARD) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """One of two relevant docs' shards is selected → 0.5."""
        # shard 0 has doc-a (hit), shard 1 has doc-c (missed)
        result = shard_recall_at_n([0], {"doc-a", "doc-c"}, _DOC_TO_SHARD)
        assert result == pytest.approx(0.5)

    def test_empty_relevant_returns_zero(self) -> None:
        """Empty relevant set → 0.0 (guard against ZeroDivisionError)."""
        assert shard_recall_at_n([0, 1], set(), _DOC_TO_SHARD) == pytest.approx(0.0)

    def test_empty_selection_returns_zero(self) -> None:
        """No shards selected → 0.0 for any non-empty relevant set."""
        assert shard_recall_at_n([], {"doc-a"}, _DOC_TO_SHARD) == pytest.approx(0.0)

    def test_doc_absent_from_index_counts_as_miss(self) -> None:
        """Relevant doc not in doc_to_shard is a miss, not skipped.

        A document the index never saw cannot be retrieved and should penalise
        the routing score — skipping it would make recall look better than it is.
        """
        # doc-unknown not in _DOC_TO_SHARD → miss even though all shards selected
        result = shard_recall_at_n([0, 1], {"doc-a", "doc-unknown"}, _DOC_TO_SHARD)
        assert result == pytest.approx(0.5)

    def test_two_relevant_in_same_shard(self) -> None:
        """Both relevant docs in shard 0, shard 0 selected → 1.0 (not 0.5)."""
        result = shard_recall_at_n([0], {"doc-a", "doc-b"}, _DOC_TO_SHARD)
        assert result == pytest.approx(1.0)

    def test_single_relevant_found(self) -> None:
        assert shard_recall_at_n([1], {"doc-c"}, _DOC_TO_SHARD) == pytest.approx(1.0)

    def test_single_relevant_missed(self) -> None:
        assert shard_recall_at_n([1], {"doc-a"}, _DOC_TO_SHARD) == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# ModeMetrics serialisation                                           #
# ------------------------------------------------------------------ #


class TestModeMetricsDict:
    def test_as_dict_includes_shard_recall(self) -> None:
        """as_dict() must carry shard_recall_at_n for the Kedro pipeline."""
        m = ModeMetrics(mode="hybrid", shard_recall_at_n=0.75)
        d = m.as_dict()
        assert "shard_recall_at_n" in d
        assert d["shard_recall_at_n"] == pytest.approx(0.75)

    def test_as_dict_default_shard_recall_zero(self) -> None:
        assert ModeMetrics(mode="lexical").as_dict()["shard_recall_at_n"] == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# evaluate() runner tests                                             #
# ------------------------------------------------------------------ #


def _make_tiny_judgments(doc_ids: list[str]) -> list[Judgment]:
    """Build a minimal set of judgments using known doc IDs from built_index."""
    return [
        Judgment(
            query_id="q-lex",
            question="zylophristine compound assay methods spectrophotometry",
            relevant_document_ids={doc_ids[0]: 2, doc_ids[1]: 1},
        ),
        Judgment(
            query_id="q-dense",
            question="gene edit molecular scissors genomes",
            relevant_document_ids={doc_ids[1]: 2, doc_ids[0]: 1},
        ),
    ]


class TestEvaluateRunner:
    """Integration-level tests using the session built_index fixture."""

    def test_returns_one_metrics_per_mode(self, built_index, hash_embedder) -> None:
        from cybernaut_mini.config import AppConfig
        from cybernaut_mini.text import TextProcessor

        doc_ids = list(built_index.by_id.keys())[:2]
        judgments = _make_tiny_judgments(doc_ids)
        processor = TextProcessor(use_spacy=False)
        config = AppConfig()
        config.embedding.provider = "hash"  # type: ignore[assignment]

        metrics = evaluate(
            built_index,
            judgments,
            config=config,
            processor=processor,
            provider=hash_embedder,
        )

        assert len(metrics) == 4
        modes = {m.mode for m in metrics}
        assert modes == {"lexical", "dense", "hybrid", "agent"}

    def test_all_metrics_in_unit_interval(self, built_index, hash_embedder) -> None:
        from cybernaut_mini.config import AppConfig
        from cybernaut_mini.text import TextProcessor

        doc_ids = list(built_index.by_id.keys())[:2]
        judgments = _make_tiny_judgments(doc_ids)
        processor = TextProcessor(use_spacy=False)
        config = AppConfig()

        metrics = evaluate(
            built_index,
            judgments,
            config=config,
            processor=processor,
            provider=hash_embedder,
        )

        for m in metrics:
            assert 0.0 <= m.recall_at_5 <= 1.0, f"{m.mode} recall@5 out of range"
            assert 0.0 <= m.recall_at_10 <= 1.0, f"{m.mode} recall@10 out of range"
            assert 0.0 <= m.mrr_at_10 <= 1.0, f"{m.mode} mrr@10 out of range"
            assert 0.0 <= m.ndcg_at_10 <= 1.0, f"{m.mode} ndcg@10 out of range"
            assert 0.0 <= m.shard_recall_at_n <= 1.0, f"{m.mode} shard_recall_at_n out of range"

    def test_agent_mean_retrieval_calls_bounded(self, built_index, hash_embedder) -> None:
        from cybernaut_mini.config import AppConfig
        from cybernaut_mini.text import TextProcessor

        doc_ids = list(built_index.by_id.keys())[:2]
        judgments = _make_tiny_judgments(doc_ids)
        processor = TextProcessor(use_spacy=False)
        config = AppConfig()

        metrics = evaluate(
            built_index,
            judgments,
            config=config,
            processor=processor,
            provider=hash_embedder,
            modes=("agent",),
        )

        agent_m = metrics[0]
        assert agent_m.mean_retrieval_calls <= 18

    def test_determinism(self, built_index, hash_embedder) -> None:
        """Two evaluate() calls with the same inputs produce identical metrics."""
        from cybernaut_mini.config import AppConfig
        from cybernaut_mini.text import TextProcessor

        doc_ids = list(built_index.by_id.keys())[:2]
        judgments = _make_tiny_judgments(doc_ids)
        processor = TextProcessor(use_spacy=False)
        config = AppConfig()

        run1 = evaluate(
            built_index,
            judgments,
            config=config,
            processor=processor,
            provider=hash_embedder,
        )
        run2 = evaluate(
            built_index,
            judgments,
            config=config,
            processor=processor,
            provider=hash_embedder,
        )

        for m1, m2 in zip(run1, run2, strict=True):
            assert m1.mode == m2.mode
            assert m1.recall_at_5 == m2.recall_at_5
            assert m1.recall_at_10 == m2.recall_at_10
            assert m1.mrr_at_10 == m2.mrr_at_10
            assert m1.ndcg_at_10 == m2.ndcg_at_10
            assert m1.mean_retrieval_calls == m2.mean_retrieval_calls
            assert m1.mean_llm_calls == m2.mean_llm_calls

    def test_baseline_retrieval_calls_is_one(self, built_index, hash_embedder) -> None:
        from cybernaut_mini.config import AppConfig
        from cybernaut_mini.text import TextProcessor

        doc_ids = list(built_index.by_id.keys())[:2]
        judgments = _make_tiny_judgments(doc_ids)
        processor = TextProcessor(use_spacy=False)
        config = AppConfig()

        for mode in ("lexical", "dense", "hybrid"):
            metrics = evaluate(
                built_index,
                judgments,
                config=config,
                processor=processor,
                provider=hash_embedder,
                modes=(mode,),
            )
            assert metrics[0].mean_retrieval_calls == 1.0, (
                f"baseline mode={mode!r} should have mean_retrieval_calls=1"
            )

    def test_llm_calls_always_zero(self, built_index, hash_embedder) -> None:
        from cybernaut_mini.config import AppConfig
        from cybernaut_mini.text import TextProcessor

        doc_ids = list(built_index.by_id.keys())[:2]
        judgments = _make_tiny_judgments(doc_ids)
        processor = TextProcessor(use_spacy=False)
        config = AppConfig()

        metrics = evaluate(
            built_index,
            judgments,
            config=config,
            processor=processor,
            provider=hash_embedder,
        )
        for m in metrics:
            assert m.mean_llm_calls == 0.0, f"{m.mode} should have 0 LLM calls"

    def test_empty_judgments_returns_zero_metrics(self, built_index, hash_embedder) -> None:
        from cybernaut_mini.config import AppConfig
        from cybernaut_mini.text import TextProcessor

        processor = TextProcessor(use_spacy=False)
        config = AppConfig()

        metrics = evaluate(
            built_index,
            [],
            config=config,
            processor=processor,
            provider=hash_embedder,
            modes=("lexical",),
        )
        assert len(metrics) == 1
        m = metrics[0]
        assert m.recall_at_5 == 0.0
        assert m.ndcg_at_10 == 0.0
