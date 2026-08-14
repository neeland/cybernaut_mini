"""Evaluation metrics and runner for cybernaut-mini retrieval modes.

Metrics implemented
-------------------
* ``dcg`` / ``ndcg_at_k`` — graded relevance; ideal DCG from sorted true grades.
* ``recall_at_k`` — fraction of relevant docs found in top-k.
* ``mrr_at_k`` — reciprocal rank of first relevant doc in top-k (0 if none).
* ``shard_recall_at_n`` — fraction of qrels-positive docs whose home shard is
  among the N shards the router selected.

Runner
------
``evaluate()`` runs all four modes (lexical, dense, hybrid, agent) over a list
of :class:`~cybernaut_mini.models.Judgment` objects and returns one
:class:`ModeMetrics` per mode, averaged over queries.

Retrieval call accounting
--------------------------
* Baseline modes (lexical/dense/hybrid): 1 call per query (one ``retrieve`` call).
* Agent mode: ``result.trace.retrieval_calls`` (up to 18 per query).
* LLM calls: 0 for all modes (heuristic judge + generator, no LLM).

Wall-clock time is measured with ``time.perf_counter`` and labelled informational
in the output — it captures real elapsed time but is not reproducible.

Shard-recall accounting
-----------------------
* Baseline modes: one ``route()`` call per query (pre-computed before the mode
  loop, so all three static modes share the same routing result).
* Agent mode: the shards the agent actually searched (``AgentResult.shard_ids``),
  which differs from the static router call — it measures *exploration coverage*,
  not *router quality*. Do not average or compare the two directly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cybernaut_mini.config import AppConfig
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.models import Judgment
    from cybernaut_mini.providers.embeddings import EmbeddingProvider
    from cybernaut_mini.text import TextProcessor


# ------------------------------------------------------------------ #
# Pure metric functions                                               #
# ------------------------------------------------------------------ #


def dcg(gains: list[float]) -> float:
    """Discounted Cumulative Gain.

    ``gains[i]`` is the relevance grade at rank i+1 (1-indexed position i+1).
    Position discount = 1 / log2(i + 2) for i >= 0.
    Returns 0.0 for an empty list.
    """
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_doc_ids: list[str], relevance: dict[str, int], k: int) -> float:
    """Normalised DCG at rank k.

    ``relevance`` maps doc_id -> grade (positive int means relevant).
    Returns 0.0 when no relevant docs exist in the judgment.
    """
    if not relevance:
        return 0.0

    # Ideal DCG: sort all known relevant grades descending.
    ideal_grades = sorted(relevance.values(), reverse=True)
    idcg = dcg([float(g) for g in ideal_grades[:k]])
    if idcg == 0.0:
        return 0.0

    # Actual DCG from the ranked list.
    actual_grades = [float(relevance.get(doc_id, 0)) for doc_id in ranked_doc_ids[:k]]
    return dcg(actual_grades) / idcg


def recall_at_k(ranked_doc_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k ranked results.

    Returns 0.0 when ``relevant_ids`` is empty.
    """
    if not relevant_ids:
        return 0.0
    found = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant_ids)
    return found / len(relevant_ids)


def mrr_at_k(ranked_doc_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Mean Reciprocal Rank (reciprocal rank of the first relevant doc in top-k).

    Returns 0.0 if no relevant doc appears in the top-k slice.
    """
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def shard_recall_at_n(
    selected_shard_ids: list[int],
    relevant_doc_ids: set[str],
    doc_to_shard: dict[str, int],
) -> float:
    """Fraction of qrels-positive docs whose home shard is in the router's selection.

    Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 5 "Shard
        Selection" exists precisely because "if we route the question to the wrong
        shards, we won't return the best document". This metric makes that failure
        mode measurable: 1.0 means every relevant document's shard was selected;
        0.0 means none were. Local copy:
        ``docs/blog-archive/the-road-to-cybernaut-1.md``.

    Assumptions:
        - A relevant document absent from ``doc_to_shard`` is counted as a miss,
          not skipped. A document the index never saw cannot be retrieved by any
          mode and should penalise routing accordingly.
        - N is implicit in ``len(selected_shard_ids)``; the caller's
          ``shard_beam_n`` parameter determines it for static modes.
        - Static modes (lexical/dense/hybrid) supply the router's top-N selection,
          measuring *router quality*. Agent mode supplies the shards the agent
          actually searched (``AgentResult.shard_ids``), measuring *exploration
          coverage*. The two are computed the same way but measure different
          behaviours; do not average or compare them directly.

    Alternatives rejected:
        - Skipping docs absent from ``doc_to_shard``: silently hides corpus gaps
          and makes recall look better than it is.
        - Computing per-shard recall and averaging: double-counts queries where
          many relevant docs share one shard, inflating the number.

    Returns 0.0 when ``relevant_doc_ids`` is empty.
    """
    if not relevant_doc_ids:
        return 0.0
    selected_set = set(selected_shard_ids)
    found = sum(1 for doc_id in relevant_doc_ids if doc_to_shard.get(doc_id) in selected_set)
    return found / len(relevant_doc_ids)


# ------------------------------------------------------------------ #
# ModeMetrics dataclass                                               #
# ------------------------------------------------------------------ #


@dataclass
class ModeMetrics:
    """Aggregated metrics for one retrieval mode, averaged over all queries."""

    mode: str
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    shard_recall_at_n: float = 0.0
    mean_retrieval_calls: float = 0.0
    mean_llm_calls: float = 0.0
    wall_clock_seconds: float = 0.0  # informational

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr_at_10": self.mrr_at_10,
            "ndcg_at_10": self.ndcg_at_10,
            "shard_recall_at_n": self.shard_recall_at_n,
            "mean_retrieval_calls": self.mean_retrieval_calls,
            "mean_llm_calls": self.mean_llm_calls,
            "wall_clock_seconds": self.wall_clock_seconds,
        }


# ------------------------------------------------------------------ #
# Evaluate runner                                                     #
# ------------------------------------------------------------------ #


def evaluate(
    index: LoadedIndex,
    judgments: list[Judgment],
    *,
    config: AppConfig,
    processor: TextProcessor,
    provider: EmbeddingProvider,
    modes: tuple[str, ...] = ("lexical", "dense", "hybrid", "agent"),
    shard_beam_n: int = 100,
) -> list[ModeMetrics]:
    """Run all specified modes over every judgment and return averaged metrics.

    For baseline modes (lexical/dense/hybrid) uses ``retrieval.retrieve`` with
    ``top_k=10``.  For agent mode uses ``agent.search.run_agent_search`` with
    ``output_top_k=10``.

    Retrieval call count:
      - Baseline: 1 per query.
      - Agent: ``result.trace.retrieval_calls``.

    LLM call count: always 0 (heuristic providers only).

    Shard-recall computation
    -------------------------
    ``shard_beam_n`` is the number of shards the router is allowed to select for
    static modes; it defaults to 100, matching
    :data:`~cybernaut_mini.query.s5_select.DEFAULT_CANDIDATE_DEPTH` — the width
    the system actually runs with. Routing is pre-computed once per query before
    the mode loop so lexical/dense/hybrid share one ``route()`` call per query.

    The ``shard_recall_at_n`` field in :class:`ModeMetrics` means different things
    per mode:

    * **lexical / dense / hybrid** — *router quality*: fraction of relevant docs
      whose shard is among the top ``shard_beam_n`` shards the static router
      selected for this question.
    * **agent** — *exploration coverage*: fraction of relevant docs whose shard
      is among the shards the agent actually searched (``AgentResult.shard_ids``),
      which varies per query and is not capped at ``shard_beam_n``.

    Do not average or compare static-mode and agent-mode shard_recall values —
    they are computed the same way but measure genuinely different behaviours.
    """
    from cybernaut_mini.agent.search import run_agent_search
    from cybernaut_mini.retrieval import retrieve
    from cybernaut_mini.routing import route

    # doc → shard map, built once from the resident manifests.
    doc_to_shard: dict[str, int] = {
        doc_id: shard_id
        for shard_id, manifest in index.manifests.items()
        for doc_id in manifest.document_ids
    }

    n = len(judgments)
    results: list[ModeMetrics] = []

    # Pre-route every query once so all three static modes share the result.
    # route() caches its ShardSelector per (index, model), so the HNSW build
    # is paid at most once per evaluate() call regardless of query count.
    routed_shard_ids: list[list[int]] = []
    for judgment in judgments:
        ids, _ = route(
            index,
            judgment.question,
            processor=processor,
            provider=provider,
            rrf_config=config.rrf,
            top_n=shard_beam_n,
        )
        routed_shard_ids.append(ids)

    for mode in modes:
        total_r5 = 0.0
        total_r10 = 0.0
        total_mrr = 0.0
        total_ndcg = 0.0
        total_shard_recall = 0.0
        total_ret_calls = 0.0
        total_llm_calls = 0.0
        t_start = time.perf_counter()

        if n == 0:
            results.append(ModeMetrics(mode=mode))
            continue

        for idx, judgment in enumerate(judgments):
            question = judgment.question
            relevance = judgment.relevant_document_ids
            relevant_ids = set(relevance.keys())

            if mode in ("lexical", "dense", "hybrid"):
                hits = retrieve(
                    index,
                    question,
                    mode=mode,  # type: ignore[arg-type]
                    processor=processor,
                    provider=provider,
                    rrf_config=config.rrf,
                    top_k=10,
                )
                ranked_ids = [h.document.id for h in hits]
                ret_calls = 1
                llm_calls = 0
                query_shard_ids = routed_shard_ids[idx]

            else:  # agent
                result, _ = run_agent_search(
                    index,
                    question,
                    config=config,
                    processor=processor,
                    provider=provider,
                    output_top_k=10,
                )
                ranked_ids = [h.document.id for h in result.hits]
                ret_calls = result.trace.retrieval_calls
                llm_calls = result.trace.llm_calls
                query_shard_ids = result.shard_ids

            total_r5 += recall_at_k(ranked_ids, relevant_ids, 5)
            total_r10 += recall_at_k(ranked_ids, relevant_ids, 10)
            total_mrr += mrr_at_k(ranked_ids, relevant_ids, 10)
            total_ndcg += ndcg_at_k(ranked_ids, relevance, 10)
            total_shard_recall += shard_recall_at_n(query_shard_ids, relevant_ids, doc_to_shard)
            total_ret_calls += ret_calls
            total_llm_calls += llm_calls

        wall = time.perf_counter() - t_start
        results.append(
            ModeMetrics(
                mode=mode,
                recall_at_5=total_r5 / n,
                recall_at_10=total_r10 / n,
                mrr_at_10=total_mrr / n,
                ndcg_at_10=total_ndcg / n,
                shard_recall_at_n=total_shard_recall / n,
                mean_retrieval_calls=total_ret_calls / n,
                mean_llm_calls=total_llm_calls / n,
                wall_clock_seconds=wall,
            )
        )

    return results
