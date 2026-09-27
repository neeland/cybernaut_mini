"""Rendering the per-hit signal bundle for LLM prompts and judge inputs.

Every :class:`~cybernaut_mini.models.SearchHit` already carries its full signal
bundle — BM25 score and rank, dense score and rank, the fused RRF score, and the
per-ranker RRF contributions. Until now those numbers stopped at the trace: the
query generator and the judge saw only document text. This module renders the
bundle two ways:

* :func:`render_signal_table` — a compact fixed-width table (3-5 lines by
  default) for inclusion in an LLM prompt, so a model-backed generator or judge
  conditions on the retriever's own evidence rather than re-deriving relevance
  from prose;
* :func:`signal_summary` — a flat ``dict[str, float]`` aggregate matching the
  ``signals`` parameter the :class:`~cybernaut_mini.providers.judge.Judge`
  protocol already accepts.

Blog ref: https://nosible.com/blog/introducing-cybernaut-1-agentic-search-with-mcts
    — the agent's LLM components are "high-trust": they are shown the ranking
    signals (BM25, dense similarity, reranker scores, fused RRF) for each result,
    not just the text, when judging progress and proposing the next action. Local
    copy: ``docs/blog-archive/introducing-cybernaut-1-agentic-search-with-mcts.md``.

Assumptions:
    - [inferred] the post does not print its table format, so this one is chosen
      for token economy: one header line, one line per hit, ``-`` for a signal a
      ranker did not produce. Scores are rendered at 3 decimals — enough to
      distinguish neighbours, cheap in tokens, and matching the repo-wide float
      rounding in canonical artifacts.
    - The "rerank" column shows the summed RRF contribution of every ranker other
      than ``lexical``/``dense`` (today: the stage-8 intent scan). Per-document
      neural rerank scores would land in the same column when a document-level
      reranker is wired in; shard-level rerankers (stage 6) act before documents
      exist and cannot appear per hit.
    - Determinism: pure string formatting over the hits given, no model, no clock.

Alternatives rejected:
    - JSON blobs in the prompt: unambiguous but 3-4x the tokens of a table, and
      small instruct models follow tables at least as well.
    - Rendering inside ``build_generator_prompt`` directly: the judge needs the
      same rendering, and two hand-rolled formats would drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cybernaut_mini.models import SearchHit

__all__ = ["render_signal_table", "signal_summary"]

#: Default number of hits rendered — the post's "3-5 line" compact table.
DEFAULT_TABLE_ROWS = 5

_HEADER = "rank | doc | bm25 | dense | rerank | rrf"


def _fmt(value: float | None, rank: int | None = None) -> str:
    """``0.812@3`` for a scored+ranked signal, ``0.812`` scored only, ``-`` absent."""
    if value is None:
        return "-"
    if rank is None:
        return f"{value:.3f}"
    return f"{value:.3f}@{rank}"


def _rerank_contribution(hit: SearchHit) -> float | None:
    """Summed RRF contribution of every non-lexical, non-dense ranker, or None."""
    extras = [
        contribution
        for name, contribution in hit.rrf_contributions.items()
        if name not in ("lexical", "dense")
    ]
    if not extras:
        return None
    return sum(extras)


def render_signal_table(hits: Sequence[SearchHit], limit: int = DEFAULT_TABLE_ROWS) -> str:
    """Render up to *limit* hits as a compact signal table, or ``""`` for no hits."""
    rows = [_HEADER]
    for hit in hits[:limit]:
        rows.append(
            " | ".join(
                (
                    str(hit.rank),
                    hit.document.id,
                    _fmt(hit.bm25_score, hit.bm25_rank),
                    _fmt(hit.dense_score, hit.dense_rank),
                    _fmt(_rerank_contribution(hit)),
                    _fmt(hit.score),
                )
            )
        )
    if len(rows) == 1:
        return ""
    return "\n".join(rows)


def signal_summary(hits: Sequence[SearchHit], limit: int = DEFAULT_TABLE_ROWS) -> dict[str, float]:
    """Aggregate signal stats over the top hits, for ``Judge.score(signals=...)``.

    Keys are always present (0.0 when the ranker produced nothing) so a judge can
    rely on the shape: ``bm25_max``, ``bm25_mean``, ``dense_max``, ``dense_mean``,
    ``rrf_max``, ``rrf_mean``, ``n_hits``.
    """
    top = hits[:limit]
    bm25 = [h.bm25_score for h in top if h.bm25_score is not None]
    dense = [h.dense_score for h in top if h.dense_score is not None]
    rrf = [h.score for h in top]

    def _stats(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        return max(values), sum(values) / len(values)

    bm25_max, bm25_mean = _stats(bm25)
    dense_max, dense_mean = _stats(dense)
    rrf_max, rrf_mean = _stats(rrf)
    return {
        "bm25_max": bm25_max,
        "bm25_mean": bm25_mean,
        "dense_max": dense_max,
        "dense_mean": dense_mean,
        "rrf_max": rrf_max,
        "rrf_mean": rrf_mean,
        "n_hits": float(len(top)),
    }
