"""Result judging behind a protocol; heuristic by default, cross-encoder opt-in.

``HeuristicJudge`` derives relevance from query-token coverage in the top hits,
coverage from unique question-token coverage across the top five, and redundancy
from mean pairwise Jaccard similarity of the top-five title token sets. It is
deterministic, offline, and free — the default so a bare install keeps working.

``CrossEncoderJudge`` is the model-backed judge (``agent.judge: cross_encoder``):
a Hugging Face cross-encoder scores every (question, document) pair, so relevance
is judged against the *user's question* rather than the current query rewrite —
the reward stays anchored to what the searcher actually asked. Redundancy comes
from mean pairwise cosine of document embeddings when an embedding provider is
supplied (semantic near-duplicates, not just shared title tokens). Runs on MPS
on Apple silicon via ``accel.resolve_device``; needs the 'st' extra.

Reasons are short structured strings, never chain-of-thought.

Blog ref: https://nosible.com/blog/introducing-cybernaut-1-agentic-search-with-mcts —
    the agent scores retrieved evidence to decide whether a branch improved. The
    post discloses no judge and no reward function; the build guide's §2 lists the
    reward/value function as an explicit gap. Local copy:
    ``docs/blog-archive/introducing-cybernaut-1-agentic-search-with-mcts.md``.

Assumptions:
    - ``HeuristicJudge`` is the default because a bare install must run with no
      downloads, no key and zero model calls. It is a stand-in for the
      undisclosed reward, not a claim about how NOSIBLE scores results:
      relevance is query-token coverage across the top five documents, coverage
      is unique question-token coverage over all five, and redundancy is mean
      pairwise Jaccard similarity of title token sets.
    - ``CrossEncoderJudge`` scores ``(question, title + body)`` pairs, not
      ``(current query, document)`` pairs. The reward stays anchored to what the
      user asked even after the agent has rewritten the query several times.
    - ``coverage`` is the maximum pair score while ``relevance`` is the mean, so
      ``coverage >= relevance`` always holds and the two terms mean "at least one
      hit answers this" and "the set is good on average" respectively.
    - Model scores are sigmoid-remapped only when the raw output falls outside
      ``[0, 1]``. A reranker that already applies sigmoid must not have it applied
      twice, and a model returning raw logits must not be read as a probability.
    - Semantic redundancy uses the session's embedding provider when one is given
      and falls back to title Jaccard otherwise, so the judge degrades instead of
      requiring a second model. Cosine is clamped to ``[0, 1]``: anti-correlated
      documents are "not redundant", not negatively redundant.
    - ``calls`` counts model invocations so the agent's trace reports honest
      ``llm_calls``; the heuristic has no ``calls`` attribute and reports none.

Alternatives considered:
    - LLM-as-judge through a hosted API: the build guide's first suggestion and
      the most faithful reading of "LLM-guided". Rejected because it makes the
      default install non-functional offline, spends a network round-trip per
      node inside an 18-call budget, and puts the reward behind a key this repo
      does not require.
    - Scoring every hit instead of the top five: more faithful to the whole result
      set, but the reward's job is to rank *branches*, and hits below rank five
      are usually the same evidence for every branch that routed to the same
      shards. Scoring them multiplies judge cost for little discrimination.
    - Returning a bare float rather than ``JudgeScore``: simpler, but the trace
      would lose the reason string and the component breakdown that make a reward
      auditable, and ``policy.compute_reward`` consumes those components.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from itertools import combinations
from typing import TYPE_CHECKING, Protocol

import numpy as np

from cybernaut_mini.config import AgentConfig, ConfigError
from cybernaut_mini.models import JudgeScore, SearchHit
from cybernaut_mini.text import TextProcessor

if TYPE_CHECKING:
    from cybernaut_mini.providers.embeddings import EmbeddingProvider

#: Characters of body text paired with the title for each judged document. The
#: cross-encoder truncates to its own max length anyway; this just bounds the
#: tokenizer work per hit.
_JUDGE_DOC_CHARS = 1000


class Judge(Protocol):
    def score(
        self,
        question: str,
        query: str,
        hits: list[SearchHit],
        signals: dict[str, float] | None = None,
    ) -> JudgeScore: ...


def title_redundancy(processor: TextProcessor, hits: list[SearchHit]) -> float:
    """Mean pairwise Jaccard similarity of title token sets — the lexical fallback."""
    title_sets = [set(processor.content_tokens(hit.document.title)) for hit in hits]
    pairs = list(combinations(range(len(title_sets)), 2))
    if not pairs:
        return 0.0
    jaccards = [
        len(title_sets[i] & title_sets[j]) / len(title_sets[i] | title_sets[j])
        if title_sets[i] | title_sets[j]
        else 0.0
        for i, j in pairs
    ]
    return sum(jaccards) / len(jaccards)


class HeuristicJudge:
    def __init__(self, processor: TextProcessor) -> None:
        self._processor = processor

    def score(
        self,
        question: str,
        query: str,
        hits: list[SearchHit],
        signals: dict[str, float] | None = None,
    ) -> JudgeScore:
        if not hits:
            return JudgeScore(relevance=0.0, coverage=0.0, redundancy=0.0, reason="no results")

        top5 = hits[:5]
        query_tokens = set(self._processor.content_tokens(query))
        question_tokens = set(self._processor.content_tokens(question))
        doc_token_sets = [
            set(self._processor.content_tokens(f"{hit.document.title} {hit.document.text}"))
            for hit in top5
        ]

        if query_tokens:
            per_hit = [
                len(query_tokens & doc_tokens) / len(query_tokens)
                for doc_tokens in doc_token_sets
            ]
            relevance = sum(per_hit) / len(per_hit)
        else:
            relevance = 0.0

        if question_tokens:
            union: set[str] = set().union(*doc_token_sets)
            coverage = len(question_tokens & union) / len(question_tokens)
        else:
            coverage = 0.0

        redundancy = title_redundancy(self._processor, top5)

        reason = (
            f"heuristic: rel={relevance:.2f} cov={coverage:.2f} red={redundancy:.2f} "
            f"over {len(top5)} hits"
        )
        return JudgeScore(
            relevance=min(1.0, relevance),
            coverage=min(1.0, coverage),
            redundancy=min(1.0, redundancy),
            reason=reason[:240],
        )


class CrossEncoderJudge:
    """Cross-encoder relevance judging on MPS/CUDA/CPU.

    Per top-5 hit the model scores the pair ``(question, title + body)``; scores
    are sigmoid-calibrated into [0, 1] (sentence-transformers applies sigmoid for
    single-label rerankers; the defensive re-map only triggers on raw logits).

    - relevance = mean pair score — how good the result *set* is.
    - coverage = max pair score — whether at least one hit really answers it.
      ``coverage >= relevance`` always holds, which keeps the reward's coverage
      term meaning "the answer is in there" rather than token overlap.
    - redundancy = mean pairwise cosine of document embeddings (clamped to
      [0, 1]) when an embedding provider is given, else title Jaccard.

    ``calls`` counts model invocations so the agent can report honest
    ``llm_calls`` in its trace.
    """

    def __init__(
        self,
        processor: TextProcessor,
        *,
        model_name: str,
        revision: str | None = None,
        device: str = "auto",
        embedder: EmbeddingProvider | None = None,
        predict_fn: Callable[[list[tuple[str, str]]], np.ndarray] | None = None,
    ) -> None:
        self._processor = processor
        self._embedder = embedder
        self._model_label = model_name.rsplit("/", 1)[-1]
        self.calls = 0
        if predict_fn is not None:
            self._predict = predict_fn
            self.device = "injected"
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            msg = (
                "agent judge 'cross_encoder' needs the optional 'st' extra "
                "(sentence-transformers + torch), which the default install does "
                "not include.\n"
                "  install it   : uv sync --extra st --extra mps   (Mac; drop "
                "--extra mps elsewhere)\n"
                "  or stay local: agent.judge: heuristic"
            )
            raise ConfigError(msg) from exc
        from cybernaut_mini.accel import resolve_device

        self.device = resolve_device(device)  # type: ignore[arg-type]
        model = CrossEncoder(
            model_name,
            revision=revision,
            device=self.device,
            token=os.environ.get("HF_TOKEN") or None,
        )

        def predict(pairs: list[tuple[str, str]]) -> np.ndarray:
            return np.asarray(
                model.predict(pairs, convert_to_numpy=True, show_progress_bar=False),
                dtype=np.float32,
            )

        self._predict = predict

    def score(
        self,
        question: str,
        query: str,
        hits: list[SearchHit],
        signals: dict[str, float] | None = None,
    ) -> JudgeScore:
        if not hits:
            return JudgeScore(relevance=0.0, coverage=0.0, redundancy=0.0, reason="no results")

        top5 = hits[:5]
        pairs = [
            (question, f"{hit.document.title}. {hit.document.text[:_JUDGE_DOC_CHARS]}")
            for hit in top5
        ]
        scores = np.asarray(self._predict(pairs), dtype=np.float32).reshape(-1)
        self.calls += 1
        if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
            # Raw logits (identity activation): calibrate the same way the
            # model's own single-label default would.
            scores = 1.0 / (1.0 + np.exp(-scores))

        relevance = float(scores.mean())
        coverage = float(scores.max())
        redundancy = self._redundancy(top5)

        reason = (
            f"cross-encoder({self._model_label}): rel={relevance:.2f} "
            f"cov={coverage:.2f} red={redundancy:.2f} over {len(top5)} hits"
        )
        return JudgeScore(
            relevance=min(1.0, max(0.0, relevance)),
            coverage=min(1.0, max(0.0, coverage)),
            redundancy=min(1.0, max(0.0, redundancy)),
            reason=reason[:240],
        )

    def _redundancy(self, hits: list[SearchHit]) -> float:
        if self._embedder is None or len(hits) < 2:
            return title_redundancy(self._processor, hits)
        texts = [
            f"{hit.document.title} {hit.document.text[:_JUDGE_DOC_CHARS]}" for hit in hits
        ]
        vectors = self._embedder.embed_documents(texts)  # already L2-normalized
        sims = vectors @ vectors.T
        upper = sims[np.triu_indices(len(hits), k=1)]
        # Cosine lands in [-1, 1]; anti-correlated docs are "not redundant", not
        # negatively redundant, so clamp rather than rescale.
        return float(np.clip(upper, 0.0, 1.0).mean())


def create_judge(
    config: AgentConfig,
    processor: TextProcessor,
    *,
    embedder: EmbeddingProvider | None = None,
) -> Judge:
    """Build the judge the config asks for; the heuristic needs no downloads."""
    if config.judge == "heuristic":
        return HeuristicJudge(processor)
    return CrossEncoderJudge(
        processor,
        model_name=config.judge_model_name,
        revision=config.judge_revision,
        device=config.device,
        embedder=embedder,
    )
