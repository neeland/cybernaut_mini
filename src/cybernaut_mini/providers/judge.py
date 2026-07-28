"""Result judging behind a protocol; the deterministic heuristic is the default.

The heuristic judge derives relevance from query-token coverage in the top hits,
coverage from unique question-token coverage across the top five, and redundancy
from mean pairwise Jaccard similarity of the top-five title token sets.
Reasons are short structured strings, never chain-of-thought.
"""

from __future__ import annotations

from itertools import combinations
from typing import Protocol

from cybernaut_mini.models import JudgeScore, SearchHit
from cybernaut_mini.text import TextProcessor


class Judge(Protocol):
    def score(
        self,
        question: str,
        query: str,
        hits: list[SearchHit],
        signals: dict[str, float] | None = None,
    ) -> JudgeScore: ...


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

        title_sets = [
            set(self._processor.content_tokens(hit.document.title)) for hit in top5
        ]
        pairs = list(combinations(range(len(title_sets)), 2))
        if pairs:
            jaccards = [
                len(title_sets[i] & title_sets[j]) / len(title_sets[i] | title_sets[j])
                if title_sets[i] | title_sets[j]
                else 0.0
                for i, j in pairs
            ]
            redundancy = sum(jaccards) / len(jaccards)
        else:
            redundancy = 0.0

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
