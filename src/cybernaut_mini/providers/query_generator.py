"""Query generation behind a protocol; the deterministic heuristic is the default.

The heuristic generator proposes up to five variant shapes (original, expanded,
missing-keyword, keyword-only, entity-preserving), deduplicated on lexical form.
"""

from __future__ import annotations

from typing import Protocol

from cybernaut_mini.agent.state import StateSummary
from cybernaut_mini.models import QueryCandidate
from cybernaut_mini.text import TextProcessor, lexical_form


class QueryGenerator(Protocol):
    def generate(
        self, question: str, state_summary: StateSummary, n: int
    ) -> list[QueryCandidate]: ...


class HeuristicQueryGenerator:
    def __init__(self, processor: TextProcessor) -> None:
        self._processor = processor

    def generate(
        self, question: str, state_summary: StateSummary, n: int
    ) -> list[QueryCandidate]:
        query = state_summary.current_query
        candidates: list[QueryCandidate] = [
            QueryCandidate(text=query, origin="original")
        ]

        if state_summary.expansions:
            candidates.append(
                QueryCandidate(
                    text=query,
                    origin="expanded",
                    expansions=list(state_summary.expansions),
                )
            )

        if state_summary.missing_keywords:
            keyword = state_summary.missing_keywords[0]
            candidates.append(
                QueryCandidate(text=f"{query} {keyword}", origin="keyword")
            )

        content = self._processor.content_tokens(query)
        if content:
            candidates.append(
                QueryCandidate(text=" ".join(content), origin="keyword_only")
            )

        if state_summary.entities:
            entity_text = " ".join(state_summary.entities)
            remaining = [tok for tok in content if tok not in set(entity_text.split())]
            candidates.append(
                QueryCandidate(
                    text=" ".join([entity_text, *remaining]).strip(),
                    origin="entity_preserving",
                )
            )

        deduped: list[QueryCandidate] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for candidate in candidates:
            key = (lexical_form(candidate.text), tuple(candidate.expansions))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped[:n]
