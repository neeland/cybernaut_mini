"""Blog stage 8 end to end: broadcast, search, fuse, intent pass, snippet, ~100 hits.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 8, "Retrieval
    Using Map-Reduce": the package of embeddings, expansions, intents and 10-30
    shard ids is broadcast; each shard filters, searches lexically, searches
    semantically, fuses, and runs a full-text search over the top results for
    intents; results are combined, grouped by document id, snippetted to a
    user-chosen maximum length, and about 100 come back [disclosed]. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - This module is composition only. Steps 1-4 live in
      :mod:`cybernaut_mini.retrieval` (via :mod:`.mapreduce`), step 5 in
      :mod:`.intent_scan` and :mod:`.refine`, and the snippet in :mod:`.snippets`;
      keeping the wiring separate is what lets each of them be tested against known
      answers rather than against the whole pipeline.
    - The intent pass is applied to a *window* of the fused ranking, defaulting to
      twice the result limit [inferred]. The post says "top results" without a
      number; twice the returned count is the smallest window in which a promotion
      can still change which documents make the cut, and it bounds the number of
      documents whose full text has to be read to ``2 * result_limit`` — the reason
      the pass is affordable at all at a million documents.
    - ``max_snippet_chars`` is the post's user-decided maximum, capped at 500
      because :class:`~cybernaut_mini.models.SearchHit` validates the field there.
      Asking for more raises rather than silently truncating.

Alternatives rejected:
    - Returning bare :class:`~cybernaut_mini.models.SearchHit` objects. The intent
      evidence is the interesting part of this stage and ``SearchHit`` forbids extra
      fields, so the result object carries the scans alongside the hits instead of
      losing them.
    - Re-embedding or re-tokenising the question here. Stages 2-4 own that; stage 8
      receives their output in the broadcast package, exactly as the post describes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cybernaut_mini.models import MetadataFilter, SearchHit
from cybernaut_mini.query.s3_intents import SearchIntent
from cybernaut_mini.query.s8_retrieve.intent_scan import IntentMatcher, IntentScan
from cybernaut_mini.query.s8_retrieve.mapreduce import (
    DEFAULT_PER_SHARD_LIMIT,
    DEFAULT_RESULT_LIMIT,
    BroadcastRequest,
    FusedCandidate,
    FusionScope,
    RetrievalMode,
    ShardResponse,
    broadcast,
    reduce_responses,
)
from cybernaut_mini.query.s8_retrieve.refine import (
    DEFAULT_INTENT_WEIGHT,
    RefinedCandidate,
    refine_with_intents,
)
from cybernaut_mini.query.s8_retrieve.snippets import (
    DEFAULT_SNIPPET_MAX_CHARS,
    SNIPPET_HARD_MAX_CHARS,
    make_snippet,
)

if TYPE_CHECKING:
    from cybernaut_mini.config import RRFConfig
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.providers.embeddings import EmbeddingProvider
    from cybernaut_mini.text import TextProcessor

__all__ = [
    "MapReduceResult",
    "build_request",
    "retrieve_map_reduce",
]


@dataclass(frozen=True)
class MapReduceResult:
    """The stage's output: the hits plus enough evidence to explain them."""

    hits: list[SearchHit]
    shard_ids: tuple[int, ...] = ()
    candidates: int = 0
    refined: int = 0
    intent_scans: dict[str, IntentScan] = field(default_factory=dict)
    responses: list[ShardResponse] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.hits)

    @property
    def matched_doc_ids(self) -> tuple[str, ...]:
        """Returned documents that contained at least one search intent, in rank order."""
        return tuple(
            hit.document.id
            for hit in self.hits
            if self.intent_scans.get(hit.document.id) is not None
            and bool(self.intent_scans[hit.document.id])
        )


def build_request(
    question: str,
    *,
    processor: TextProcessor,
    provider: EmbeddingProvider | None = None,
    intents: Sequence[SearchIntent | str] = (),
    expansions: Sequence[str] = (),
    shard_ids: Sequence[int] = (),
    metadata_filter: MetadataFilter | None = None,
    mode: RetrievalMode = "hybrid",
    result_limit: int = DEFAULT_RESULT_LIMIT,
    per_shard_limit: int = DEFAULT_PER_SHARD_LIMIT,
) -> BroadcastRequest:
    """Assemble the broadcast package from a question the earlier stages processed.

    A convenience for callers that do not already hold stage 2-4 output: it
    tokenises with ``processor`` and embeds with ``provider`` (omit it for
    ``mode="lexical"``). Bare strings in ``intents`` are accepted and weighted 1.0.
    """
    query_tokens = processor.content_tokens(question) or processor.tokenize(question)
    query_vector = None if mode == "lexical" or provider is None else provider.embed_queries(
        [question]
    )[0]
    resolved = tuple(
        intent
        if isinstance(intent, SearchIntent)
        else SearchIntent(text=intent, tokens=tuple(intent.split()), score=1.0, span=(0, 0))
        for intent in intents
    )
    return BroadcastRequest(
        query_tokens=tuple(query_tokens),
        expansion_tokens=tuple(expansions),
        query_vector=query_vector,
        intents=resolved,
        shard_ids=tuple(shard_ids),
        metadata_filter=metadata_filter,
        mode=mode,
        result_limit=result_limit,
        per_shard_limit=per_shard_limit,
        query_variant=question,
    )


def _scan_window(
    index: LoadedIndex,
    matcher: IntentMatcher,
    window: Sequence[FusedCandidate],
) -> dict[str, IntentScan]:
    """Run the Aho-Corasick pass over the window's full text, one document at a time.

    Documents are pulled through :attr:`LoadedIndex.by_id`, which reads from the
    JSONL store behind an LRU — the window, not the corpus, is what ends up
    resident. Title and body are scanned together because an intent that appears
    only in the headline is still an intent the document satisfies.
    """
    scans: dict[str, IntentScan] = {}
    for candidate in window:
        document = index.by_id[candidate.doc_id]
        scans[candidate.doc_id] = matcher.scan(
            candidate.doc_id, f"{document.title}\n{document.text}"
        )
    return scans


def _to_hits(
    index: LoadedIndex,
    refined: Sequence[RefinedCandidate],
    request: BroadcastRequest,
    *,
    max_snippet_chars: int,
) -> list[SearchHit]:
    """Turn the refined ranking into ``SearchHit`` objects with snippets."""
    hits: list[SearchHit] = []
    anchors = list(request.all_query_tokens)
    for rank, refined_candidate in enumerate(refined, start=1):
        document = index.by_id[refined_candidate.doc_id]
        scan = refined_candidate.scan
        snippet = make_snippet(
            document.text,
            intent_matches=scan.matches if scan is not None else (),
            query_tokens=anchors,
            max_chars=max_snippet_chars,
        )
        candidate = refined_candidate.candidate
        hits.append(
            SearchHit(
                document=document,
                rank=rank,
                score=refined_candidate.score,
                shard_id=candidate.shard_id,
                bm25_score=candidate.lexical_score,
                bm25_rank=candidate.lexical_rank,
                dense_score=candidate.dense_score,
                dense_rank=candidate.dense_rank,
                rrf_contributions=refined_candidate.contributions,
                query_variant=request.query_variant,
                expansion_terms=list(request.expansion_tokens),
                snippet=snippet,
            )
        )
    return hits


def retrieve_map_reduce(
    index: LoadedIndex,
    request: BroadcastRequest,
    *,
    rrf_config: RRFConfig,
    fusion: FusionScope = "global",
    intent_weight: float = DEFAULT_INTENT_WEIGHT,
    refine_top_n: int | None = None,
    max_snippet_chars: int = DEFAULT_SNIPPET_MAX_CHARS,
) -> MapReduceResult:
    """Run blog stage 8 over ``index`` and return at most ``request.result_limit`` hits.

    ``refine_top_n`` defaults to ``2 * request.result_limit``: the number of top
    results whose full text is read and scanned for intents. Passing ``0`` disables
    the intent pass entirely, which reproduces the pre-stage-8 behaviour exactly.

    Raises :class:`ValueError` when ``max_snippet_chars`` is outside 1..500 or
    ``refine_top_n`` is negative.
    """
    if not 1 <= max_snippet_chars <= SNIPPET_HARD_MAX_CHARS:
        msg = (
            f"max_snippet_chars must be in 1..{SNIPPET_HARD_MAX_CHARS} "
            f"(SearchHit.snippet's limit), got {max_snippet_chars}"
        )
        raise ValueError(msg)
    if refine_top_n is not None and refine_top_n < 0:
        msg = f"refine_top_n must be >= 0, got {refine_top_n}"
        raise ValueError(msg)

    window_size = 2 * request.result_limit if refine_top_n is None else refine_top_n

    responses = broadcast(index, request)
    # Candidates below both the window and the result limit can neither be promoted
    # by an intent nor survive to the output, so they are dropped before the reduce
    # allocates records for them.
    candidates = reduce_responses(
        responses,
        mode=request.mode,
        rrf_config=rrf_config,
        fusion=fusion,
        limit=window_size + request.result_limit,
    )

    matcher = IntentMatcher(request.intents)
    scans: dict[str, IntentScan] = {}
    if not matcher.is_empty() and window_size > 0:
        scans = _scan_window(index, matcher, candidates[:window_size])

    refined = refine_with_intents(
        candidates,
        scans,
        rrf_config=rrf_config,
        intent_weight=intent_weight,
    )[: request.result_limit]

    return MapReduceResult(
        hits=_to_hits(index, refined, request, max_snippet_chars=max_snippet_chars),
        shard_ids=tuple(response.shard_id for response in responses),
        candidates=len(candidates),
        refined=len(scans),
        intent_scans=scans,
        responses=responses,
    )
