"""Tests for blog stage 8: map-reduce retrieval and the Aho-Corasick intent pass.

Everything here runs on HashEmbedder(dim=64) and TextProcessor(use_spacy=False):
no network, no spaCy, no torch.
"""

from __future__ import annotations

import doctest
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import pytest

from cybernaut_mini.config import RRFConfig
from cybernaut_mini.indexing import LoadedIndex
from cybernaut_mini.models import Document, IndexMeta, MetadataFilter, ShardManifest
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.query.s3_intents import SearchIntent, compute_idf, predict_intents
from cybernaut_mini.query.s8_retrieve import (
    DEFAULT_MAX_SHARDS,
    DEFAULT_MIN_SHARDS,
    DEFAULT_RESULT_LIMIT,
    BroadcastRequest,
    FusedCandidate,
    IntentMatcher,
    ShardResponse,
    broadcast,
    build_request,
    fold,
    group_by_document,
    intent_patterns,
    make_snippet,
    map_shard,
    reduce_responses,
    refine_with_intents,
    retrieve_map_reduce,
    select_shards,
)
from cybernaut_mini.query.s8_retrieve import intent_scan as intent_scan_module
from cybernaut_mini.query.s8_retrieve import pipeline as pipeline_module
from cybernaut_mini.text import TextProcessor

# --------------------------------------------------------------------------- #
# Fixtures: a two-shard index with known-correct answers.                      #
# --------------------------------------------------------------------------- #

#: doc-dup is deliberately listed by BOTH shards, so the reduce has to group it.
_DOCS: list[Document] = [
    Document(
        id="doc-phrase",
        title="Safety review of gene editing trials",
        text=(
            "Regulators published a safety review of gene editing trials this year. "
            "The gene editing trials covered somatic therapies only, and the review "
            "found that gene editing remains tightly controlled in clinical use."
        ),
        language="en",
        metadata={"category": "biotech"},
    ),
    Document(
        id="doc-scattered",
        title="Editing, genes, and safety in the laboratory",
        text=(
            "A gene is a unit of heredity. Editing of genomes is routine. Safety "
            "officers audit the laboratory. Editing, safety, and the gene registry "
            "are reviewed. Gene registries and editing logs are audited for safety."
        ),
        language="en",
        metadata={"category": "biotech"},
    ),
    Document(
        id="doc-mid-word",
        title="Polygene editingness and other coinages",
        text=(
            "Polygene editingness is not a real phrase, and neither is genesafety. "
            "This document exists to prove that a phrase must not match mid-word."
        ),
        language="en",
        metadata={"category": "linguistics"},
    ),
    Document(
        id="doc-dup",
        title="Shared document present in two shards",
        text=(
            "This document is listed by two shard manifests at once so that the "
            "reduce step has to group results by document identifier."
        ),
        language="en",
        metadata={"category": "meta"},
    ),
    Document(
        id="doc-yeast",
        title="Yeast fermentation improves flavour",
        text="Yeast fermentation improves flavour compounds in industrial brewing.",
        language="en",
        metadata={"category": "food"},
    ),
    Document(
        id="doc-finance",
        title="Bond yields fall on inflation data",
        text="Bond yields fall on inflation data signalling potential rate cuts.",
        language="fr",
        metadata={"category": "finance"},
    ),
    # Filler. BM25's IDF goes negative for a term carried by most of a shard, so a
    # four-document shard would score every query term at zero and quietly turn the
    # lexical leg off. These documents keep the query terms rare.
    *[
        Document(
            id=f"doc-filler-{i:02d}",
            title=f"Unrelated bulletin {i}",
            text=(
                f"Bulletin {i} covers shipping schedules, weather patterns, and "
                "quarterly logistics updates with no biology content whatsoever."
            ),
            language="en",
            metadata={"category": "filler"},
        )
        for i in range(1, 13)
    ],
]

_SHARD_DOCS: dict[int, list[str]] = {
    0: [
        "doc-phrase",
        "doc-scattered",
        "doc-mid-word",
        "doc-dup",
        *[f"doc-filler-{i:02d}" for i in range(1, 7)],
    ],
    1: [
        "doc-dup",
        "doc-yeast",
        "doc-finance",
        *[f"doc-filler-{i:02d}" for i in range(7, 13)],
    ],
}


def _manifest(shard_id: int, doc_ids: list[str], dim: int) -> ShardManifest:
    return ShardManifest(
        shard_id=shard_id,
        document_ids=doc_ids,
        centroid=[0.0] * dim,
        title=f"shard-{shard_id}",
        summary=f"summary for shard {shard_id}",
        keywords=[],
        entities=[],
        term_graph={},
        document_count=len(doc_ids),
        embedding_model=f"hash-{dim}",
    )


@pytest.fixture(scope="module")
def processor() -> TextProcessor:
    return TextProcessor(use_spacy=False)


@pytest.fixture(scope="module")
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=64)


@pytest.fixture(scope="module")
def rrf_config() -> RRFConfig:
    return RRFConfig()


@pytest.fixture(scope="module")
def small_index(processor: TextProcessor, embedder: HashEmbedder) -> LoadedIndex:
    """Two shards, six documents, one of them shared by both shards."""
    texts = [f"{doc.title}\n{doc.text}" for doc in _DOCS]
    vectors = embedder.embed_documents(texts)
    doc_tokens = {
        doc.id: processor.content_tokens(f"{doc.title}\n{doc.text}") for doc in _DOCS
    }
    meta = IndexMeta(
        embedding_model=f"hash-{embedder.dim}",
        embedding_dim=embedder.dim,
        n_shards=len(_SHARD_DOCS),
        n_documents=len(_DOCS),
        seed=0,
    )
    return LoadedIndex(
        meta=meta,
        documents=_DOCS,
        vectors=vectors,
        row_map={doc.id: i for i, doc in enumerate(_DOCS)},
        manifests={
            shard_id: _manifest(shard_id, doc_ids, embedder.dim)
            for shard_id, doc_ids in _SHARD_DOCS.items()
        },
        doc_tokens=doc_tokens,
    )


def _request(
    processor: TextProcessor,
    embedder: HashEmbedder,
    *,
    intents: tuple[str, ...] = (),
    **kwargs: object,
) -> BroadcastRequest:
    return build_request(
        "gene editing safety",
        processor=processor,
        provider=embedder,
        intents=intents,
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# intent_scan.py — the Aho-Corasick pass                                       #
# --------------------------------------------------------------------------- #


def test_multi_word_intent_matches_exact_phrase() -> None:
    matcher = IntentMatcher(["gene editing", "safety review"])
    scan = matcher.scan("d1", "The safety review covered gene editing trials.")
    assert scan.counts == {"safety review": 1, "gene editing": 1}
    assert scan.matched_intents == ("safety review", "gene editing")
    assert scan.coverage == 1.0
    assert scan.evidence > 0.0


def test_phrase_does_not_match_mid_word() -> None:
    matcher = IntentMatcher(["gene editing"])
    assert matcher.scan("d", "Polygene editing is different.").counts == {}
    assert matcher.scan("d", "gene editingness is different.").counts == {}
    assert matcher.scan("d", "polygene editingness").counts == {}
    # ... but a punctuation or whitespace neighbour is a boundary.
    assert matcher.scan("d", '"gene editing", they said.').counts == {"gene editing": 1}


def test_word_boundary_is_not_applied_to_scriptio_continua() -> None:
    """Japanese is one of the post's worked examples and writes without spaces."""
    matcher = IntentMatcher(["遺伝子編集"])
    scan = matcher.scan("jp", "より安全な遺伝子編集医薬品について。")
    assert scan.counts == {"遺伝子編集": 1}


def test_overlapping_and_nested_intents_are_all_reported() -> None:
    matcher = IntentMatcher(["gene editing", "editing medicine", "gene editing medicine"])
    scan = matcher.scan("d", "safer gene editing medicine")
    assert set(scan.counts) == {"gene editing", "editing medicine", "gene editing medicine"}
    # Earliest-first, then by end offset: the nesting order is stable.
    assert [m.intent for m in scan.matches] == [
        "gene editing",
        "gene editing medicine",
        "editing medicine",
    ]


def test_repeats_are_counted_but_damped() -> None:
    matcher = IntentMatcher(["gene editing"])
    once = matcher.scan("a", "gene editing").evidence
    thrice = matcher.scan("b", "gene editing. gene editing! gene editing?").evidence
    assert matcher.scan("b", "gene editing. gene editing! gene editing?").counts == {
        "gene editing": 3
    }
    assert once < thrice < 3 * once


def test_two_distinct_intents_beat_any_number_of_repeats() -> None:
    matcher = IntentMatcher(["gene editing", "safety review"])
    two = matcher.scan("two", "gene editing and the safety review").evidence
    repeated = matcher.scan("one", "gene editing " * 50).evidence
    assert two > repeated


def test_intent_patterns_deduplicate_and_order_deterministically() -> None:
    intents = [
        SearchIntent(text="Gene Editing", tokens=("gene", "editing"), score=0.4, span=(0, 2)),
        SearchIntent(text="gene editing", tokens=("gene", "editing"), score=0.9, span=(0, 2)),
        SearchIntent(text="safety review", tokens=("safety", "review"), score=0.6, span=(2, 4)),
    ]
    patterns = intent_patterns(intents)
    assert patterns == (("gene editing", 0.9), ("safety review", 0.6))
    assert intent_patterns(list(reversed(intents))) == patterns


def test_empty_intent_set_degrades_cleanly() -> None:
    matcher = IntentMatcher([])
    assert matcher.is_empty()
    assert matcher.patterns == ()
    scan = matcher.scan("d", "any text at all")
    assert not scan
    assert scan.evidence == 0.0
    assert scan.coverage == 0.0
    # Blank and whitespace-only intents are dropped rather than crashing the automaton.
    assert IntentMatcher(["", "   "]).is_empty()


def test_fold_preserves_length_for_offset_mapping() -> None:
    # The dotted capital I lowercases to two characters; fold must not do that,
    # because match offsets are reused to slice the same-length normalised text.
    assert len("İ".lower()) == 2
    assert len(fold("İ")) == 1
    for text in ("İstanbul gene editing", "GENE editing İ trials"):
        assert len(fold(text)) == len(text)
    # Case and full-width forms still fold together.
    assert fold("GENE Editing") == "gene editing"
    # Full-width GENE: the ambiguity RUF001 flags is exactly what NFKC removes.
    assert fold("ＧＥＮＥ") == "gene"  # noqa: RUF001


def test_offsets_survive_a_length_changing_uppercase(rrf_config: RRFConfig) -> None:
    """A phrase after an 'İ' is still located exactly, not one character out."""
    text = "İstanbul report: the gene editing trials continue."
    matcher = IntentMatcher(["gene editing"])
    match = matcher.find(text)[0]
    assert fold(text)[match.start : match.end] == "gene editing"
    assert text[match.start : match.end] == "gene editing"


def test_scan_is_deterministic() -> None:
    matcher = IntentMatcher(["gene editing", "safety review"])
    text = "gene editing and safety review and gene editing again"
    first = matcher.scan("d", text)
    second = matcher.scan("d", text)
    assert first == second


def test_intent_scan_module_doctests_pass() -> None:
    results = doctest.testmod(intent_scan_module)
    assert results.attempted > 0
    assert results.failed == 0


def test_stage3_intents_feed_the_matcher_directly() -> None:
    """Stage 3's output is the matcher's input, with no adaptation layer."""
    idf = compute_idf([["gene", "editing"], ["safer", "medicine"], ["yeast"]])
    intents = predict_intents("safer gene-editing medicines", idf)
    matcher = IntentMatcher(intents)
    assert matcher.patterns
    scan = matcher.scan("d", "A safer gene-editing medicine reached the clinic.")
    assert scan.matched_intents


# --------------------------------------------------------------------------- #
# snippets.py                                                                  #
# --------------------------------------------------------------------------- #

_LONG_TEXT = (
    "Introductory filler. " * 40
    + "The regulator confirmed that gene editing trials continue. "
    + "Trailing filler. " * 40
)


def test_snippet_is_centred_on_the_intent_match() -> None:
    matcher = IntentMatcher(["gene editing"])
    matches = matcher.find(_LONG_TEXT)
    snippet = make_snippet(_LONG_TEXT, intent_matches=matches, max_chars=200)
    assert "gene editing" in snippet
    assert len(snippet) <= 200


@pytest.mark.parametrize("max_chars", [1, 20, 80, 200, 500])
def test_snippet_respects_the_callers_maximum(max_chars: int) -> None:
    matcher = IntentMatcher(["gene editing"])
    matches = matcher.find(_LONG_TEXT)
    snippet = make_snippet(
        _LONG_TEXT,
        intent_matches=matches,
        query_tokens=["gene", "editing"],
        max_chars=max_chars,
    )
    assert len(snippet) <= max_chars


def test_snippet_falls_back_to_query_tokens_then_to_the_lead() -> None:
    on_token = make_snippet(_LONG_TEXT, query_tokens=["regulator"], max_chars=120)
    assert "regulator" in on_token.lower()
    lead = make_snippet(_LONG_TEXT, query_tokens=["nonexistent"], max_chars=120)
    assert lead.startswith("Introductory filler.")
    assert len(lead) <= 120


def test_snippet_does_not_split_words() -> None:
    snippet = make_snippet(_LONG_TEXT, query_tokens=["regulator"], max_chars=137)
    assert snippet == snippet.strip()
    assert snippet in " ".join(_LONG_TEXT.split())
    words = snippet.split()
    assert all(word in _LONG_TEXT for word in words)


def test_snippet_rejects_a_nonsense_maximum() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        make_snippet("text", max_chars=0)


def test_snippet_of_short_text_is_the_whole_text() -> None:
    assert make_snippet("Short body text.", max_chars=500) == "Short body text."
    assert make_snippet("", max_chars=500) == ""


# --------------------------------------------------------------------------- #
# mapreduce.py — shard selection, broadcast, grouping, fusion                  #
# --------------------------------------------------------------------------- #


def test_select_shards_defaults_to_the_posts_ten_to_thirty() -> None:
    assert (DEFAULT_MIN_SHARDS, DEFAULT_MAX_SHARDS) == (10, 30)
    assert DEFAULT_RESULT_LIMIT == 100
    routed = list(range(50))
    selected = select_shards(routed, available=set(range(100)))
    assert len(selected) == DEFAULT_MAX_SHARDS
    assert selected == tuple(range(30))  # routing order preserved


def test_select_shards_tops_up_to_the_minimum() -> None:
    selected = select_shards([7, 3], available=set(range(20)))
    assert selected[:2] == (7, 3)
    assert len(selected) == DEFAULT_MIN_SHARDS
    assert len(set(selected)) == DEFAULT_MIN_SHARDS


def test_select_shards_drops_unknown_and_duplicate_ids() -> None:
    assert select_shards([3, 3, 99, 1], available={1, 3}, min_shards=1) == (3, 1)


def test_select_shards_never_exceeds_a_small_index() -> None:
    assert select_shards([1, 0], available={0, 1}) == (1, 0)


def test_select_shards_validates_bounds() -> None:
    with pytest.raises(ValueError, match="min_shards"):
        select_shards([0], min_shards=0)
    with pytest.raises(ValueError, match="max_shards"):
        select_shards([0], min_shards=5, max_shards=2)


def test_group_by_document_merges_shards() -> None:
    responses = [
        ShardResponse(shard_id=1, lexical=(("a", 2.0), ("b", 1.0))),
        ShardResponse(shard_id=0, lexical=(("a", 3.0),), dense=(("c", 0.5),)),
    ]
    assert group_by_document(responses) == {"a": (0, 1), "b": (1,), "c": (0,)}


def test_broadcast_visits_shards_in_a_stable_order(
    small_index: LoadedIndex, processor: TextProcessor, embedder: HashEmbedder
) -> None:
    request = _request(processor, embedder)
    responses = broadcast(small_index, request)
    assert [r.shard_id for r in responses] == [0, 1]
    assert broadcast(small_index, request) == responses


def test_map_shard_applies_the_metadata_filter(
    small_index: LoadedIndex, processor: TextProcessor, embedder: HashEmbedder
) -> None:
    unfiltered = map_shard(small_index, 0, _request(processor, embedder))
    filtered = map_shard(
        small_index,
        0,
        _request(
            processor,
            embedder,
            metadata_filter=MetadataFilter(metadata_equals={"category": "biotech"}),
        ),
    )
    assert set(unfiltered.doc_ids) > set(filtered.doc_ids)
    assert set(filtered.doc_ids) == {"doc-phrase", "doc-scattered"}


def test_map_shard_honours_the_per_shard_limit(
    small_index: LoadedIndex, processor: TextProcessor, embedder: HashEmbedder
) -> None:
    response = map_shard(small_index, 0, _request(processor, embedder, per_shard_limit=2))
    assert len(response.lexical) <= 2
    assert len(response.dense) <= 2


def test_reduce_groups_a_duplicated_document_once(
    small_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    responses = broadcast(small_index, _request(processor, embedder))
    candidates = reduce_responses(responses, mode="hybrid", rrf_config=rrf_config)
    doc_ids = [candidate.doc_id for candidate in candidates]
    assert len(doc_ids) == len(set(doc_ids))
    dup = next(c for c in candidates if c.doc_id == "doc-dup")
    assert dup.shard_ids == (0, 1)


def test_global_fusion_avoids_the_per_shard_rank_collapse(
    small_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    """Per-shard fusion restarts ranks at 1, so every shard's best document ties."""
    responses = broadcast(small_index, _request(processor, embedder))
    per_shard = reduce_responses(
        responses, mode="hybrid", rrf_config=rrf_config, fusion="per_shard"
    )
    global_ = reduce_responses(responses, mode="hybrid", rrf_config=rrf_config)
    assert {c.doc_id for c in per_shard} == {c.doc_id for c in global_}
    per_shard_scores = [round(c.score, 12) for c in per_shard]
    global_scores = [round(c.score, 12) for c in global_]
    assert len(set(per_shard_scores)) < len(set(global_scores))
    # ... and the collapse has teeth: the best document of the shard that holds no
    # query term at all outranks a shard-0 document under per-shard fusion, purely
    # because its ranks restarted at 1. Global fusion puts them back in order.
    per_shard_order = [c.doc_id for c in per_shard]
    global_order = [c.doc_id for c in global_]
    assert per_shard_order.index("doc-filler-09") < per_shard_order.index("doc-mid-word")
    assert global_order.index("doc-mid-word") < global_order.index("doc-filler-09")


def test_reduce_in_lexical_mode_ranks_on_bm25(
    small_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    request = build_request(
        "gene editing safety", processor=processor, provider=embedder, mode="lexical"
    )
    candidates = reduce_responses(
        broadcast(small_index, request), mode="lexical", rrf_config=rrf_config
    )
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)
    assert all(c.lexical_score == c.score for c in candidates)
    assert all(c.dense_score is None for c in candidates)


def test_request_validates_its_package() -> None:
    with pytest.raises(ValueError, match="query_vector"):
        BroadcastRequest(query_tokens=("gene",), mode="hybrid")
    with pytest.raises(ValueError, match="query_tokens"):
        BroadcastRequest(query_vector=np.zeros(4, dtype=np.float32), mode="hybrid")
    with pytest.raises(ValueError, match="result_limit"):
        BroadcastRequest(query_tokens=("gene",), mode="lexical", result_limit=0)
    with pytest.raises(ValueError, match="per_shard_limit"):
        BroadcastRequest(query_tokens=("gene",), mode="lexical", per_shard_limit=0)


# --------------------------------------------------------------------------- #
# refine.py — what an intent match does to the ranking                         #
# --------------------------------------------------------------------------- #


def _candidates(doc_ids: list[str]) -> list[FusedCandidate]:
    return [
        FusedCandidate(doc_id=doc_id, shard_id=0, shard_ids=(0,), score=1.0 / (60 + rank))
        for rank, doc_id in enumerate(doc_ids, start=1)
    ]


def test_intent_match_promotes_but_cannot_leapfrog_the_head(rrf_config: RRFConfig) -> None:
    """A bounded rerank: rank 100 with the best intent still sits below rank 1."""
    candidates = _candidates([f"doc-{i:03d}" for i in range(1, 101)])
    scans = {
        "doc-100": intent_scan_module.IntentScan(
            doc_id="doc-100", matches=(), counts={"x": 1}, evidence=5.0, coverage=1.0
        )
    }
    refined = refine_with_intents(candidates, scans, rrf_config=rrf_config)
    order = [r.doc_id for r in refined]
    assert order[0] == "doc-001"
    assert order.index("doc-100") < 99  # it climbed
    assert set(order) == {c.doc_id for c in candidates}  # nothing dropped


def test_no_intent_matches_leaves_the_fusion_untouched(rrf_config: RRFConfig) -> None:
    candidates = _candidates(["a", "b", "c"])
    refined = refine_with_intents(candidates, {}, rrf_config=rrf_config)
    assert [r.doc_id for r in refined] == ["a", "b", "c"]
    assert [r.score for r in refined] == [c.score for c in candidates]


def test_zero_intent_weight_is_a_no_op(rrf_config: RRFConfig) -> None:
    candidates = _candidates(["a", "b", "c"])
    scans = {
        "c": intent_scan_module.IntentScan(
            doc_id="c", matches=(), counts={"x": 1}, evidence=9.0, coverage=1.0
        )
    }
    refined = refine_with_intents(candidates, scans, rrf_config=rrf_config, intent_weight=0.0)
    assert [r.doc_id for r in refined] == ["a", "b", "c"]
    assert [r.score for r in refined] == [c.score for c in candidates]


def test_refine_rejects_a_negative_weight(rrf_config: RRFConfig) -> None:
    with pytest.raises(ValueError, match="intent_weight"):
        refine_with_intents(_candidates(["a"]), {}, rrf_config=rrf_config, intent_weight=-1.0)


def test_equally_matched_documents_keep_their_fused_order(rrf_config: RRFConfig) -> None:
    """When every candidate matches equally, the intent list cannot reorder anything."""
    candidates = _candidates(["b", "a"])
    scans = {
        doc_id: intent_scan_module.IntentScan(
            doc_id=doc_id, matches=(), counts={"x": 1}, evidence=1.0, coverage=1.0
        )
        for doc_id in ("a", "b")
    }
    refined = refine_with_intents(candidates, scans, rrf_config=rrf_config)
    # The intent list breaks its evidence tie on the id, so "a" ranks first there
    # and "b" first in the fusion; the primary list wins, and the outcome is the
    # same on every run rather than depending on dict order.
    assert [r.doc_id for r in refined] == ["b", "a"]
    assert [r.intent_rank for r in refined] == [2, 1]
    assert refined == refine_with_intents(candidates, scans, rrf_config=rrf_config)


# --------------------------------------------------------------------------- #
# pipeline.py — the whole stage against known-correct answers                  #
# --------------------------------------------------------------------------- #


def test_intent_pass_promotes_the_exact_phrase_document(
    small_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    """doc-scattered has every query token; only doc-phrase has the intent phrase."""
    without = retrieve_map_reduce(
        small_index, _request(processor, embedder), rrf_config=rrf_config
    )
    with_intents = retrieve_map_reduce(
        small_index,
        _request(processor, embedder, intents=("gene editing",)),
        rrf_config=rrf_config,
    )
    plain_order = [hit.document.id for hit in without.hits]
    intent_order = [hit.document.id for hit in with_intents.hits]

    assert plain_order[0] != "doc-phrase", "fixture no longer exercises a promotion"
    assert intent_order[0] == "doc-phrase"
    # A rerank, not a filter: the same documents come back, in a different order.
    assert set(plain_order) == set(intent_order)
    assert with_intents.matched_doc_ids == ("doc-phrase",)


def test_mid_word_document_is_not_promoted(
    small_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    result = retrieve_map_reduce(
        small_index,
        _request(processor, embedder, intents=("gene editing",)),
        rrf_config=rrf_config,
    )
    assert "doc-mid-word" not in result.matched_doc_ids
    scan = result.intent_scans["doc-mid-word"]
    assert scan.counts == {}


def test_results_are_grouped_by_document_id(
    small_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    result = retrieve_map_reduce(
        small_index, _request(processor, embedder), rrf_config=rrf_config
    )
    ids = [hit.document.id for hit in result.hits]
    assert len(ids) == len(set(ids))
    assert "doc-dup" in ids
    assert [hit.rank for hit in result.hits] == list(range(1, len(ids) + 1))
    assert [hit.score for hit in result.hits] == sorted(
        (hit.score for hit in result.hits), reverse=True
    )


def test_result_limit_is_honoured(
    small_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    result = retrieve_map_reduce(
        small_index,
        _request(processor, embedder, intents=("gene editing",), result_limit=2),
        rrf_config=rrf_config,
    )
    assert len(result) == 2


def test_snippets_are_bounded_by_the_callers_maximum(
    small_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    for max_chars in (40, 120, 500):
        result = retrieve_map_reduce(
            small_index,
            _request(processor, embedder, intents=("gene editing",)),
            rrf_config=rrf_config,
            max_snippet_chars=max_chars,
        )
        assert result.hits
        assert all(len(hit.snippet) <= max_chars for hit in result.hits)
    with pytest.raises(ValueError, match="max_snippet_chars"):
        retrieve_map_reduce(
            small_index,
            _request(processor, embedder),
            rrf_config=rrf_config,
            max_snippet_chars=501,
        )


def test_snippet_of_a_matched_document_shows_the_intent(
    small_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    result = retrieve_map_reduce(
        small_index,
        _request(processor, embedder, intents=("gene editing",)),
        rrf_config=rrf_config,
        max_snippet_chars=120,
    )
    top = result.hits[0]
    assert top.document.id == "doc-phrase"
    assert "gene editing" in top.snippet.lower()


def test_pipeline_is_deterministic(
    small_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    def run() -> list[tuple[str, float, str]]:
        result = retrieve_map_reduce(
            small_index,
            _request(processor, embedder, intents=("gene editing", "safety review")),
            rrf_config=rrf_config,
        )
        return [(hit.document.id, hit.score, hit.snippet) for hit in result.hits]

    assert run() == run()


def test_disabling_the_intent_pass_reproduces_plain_fusion(
    small_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    baseline = retrieve_map_reduce(
        small_index, _request(processor, embedder), rrf_config=rrf_config
    )
    disabled = retrieve_map_reduce(
        small_index,
        _request(processor, embedder, intents=("gene editing",)),
        rrf_config=rrf_config,
        refine_top_n=0,
    )
    assert [h.document.id for h in disabled.hits] == [h.document.id for h in baseline.hits]
    assert [h.score for h in disabled.hits] == [h.score for h in baseline.hits]
    assert disabled.refined == 0


def test_metadata_filter_removes_documents_before_search(
    small_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    result = retrieve_map_reduce(
        small_index,
        _request(
            processor,
            embedder,
            intents=("gene editing",),
            metadata_filter=MetadataFilter(language=["fr"]),
        ),
        rrf_config=rrf_config,
    )
    assert [hit.document.id for hit in result.hits] == ["doc-finance"]


def test_lexical_mode_needs_no_embedding_provider(
    small_index: LoadedIndex, processor: TextProcessor, rrf_config: RRFConfig
) -> None:
    request = build_request("gene editing safety", processor=processor, mode="lexical")
    assert request.query_vector is None
    result = retrieve_map_reduce(small_index, request, rrf_config=rrf_config)
    assert result.hits
    assert all(hit.dense_score is None for hit in result.hits)


def test_the_intent_pass_reads_only_the_window(
    small_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    """The full text of the whole corpus must never be materialised."""
    request = _request(
        processor, embedder, intents=("gene editing",), result_limit=1, per_shard_limit=100
    )
    result = retrieve_map_reduce(small_index, request, rrf_config=rrf_config, refine_top_n=2)
    assert result.refined == 2
    assert len(result.intent_scans) == 2
    assert len(result) == 1
    assert result.candidates <= 3  # window + result_limit


class _CountingDocuments(Mapping[str, Document]):
    """A ``by_id`` mapping that records every document actually read."""

    def __init__(self, inner: Mapping[str, Document]) -> None:
        self._inner = inner
        self.requested: list[str] = []

    def __getitem__(self, doc_id: str) -> Document:
        self.requested.append(doc_id)
        return self._inner[doc_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._inner)

    def __len__(self) -> int:
        return len(self._inner)


def test_only_the_window_and_the_results_are_read_from_disk(
    built_index_path: Path,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    """The 1M-document target forbids materialising the corpus to run the pass."""
    index = LoadedIndex.load(built_index_path)
    try:
        counting = _CountingDocuments(index.by_id)
        index.by_id = counting  # type: ignore[assignment]
        request = build_request(
            "gene editing safety",
            processor=processor,
            provider=embedder,
            intents=("gene editing",),
            result_limit=3,
        )
        result = retrieve_map_reduce(
            index, request, rrf_config=rrf_config, refine_top_n=6
        )
        assert len(result) == 3
        distinct = set(counting.requested)
        assert len(distinct) <= 6  # the window; the three hits are inside it
        assert len(distinct) < index.meta.n_documents
    finally:
        index.close()


def test_pipeline_module_doctests_pass() -> None:
    results = doctest.testmod(pipeline_module)
    assert results.failed == 0


def test_negative_refine_window_is_rejected(
    small_index: LoadedIndex,
    processor: TextProcessor,
    embedder: HashEmbedder,
    rrf_config: RRFConfig,
) -> None:
    with pytest.raises(ValueError, match="refine_top_n"):
        retrieve_map_reduce(
            small_index, _request(processor, embedder), rrf_config=rrf_config, refine_top_n=-1
        )
