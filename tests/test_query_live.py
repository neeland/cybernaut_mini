"""Tests for the live query path: stages 1, 2 and 4 wired into retrieve()/route().

The headline test is the end-to-end regression the integration exists for:
the post's own Japanese worked-example question — which the latin-only
``[a-z0-9]+`` tokenizer used to delete entirely — now yields non-empty query
tokens and non-empty retrieval results. The rest pins each seam individually:
ASCII queries keep index-compatible tokens, the E5-instruct wire format is
composed by stage 4 (and only for instruct checkpoints), and the corpus-global
IDF sidecar written at build time replaces uniform IDF in stage-3 intents.

Every test runs offline. No API key, no model download, no network.
"""

# The Japanese assertions quote the archived post's worked example verbatim;
# RUF001 has nothing to catch here.
# ruff: noqa: RUF001

from __future__ import annotations

from typing import TYPE_CHECKING

from cybernaut_mini.config import AppConfig, EmbeddingConfig
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.query.live import (
    embedding_query_text,
    is_e5_instruct,
    live_query_tokens,
    prepare,
)
from cybernaut_mini.retrieval import retrieve
from cybernaut_mini.routing import query_intents, route
from cybernaut_mini.text import TextProcessor

if TYPE_CHECKING:
    from pathlib import Path

    from cybernaut_mini.indexing import LoadedIndex

#: The post's Japanese worked example, verbatim from
#: docs/blog-archive/the-road-to-cybernaut-1.md ("What lessons from bacteria and
#: yeast actually translate into safer gene-editing medicines?").
JAPANESE_QUESTION = (
    "細菌と酵母から得られる教訓は、より安全な遺伝子編集医薬品に"
    "どのように応用されるのでしょうか？"
)


def app_config() -> AppConfig:
    return AppConfig(embedding=EmbeddingConfig(provider="hash", dim=64))


# ----------------------------- stage 1 -------------------------------- #


def test_prepare_is_the_identity_path_without_a_translator() -> None:
    prepared = prepare("what is gene editing")
    assert prepared.text_for_retrieval == "what is gene editing"


# ----------------------------- stage 2 -------------------------------- #


def test_ascii_query_keeps_index_compatible_tokens(text_processor: TextProcessor) -> None:
    """English tokens must stay exactly what the BM25 index was built from."""
    text = "gene editing medicine"
    tokens = live_query_tokens(text, language="en", processor=text_processor)
    assert tokens == text_processor.content_tokens(text)


def test_japanese_query_tokens_are_not_deleted(text_processor: TextProcessor) -> None:
    """The latin-only regex returns nothing here; stage 2 must take over."""
    assert text_processor.content_tokens(JAPANESE_QUESTION) == []
    tokens = live_query_tokens(JAPANESE_QUESTION, language="ja", processor=text_processor)
    assert tokens
    # The post's own worked-example tokens for this question.
    assert "遺伝子" in tokens
    assert "編集" in tokens


def test_undetermined_language_still_tokenizes_by_script(
    text_processor: TextProcessor,
) -> None:
    """Stage 1 without fastText reports ``und``; stage 2 guesses from the script."""
    tokens = live_query_tokens(JAPANESE_QUESTION, language="und", processor=text_processor)
    assert tokens


# ----------------------------- stage 4 -------------------------------- #


def test_is_e5_instruct_predicate() -> None:
    assert is_e5_instruct("intfloat/multilingual-e5-large-instruct")
    assert not is_e5_instruct("intfloat/multilingual-e5-small")
    assert not is_e5_instruct("hash-64")


def test_embedding_query_text_is_passthrough_for_non_instruct_providers() -> None:
    assert embedding_query_text("hash-64", "what is gene editing") == "what is gene editing"


def test_embedding_query_text_composes_the_instruct_wire_format() -> None:
    wire = embedding_query_text(
        "intfloat/multilingual-e5-large-instruct", "what is gene editing", language="en"
    )
    assert wire.startswith("Instruct: ")
    assert "\nQuery: what is gene editing" in wire


def test_unknown_language_falls_back_to_the_selector_default() -> None:
    wire = embedding_query_text(
        "intfloat/multilingual-e5-large-instruct", "what is gene editing", language="und"
    )
    assert wire.startswith("Instruct: ")
    assert "\nQuery: what is gene editing" in wire


# ----------------------- end-to-end regression ------------------------ #


def test_japanese_query_yields_tokens_and_results_end_to_end(
    built_index: LoadedIndex,
) -> None:
    """The regression that motivated the wiring: a Japanese question used to
    tokenize to nothing and lexically retrieve nothing. It must now produce
    non-empty tokens and non-empty hybrid results."""
    processor = TextProcessor(use_spacy=False)
    tokens = live_query_tokens(JAPANESE_QUESTION, language="ja", processor=processor)
    assert tokens

    hits = retrieve(
        built_index,
        JAPANESE_QUESTION,
        mode="hybrid",
        processor=processor,
        provider=HashEmbedder(dim=64),
        rrf_config=app_config().rrf,
        top_k=5,
    )
    assert hits
    assert all(hit.document.id for hit in hits)


def test_japanese_query_routes_to_shards(built_index: LoadedIndex) -> None:
    shard_ids, signals = route(
        built_index,
        JAPANESE_QUESTION,
        processor=TextProcessor(use_spacy=False),
        provider=HashEmbedder(dim=64),
        rrf_config=app_config().rrf,
        top_n=3,
    )
    assert shard_ids
    assert signals.fused


# ------------------------- corpus-global IDF -------------------------- #


def test_build_writes_the_global_idf_sidecar(
    built_index_path: Path, built_index: LoadedIndex
) -> None:
    assert (built_index_path / "global_idf.json").exists()
    table = built_index.global_idf
    assert table
    # The planted single-document token must be rarer than a token shared by
    # several documents.
    assert table["zylophristine"] > table["immune"]


def test_query_intents_rank_by_corpus_rarity_with_the_sidecar(
    built_index: LoadedIndex,
) -> None:
    question = "zylophristine compound assay immune response"
    uniform = query_intents(question)
    weighted = query_intents(question, idf_lookup=built_index.global_idf)

    # Uniform IDF cannot rank: every intent scores identically.
    assert len({intent.score for intent in uniform}) == 1
    # The corpus table can: rare-term intents outscore common-term intents.
    assert len({intent.score for intent in weighted}) > 1
    by_score = sorted(weighted, key=lambda intent: intent.score, reverse=True)
    assert "zylophristine" in " ".join(intent.text for intent in by_score[:3])


def test_retrieve_derives_intents_from_the_sidecar(built_index: LoadedIndex) -> None:
    """retrieve() without explicit intents must not crash on the sidecar path and
    must still return the planted lexical target."""
    hits = retrieve(
        built_index,
        "zylophristine compound assay",
        mode="lexical",
        processor=TextProcessor(use_spacy=False),
        provider=HashEmbedder(dim=64),
        rrf_config=app_config().rrf,
        top_k=3,
    )
    assert hits
    assert hits[0].document.id == "doc-lex"
