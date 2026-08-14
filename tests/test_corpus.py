"""Tests for corpus normalisation and the corpus_ingest nodes."""

from __future__ import annotations

from typing import Any

import pytest

from cybernaut_mini.corpus import (
    CorpusNormalizeError,
    CorpusSourceConfig,
    derive_title,
    load_miracl_judgments,
    make_document_id,
    normalize_rows,
    validate_judgments_against_corpus,
)
from cybernaut_mini.models import Judgment
from cybernaut_mini.pipelines.corpus_ingest.nodes import (
    build_miracl_judgments,
    normalize_corpus,
    select_documents,
    snapshot_raw_corpus,
    validate_judgments,
)

# Mirrors the NOSIBLE Hugging Face schema: text / label / netloc / url.
_HF_CONFIG = CorpusSourceConfig(
    field_map={"text": "text", "url": "url"},
    metadata_fields=["label", "netloc"],
    default_language="en",
    id_prefix="pred",
    min_text_chars=32,
)


def _row(index: int, text: str | None = None, **overrides: Any) -> dict[str, Any]:
    row = {
        "text": text
        if text is not None
        else f"Analysts expect conditions to improve materially during the coming quarter {index}.",
        "label": "prediction",
        "netloc": "example.com",
        "url": f"https://example.com/article-{index}",
    }
    row.update(overrides)
    return row


# ------------------------------------------------------------------ #
# Title derivation                                                    #
# ------------------------------------------------------------------ #


def test_title_is_the_first_sentence() -> None:
    assert derive_title("First sentence here. Second one follows.") == "First sentence here."


def test_long_title_is_cut_on_a_word_boundary() -> None:
    text = " ".join(["word"] * 60) + "."
    title = derive_title(text)
    assert len(title) <= 120
    assert not title.endswith("wor")  # never mid-word


def test_single_overlong_token_is_still_truncated() -> None:
    title = derive_title("x" * 400)
    assert len(title) == 120


def test_title_falls_back_to_whole_text_without_punctuation() -> None:
    assert derive_title("no terminal punctuation here") == "no terminal punctuation here"


# ------------------------------------------------------------------ #
# Id stability                                                        #
# ------------------------------------------------------------------ #


def test_document_id_is_stable_across_runs() -> None:
    first = make_document_id("pred", "https://example.com/a")
    second = make_document_id("pred", "https://example.com/a")
    assert first == second
    assert first.startswith("pred-")


def test_different_keys_yield_different_ids() -> None:
    assert make_document_id("pred", "a") != make_document_id("pred", "b")


def test_ids_are_independent_of_row_order() -> None:
    rows = [_row(i) for i in range(5)]
    forward = [d.id for d in normalize_rows(rows, _HF_CONFIG)]
    backward = [d.id for d in normalize_rows(list(reversed(rows)), _HF_CONFIG)]
    assert forward == backward


# ------------------------------------------------------------------ #
# Normalisation                                                       #
# ------------------------------------------------------------------ #


def test_hf_schema_maps_onto_document() -> None:
    documents = normalize_rows([_row(1)], _HF_CONFIG)
    assert len(documents) == 1
    doc = documents[0]
    assert doc.url == "https://example.com/article-1"
    assert doc.language == "en"
    assert doc.metadata == {"label": "prediction", "netloc": "example.com"}
    assert doc.title
    assert doc.text


def test_short_rows_are_skipped_not_raised() -> None:
    """A 100k-row public corpus contains junk; one bad row must not fail a build."""
    rows = [_row(1), _row(2, text="too short")]
    documents = normalize_rows(rows, _HF_CONFIG)
    assert len(documents) == 1


def test_duplicate_urls_collapse() -> None:
    rows = [_row(1), _row(1)]
    assert len(normalize_rows(rows, _HF_CONFIG)) == 1


def test_rows_without_url_still_get_stable_distinct_ids() -> None:
    rows = [_row(1, url=""), _row(2, url="")]
    documents = normalize_rows(rows, _HF_CONFIG)
    assert len({d.id for d in documents}) == 2


def test_empty_source_returns_empty() -> None:
    assert normalize_rows([], _HF_CONFIG) == []


def test_missing_text_column_is_a_hard_error() -> None:
    with pytest.raises(CorpusNormalizeError, match="absent"):
        normalize_rows([{"body": "x" * 100}], _HF_CONFIG)


def test_field_map_without_text_target_is_rejected() -> None:
    config = CorpusSourceConfig(field_map={"url": "url"})
    with pytest.raises(CorpusNormalizeError, match="exactly one"):
        normalize_rows([{"url": "https://example.com"}], config)


def test_whitespace_is_collapsed() -> None:
    rows = [_row(1, text="Lots   of \n\n irregular \t whitespace in this passage body here.")]
    doc = normalize_rows(rows, _HF_CONFIG)[0]
    assert "  " not in doc.text
    assert "\n" not in doc.text


# ------------------------------------------------------------------ #
# Nodes                                                               #
# ------------------------------------------------------------------ #


def test_snapshot_refuses_an_empty_source() -> None:
    with pytest.raises(ValueError, match="zero rows"):
        snapshot_raw_corpus([])


def test_snapshot_passes_rows_through_unchanged() -> None:
    rows = [_row(1)]
    assert snapshot_raw_corpus(rows) == rows


def test_normalize_node_reports_total_wipeout() -> None:
    """Every row filtered out is a config error, not an empty result."""
    params = _HF_CONFIG.model_dump()
    params["min_text_chars"] = 10_000
    with pytest.raises(ValueError, match="zero documents"):
        normalize_corpus([_row(1)], params)


def test_select_caps_deterministically() -> None:
    normalized = normalize_rows([_row(i) for i in range(10)], _HF_CONFIG)
    documents = [d.model_dump(mode="json") for d in normalized]
    first = select_documents(documents, {"max_documents": 4})
    second = select_documents(list(reversed(documents)), {"max_documents": 4})
    assert len(first) == 4
    assert [d["id"] for d in first] == [d["id"] for d in second]


def test_select_filters_by_language() -> None:
    normalized = normalize_rows([_row(i) for i in range(3)], _HF_CONFIG)
    documents = [d.model_dump(mode="json") for d in normalized]
    documents[0]["language"] = "de"
    kept = select_documents(documents, {"languages": ["en"]})
    assert len(kept) == 2


def test_select_filters_by_metadata() -> None:
    rows = [_row(0), _row(1, label="not-prediction")]
    documents = [d.model_dump(mode="json") for d in normalize_rows(rows, _HF_CONFIG)]
    kept = select_documents(documents, {"metadata_equals": {"label": "prediction"}})
    assert len(kept) == 1
    assert kept[0]["metadata"]["label"] == "prediction"


def test_select_rejects_an_empty_result() -> None:
    documents = [d.model_dump(mode="json") for d in normalize_rows([_row(1)], _HF_CONFIG)]
    with pytest.raises(ValueError, match="zero documents"):
        select_documents(documents, {"languages": ["fr"]})


def test_normalized_documents_survive_a_real_index_build() -> None:
    """Normalised output must satisfy the same validation the build applies."""
    from cybernaut_mini.pipelines.index_build.nodes import ingest_documents

    documents = [
        d.model_dump(mode="json") for d in normalize_rows([_row(i) for i in range(6)], _HF_CONFIG)
    ]
    assert len(ingest_documents(documents)) == 6


# ------------------------------------------------------------------ #
# MIRACL → Judgment loader                                            #
# ------------------------------------------------------------------ #


def _miracl_row(
    query_id: str,
    question: str,
    positives: list[str],
    negatives: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal structured MIRACL row (as load_dataset would return)."""
    return {
        "query_id": query_id,
        "query": question,
        "positive_passages": [{"docid": d, "title": "", "text": ""} for d in positives],
        "negative_passages": [{"docid": d, "title": "", "text": ""} for d in (negatives or [])],
    }


def test_load_miracl_judgments_maps_query_id_and_question() -> None:
    rows = [_miracl_row("42", "What is the speed of light?", ["12#0"])]
    judgments = load_miracl_judgments(rows)
    assert len(judgments) == 1
    assert judgments[0].query_id == "42"
    assert judgments[0].question == "What is the speed of light?"


def test_load_miracl_judgments_prefixes_docids_with_mir() -> None:
    rows = [_miracl_row("1", "Q?", positives=["12#0"], negatives=["99#1"])]
    judgments = load_miracl_judgments(rows)
    doc_ids = set(judgments[0].relevant_document_ids.keys())
    expected_pos = make_document_id("mir", "12#0")
    expected_neg = make_document_id("mir", "99#1")
    assert expected_pos in doc_ids
    assert expected_neg in doc_ids
    # All ids start with the mir prefix
    assert all(k.startswith("mir-") for k in doc_ids)


def test_load_miracl_judgments_preserves_grades() -> None:
    rows = [_miracl_row("1", "Q?", positives=["12#0"], negatives=["99#1"])]
    judgments = load_miracl_judgments(rows)
    pos_id = make_document_id("mir", "12#0")
    neg_id = make_document_id("mir", "99#1")
    assert judgments[0].relevant_document_ids[pos_id] == 1
    assert judgments[0].relevant_document_ids[neg_id] == 0


def test_load_miracl_judgments_skips_rows_with_no_assessments() -> None:
    rows = [
        _miracl_row("1", "has passages", positives=["12#0"]),
        _miracl_row("2", "no passages", positives=[], negatives=[]),
    ]
    judgments = load_miracl_judgments(rows)
    assert len(judgments) == 1
    assert judgments[0].query_id == "1"


def test_load_miracl_judgments_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="zero rows"):
        load_miracl_judgments([])


def test_load_miracl_judgments_returns_valid_judgment_models() -> None:
    rows = [_miracl_row("7", "Q?", positives=["5#0", "5#1"])]
    judgments = load_miracl_judgments(rows)
    assert all(isinstance(j, Judgment) for j in judgments)


def test_validate_judgments_against_corpus_passes_when_all_present() -> None:
    rows = [_miracl_row("1", "Q?", positives=["12#0"])]
    judgments = load_miracl_judgments(rows)
    doc_id = make_document_id("mir", "12#0")
    # No exception should be raised when the corpus contains the id.
    validate_judgments_against_corpus(judgments, corpus_doc_ids={doc_id})


def test_validate_judgments_against_corpus_raises_on_missing_docid() -> None:
    rows = [_miracl_row("1", "Q?", positives=["12#0"])]
    judgments = load_miracl_judgments(rows)
    with pytest.raises(ValueError, match="absent from corpus"):
        validate_judgments_against_corpus(judgments, corpus_doc_ids=set())


def test_validate_judgments_against_corpus_lists_missing_ids_in_error() -> None:
    rows = [_miracl_row("1", "Q?", positives=["12#0", "13#0"])]
    judgments = load_miracl_judgments(rows)
    with pytest.raises(ValueError, match="mir-"):
        validate_judgments_against_corpus(judgments, corpus_doc_ids=set())


# ------------------------------------------------------------------ #
# Kedro node wrappers                                                 #
# ------------------------------------------------------------------ #


def test_build_miracl_judgments_node_returns_serialisable_dicts() -> None:
    rows = [_miracl_row("1", "Q?", positives=["12#0"])]
    result = build_miracl_judgments(rows)
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert result[0]["query_id"] == "1"
    assert isinstance(result[0]["relevant_document_ids"], dict)


def test_validate_judgments_node_passes_through_when_corpus_matches() -> None:
    rows = [_miracl_row("1", "Q?", positives=["12#0"])]
    judgment_dicts = build_miracl_judgments(rows)
    doc_id = make_document_id("mir", "12#0")
    documents = [{"id": doc_id, "title": "T", "text": "Body text here.", "language": "en"}]
    result = validate_judgments(judgment_dicts, documents)
    # Node is a pass-through on success
    assert result == judgment_dicts


def test_validate_judgments_node_raises_on_missing_doc() -> None:
    rows = [_miracl_row("1", "Q?", positives=["missing#0"])]
    judgment_dicts = build_miracl_judgments(rows)
    with pytest.raises(ValueError, match="absent from corpus"):
        validate_judgments(judgment_dicts, documents=[])
