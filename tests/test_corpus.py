"""Tests for corpus normalisation and the corpus_ingest nodes."""

from __future__ import annotations

from typing import Any

import pytest

from cybernaut_mini.corpus import (
    CorpusNormalizeError,
    CorpusSourceConfig,
    derive_title,
    make_document_id,
    normalize_rows,
)
from cybernaut_mini.pipelines.corpus_ingest.nodes import (
    normalize_corpus,
    select_documents,
    snapshot_raw_corpus,
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
