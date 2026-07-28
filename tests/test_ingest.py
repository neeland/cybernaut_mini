"""Tests for cybernaut_mini.ingest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cybernaut_mini.ingest import IngestError, load_documents
from cybernaut_mini.models import Document


def _write_jsonl(path: Path, records: list) -> None:
    lines = "".join(json.dumps(r) + "\n" for r in records)
    path.write_text(lines, encoding="utf-8")


def _doc_dict(idx: int, **overrides) -> dict:
    base = {
        "id": f"doc-{idx:03d}",
        "title": f"Title {idx}",
        "text": f"Body text for document {idx}.",
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------ #
# Happy path                                                          #
# ------------------------------------------------------------------ #


def test_load_valid_file(tmp_path: Path) -> None:
    p = tmp_path / "docs.jsonl"
    _write_jsonl(p, [_doc_dict(1), _doc_dict(2), _doc_dict(3)])
    docs = load_documents(p)
    assert len(docs) == 3
    assert isinstance(docs[0], Document)
    assert docs[0].id == "doc-001"


def test_load_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "docs.jsonl"
    p.write_text(
        json.dumps(_doc_dict(1)) + "\n\n  \n" + json.dumps(_doc_dict(2)) + "\n",
        encoding="utf-8",
    )
    docs = load_documents(p)
    assert len(docs) == 2


def test_load_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert load_documents(p) == []


# ------------------------------------------------------------------ #
# Duplicate id                                                        #
# ------------------------------------------------------------------ #


def test_duplicate_id_raises(tmp_path: Path) -> None:
    p = tmp_path / "docs.jsonl"
    _write_jsonl(p, [_doc_dict(1), _doc_dict(1)])  # same id twice
    with pytest.raises(IngestError, match="doc-001"):
        load_documents(p)


def test_duplicate_id_message_names_record(tmp_path: Path) -> None:
    p = tmp_path / "docs.jsonl"
    _write_jsonl(p, [_doc_dict(5), _doc_dict(5)])
    with pytest.raises(IngestError) as exc_info:
        load_documents(p)
    assert "doc-005" in str(exc_info.value)


# ------------------------------------------------------------------ #
# Empty required field                                                #
# ------------------------------------------------------------------ #


def test_empty_text_raises(tmp_path: Path) -> None:
    p = tmp_path / "docs.jsonl"
    _write_jsonl(p, [_doc_dict(1, text="   ")])
    with pytest.raises(IngestError) as exc_info:
        load_documents(p)
    # Should name the record
    assert "doc-001" in str(exc_info.value) or "line 1" in str(exc_info.value)


def test_empty_title_raises(tmp_path: Path) -> None:
    p = tmp_path / "docs.jsonl"
    _write_jsonl(p, [_doc_dict(1, title="")])
    with pytest.raises(IngestError):
        load_documents(p)


def test_missing_required_field_raises(tmp_path: Path) -> None:
    p = tmp_path / "docs.jsonl"
    # Missing 'text' field
    p.write_text(json.dumps({"id": "doc-001", "title": "A title"}) + "\n", encoding="utf-8")
    with pytest.raises(IngestError):
        load_documents(p)


# ------------------------------------------------------------------ #
# Invalid JSON                                                        #
# ------------------------------------------------------------------ #


def test_invalid_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "docs.jsonl"
    p.write_text(json.dumps(_doc_dict(1)) + "\n{bad json\n", encoding="utf-8")
    with pytest.raises(IngestError) as exc_info:
        load_documents(p)
    # Should name the line number
    assert "line 2" in str(exc_info.value)


def test_invalid_json_first_line_raises(tmp_path: Path) -> None:
    p = tmp_path / "docs.jsonl"
    p.write_text("{not valid}\n", encoding="utf-8")
    with pytest.raises(IngestError) as exc_info:
        load_documents(p)
    assert "line 1" in str(exc_info.value)
