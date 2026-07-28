from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cybernaut_mini.models import (
    Document,
    MetadataFilter,
    canonical_dumps,
)


def make_doc(**overrides: object) -> Document:
    payload: dict[str, object] = {
        "id": "doc-001",
        "title": "A title",
        "text": "A body",
        "language": "en",
        "published_at": "2025-01-12T00:00:00Z",
        "metadata": {"category": "biotech"},
    }
    payload.update(overrides)
    return Document.model_validate(payload)


def test_document_requires_non_empty_text() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        make_doc(text="   ")


def test_document_preserves_unknown_metadata_keys() -> None:
    doc = make_doc(metadata={"category": "biotech", "custom_key": [1, 2]})
    assert doc.metadata["custom_key"] == [1, 2]


def test_metadata_filter_rejects_unknown_operators() -> None:
    with pytest.raises(ValidationError):
        MetadataFilter.model_validate({"language_not": ["en"]})


def test_metadata_filter_combines_with_and() -> None:
    doc = make_doc()
    matching = MetadataFilter(
        language=["en"],
        published_from=datetime(2024, 1, 1, tzinfo=UTC),
        published_to=datetime(2026, 1, 1, tzinfo=UTC),
        metadata_equals={"category": "biotech"},
    )
    assert matching.matches(doc)
    assert not MetadataFilter(language=["de"]).matches(doc)
    assert not MetadataFilter(metadata_equals={"category": "energy"}).matches(doc)
    assert not MetadataFilter(published_from=datetime(2025, 6, 1, tzinfo=UTC)).matches(doc)


def test_metadata_filter_date_bounds_require_published_at() -> None:
    doc = make_doc(published_at=None)
    assert not MetadataFilter(published_from=datetime(2024, 1, 1, tzinfo=UTC)).matches(doc)
    assert MetadataFilter(language=["en"]).matches(doc)


def test_canonical_dumps_is_sorted_compact_and_rounded() -> None:
    payload = {"b": 0.123456789123, "a": [1.0000000001, -0.0]}
    assert canonical_dumps(payload) == '{"a":[1.0,0.0],"b":0.12345679}'


def test_canonical_dumps_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_dumps({"x": float("inf")})


def test_canonical_dumps_is_stable_across_key_order() -> None:
    assert canonical_dumps({"a": 1, "b": 2}) == canonical_dumps({"b": 2, "a": 1})
