from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cybernaut_mini.models import (
    ARTIFACT_VERSION,
    Document,
    MetadataFilter,
    ShardManifest,
    canonical_dumps,
)
from cybernaut_mini.shard_artifacts import PhraseBloom, Vocabulary, build_shard_artifacts


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


# ------------------------------------------------------------------ #
# ShardManifest artifact fields                                       #
# ------------------------------------------------------------------ #


def make_manifest(**overrides: object) -> ShardManifest:
    payload: dict[str, object] = {
        "shard_id": 0,
        "document_ids": ["doc-001"],
        "centroid": [0.5, 0.5],
        "title": "t",
        "summary": "s",
        "keywords": [],
        "entities": [],
        "term_graph": {},
        "document_count": 1,
        "embedding_model": "hash-2",
    }
    payload.update(overrides)
    return ShardManifest.model_validate(payload)


def test_artifact_version_is_two() -> None:
    """The bump is load-bearing: v1 indexes have no shard artifacts on disk."""
    assert ARTIFACT_VERSION == "2"
    assert make_manifest().artifact_version == "2"


def test_manifest_artifact_fields_default_to_none() -> None:
    manifest = make_manifest()
    assert manifest.phrase_bloom is None
    assert manifest.zstd_dictionary is None
    assert manifest.vocabulary is None
    assert manifest.has_artifacts() is False


def test_manifest_carries_real_shard_artifacts_and_round_trips() -> None:
    """The three fields hold exactly what shard_artifacts produces, through JSON."""
    texts = [f"shard text number {i} about gene editing and genomes" for i in range(12)]
    artifacts = build_shard_artifacts(
        phrases=["gene editing", "genomes"],
        texts=texts,
        token_streams=[text.split() for text in texts],
    )
    payload = artifacts.to_payload()
    manifest = make_manifest(
        phrase_bloom=payload["phrase_bloom"],
        zstd_dictionary=payload["zstd_dictionary"],
        vocabulary=payload["vocabulary"],
    )
    assert manifest.has_artifacts() is True

    revived = ShardManifest.model_validate_json(canonical_dumps(manifest.model_dump(mode="json")))
    assert revived.phrase_bloom == payload["phrase_bloom"]
    assert revived.zstd_dictionary == payload["zstd_dictionary"]
    assert revived.vocabulary == payload["vocabulary"]

    # A field that round-trips as bytes but not as a working filter would be worse
    # than useless, so rebuild the real objects and query them.
    assert revived.phrase_bloom is not None
    assert revived.vocabulary is not None
    assert "gene editing" in PhraseBloom.from_payload(revived.phrase_bloom)
    assert "sasquatch" not in PhraseBloom.from_payload(revived.phrase_bloom)
    assert "genomes" in Vocabulary.from_payload(revived.vocabulary)


def test_manifest_canonical_json_is_byte_identical_for_equal_artifacts() -> None:
    """Two manifests built from the same shard content serialise identically."""
    texts = [f"document {i} about solar panels and grid storage" for i in range(12)]
    kwargs = {
        "phrases": ["solar panels", "grid storage"],
        "texts": texts,
        "token_streams": [text.split() for text in texts],
    }
    first = build_shard_artifacts(**kwargs).to_payload()  # type: ignore[arg-type]
    second = build_shard_artifacts(**kwargs).to_payload()  # type: ignore[arg-type]

    def dumped(payload: dict[str, object]) -> str:
        manifest = make_manifest(
            phrase_bloom=payload["phrase_bloom"],
            zstd_dictionary=payload["zstd_dictionary"],
            vocabulary=payload["vocabulary"],
        )
        return canonical_dumps(manifest.model_dump(mode="json"))

    assert dumped(first) == dumped(second)


def test_without_artifacts_clears_only_the_three_fields() -> None:
    manifest = make_manifest(
        phrase_bloom={"bits": "AA=="},
        zstd_dictionary="AA==",
        vocabulary={"terms": ["a"], "unique_terms": 1, "truncated": False},
        summary="kept",
    )
    stripped = manifest.without_artifacts()
    assert (stripped.phrase_bloom, stripped.zstd_dictionary, stripped.vocabulary) == (
        None,
        None,
        None,
    )
    assert stripped.summary == "kept"
    assert stripped.document_ids == manifest.document_ids
    # The original is untouched — model_copy, not mutation.
    assert manifest.zstd_dictionary == "AA=="


def test_manifest_still_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        make_manifest(bloom_filter={"bits": "AA=="})
