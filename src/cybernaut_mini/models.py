"""Pydantic models for all persisted records, plus the canonical JSON writer.

Every artifact on disk is produced through :func:`canonical_dumps` so that two
builds with the same seed and provider are byte-for-byte identical.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — the post lists what a
    shard actually holds: "Bloom Filters: used to quickly check if phrases exist",
    "Each shard has a trained Zstandard text compression dictionary", and
    "Vocabulary: the set of unique words in this shard" (shard 11,343 is quoted at a
    vocabulary of 64,992 terms). :class:`ShardManifest` had none of those three, so
    they are added here as first-class manifest fields. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``

Assumptions: the post does not say how the three artifacts are serialised, so the
    two binary ones travel base64 inside JSON — the bloom bit array as the payload
    produced by :meth:`cybernaut_mini.shard_artifacts.PhraseBloom.to_payload`, the
    zstd dictionary as a bare base64 string. All three fields default to ``None``
    because a shard too small to train a dictionary on is legitimate (the tail of a
    k-means split) and because a manifest held in memory usually does *not* carry
    them: :class:`~cybernaut_mini.indexing.LoadedIndex` deliberately leaves them
    unpopulated and fetches them per shard on demand, since 1,000 resident
    vocabularies of 65,536 terms is gigabytes of heap.

Alternatives rejected: a single ``artifacts`` sub-model holding all three. Rejected
    because the three have genuinely different lifetimes — the bloom is ~1.2 KB and
    cheap to keep, the vocabulary is the one that cannot be held 1,000 times over —
    and one field would force an all-or-nothing load. Also rejected: validating
    ``artifact_version`` inside the models. A model that refuses to parse an old
    record cannot produce the error message that tells the user which version they
    have; that check belongs at the index boundary, in
    :meth:`~cybernaut_mini.indexing.LoadedIndex.load`.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Bumped to "2" when shards gained their bloom filter, zstd dictionary and
#: vocabulary, and the index layout gained per-shard artifact sidecars and
#: JSONL offset sidecars. A "1" index has none of those files, so it is rejected
#: at load rather than read as a shard set with three permanently-absent
#: artifacts — see :meth:`cybernaut_mini.indexing.LoadedIndex.load`.
ARTIFACT_VERSION = "2"

FLOAT_DECIMALS = 8


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = f"non-finite float in artifact: {value!r}"
            raise ValueError(msg)
        rounded = round(value, FLOAT_DECIMALS)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def canonical_dumps(obj: Any) -> str:
    """Serialize to deterministic JSON: sorted keys, compact, floats at 8 decimals."""
    return json.dumps(_round_floats(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def model_canonical_dumps(model: BaseModel) -> str:
    return canonical_dumps(model.model_dump(mode="json"))


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str
    text: str
    url: str | None = None
    language: str | None = None
    published_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title", "text")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            msg = "must be non-empty"
            raise ValueError(msg)
        return value


class MetadataFilter(BaseModel):
    """Typed metadata filter; fields combine with logical AND.

    ``extra="forbid"`` makes any unsupported operator a validation error.
    """

    model_config = ConfigDict(extra="forbid")

    language: list[str] | None = None
    published_from: datetime | None = None
    published_to: datetime | None = None
    metadata_equals: dict[str, Any] | None = None

    def is_empty(self) -> bool:
        return (
            self.language is None
            and self.published_from is None
            and self.published_to is None
            and self.metadata_equals is None
        )

    def matches(self, document: Document) -> bool:
        if self.language is not None and document.language not in self.language:
            return False
        if self.published_from is not None and (
            document.published_at is None or document.published_at < self.published_from
        ):
            return False
        if self.published_to is not None and (
            document.published_at is None or document.published_at >= self.published_to
        ):
            return False
        if self.metadata_equals is not None:
            for key, expected in self.metadata_equals.items():
                if document.metadata.get(key) != expected:
                    return False
        return True


class ShardKeyword(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: str
    weight: float


class ShardEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    count: int


class ShardManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shard_id: int
    document_ids: list[str]
    centroid: list[float]
    title: str
    summary: str
    keywords: list[ShardKeyword]
    entities: list[ShardEntity]
    term_graph: dict[str, dict[str, float]]
    document_count: int
    artifact_version: str = ARTIFACT_VERSION
    embedding_model: str

    #: Bloom filter over this shard's important phrases, as produced by
    #: :meth:`cybernaut_mini.shard_artifacts.PhraseBloom.to_payload` (the bit array
    #: is base64 inside it). ``None`` means "not loaded / not built", never "empty":
    #: an empty shard still gets a real, always-false filter.
    phrase_bloom: dict[str, Any] | None = None
    #: Base64 of this shard's trained Zstandard dictionary. ``None`` when zstd
    #: refused to train — a shard of a handful of short documents does not contain
    #: enough repeated material — in which case
    #: :func:`cybernaut_mini.shard_artifacts.compression_score` scores it 0.0.
    zstd_dictionary: str | None = None
    #: This shard's unique terms, as produced by
    #: :meth:`cybernaut_mini.shard_artifacts.Vocabulary.to_payload`. The single
    #: largest field a manifest can carry, which is why it is loaded per shard on
    #: demand rather than with the manifest.
    vocabulary: dict[str, Any] | None = None

    def without_artifacts(self) -> ShardManifest:
        """A copy with the three artifact fields cleared.

        This is the form the manifest takes on disk and in memory: the payloads live
        in a per-shard sidecar so that loading 1,000 manifests stays cheap.
        """
        return self.model_copy(
            update={"phrase_bloom": None, "zstd_dictionary": None, "vocabulary": None}
        )

    def has_artifacts(self) -> bool:
        """True when all three artifact fields are populated on this instance."""
        return self.phrase_bloom is not None and self.vocabulary is not None

    @field_validator("document_ids")
    @classmethod
    def _non_empty_shard(cls, value: list[str]) -> list[str]:
        if not value:
            msg = "shard must contain at least one document"
            raise ValueError(msg)
        return value


class IndexMeta(BaseModel):
    """Top-level index metadata; presence of the `_VALID` marker refers to this record."""

    model_config = ConfigDict(extra="forbid")

    artifact_version: str = ARTIFACT_VERSION
    embedding_model: str
    #: Weights revision used at build time. Queries must embed with the same
    #: revision or their vectors are not comparable to the stored ones.
    #: None means the build did not pin one (indexes written before this field
    #: existed also load as None).
    embedding_revision: str | None = None
    embedding_dim: int
    n_shards: int
    n_documents: int
    seed: int


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document: Document
    rank: int
    score: float
    shard_id: int
    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    rrf_contributions: dict[str, float] = Field(default_factory=dict)
    query_variant: str
    expansion_terms: list[str] = Field(default_factory=list)
    snippet: str = Field(max_length=500)


class JudgeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevance: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    redundancy: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=240)


class QueryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    origin: str
    expansions: list[str] = Field(default_factory=list)


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    question: str
    relevant_document_ids: dict[str, int]
