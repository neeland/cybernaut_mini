"""Pydantic models for all persisted records, plus the canonical JSON writer.

Every artifact on disk is produced through :func:`canonical_dumps` so that two
builds with the same seed and provider are byte-for-byte identical.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

ARTIFACT_VERSION = "1"

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
