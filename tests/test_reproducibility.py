"""Tests for the pinning guarantees that make a production index rebuildable."""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from typing import Any

import pytest

from cybernaut_mini.config import AppConfig, ConfigError, EmbeddingConfig
from cybernaut_mini.hooks import ReproducibilityHooks
from cybernaut_mini.models import IndexMeta

_SHA = "0123456789abcdef0123456789abcdef01234567"


# ------------------------------------------------------------------ #
# EmbeddingConfig.is_pinned                                           #
# ------------------------------------------------------------------ #


def test_hash_provider_is_always_pinned() -> None:
    """The hash embedder is pure code, so it has no weights that can drift."""
    assert EmbeddingConfig(provider="hash", dim=256).is_pinned()


@pytest.mark.parametrize("revision", [None, "main", "master", "HEAD", "  main  "])
def test_moving_or_absent_revision_is_not_pinned(revision: str | None) -> None:
    config = EmbeddingConfig(provider="sentence_transformers", revision=revision)
    assert not config.is_pinned()


def test_commit_sha_is_pinned() -> None:
    assert EmbeddingConfig(provider="sentence_transformers", revision=_SHA).is_pinned()


def test_require_reproducible_names_the_offending_model() -> None:
    config = AppConfig(embedding=EmbeddingConfig(provider="sentence_transformers"))
    with pytest.raises(ConfigError, match="not pinned"):
        config.require_reproducible()


def test_require_reproducible_passes_when_pinned() -> None:
    config = AppConfig(
        embedding=EmbeddingConfig(provider="sentence_transformers", revision=_SHA)
    )
    config.require_reproducible()  # must not raise


# ------------------------------------------------------------------ #
# ReproducibilityHooks                                                #
# ------------------------------------------------------------------ #


def _context(env: str, embedding: dict[str, Any] | None) -> SimpleNamespace:
    return SimpleNamespace(env=env, params={"embedding": embedding} if embedding else {})


def test_prod_run_aborts_when_unpinned() -> None:
    hooks = ReproducibilityHooks()
    context = _context("prod", {"provider": "sentence_transformers", "revision": None})
    with pytest.raises(ConfigError, match=re.escape("conf/prod/parameters.yml")):
        hooks.after_context_created(context)


def test_local_run_warns_but_proceeds(caplog: pytest.LogCaptureFixture) -> None:
    hooks = ReproducibilityHooks()
    context = _context("local", {"provider": "sentence_transformers", "revision": None})
    with caplog.at_level(logging.WARNING):
        hooks.after_context_created(context)
    assert "not pinned" in caplog.text


def test_prod_run_proceeds_when_pinned() -> None:
    hooks = ReproducibilityHooks()
    context = _context("prod", {"provider": "sentence_transformers", "revision": _SHA})
    hooks.after_context_created(context)  # must not raise


def test_prod_run_proceeds_with_hash_provider() -> None:
    hooks = ReproducibilityHooks()
    context = _context("prod", {"provider": "hash", "dim": 256})
    hooks.after_context_created(context)  # must not raise


def test_hook_ignores_a_run_without_embedding_params() -> None:
    """Pipelines that never embed must not be blocked by an embedding check."""
    ReproducibilityHooks().after_context_created(_context("prod", None))


# ------------------------------------------------------------------ #
# The revision travels with the index                                 #
# ------------------------------------------------------------------ #


def test_index_meta_records_the_build_revision() -> None:
    meta = IndexMeta(
        embedding_model="intfloat/multilingual-e5-small",
        embedding_revision=_SHA,
        embedding_dim=384,
        n_shards=4,
        n_documents=10,
        seed=42,
    )
    assert meta.embedding_revision == _SHA


def test_index_meta_revision_defaults_to_none_for_older_artifacts() -> None:
    """Indexes written before this field existed must still load."""
    meta = IndexMeta.model_validate(
        {
            "artifact_version": "1",
            "embedding_model": "hash-256",
            "embedding_dim": 256,
            "n_shards": 4,
            "n_documents": 10,
            "seed": 42,
        }
    )
    assert meta.embedding_revision is None


def test_query_provider_reuses_the_build_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Query vectors must come from the same weights as the stored document vectors."""
    from cybernaut_mini import retrieval

    captured: dict[str, Any] = {}

    class _Recorder:
        def __init__(self, model_name: str, revision: str | None = None) -> None:
            captured["model"] = model_name
            captured["revision"] = revision

    monkeypatch.setattr(
        "cybernaut_mini.providers.embeddings.SentenceTransformersEmbedder", _Recorder
    )
    meta = IndexMeta(
        embedding_model="intfloat/multilingual-e5-small",
        embedding_revision=_SHA,
        embedding_dim=384,
        n_shards=4,
        n_documents=10,
        seed=42,
    )
    retrieval.provider_from_meta(meta)
    assert captured == {"model": "intfloat/multilingual-e5-small", "revision": _SHA}


def test_build_payload_records_the_revision() -> None:
    from cybernaut_mini.pipelines.index_build.nodes import build_index_payload

    payload = build_index_payload(
        documents=[{"id": "a", "title": "t", "text": "body"}],
        vectors_list=[[1.0, 0.0]],
        manifests_list=[],
        text_result={"tokens": {"a": ["body"]}},
        embedding_params={"provider": "sentence_transformers", "revision": _SHA},
        seed=42,
    )
    assert payload["meta"]["embedding_revision"] == _SHA


def test_hash_builds_record_no_revision() -> None:
    from cybernaut_mini.pipelines.index_build.nodes import build_index_payload

    payload = build_index_payload(
        documents=[{"id": "a", "title": "t", "text": "body"}],
        vectors_list=[[1.0, 0.0]],
        manifests_list=[],
        text_result={"tokens": {"a": ["body"]}},
        embedding_params={"provider": "hash", "dim": 8, "revision": _SHA},
        seed=42,
    )
    assert payload["meta"]["embedding_revision"] is None
