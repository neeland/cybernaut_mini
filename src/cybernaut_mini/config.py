"""Configuration models and layered loading.

Precedence (highest wins): CLI overrides > environment variables > YAML file > defaults.
Environment variables use the prefix ``CYBERNAUT_MINI__`` with ``__`` as the section
separator, e.g. ``CYBERNAUT_MINI__EMBEDDING__PROVIDER=hash``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ENV_PREFIX = "CYBERNAUT_MINI__"


class ConfigError(ValueError):
    """Raised for invalid or unusable configuration."""


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Defaults to the offline provider so a bare `kedro run` or `cybernaut-mini build`
    # works on the default install. 'sentence_transformers' is the quality option but
    # lives behind the optional 'st' extra, so it must be opted into explicitly
    # (configs/default.yaml, or --env prod).
    provider: Literal["hash", "sentence_transformers"] = "hash"
    model: str = "intfloat/multilingual-e5-small"
    revision: str | None = Field(
        default=None,
        description=(
            "Commit SHA or tag of the model weights. None resolves to the repo's "
            "default branch and is not reproducible; pin it for production builds."
        ),
    )
    dim: int = Field(default=256, ge=8, description="Vector size for the hash provider")

    def is_pinned(self) -> bool:
        """True when this configuration identifies exactly one set of weights."""
        if self.provider == "hash":
            # The hash embedder is pure code — no weights to drift.
            return True
        revision = self.revision
        if revision is None:
            return False
        return revision.strip().lower() not in {"main", "master", "head"}


class IndexConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_shards: int = Field(default=12, ge=1)
    max_keywords: int = Field(default=30, ge=1)
    max_entities: int = Field(default=30, ge=1)
    cooccurrence_window: int = Field(default=5, ge=2)
    min_edge_count: int = Field(default=2, ge=1)


class RRFConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=1.0, ge=0.0)
    lexical_weight: float = Field(default=1.0, ge=0.0)
    entity_weight: float = Field(default=0.5, ge=0.0)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exploration_constant: float = Field(default=1.2, ge=0.0)
    max_expansions: int = Field(default=5, ge=0)
    max_retrieval_calls: int = Field(default=18, ge=1)
    judge: Literal["heuristic"] = "heuristic"
    query_generator: Literal["heuristic"] = "heuristic"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    rrf: RRFConfig = Field(default_factory=RRFConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    def require_offline_compatible(self) -> None:
        """Reject configurations that would require network access (``--offline``)."""
        if self.embedding.provider == "sentence_transformers":
            msg = (
                "offline mode: embedding provider 'sentence_transformers' may require a "
                "model download; use provider 'hash' (e.g. configs/tiny.yaml)"
            )
            raise ConfigError(msg)

    def require_reproducible(self) -> None:
        """Reject configurations that cannot be rebuilt to the same bytes later.

        An index is only reproducible from ``(corpus, config, seed)`` if the weights
        that produced its vectors are pinned. Enforced for production builds, where
        an unnoticed weight change would silently invalidate every stored vector.
        """
        if not self.embedding.is_pinned():
            msg = (
                f"embedding model {self.embedding.model!r} is not pinned "
                f"(revision={self.embedding.revision!r}); set embedding.revision to a "
                f"commit SHA so the index can be rebuilt byte-for-byte"
            )
            raise ConfigError(msg)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_overrides(environ: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        path = [part.lower() for part in name[len(ENV_PREFIX) :].split("__") if part]
        if not path:
            continue
        node = result
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = yaml.safe_load(raw)
    return result


def load_config(
    config_path: Path | None = None,
    *,
    environ: dict[str, str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    layers: dict[str, Any] = {}
    if config_path is not None:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            msg = f"config file {config_path} must contain a mapping"
            raise ConfigError(msg)
        layers = _deep_merge(layers, raw)
    env = environ if environ is not None else dict(os.environ)
    layers = _deep_merge(layers, _env_overrides(env))
    if overrides:
        layers = _deep_merge(layers, overrides)
    try:
        return AppConfig.model_validate(layers)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
