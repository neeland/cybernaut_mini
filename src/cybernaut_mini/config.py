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


#: Default weights per provider. Kept here rather than on the `model` field so the
#: two downloadable providers can have different defaults without a caller who sets
#: only `provider` silently inheriting the other one's model name.
PROVIDER_DEFAULT_MODEL: dict[str, str] = {
    "model2vec": "minishlab/potion-multilingual-128M",
    "sentence_transformers": "intfloat/multilingual-e5-small",
}

#: Prefix marking a model2vec index inside ``IndexMeta.embedding_model``. Lives here,
#: not on the embedder, so config/retrieval/index_build share one definition and the
#: providers module stays importable without pulling in the model2vec package.
MODEL2VEC_PREFIX = "model2vec:"


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Defaults to 'model2vec': real distilled embeddings that carry semantic
    # structure, with no torch, so a bare `kedro run` still works on the default
    # install. It is NOT the hash embedder, because shards are clusters over these
    # vectors and feature hashing has no semantics to cluster on.
    # 'hash' remains a first-class OFFLINE provider (no download, byte-deterministic)
    # and is what --offline and configs/tiny.yaml select.
    # 'sentence_transformers' is the quality ceiling, behind the optional 'st' extra.
    provider: Literal["hash", "model2vec", "sentence_transformers"] = "model2vec"
    model: str = Field(
        default="",
        description=(
            "Model repo id. Empty resolves to this provider's default from "
            "PROVIDER_DEFAULT_MODEL; read it through `model_name`, never directly."
        ),
    )
    revision: str | None = Field(
        default=None,
        description=(
            "Commit SHA or tag of the model weights. None resolves to the repo's "
            "default branch and is not reproducible; pin it for production builds."
        ),
    )
    dim: int = Field(
        default=256,
        ge=8,
        description=(
            "Vector size for the hash provider only. The downloadable providers "
            "report their own dimension from the loaded weights."
        ),
    )
    device: Literal["auto", "mps", "cuda", "cpu"] = Field(
        default="auto",
        description=(
            "Torch device for the 'sentence_transformers' provider; ignored by "
            "'hash' and 'model2vec', which never import torch. 'auto' prefers MPS "
            "on Apple silicon, then CUDA, then CPU. An unavailable explicit choice "
            "degrades to CPU rather than raising, so a config that travels between "
            "a Mac and a Linux box still builds on both. Note that MPS and CPU do "
            "not produce bit-identical vectors: byte-identical rebuilds are "
            "guaranteed per-device, not across devices."
        ),
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description=(
            "Mini-batch size passed to sentence_transformers encode(). Ignored by "
            "'hash' and 'model2vec', which process all inputs in one numpy call. "
            "Benchmarked on 2,048 MIRACL passages with intfloat/multilingual-e5-small "
            "on Apple M-series MPS: 32→269 docs/s, 64→267, 128→256, 256→196. MPS "
            "unified memory does NOT benefit from larger batches — 32 is optimal. "
            "For discrete CUDA GPUs larger batches (128-256) typically improve "
            "throughput; override via conf/prod/parameters.yml if the target machine "
            "has a CUDA GPU."
        ),
    )

    @property
    def model_name(self) -> str:
        """The model repo id to load, resolving the per-provider default."""
        if self.model:
            return self.model
        return PROVIDER_DEFAULT_MODEL.get(self.provider, "")

    def identifier(self) -> str:
        """The ``IndexMeta.embedding_model`` value this configuration produces.

        Must stay in step with each provider's ``identifier`` property: the build
        writes this string and ``retrieval.provider_from_meta`` dispatches on it, so
        a mismatch reopens an index with the wrong provider and silently yields
        vectors that are not comparable to the stored ones.
        """
        if self.provider == "hash":
            return f"hash-{self.dim}"
        if self.provider == "model2vec":
            return f"{MODEL2VEC_PREFIX}{self.model_name}"
        return self.model_name

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


#: Default Hugging Face models for the model-backed agent providers. Kept beside
#: PROVIDER_DEFAULT_MODEL for the same reason: the config field stays empty-able
#: without one provider inheriting another's default.
AGENT_DEFAULT_MODEL: dict[str, str] = {
    # Multilingual MiniLM reranker: the corpus mixes CC-News and MIRACL, so an
    # English-only ms-marco cross-encoder would misjudge half the shards.
    "cross_encoder": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    # Small enough to generate a handful of query rewrites in well under a
    # second on MPS; instruct-tuned so it follows the one-query-per-line format.
    "llm": "Qwen/Qwen2.5-0.5B-Instruct",
}


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exploration_constant: float = Field(default=1.2, ge=0.0)
    max_expansions: int = Field(default=5, ge=0)
    max_retrieval_calls: int = Field(default=18, ge=1)
    # 'heuristic' stays the default so a bare install (no torch) keeps working
    # and tests stay offline-deterministic. 'cross_encoder' / 'llm' are the real
    # model-backed providers; both need the 'st' extra ('mps' extra on a Mac for
    # the GPU) and download weights on first use.
    judge: Literal["heuristic", "cross_encoder"] = "heuristic"
    query_generator: Literal["heuristic", "llm"] = "heuristic"
    judge_model: str = Field(
        default="",
        description=(
            "Cross-encoder repo id for judge='cross_encoder'. Empty resolves to "
            "AGENT_DEFAULT_MODEL['cross_encoder']; read via `judge_model_name`."
        ),
    )
    judge_revision: str | None = None
    generator_model: str = Field(
        default="",
        description=(
            "Causal-LM repo id for query_generator='llm'. Empty resolves to "
            "AGENT_DEFAULT_MODEL['llm']; read via `generator_model_name`."
        ),
    )
    generator_revision: str | None = None
    device: Literal["auto", "mps", "cuda", "cpu"] = Field(
        default="auto",
        description=(
            "Torch device for the model-backed judge and generator; ignored by the "
            "heuristic providers. Same semantics as embedding.device: 'auto' "
            "prefers MPS on Apple silicon, and an unavailable explicit choice "
            "degrades to CPU."
        ),
    )
    generator_max_new_tokens: int = Field(default=192, ge=16)

    @property
    def judge_model_name(self) -> str:
        return self.judge_model or AGENT_DEFAULT_MODEL["cross_encoder"]

    @property
    def generator_model_name(self) -> str:
        return self.generator_model or AGENT_DEFAULT_MODEL["llm"]


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    rrf: RRFConfig = Field(default_factory=RRFConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    def require_offline_compatible(self) -> None:
        """Reject configurations that would require network access (``--offline``).

        Both downloadable providers are rejected: a warm cache is not something the
        config can verify, so 'may require a download' is treated as 'does'.
        """
        if self.embedding.provider != "hash":
            msg = (
                f"offline mode: embedding provider {self.embedding.provider!r} may "
                "require a model download; use provider 'hash' (e.g. configs/tiny.yaml)"
            )
            raise ConfigError(msg)
        if self.agent.judge != "heuristic" or self.agent.query_generator != "heuristic":
            msg = (
                f"offline mode: agent judge {self.agent.judge!r} / query generator "
                f"{self.agent.query_generator!r} may require a model download; use "
                "'heuristic' for both"
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
                f"embedding model {self.embedding.model_name!r} is not pinned "
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
