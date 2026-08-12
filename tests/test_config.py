from __future__ import annotations

from pathlib import Path

import pytest

from cybernaut_mini.config import ConfigError, load_config


def test_defaults_match_spec() -> None:
    config = load_config(environ={})
    assert config.seed == 42
    # The baseline is model2vec: distilled static embeddings that carry semantic
    # structure without torch. Not 'hash' — shards are k-means clusters over these
    # vectors, and feature hashing has no semantics to cluster on. Not
    # 'sentence_transformers' — that lives behind the optional 'st' extra; see
    # test_default_config_runs_on_a_bare_install below.
    assert config.embedding.provider == "model2vec"
    assert config.embedding.model_name == "minishlab/potion-multilingual-128M"
    assert config.index.n_shards == 12
    assert config.rrf.k == 60
    assert config.agent.max_retrieval_calls == 18


def test_default_config_runs_on_a_bare_install() -> None:
    """The default config must not require an optional extra.

    Regression: the baseline selected 'sentence_transformers', which lives behind the
    optional 'st' extra that neither `uv sync` nor the devcontainer installs. A bare
    `kedro run` or `cybernaut-mini build` therefore died with "sentence-transformers is
    not installed" on a freshly built container.

    This asserts the property directly by importing the default provider's backing
    package. It deliberately does NOT use ``require_offline_compatible``: since the
    default became model2vec those are two different properties — model2vec ships in
    the core dependencies (bare install works) but downloads weights on first use
    (not offline-safe). Conflating them is what this test used to do.
    """
    provider = load_config(environ={}).embedding.provider
    assert provider != "sentence_transformers", "default must not need the 'st' extra"
    if provider == "model2vec":
        import model2vec  # noqa: F401  — must be a core dependency, not an extra


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("index:\n  n_shards: 8\n", encoding="utf-8")
    config = load_config(config_file, environ={})
    assert config.index.n_shards == 8
    assert config.rrf.k == 60


def test_env_overrides_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("index:\n  n_shards: 8\n", encoding="utf-8")
    config = load_config(
        config_file,
        environ={"CYBERNAUT_MINI__INDEX__N_SHARDS": "16"},
    )
    assert config.index.n_shards == 16


def test_cli_overrides_env(tmp_path: Path) -> None:
    config = load_config(
        environ={"CYBERNAUT_MINI__SEED": "7"},
        overrides={"seed": 11},
    )
    assert config.seed == 11


def test_unknown_key_fails_validation(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("not_a_section: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_file, environ={})


def test_offline_rejects_sentence_transformers() -> None:
    # Explicit rather than AppConfig(): the default provider is 'hash' now, so this
    # must opt into sentence_transformers to exercise the rejection at all.
    config = load_config(
        environ={}, overrides={"embedding": {"provider": "sentence_transformers"}}
    )
    with pytest.raises(ConfigError, match="offline"):
        config.require_offline_compatible()


def test_offline_accepts_hash_provider() -> None:
    config = load_config(environ={}, overrides={"embedding": {"provider": "hash"}})
    config.require_offline_compatible()
