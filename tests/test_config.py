from __future__ import annotations

from pathlib import Path

import pytest

from cybernaut_mini.config import AppConfig, ConfigError, load_config


def test_defaults_match_spec() -> None:
    config = load_config(environ={})
    assert config.seed == 42
    assert config.embedding.provider == "sentence_transformers"
    assert config.index.n_shards == 12
    assert config.rrf.k == 60
    assert config.agent.max_retrieval_calls == 18


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
    config = AppConfig()
    with pytest.raises(ConfigError, match="offline"):
        config.require_offline_compatible()


def test_offline_accepts_hash_provider() -> None:
    config = load_config(environ={}, overrides={"embedding": {"provider": "hash"}})
    config.require_offline_compatible()
