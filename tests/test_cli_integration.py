"""End-to-end offline CLI integration tests: build -> inspect -> search -> eval.

Uses CliRunner (no subprocess) and tmp_path so every test is fully isolated.
All tests use hash provider + tiny config (no network, no downloads).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from cybernaut_mini.cli import app

runner = CliRunner()

# ------------------------------------------------------------------ #
# Corpus / config helpers                                             #
# ------------------------------------------------------------------ #

# ~24-doc corpus spanning 4 topics — reuse conftest _CORPUS style.
_TOPICS = {
    "biotech": [
        "CRISPR screening maps immune pathways in T-cell activation",
        "Gut microbiome composition predicts inflammatory bowel disease",
        "Zylophristine bioassay methods use spectrophotometric detection for quantification",
        "Protein folding prediction enables rational drug discovery pipelines",
        "mRNA delivery platforms extend therapeutic gene expression duration",
        "Epigenetic reprogramming reverses cellular senescence markers",
    ],
    "energy": [
        "Perovskite solar cells achieve record photovoltaic conversion efficiency",
        "Grid-scale lithium batteries stabilise renewable energy output",
        "Green hydrogen electrolysers reach cost parity with fossil fuels",
        "Offshore floating wind turbines expand deep-water generation capacity",
        "Molten salt thermal storage extends concentrated solar plant operation",
        "Solid-state electrolytes improve lithium-metal battery safety margins",
    ],
    "space": [
        "Perseverance rover collects rock samples from ancient Martian river delta",
        "James Webb Telescope resolves distant galaxy clusters at cosmic dawn",
        "Reusable rockets reduce orbital launch costs for commercial access",
        "Gravitational wave detectors capture neutron-star merger ringdown signals",
        "Artemis programme returns crewed missions to the lunar south pole region",
        "Europa Clipper investigates subsurface ocean habitability potential",
    ],
    "finance": [
        "Central bank raises policy rates to combat persistent core inflation",
        "Sovereign bond yields invert signalling recession probability increases",
        "Algorithmic trading strategies exploit limit-order-book microstructure patterns",
        "Retail investors amplify momentum via social media coordination",
        "Venture capital shifts deployment toward climate-tech sector allocations",
        "Stablecoin regulatory frameworks address systemic financial stability risks",
    ],
}


def _make_corpus(tmp_path: Path) -> Path:
    docs: list[dict[str, Any]] = []
    idx = 1
    for topic, titles in _TOPICS.items():
        for title in titles:
            docs.append(
                {
                    "id": f"doc-{idx:03d}",
                    "title": title,
                    "text": f"{title}. Extended body text about {topic} research topic {idx}.",
                    "language": "en",
                    "metadata": {"category": topic, "source": "test"},
                }
            )
            idx += 1
    p = tmp_path / "corpus.jsonl"
    p.write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
    return p


def _make_config(tmp_path: Path, n_shards: int = 4) -> Path:
    cfg = {
        "seed": 42,
        "embedding": {"provider": "hash", "dim": 64},
        "index": {
            "n_shards": n_shards,
            "max_keywords": 10,
            "max_entities": 10,
            "cooccurrence_window": 3,
            "min_edge_count": 1,
        },
        "rrf": {"k": 60, "dense_weight": 1.0, "lexical_weight": 1.0},
        "agent": {
            "exploration_constant": 1.2,
            "max_expansions": 5,
            "max_retrieval_calls": 18,
        },
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


def _make_judgments(tmp_path: Path) -> Path:
    judgments = [
        {
            "query_id": "q1",
            "question": "CRISPR immune screening pathways",
            "relevant_document_ids": {"doc-001": 2, "doc-002": 1},
        },
        {
            "query_id": "q2",
            "question": "solar photovoltaic perovskite efficiency",
            "relevant_document_ids": {"doc-007": 2, "doc-010": 1},
        },
        {
            "query_id": "q3",
            "question": "Mars rover ancient river delta rock samples",
            "relevant_document_ids": {"doc-013": 2, "doc-016": 1},
        },
    ]
    p = tmp_path / "judgments.jsonl"
    p.write_text("".join(json.dumps(j) + "\n" for j in judgments), encoding="utf-8")
    return p


# ------------------------------------------------------------------ #
# Session-scoped built index for integration tests                    #
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def integration_index(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build an index once for all integration tests in this module."""
    base = tmp_path_factory.mktemp("integration")
    corpus_path = _make_corpus(base)
    config_path = _make_config(base)
    index_path = base / "idx"

    result = runner.invoke(
        app,
        [
            "build",
            "--input",
            str(corpus_path),
            "--index",
            str(index_path),
            "--config",
            str(config_path),
            "--offline",
        ],
    )
    assert result.exit_code == 0, f"build failed:\n{result.output}\n{result.stderr}"
    assert (index_path / "_VALID").exists()

    judgments_path = _make_judgments(base)

    return {
        "index": index_path,
        "corpus": corpus_path,
        "config": config_path,
        "judgments": judgments_path,
    }


# ------------------------------------------------------------------ #
# build                                                               #
# ------------------------------------------------------------------ #


def test_build_produces_valid_index(integration_index: dict[str, Path]) -> None:
    assert (integration_index["index"] / "_VALID").exists()


def test_build_json_output(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    config = _make_config(tmp_path)
    index = tmp_path / "idx2"
    result = runner.invoke(
        app,
        [
            "build",
            "--input", str(corpus),
            "--index", str(index),
            "--config", str(config),
            "--offline",
            "--json",
        ],
    )
    assert result.exit_code == 0
    # Kedro logs also go to stdout; find the last line starting with '{'.
    json_line = next(
        (line for line in reversed(result.output.splitlines()) if line.strip().startswith("{")),
        None,
    )
    assert json_line is not None, f"No JSON line in output: {result.output!r}"
    data = json.loads(json_line)
    assert data["n_documents"] == 24
    assert data["n_shards"] == 4


# ------------------------------------------------------------------ #
# inspect-shards                                                      #
# ------------------------------------------------------------------ #


def test_inspect_shards_exits_0(integration_index: dict[str, Path]) -> None:
    result = runner.invoke(
        app, ["inspect-shards", "--index", str(integration_index["index"])]
    )
    assert result.exit_code == 0
    assert "shard 000" in result.output


def test_inspect_shards_json(integration_index: dict[str, Path]) -> None:
    result = runner.invoke(
        app,
        ["inspect-shards", "--index", str(integration_index["index"]), "--json"],
    )
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert "shards" in data
    assert len(data["shards"]) == 4


# ------------------------------------------------------------------ #
# search                                                              #
# ------------------------------------------------------------------ #


def test_search_hybrid_exits_0(integration_index: dict[str, Path]) -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(integration_index["index"]),
            "--question", "CRISPR immune screening",
            "--mode", "hybrid",
            "--offline",
        ],
    )
    assert result.exit_code == 0


def test_search_lexical_exits_0(integration_index: dict[str, Path]) -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(integration_index["index"]),
            "--question", "zylophristine spectrophotometric quantification",
            "--mode", "lexical",
            "--offline",
        ],
    )
    assert result.exit_code == 0


def test_search_agent_exits_0_and_trace(
    integration_index: dict[str, Path], tmp_path: Path
) -> None:
    trace_path = tmp_path / "trace.json"
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(integration_index["index"]),
            "--question", "Mars rover ancient delta",
            "--mode", "agent",
            "--offline",
            "--trace-out", str(trace_path),
            "--config", str(integration_index["config"]),
        ],
    )
    assert result.exit_code == 0, f"search agent failed:\n{result.output}"
    assert trace_path.exists()
    trace_data = json.loads(trace_path.read_text())
    assert "retrieval_calls" in trace_data
    assert trace_data["retrieval_calls"] <= 18


def test_search_json_mode(integration_index: dict[str, Path]) -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(integration_index["index"]),
            "--question", "solar panel efficiency",
            "--mode", "hybrid",
            "--json",
            "--offline",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert "hits" in data
    assert "question" in data


# ------------------------------------------------------------------ #
# eval                                                                #
# ------------------------------------------------------------------ #


def test_eval_exits_0(integration_index: dict[str, Path]) -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "--index", str(integration_index["index"]),
            "--judgments", str(integration_index["judgments"]),
            "--config", str(integration_index["config"]),
            "--offline",
        ],
    )
    assert result.exit_code == 0, f"eval failed:\n{result.output}\n{result.stderr}"
    # Table output should contain all 4 modes.
    for mode in ("lexical", "dense", "hybrid", "agent"):
        assert mode in result.output


def test_eval_json_output(integration_index: dict[str, Path]) -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "--index", str(integration_index["index"]),
            "--judgments", str(integration_index["judgments"]),
            "--config", str(integration_index["config"]),
            "--offline",
            "--json",
        ],
    )
    assert result.exit_code == 0, f"eval --json failed:\n{result.output}"
    data = json.loads(result.output.strip())
    assert "metrics" in data
    assert len(data["metrics"]) == 4
    for m in data["metrics"]:
        assert "mode" in m
        assert "recall_at_5" in m
        assert "recall_at_10" in m
        assert "mrr_at_10" in m
        assert "ndcg_at_10" in m
        assert "mean_retrieval_calls" in m
        assert "mean_llm_calls" in m
        assert "wall_clock_seconds" in m


def test_eval_bad_judgments_exits_nonzero(
    integration_index: dict[str, Path], tmp_path: Path
) -> None:
    """A judgments file with missing required field should produce exit != 0 and stderr."""
    bad_judg = tmp_path / "bad_judgments.jsonl"
    # Missing 'question' field (required by Judgment model).
    bad_judg.write_text(
        json.dumps({"query_id": "q1", "relevant_document_ids": {"doc-001": 2}}) + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "eval",
            "--index", str(integration_index["index"]),
            "--judgments", str(bad_judg),
            "--config", str(integration_index["config"]),
            "--offline",
        ],
    )
    assert result.exit_code != 0
    # Error message should mention the problem (query_id or field name).
    combined = (result.output or "") + (result.stderr or "")
    assert "q1" in combined or "question" in combined or "judgment" in combined.lower()


def test_eval_offline_rejects_st_config(
    integration_index: dict[str, Path], tmp_path: Path
) -> None:
    """--offline with a sentence_transformers config should exit != 0."""
    st_config = tmp_path / "st.yaml"
    st_config.write_text(
        yaml.dump({"embedding": {"provider": "sentence_transformers"}}),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "eval",
            "--index", str(integration_index["index"]),
            "--judgments", str(integration_index["judgments"]),
            "--config", str(st_config),
            "--offline",
        ],
    )
    # provider_from_meta reads from index meta (hash provider), so --offline
    # does not fail at provider_from_meta; but require_offline_compatible() on
    # load_config should fail.
    assert result.exit_code != 0
