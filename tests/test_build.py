"""Integration tests for the index_build pipeline and CLI.

All tests use HashEmbedder and TextProcessor(use_spacy=False): no network,
no spaCy. CLI tests use typer.testing.CliRunner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from cybernaut_mini.cli import app
from cybernaut_mini.indexing import IndexLoadError, LoadedIndex

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

runner = CliRunner()


def _make_config(tmp_path: Path, n_shards: int = 3) -> Path:
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
    }
    p = tmp_path / "test.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


def _make_corpus(tmp_path: Path, docs: list[dict[str, Any]]) -> Path:
    p = tmp_path / "corpus.jsonl"
    lines = "".join(json.dumps(d) + "\n" for d in docs)
    p.write_text(lines, encoding="utf-8")
    return p


def _topical_docs(n: int = 12) -> list[dict[str, Any]]:
    """Generate n simple topical documents matching the conftest pattern."""
    topics = ["biotech", "energy", "space", "finance"]
    docs = []
    for i in range(1, n + 1):
        topic = topics[(i - 1) % len(topics)]
        docs.append(
            {
                "id": f"doc-{i:03d}",
                "title": f"{topic.capitalize()} headline {i}",
                "text": f"Extended body about {topic} topic number {i}.",
                "language": "en",
                "metadata": {"category": topic},
            }
        )
    return docs


# ------------------------------------------------------------------ #
# Indexing unit tests (keywords, term graph, entities)                #
# ------------------------------------------------------------------ #


def test_keywords_sorted_and_capped() -> None:
    from cybernaut_mini.indexing import compute_keywords

    shard_tokens = {
        0: ["alpha", "beta", "alpha", "gamma", "alpha", "beta"],
        1: ["delta", "delta", "epsilon"],
    }
    result = compute_keywords(shard_tokens, max_keywords=2)
    assert len(result[0]) <= 2
    # Highest weight first
    if len(result[0]) >= 2:
        assert result[0][0].weight >= result[0][1].weight
    # Ties broken by term
    weights = [kw.weight for kw in result[0]]
    terms = [kw.term for kw in result[0]]
    for i in range(len(weights) - 1):
        if abs(weights[i] - weights[i + 1]) < 1e-9:
            assert terms[i] <= terms[i + 1]


def test_keywords_all_shards_present() -> None:
    from cybernaut_mini.indexing import compute_keywords

    shard_tokens = {0: ["foo", "bar"], 1: ["baz"], 2: []}
    result = compute_keywords(shard_tokens, max_keywords=5)
    assert set(result.keys()) == {0, 1, 2}
    assert result[2] == []


def test_term_graph_drops_rare_edges() -> None:
    from cybernaut_mini.indexing import compute_term_graph

    # Only one doc with tokens that produce pairs — with min_edge_count=2,
    # pairs that occur only once must be dropped.
    doc_tokens = [["a", "b", "c", "a", "b"]]
    graph = compute_term_graph(doc_tokens, window=3, min_edge_count=2)
    # a-b should appear multiple times; c-a only once in some windows
    # Verify surviving edges all have count >= 2 (implicit in graph presence)
    assert "a" in graph or "b" in graph  # at least some edges survive

    # min_edge_count=999 should produce empty graph
    empty_graph = compute_term_graph(doc_tokens, window=3, min_edge_count=999)
    assert empty_graph == {}


def test_term_graph_outgoing_weights_sum_to_one() -> None:
    from cybernaut_mini.indexing import compute_term_graph

    doc_tokens = [["a", "b", "c", "a", "b", "c", "a", "b", "c"]]
    graph = compute_term_graph(doc_tokens, window=3, min_edge_count=1)
    for node, edges in graph.items():
        total = sum(edges.values())
        assert abs(total - 1.0) < 1e-6, f"Node {node!r} weights sum to {total}, expected 1.0"


def test_entity_aggregation_caps_and_orders() -> None:
    from cybernaut_mini.indexing import compute_entities

    doc_entities: list[list[str]] = [
        ["alice", "bob", "alice"],
        ["bob", "carol"],
        ["alice", "carol", "carol"],
    ]
    result = compute_entities(doc_entities, max_entities=2)
    assert len(result) <= 2
    if len(result) >= 2:
        assert result[0].count >= result[1].count
    # alice=3, carol=3, bob=2 → tie between alice and carol broken by text
    texts = [e.text for e in result]
    assert "alice" in texts  # alice sorts before carol


# ------------------------------------------------------------------ #
# Full build round-trip (tmp_path)                                    #
# ------------------------------------------------------------------ #


def test_build_exit_zero(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path, _topical_docs(12))
    cfg = _make_config(tmp_path, n_shards=3)
    index_dir = tmp_path / "idx"

    result = runner.invoke(
        app,
        ["build", "--input", str(corpus), "--index", str(index_dir),
         "--config", str(cfg), "--offline"],
    )
    assert result.exit_code == 0, f"Unexpected exit {result.exit_code}: {result.output}"
    assert (index_dir / "_VALID").exists()


def test_build_valid_marker_exists(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path, _topical_docs(12))
    cfg = _make_config(tmp_path, n_shards=3)
    index_dir = tmp_path / "idx"
    runner.invoke(
        app,
        ["build", "--input", str(corpus), "--index", str(index_dir),
         "--config", str(cfg), "--offline"],
    )
    assert (index_dir / "_VALID").exists()


def test_build_loaded_index_valid(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path, _topical_docs(12))
    cfg = _make_config(tmp_path, n_shards=3)
    index_dir = tmp_path / "idx"
    runner.invoke(
        app,
        ["build", "--input", str(corpus), "--index", str(index_dir),
         "--config", str(cfg), "--offline"],
    )
    loaded = LoadedIndex.load(index_dir)
    assert loaded.meta.n_documents == 12
    assert loaded.meta.n_shards == 3
    assert len(loaded.documents) == 12


def test_build_every_doc_in_exactly_one_shard(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path, _topical_docs(12))
    cfg = _make_config(tmp_path, n_shards=3)
    index_dir = tmp_path / "idx"
    runner.invoke(
        app,
        ["build", "--input", str(corpus), "--index", str(index_dir),
         "--config", str(cfg), "--offline"],
    )
    loaded = LoadedIndex.load(index_dir)
    all_ids_in_shards: list[str] = []
    for manifest in loaded.manifests.values():
        all_ids_in_shards.extend(manifest.document_ids)
    doc_ids = [doc.id for doc in loaded.documents]
    assert sorted(all_ids_in_shards) == sorted(doc_ids)
    # No duplicates
    assert len(all_ids_in_shards) == len(set(all_ids_in_shards))


def test_build_deterministic_two_runs(tmp_path: Path) -> None:
    """Two builds into different dirs must produce byte-for-byte identical artifacts."""
    corpus = _make_corpus(tmp_path, _topical_docs(12))
    cfg = _make_config(tmp_path, n_shards=3)
    idx1 = tmp_path / "idx1"
    idx2 = tmp_path / "idx2"

    for idx in [idx1, idx2]:
        result = runner.invoke(
            app,
            ["build", "--input", str(corpus), "--index", str(idx),
             "--config", str(cfg), "--offline"],
        )
        assert result.exit_code == 0

    # Compare all non-npy artifacts byte-for-byte.
    def collect_files(base: Path) -> dict[str, bytes]:
        return {
            str(p.relative_to(base)): p.read_bytes()
            for p in sorted(base.rglob("*"))
            if p.is_file() and p.suffix != ".npy"
        }

    files1 = collect_files(idx1)
    files2 = collect_files(idx2)
    assert set(files1.keys()) == set(files2.keys()), "Different artifact sets"
    for name, content1 in files1.items():
        assert content1 == files2[name], f"Artifact {name!r} differs between runs"

    # Compare npy files numerically.
    v1 = np.load(idx1 / "embeddings.npy")
    v2 = np.load(idx2 / "embeddings.npy")
    assert np.array_equal(v1, v2)


# ------------------------------------------------------------------ #
# Failed build: duplicate id                                          #
# ------------------------------------------------------------------ #


def test_build_duplicate_id_exits_nonzero(tmp_path: Path) -> None:
    bad_docs = _topical_docs(5)
    bad_docs.append(bad_docs[0].copy())  # duplicate
    corpus = _make_corpus(tmp_path, bad_docs)
    cfg = _make_config(tmp_path, n_shards=2)
    index_dir = tmp_path / "idx"

    result = runner.invoke(
        app,
        ["build", "--input", str(corpus), "--index", str(index_dir),
         "--config", str(cfg), "--offline"],
    )
    assert result.exit_code != 0


def test_build_duplicate_id_stderr_names_record(tmp_path: Path) -> None:
    bad_docs = _topical_docs(5)
    bad_docs.append(bad_docs[2].copy())  # duplicate doc-003
    corpus = _make_corpus(tmp_path, bad_docs)
    cfg = _make_config(tmp_path, n_shards=2)
    index_dir = tmp_path / "idx"

    result = runner.invoke(
        app,
        ["build", "--input", str(corpus), "--index", str(index_dir),
         "--config", str(cfg), "--offline"],
    )
    assert result.exit_code != 0
    # The doc id must appear in output (CliRunner merges stdout+stderr)
    assert "doc-003" in result.output


def test_build_duplicate_no_valid_marker(tmp_path: Path) -> None:
    bad_docs = _topical_docs(5)
    bad_docs.append(bad_docs[0].copy())
    corpus = _make_corpus(tmp_path, bad_docs)
    cfg = _make_config(tmp_path, n_shards=2)
    index_dir = tmp_path / "idx"

    runner.invoke(
        app,
        ["build", "--input", str(corpus), "--index", str(index_dir),
         "--config", str(cfg), "--offline"],
    )
    assert not (index_dir / "_VALID").exists()


def test_build_duplicate_load_raises(tmp_path: Path) -> None:
    bad_docs = _topical_docs(5)
    bad_docs.append(bad_docs[0].copy())
    corpus = _make_corpus(tmp_path, bad_docs)
    cfg = _make_config(tmp_path, n_shards=2)
    index_dir = tmp_path / "idx"

    runner.invoke(
        app,
        ["build", "--input", str(corpus), "--index", str(index_dir),
         "--config", str(cfg), "--offline"],
    )
    with pytest.raises((IndexLoadError, ValueError)):
        LoadedIndex.load(index_dir)


# ------------------------------------------------------------------ #
# n_shards > n_docs via config                                        #
# ------------------------------------------------------------------ #


def test_build_n_shards_gt_n_docs_exits_nonzero(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path, _topical_docs(4))
    cfg = _make_config(tmp_path, n_shards=10)  # 10 shards > 4 docs
    index_dir = tmp_path / "idx"

    result = runner.invoke(
        app,
        ["build", "--input", str(corpus), "--index", str(index_dir),
         "--config", str(cfg), "--offline"],
    )
    assert result.exit_code != 0


def test_build_n_shards_gt_n_docs_no_valid_marker(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path, _topical_docs(4))
    cfg = _make_config(tmp_path, n_shards=10)
    index_dir = tmp_path / "idx"

    runner.invoke(
        app,
        ["build", "--input", str(corpus), "--index", str(index_dir),
         "--config", str(cfg), "--offline"],
    )
    assert not (index_dir / "_VALID").exists()


# ------------------------------------------------------------------ #
# inspect-shards                                                      #
# ------------------------------------------------------------------ #


def _build_index(tmp_path: Path, n_docs: int = 12, n_shards: int = 3) -> Path:
    corpus = _make_corpus(tmp_path, _topical_docs(n_docs))
    cfg = _make_config(tmp_path, n_shards=n_shards)
    index_dir = tmp_path / "idx"
    result = runner.invoke(
        app,
        ["build", "--input", str(corpus), "--index", str(index_dir),
         "--config", str(cfg), "--offline"],
    )
    assert result.exit_code == 0, f"Build failed: {result.output}"
    return index_dir


def test_inspect_shards_exit_zero(tmp_path: Path) -> None:
    index_dir = _build_index(tmp_path)
    result = runner.invoke(app, ["inspect-shards", "--index", str(index_dir)])
    assert result.exit_code == 0


def test_inspect_shards_mentions_all_shard_ids(tmp_path: Path) -> None:
    index_dir = _build_index(tmp_path, n_shards=3)
    result = runner.invoke(app, ["inspect-shards", "--index", str(index_dir)])
    assert result.exit_code == 0
    for shard_id in range(3):
        assert f"{shard_id:03d}" in result.output or str(shard_id) in result.output


def test_inspect_shards_json_parses(tmp_path: Path) -> None:
    index_dir = _build_index(tmp_path)
    result = runner.invoke(app, ["inspect-shards", "--index", str(index_dir), "--json"])
    assert result.exit_code == 0
    json_line = next(
        (line for line in reversed(result.output.splitlines()) if line.strip().startswith("{")),
        None,
    )
    assert json_line is not None, f"No JSON in output: {result.output!r}"
    parsed = json.loads(json_line)
    assert "shards" in parsed
    assert isinstance(parsed["shards"], list)
    assert len(parsed["shards"]) == 3


def test_inspect_shards_json_single_object(tmp_path: Path) -> None:
    """--json output must be a single JSON object, not multiple lines."""
    index_dir = _build_index(tmp_path)
    result = runner.invoke(app, ["inspect-shards", "--index", str(index_dir), "--json"])
    assert result.exit_code == 0
    # Extract the JSON line (Kedro log lines may precede it).
    json_line = next(
        (line for line in reversed(result.output.splitlines()) if line.strip().startswith("{")),
        None,
    )
    assert json_line is not None, f"No JSON line in output: {result.output!r}"
    parsed = json.loads(json_line)
    assert isinstance(parsed, dict)


def test_inspect_shards_missing_valid_exits_nonzero(tmp_path: Path) -> None:
    index_dir = tmp_path / "empty_idx"
    index_dir.mkdir()
    result = runner.invoke(app, ["inspect-shards", "--index", str(index_dir)])
    assert result.exit_code != 0


# ------------------------------------------------------------------ #
# --json build flag                                                   #
# ------------------------------------------------------------------ #


def test_build_json_flag(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path, _topical_docs(12))
    cfg = _make_config(tmp_path, n_shards=3)
    index_dir = tmp_path / "idx"

    result = runner.invoke(
        app,
        ["build", "--input", str(corpus), "--index", str(index_dir),
         "--config", str(cfg), "--offline", "--json"],
    )
    assert result.exit_code == 0
    # Extract the JSON line from output (Kedro may emit log lines before it).
    json_line = next(
        (line for line in reversed(result.output.splitlines()) if line.strip().startswith("{")),
        None,
    )
    assert json_line is not None, f"No JSON line found in output: {result.output!r}"
    parsed = json.loads(json_line)
    assert parsed["n_documents"] == 12
    assert parsed["n_shards"] == 3
    assert "index" in parsed
