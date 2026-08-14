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


# ------------------------------------------------------------------ #
# Shard file naming (the fixed-3-digit bug)                           #
# ------------------------------------------------------------------ #


def test_shard_id_width_keeps_three_digits_below_a_thousand_shards() -> None:
    from cybernaut_mini.indexing import shard_filename, shard_id_width

    assert shard_id_width(1) == 3
    assert shard_id_width(4) == 3
    assert shard_id_width(1000) == 3  # ids 0..999 still fit in three digits
    assert shard_filename(3, 4) == "shard_003.json"
    assert shard_filename(999, 1000) == "shard_999.json"


def test_shard_id_width_grows_past_a_thousand_shards() -> None:
    """The old fixed ``:03d`` produced names that no longer sort in shard order."""
    from cybernaut_mini.indexing import shard_filename, shard_id_width

    assert shard_id_width(1001) == 4
    assert shard_id_width(10_000) == 4
    assert shard_id_width(10_001) == 5

    n_shards = 1001
    names = [shard_filename(sid, n_shards) for sid in range(n_shards)]
    assert names[0] == "shard_0000.json"
    assert names[1000] == "shard_1000.json"
    # Lexicographic order and shard order agree — the property the old naming lost.
    assert sorted(names) == names

    # Demonstrate the defect being fixed: with the old fixed width they do not.
    old_style = [f"shard_{sid:03d}.json" for sid in range(n_shards)]
    assert sorted(old_style) != old_style


def test_shard_filename_rejects_out_of_range_ids() -> None:
    from cybernaut_mini.indexing import shard_filename

    with pytest.raises(ValueError, match="out of range"):
        shard_filename(4, 4)


# ------------------------------------------------------------------ #
# Direct write_index round-trips at 1000+ shards                      #
# ------------------------------------------------------------------ #


def _write_one_doc_per_shard_index(
    index_path: Path, n_shards: int, *, build_artifacts: bool = True
) -> None:
    """Write an index of ``n_shards`` single-document shards without running k-means.

    Sharding quality is irrelevant here; what is under test is that shard ids
    survive the file-name round-trip at a scale where three digits are not enough.
    """
    from cybernaut_mini.indexing import write_index
    from cybernaut_mini.models import Document, IndexMeta, ShardManifest

    documents = [
        Document(id=f"doc-{i:05d}", title=f"title {i}", text=f"body text for document {i}")
        for i in range(n_shards)
    ]
    doc_tokens = {doc.id: [f"tok{i}", "shared", f"body{i}"] for i, doc in enumerate(documents)}
    vectors = np.zeros((n_shards, 4), dtype=np.float32)
    vectors[:, 0] = 1.0
    manifests = [
        ShardManifest(
            shard_id=i,
            document_ids=[documents[i].id],
            centroid=[1.0, 0.0, 0.0, 0.0],
            title=f"shard {i}",
            summary=documents[i].title,
            keywords=[],
            entities=[],
            term_graph={},
            document_count=1,
            embedding_model="hash-4",
        )
        for i in range(n_shards)
    ]
    meta = IndexMeta(
        embedding_model="hash-4",
        embedding_dim=4,
        n_shards=n_shards,
        n_documents=n_shards,
        seed=42,
    )
    write_index(
        index_path,
        meta=meta,
        documents=documents,
        vectors=vectors,
        manifests=manifests,
        doc_tokens=doc_tokens,
        build_artifacts=build_artifacts,
    )


def test_shard_ids_round_trip_at_1030_shards(tmp_path: Path) -> None:
    """1030 > 1000, so shard ids need four digits and every one must come back."""
    index_dir = tmp_path / "wide"
    _write_one_doc_per_shard_index(index_dir, 1030)

    loaded = LoadedIndex.load(index_dir)
    assert sorted(loaded.manifests) == list(range(1030))
    for shard_id in (0, 1, 999, 1000, 1029):
        assert loaded.manifests[shard_id].shard_id == shard_id
        assert loaded.manifests[shard_id].document_ids == [f"doc-{shard_id:05d}"]

    names = sorted(p.name for p in (index_dir / "shards").glob("shard_*.json"))
    assert len(names) == 1030
    assert names[0] == "shard_0000.json"
    assert names[-1] == "shard_1029.json"
    # Sorted file order is shard order, which is what the fixed width buys.
    assert [int(name[len("shard_") : -len(".json")]) for name in names] == list(range(1030))


def test_load_ignores_stray_files_and_directory_order(tmp_path: Path) -> None:
    """Loading is driven by meta.n_shards, never by what the directory happens to hold."""
    index_dir = tmp_path / "stray"
    _write_one_doc_per_shard_index(index_dir, 1030, build_artifacts=False)

    (index_dir / "shards" / "shard_9999.json").write_text("{ not json", encoding="utf-8")
    (index_dir / "shards" / "README.txt").write_text("hello", encoding="utf-8")

    loaded = LoadedIndex.load(index_dir)
    assert sorted(loaded.manifests) == list(range(1030))


def test_write_index_rejects_a_manifest_set_that_cannot_be_loaded(tmp_path: Path) -> None:
    from cybernaut_mini.indexing import write_index
    from cybernaut_mini.models import Document, IndexMeta, ShardManifest

    documents = [
        Document(id="doc-a", title="t", text="b"),
        Document(id="doc-b", title="t", text="b"),
    ]
    manifests = [
        ShardManifest(
            shard_id=5,  # not in range(2)
            document_ids=["doc-a", "doc-b"],
            centroid=[1.0],
            title="t",
            summary="s",
            keywords=[],
            entities=[],
            term_graph={},
            document_count=2,
            embedding_model="hash-1",
        )
    ]
    meta = IndexMeta(
        embedding_model="hash-1", embedding_dim=1, n_shards=1, n_documents=2, seed=0
    )
    with pytest.raises(ValueError, match="shard ids must be exactly"):
        write_index(
            tmp_path / "bad",
            meta=meta,
            documents=documents,
            vectors=np.ones((2, 1), dtype=np.float32),
            manifests=manifests,
            doc_tokens={"doc-a": ["a"], "doc-b": ["b"]},
        )


# ------------------------------------------------------------------ #
# Artifact version rejection                                          #
# ------------------------------------------------------------------ #


def test_load_rejects_an_older_valid_marker(tmp_path: Path) -> None:
    index_dir = _build_index(tmp_path)
    (index_dir / "_VALID").write_text("1", encoding="utf-8")

    with pytest.raises(IndexLoadError) as excinfo:
        LoadedIndex.load(index_dir)
    message = str(excinfo.value)
    assert "'1'" in message and "'2'" in message
    assert "cybernaut-mini build" in message, "the error must say how to fix it"


def test_load_rejects_an_older_artifact_version_in_meta(tmp_path: Path) -> None:
    """A _VALID marker alone is not proof; index_meta.json is checked too."""
    index_dir = _build_index(tmp_path)
    meta_file = index_dir / "index_meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["artifact_version"] = "1"
    meta_file.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")

    with pytest.raises(IndexLoadError, match="cybernaut-mini build"):
        LoadedIndex.load(index_dir)


def test_missing_valid_marker_still_names_valid(tmp_path: Path) -> None:
    index_dir = _build_index(tmp_path)
    (index_dir / "_VALID").unlink()
    with pytest.raises(IndexLoadError, match="_VALID"):
        LoadedIndex.load(index_dir)


# ------------------------------------------------------------------ #
# Shard artifacts on disk                                             #
# ------------------------------------------------------------------ #


def test_build_writes_a_shard_artifact_sidecar_per_shard(tmp_path: Path) -> None:
    index_dir = _build_index(tmp_path, n_docs=40, n_shards=4)
    sidecars = sorted((index_dir / "shards" / "artifacts").glob("shard_*.json"))
    assert len(sidecars) == 4

    loaded = LoadedIndex.load(index_dir)
    for shard_id in range(4):
        artifacts = loaded.shard_artifacts(shard_id)
        assert artifacts is not None
        # Every keyword the manifest advertises must be in the shard's own filter.
        for keyword in loaded.manifests[shard_id].keywords[:5]:
            assert keyword.term in artifacts.phrase_bloom
            assert keyword.term in artifacts.vocabulary


def test_manifests_do_not_carry_artifact_payloads_but_can_be_asked_for_them(
    tmp_path: Path,
) -> None:
    """The resident manifest stays small; the full record is one call away."""
    index_dir = _build_index(tmp_path, n_docs=40, n_shards=4)
    loaded = LoadedIndex.load(index_dir)

    assert loaded.manifests[0].vocabulary is None
    assert loaded.manifests[0].phrase_bloom is None

    full = loaded.manifest_with_artifacts(0)
    assert full.phrase_bloom is not None
    assert full.vocabulary is not None
    assert full.document_ids == loaded.manifests[0].document_ids


def test_shard_artifacts_are_read_once_and_then_cached(tmp_path: Path) -> None:
    index_dir = _build_index(tmp_path, n_docs=40, n_shards=4)
    loaded = LoadedIndex.load(index_dir)

    loaded.shard_artifacts(1)
    loaded.shard_artifacts(1)
    loaded.shard_artifacts(2)
    stats = loaded.artifacts.stats()
    assert stats["misses"] == 2, "one disk read per distinct shard"
    assert stats["hits"] == 1


def test_index_without_artifacts_loads_and_reports_none(tmp_path: Path) -> None:
    index_dir = tmp_path / "bare"
    _write_one_doc_per_shard_index(index_dir, 4, build_artifacts=False)
    loaded = LoadedIndex.load(index_dir)
    assert loaded.shard_artifacts(0) is None
    assert loaded.manifest_with_artifacts(0).phrase_bloom is None


# ------------------------------------------------------------------ #
# Laziness of the loaded index                                        #
# ------------------------------------------------------------------ #


def test_loaded_documents_are_a_lazy_sequence_not_a_list(tmp_path: Path) -> None:
    from cybernaut_mini.indexing import StoredDocuments

    index_dir = _build_index(tmp_path, n_docs=12, n_shards=3)
    loaded = LoadedIndex.load(index_dir)

    assert isinstance(loaded.documents, StoredDocuments)
    assert not isinstance(loaded.documents, list)
    assert len(loaded.documents) == 12
    assert [doc.id for doc in loaded.documents] == [f"doc-{i:03d}" for i in range(1, 13)]
    assert loaded.documents[0].id == "doc-001"
    assert loaded.documents[-1].id == "doc-012"
    assert [doc.id for doc in loaded.documents[2:4]] == ["doc-003", "doc-004"]
    with pytest.raises(IndexError):
        loaded.documents[12]


def test_by_id_and_row_map_agree_with_the_written_row_map(tmp_path: Path) -> None:
    """row_map.json stays on disk; the lazy view must return exactly what it says."""
    index_dir = _build_index(tmp_path, n_docs=12, n_shards=3)
    written = json.loads((index_dir / "row_map.json").read_text(encoding="utf-8"))
    loaded = LoadedIndex.load(index_dir)

    assert dict(loaded.row_map) == written
    for doc_id, row in written.items():
        assert loaded.row_map[doc_id] == row
        assert loaded.by_id[doc_id].id == doc_id
    assert "doc-999" not in loaded.by_id
    with pytest.raises(KeyError):
        loaded.by_id["doc-999"]


def test_doc_tokens_behaves_like_the_old_dict(tmp_path: Path) -> None:
    index_dir = _build_index(tmp_path, n_docs=12, n_shards=3)
    loaded = LoadedIndex.load(index_dir)

    tokens = loaded.doc_tokens["doc-001"]
    assert isinstance(tokens, list)
    assert tokens and all(isinstance(token, str) for token in tokens)
    assert loaded.doc_tokens.get("doc-999", []) == []
    assert len(loaded.doc_tokens) == 12
    assert sorted(loaded.doc_tokens) == sorted(f"doc-{i:03d}" for i in range(1, 13))


def test_bm25_is_built_per_shard_on_first_access(tmp_path: Path) -> None:
    index_dir = _build_index(tmp_path, n_docs=12, n_shards=3)
    loaded = LoadedIndex.load(index_dir)

    assert loaded.bm25.stats()["misses"] == 0, "load must not build any BM25 index"
    assert loaded.bm25.built_shard_ids() == []

    first = loaded.bm25[1]
    assert loaded.bm25.built_shard_ids() == [1]
    assert loaded.bm25[1] is first, "second access must reuse the built object"
    assert loaded.bm25.stats() == {
        "hits": 1,
        "misses": 1,
        "evictions": 0,
        "size": 1,
        "maxsize": 32,
    }
    with pytest.raises(KeyError):
        loaded.bm25[99]


def test_bm25_cache_evicts_beyond_its_bound(tmp_path: Path) -> None:
    index_dir = _build_index(tmp_path, n_docs=12, n_shards=3)
    loaded = LoadedIndex.load(index_dir, bm25_cache_size=2)

    for shard_id in (0, 1, 2):
        loaded.bm25[shard_id]
    assert loaded.bm25.stats()["evictions"] == 1
    assert loaded.bm25.built_shard_ids() == [1, 2]


def test_bm25_scores_match_a_directly_built_index(tmp_path: Path) -> None:
    """Laziness must not change a single score."""
    from rank_bm25 import BM25Okapi

    index_dir = _build_index(tmp_path, n_docs=12, n_shards=3)
    loaded = LoadedIndex.load(index_dir)

    for shard_id, manifest in loaded.manifests.items():
        corpus = [loaded.doc_tokens.get(doc_id, []) for doc_id in manifest.document_ids]
        expected = BM25Okapi(corpus).get_scores(["biotech", "headline"])
        actual = loaded.bm25[shard_id].get_scores(["biotech", "headline"])
        assert np.allclose(actual, expected), f"shard {shard_id} scores drifted"


def test_vectors_are_memory_mapped(tmp_path: Path) -> None:
    index_dir = _build_index(tmp_path, n_docs=12, n_shards=3)
    loaded = LoadedIndex.load(index_dir)

    assert isinstance(loaded.vectors, np.memmap)
    assert loaded.vectors.shape == (12, 64)
    assert loaded.vectors.dtype == np.float32
    # Still a real array for the callers that treat it as one.
    assert len(loaded.vectors.tolist()) == 12


def test_loaded_index_can_be_constructed_from_plain_objects(tmp_path: Path) -> None:
    """The in-memory constructor still works, so small fixtures need no index on disk."""
    from cybernaut_mini.indexing import LoadedIndex as LI
    from cybernaut_mini.models import Document, IndexMeta, ShardManifest

    docs = [
        Document(id="d1", title="t1", text="alpha beta"),
        Document(id="d2", title="t2", text="beta gamma"),
    ]
    manifest = ShardManifest(
        shard_id=0,
        document_ids=["d1", "d2"],
        centroid=[1.0, 0.0],
        title="t",
        summary="s",
        keywords=[],
        entities=[],
        term_graph={},
        document_count=2,
        embedding_model="hash-2",
    )
    index = LI(
        meta=IndexMeta(
            embedding_model="hash-2", embedding_dim=2, n_shards=1, n_documents=2, seed=0
        ),
        documents=docs,
        vectors=np.eye(2, dtype=np.float32),
        row_map={"d1": 0, "d2": 1},
        manifests={0: manifest},
        doc_tokens={"d1": ["alpha", "beta"], "d2": ["beta", "gamma"]},
    )
    assert index.by_id["d1"].title == "t1"
    assert len(index.bm25[0].get_scores(["alpha"])) == 2
    assert index.shard_artifacts(0) is None


def test_write_index_names_a_shard_that_references_an_unwritten_document(tmp_path: Path) -> None:
    from cybernaut_mini.indexing import write_index
    from cybernaut_mini.models import Document, IndexMeta, ShardManifest

    manifest = ShardManifest(
        shard_id=0,
        document_ids=["doc-a", "doc-ghost"],
        centroid=[1.0],
        title="t",
        summary="s",
        keywords=[],
        entities=[],
        term_graph={},
        document_count=2,
        embedding_model="hash-1",
    )
    with pytest.raises(ValueError, match="shard 0 references document ids"):
        write_index(
            tmp_path / "ghost",
            meta=IndexMeta(
                embedding_model="hash-1", embedding_dim=1, n_shards=1, n_documents=1, seed=0
            ),
            documents=[Document(id="doc-a", title="t", text="b")],
            vectors=np.ones((1, 1), dtype=np.float32),
            manifests=[manifest],
            doc_tokens={"doc-a": ["a"]},
        )
    assert not (tmp_path / "ghost" / "_VALID").exists()


def test_cache_stats_reports_every_bounded_cache(tmp_path: Path) -> None:
    index_dir = _build_index(tmp_path, n_docs=12, n_shards=3)
    loaded = LoadedIndex.load(index_dir)
    loaded.by_id["doc-001"]
    loaded.doc_tokens["doc-001"]
    loaded.bm25[0]
    loaded.shard_artifacts(0)

    stats = loaded.cache_stats()
    assert set(stats) == {"bm25", "artifacts", "documents", "tokens"}
    assert all(entry["misses"] == 1 for entry in stats.values())


def test_close_releases_the_stores_and_is_idempotent(tmp_path: Path) -> None:
    index_dir = _build_index(tmp_path, n_docs=12, n_shards=3)
    loaded = LoadedIndex.load(index_dir)
    assert loaded.by_id["doc-001"].id == "doc-001"

    loaded.close()
    loaded.close()  # idempotent
    with pytest.raises(ValueError, match="closed"):
        loaded.by_id["doc-002"]
