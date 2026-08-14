"""The term graph belongs to the shard, not to the corpus.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 7: "we use the
    synonym graphs in each shard to probabilistically expand our search terms. Because
    each shard is lexically and semantically coherent, the synonyms in each shard are
    unambiguous. Put simply, 'gene' in shard 11,343 only has the genetic meaning. We
    don't need to worry that Gene is also a name and, in some other shards, would
    co-occur a lot with 'Willy Wonka'."
    Local copy: docs/blog-archive/the-road-to-cybernaut-1.md

These tests exist because the build used to compute ONE graph over the whole corpus
and assign it to every :class:`ShardManifest`. That was two defects wearing one hat:

* scale — 4,072 KB of manifest per shard and 301 MB of load-time Python heap at
  2,000 documents / 32 shards with production parameters (``window=5``,
  ``min_edge_count=2``); and
* fidelity — every shard got every sense of every word, which is exactly the
  ambiguity the sharding is there to remove.

:func:`test_gene_means_something_different_in_each_shard` is the fidelity half, in
the post's own example. :func:`test_shards_with_disjoint_vocabularies_share_no_terms`
and :func:`test_per_shard_manifest_stays_under_its_documented_bound` are the tripwires
that fail if the graph ever goes global again.

Everything runs offline on HashEmbedder; no model is downloaded.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from cybernaut_mini.cli import app
from cybernaut_mini.indexing import (
    TERM_GRAPH_MAX_NEIGHBOURS,
    TERM_GRAPH_MAX_TERMS,
    GlobalTermGraphWarning,
    LoadedIndex,
    compute_shard_term_graphs,
    compute_term_graph,
    foreign_term_graph_terms,
    write_index,
)
from cybernaut_mini.models import Document, IndexMeta, ShardManifest
from cybernaut_mini.text import TextProcessor

runner = CliRunner()

#: Upper bound on one shard's manifest on disk. Measured at the shape the defect was
#: measured at — 2,000 documents / 32 shards, ``cooccurrence_window=5``,
#: ``min_edge_count=2`` — a per-shard capped graph produces a 90.7 KB manifest at the
#: largest and 80.1 KB on average, against 4,072 KB per manifest for the global graph.
#: 256 KB leaves room for the parts of a manifest that legitimately grow with the
#: shard (1,000 document ids and a 384-dimension centroid are ~20 KB together) while
#: still being ~16x below what a single global graph costs.
MANIFEST_BYTES_BOUND = 256 * 1024

#: The hard structural bound the cap gives: at most this many edges in one graph.
MAX_EDGES_PER_SHARD = TERM_GRAPH_MAX_TERMS * TERM_GRAPH_MAX_NEIGHBOURS


# ------------------------------------------------------------------ #
# Fixtures: a corpus with one token that means two different things   #
# ------------------------------------------------------------------ #


def _genetics_docs(n: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"gen-{i:03d}",
            "title": f"Gene expression atlas gsuf{i}",
            "text": (
                "gene expression sequencing genome transcript assay "
                "gene expression sequencing genome transcript assay "
                f"chromosome allele ribosome gsuf{i}"
            ),
            "language": "en",
            "metadata": {"category": "genetics"},
        }
        for i in range(1, n + 1)
    ]


def _film_docs(n: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"flm-{i:03d}",
            "title": f"Gene Wilder retrospective fsuf{i}",
            "text": (
                "gene wilder wonka chocolate factory comedy "
                "gene wilder wonka chocolate factory comedy "
                f"screenplay matinee usherette fsuf{i}"
            ),
            "language": "en",
            "metadata": {"category": "film"},
        }
        for i in range(1, n + 1)
    ]


def _tokens_for(docs: list[dict[str, Any]]) -> dict[str, list[str]]:
    processor = TextProcessor(use_spacy=False)
    return {doc["id"]: processor.content_tokens(f"{doc['title']}\n{doc['text']}") for doc in docs}


# ------------------------------------------------------------------ #
# 1. Fidelity: the post's own 'gene' / 'Gene Wilder' claim            #
# ------------------------------------------------------------------ #


def test_gene_means_something_different_in_each_shard() -> None:
    """Per-shard graphs disambiguate 'gene'; the global graph merges both senses."""
    genetics = _genetics_docs(6)
    film = _film_docs(6)
    tokens = _tokens_for(genetics + film)
    shard_doc_ids = {
        0: [doc["id"] for doc in genetics],
        1: [doc["id"] for doc in film],
    }

    graphs = compute_shard_term_graphs(shard_doc_ids, tokens, window=5, min_edge_count=2)

    genetic_sense = set(graphs[0]["gene"])
    film_sense = set(graphs[1]["gene"])

    # The post's claim, literally: 'gene' in the genetics shard has only the genetic
    # meaning, and nothing from the shard where Gene is a first name.
    assert "expression" in genetic_sense
    assert "wilder" not in genetic_sense
    assert "wonka" not in genetic_sense

    assert "wilder" in film_sense
    assert "expression" not in film_sense
    assert "genome" not in film_sense

    assert genetic_sense.isdisjoint(film_sense)

    # And what the old global graph gave instead: one neighbour set, identical in
    # every manifest, carrying both senses at once.
    global_graph = compute_term_graph(
        [tokens[doc["id"]] for doc in genetics + film], window=5, min_edge_count=2
    )
    both_senses = set(global_graph["gene"])
    assert {"expression", "wilder"} <= both_senses, (
        "the global graph is supposed to be the ambiguous one; if this fails the "
        "fixture no longer demonstrates anything"
    )


def test_expansion_of_gene_differs_by_routed_shard(tmp_path: Path) -> None:
    """The disambiguation survives into stage 7: expansion depends on the shard."""
    from cybernaut_mini.expansion import expand_query

    genetics = _genetics_docs(6)
    film = _film_docs(6)
    docs = [Document.model_validate(raw) for raw in genetics + film]
    tokens = _tokens_for(genetics + film)
    shard_doc_ids = {
        0: [doc["id"] for doc in genetics],
        1: [doc["id"] for doc in film],
    }
    graphs = compute_shard_term_graphs(shard_doc_ids, tokens, window=5, min_edge_count=2)

    vectors = np.zeros((len(docs), 4), dtype=np.float32)
    vectors[:, 0] = 1.0
    manifests = {
        shard_id: ShardManifest(
            shard_id=shard_id,
            document_ids=doc_ids,
            centroid=[1.0, 0.0, 0.0, 0.0],
            title=f"shard-{shard_id}",
            summary="",
            keywords=[],
            entities=[],
            term_graph=graphs[shard_id],
            document_count=len(doc_ids),
            embedding_model="hash-4",
        )
        for shard_id, doc_ids in shard_doc_ids.items()
    }
    index = LoadedIndex(
        meta=IndexMeta(
            embedding_model="hash-4",
            embedding_dim=4,
            n_shards=2,
            n_documents=len(docs),
            seed=0,
        ),
        documents=docs,
        vectors=vectors,
        row_map={doc.id: i for i, doc in enumerate(docs)},
        manifests=manifests,
        doc_tokens=tokens,
    )

    processor = TextProcessor(use_spacy=False)
    genetics_expansion = expand_query(
        index, "gene", shard_ids=[0], processor=processor, max_terms=5
    )
    film_expansion = expand_query(index, "gene", shard_ids=[1], processor=processor, max_terms=5)

    assert "expression" in genetics_expansion
    assert "wilder" not in genetics_expansion
    assert "wilder" in film_expansion
    assert "expression" not in film_expansion


# ------------------------------------------------------------------ #
# 2. Tripwires: the graph must never go global again                  #
# ------------------------------------------------------------------ #


def test_shards_with_disjoint_vocabularies_share_no_terms() -> None:
    """Two shards with nothing in common must produce graphs with nothing in common."""
    shard_doc_ids = {0: ["a1", "a2"], 1: ["b1", "b2"]}
    tokens = {
        "a1": ["alpha", "beta", "gamma"] * 4,
        "a2": ["alpha", "beta", "delta"] * 4,
        "b1": ["kappa", "lambda", "mu"] * 4,
        "b2": ["kappa", "lambda", "nu"] * 4,
    }
    graphs = compute_shard_term_graphs(shard_doc_ids, tokens, window=3, min_edge_count=2)

    assert graphs[0], "shard 0 produced no graph at all"
    assert graphs[1], "shard 1 produced no graph at all"

    terms_0 = set(graphs[0])
    terms_1 = set(graphs[1])
    assert terms_0.isdisjoint(terms_1), (
        f"shards with disjoint vocabularies share graph terms {sorted(terms_0 & terms_1)}; "
        f"the term graph has gone global again"
    )

    # Neighbours too, not just sources.
    neighbours_0 = {n for edges in graphs[0].values() for n in edges}
    neighbours_1 = {n for edges in graphs[1].values() for n in edges}
    assert neighbours_0.isdisjoint(neighbours_1)


def test_every_graph_term_occurs_in_its_own_shard() -> None:
    """The exact property a global graph violates, checked term by term."""
    genetics = _genetics_docs(4)
    film = _film_docs(4)
    tokens = _tokens_for(genetics + film)
    shard_doc_ids = {
        0: [doc["id"] for doc in genetics],
        1: [doc["id"] for doc in film],
    }
    graphs = compute_shard_term_graphs(shard_doc_ids, tokens, window=5, min_edge_count=2)

    for shard_id, doc_ids in shard_doc_ids.items():
        local = {token for doc_id in doc_ids for token in tokens[doc_id]}
        graph = graphs[shard_id]
        for source, edges in graph.items():
            assert source in local, f"shard {shard_id} graph has foreign source {source!r}"
            for neighbour in edges:
                assert neighbour in local, (
                    f"shard {shard_id} graph has foreign neighbour {neighbour!r}"
                )


def test_write_index_warns_when_a_manifest_carries_a_global_graph(tmp_path: Path) -> None:
    """write_index is the funnel every build passes through, so it is the tripwire."""
    genetics = _genetics_docs(3)
    film = _film_docs(3)
    raw_docs = genetics + film
    docs = [Document.model_validate(raw) for raw in raw_docs]
    tokens = _tokens_for(raw_docs)
    shard_doc_ids = {
        0: [doc["id"] for doc in genetics],
        1: [doc["id"] for doc in film],
    }
    global_graph = compute_term_graph([tokens[doc.id] for doc in docs], window=5, min_edge_count=2)
    vectors = np.zeros((len(docs), 4), dtype=np.float32)
    vectors[:, 0] = 1.0
    meta = IndexMeta(
        embedding_model="hash-4",
        embedding_dim=4,
        n_shards=2,
        n_documents=len(docs),
        seed=0,
    )

    def _manifests(graphs: dict[int, dict[str, dict[str, float]]]) -> list[ShardManifest]:
        return [
            ShardManifest(
                shard_id=shard_id,
                document_ids=doc_ids,
                centroid=[1.0, 0.0, 0.0, 0.0],
                title=f"shard-{shard_id}",
                summary="",
                keywords=[],
                entities=[],
                term_graph=graphs[shard_id],
                document_count=len(doc_ids),
                embedding_model="hash-4",
            )
            for shard_id, doc_ids in shard_doc_ids.items()
        ]

    with pytest.warns(GlobalTermGraphWarning, match="term-graph terms do not occur"):
        write_index(
            tmp_path / "global",
            meta=meta,
            documents=docs,
            vectors=vectors,
            manifests=_manifests(dict.fromkeys(shard_doc_ids, global_graph)),
            doc_tokens=tokens,
            build_artifacts=False,
        )

    per_shard = compute_shard_term_graphs(shard_doc_ids, tokens, window=5, min_edge_count=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GlobalTermGraphWarning)
        write_index(
            tmp_path / "per_shard",
            meta=meta,
            documents=docs,
            vectors=vectors,
            manifests=_manifests(per_shard),
            doc_tokens=tokens,
            build_artifacts=False,
        )


def test_foreign_term_report_is_empty_for_a_shard_local_graph() -> None:
    tokens = {"a1": ["alpha", "beta"] * 4, "a2": ["alpha", "gamma"] * 4}
    graphs = compute_shard_term_graphs({0: ["a1", "a2"]}, tokens, window=3, min_edge_count=2)
    manifest = ShardManifest(
        shard_id=0,
        document_ids=["a1", "a2"],
        centroid=[1.0],
        title="t",
        summary="",
        keywords=[],
        entities=[],
        term_graph=graphs[0],
        document_count=2,
        embedding_model="hash-1",
    )
    assert foreign_term_graph_terms(manifest, [tokens["a1"], tokens["a2"]]) == []

    intruder = manifest.model_copy(
        update={"term_graph": {**graphs[0], "wonka": {"chocolate": 1.0}}}
    )
    assert foreign_term_graph_terms(intruder, [tokens["a1"], tokens["a2"]]) == ["wonka"]


# ------------------------------------------------------------------ #
# 3. The cap: bounded, deterministic, and honest about what it drops  #
# ------------------------------------------------------------------ #


def test_cap_bounds_terms_and_neighbours() -> None:
    """A pathologically dense shard still produces a bounded graph."""
    # 400 distinct terms, every one adjacent to every other inside a window.
    tokens = [[f"t{i:03d}" for i in range(400)] for _ in range(3)]
    graph = compute_term_graph(tokens, window=400, min_edge_count=2)

    assert len(graph) == TERM_GRAPH_MAX_TERMS
    assert max(len(edges) for edges in graph.values()) <= TERM_GRAPH_MAX_NEIGHBOURS
    assert sum(len(edges) for edges in graph.values()) <= MAX_EDGES_PER_SHARD

    # Uncapped, the same input is quadratic: 400 * 399 edges.
    uncapped = compute_term_graph(
        tokens, window=400, min_edge_count=2, max_terms=10**9, max_neighbours=10**9
    )
    assert len(uncapped) == 400
    assert sum(len(edges) for edges in uncapped.values()) == 400 * 399


def test_cap_keeps_the_strongest_edges() -> None:
    """Truncation drops the weakest neighbours, not an arbitrary slice."""
    # 'hub' co-occurs with 'near' constantly and with 'far' twice.
    tokens = [["hub", "near"] * 20 + ["hub", "far"] * 1 for _ in range(2)]
    graph = compute_term_graph(tokens, window=2, min_edge_count=2, max_neighbours=1)
    assert graph["hub"] == {"near": 1.0}

    both = compute_term_graph(tokens, window=2, min_edge_count=2, max_neighbours=2)
    assert set(both["hub"]) == {"near", "far"}
    assert both["hub"]["near"] > both["hub"]["far"]


def test_outgoing_weights_sum_to_one_after_truncation() -> None:
    """Normalisation happens after the cap, so weights are a distribution, not a slice."""
    tokens = [[f"t{i:02d}" for i in range(60)] * 2]
    graph = compute_term_graph(tokens, window=60, min_edge_count=2, max_neighbours=4)
    assert graph
    for term, edges in graph.items():
        assert len(edges) <= 4
        assert abs(sum(edges.values()) - 1.0) < 1e-9, f"{term} sums to {sum(edges.values())}"


def test_cap_is_deterministic() -> None:
    tokens = [[f"t{(i * 7) % 300:03d}" for i in range(600)] for _ in range(2)]
    first = compute_term_graph(tokens, window=6, min_edge_count=2)
    second = compute_term_graph(tokens, window=6, min_edge_count=2)
    assert first == second
    assert list(first) == list(second)
    assert list(first) == sorted(first)


def test_the_cap_is_not_free_and_priority_terms_buy_some_of_it_back() -> None:
    """The documented tradeoff, as a test: rare terms lose, unless they are pinned.

    Ranking by co-occurrence count keeps a shard's frequent terms and evicts its rare
    ones — which are exactly the rare-entity queries that most need expansion. The
    build pins each shard's TF-IDF keywords for that reason; here 'zylophristine' is
    the rare entity and pinning is what saves it.
    """
    common = [[f"t{i:02d}" for i in range(40)] * 6 for _ in range(4)]
    rare = [["zylophristine", "cofactor"] * 2]
    tokens = common + rare

    dropped = compute_term_graph(tokens, window=4, min_edge_count=2, max_terms=10)
    assert "zylophristine" not in dropped, (
        "fixture no longer demonstrates the tradeoff: the rare term survived on merit"
    )

    saved = compute_term_graph(
        tokens, window=4, min_edge_count=2, max_terms=10, priority_terms=["zylophristine"]
    )
    assert "zylophristine" in saved
    assert saved["zylophristine"] == {"cofactor": 1.0}
    assert len(saved) == 10, "pinning must not raise the cap, only reorder who fills it"


def test_cap_rejects_nonsense_bounds() -> None:
    with pytest.raises(ValueError, match="max_terms must be >= 1"):
        compute_term_graph([["a", "b"]], window=2, min_edge_count=1, max_terms=0)
    with pytest.raises(ValueError, match="max_neighbours must be >= 1"):
        compute_term_graph([["a", "b"]], window=2, min_edge_count=1, max_neighbours=0)


def test_priority_terms_cannot_exceed_the_term_cap() -> None:
    tokens = [[f"t{i:02d}" for i in range(30)] * 4]
    pinned = [f"t{i:02d}" for i in range(30)]
    graph = compute_term_graph(
        tokens, window=5, min_edge_count=2, max_terms=5, priority_terms=pinned
    )
    assert len(graph) == 5


# ------------------------------------------------------------------ #
# 4. End to end: the real CLI build, and the manifest size bound      #
# ------------------------------------------------------------------ #


def _write_corpus(path: Path, docs: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
    return path


def _write_config(path: Path, n_shards: int) -> Path:
    path.write_text(
        yaml.dump(
            {
                "seed": 42,
                "embedding": {"provider": "hash", "dim": 64},
                "index": {
                    "n_shards": n_shards,
                    "max_keywords": 30,
                    "max_entities": 30,
                    # Production parameters: the shape the defect was measured at.
                    "cooccurrence_window": 5,
                    "min_edge_count": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _varied_corpus(n_per_topic: int = 30) -> list[dict[str, Any]]:
    """A corpus with enough distinct vocabulary that a global graph would be large."""
    topics = {
        "biotech": ["crispr", "genome", "enzyme", "protein", "assay", "clinical", "vaccine"],
        "energy": ["solar", "turbine", "battery", "grid", "photovoltaic", "reactor"],
        "space": ["rover", "telescope", "orbit", "launch", "asteroid", "galaxy"],
        "finance": ["yields", "inflation", "equities", "custody", "arbitrage", "coupon"],
        "climate": ["glacier", "monsoon", "carbon", "permafrost", "drought", "reef"],
        "materials": ["graphene", "lattice", "annealing", "ceramic", "alloy", "wafer"],
    }
    docs: list[dict[str, Any]] = []
    for topic, vocab in sorted(topics.items()):
        for i in range(n_per_topic):
            body = " ".join(vocab[(i + j) % len(vocab)] for j in range(24))
            docs.append(
                {
                    "id": f"{topic}-{i:03d}",
                    "title": f"{topic} report {i}",
                    "text": f"{body} filler{i % 17} filler{i % 23}",
                    "language": "en",
                    "metadata": {"category": topic},
                }
            )
    return docs


@pytest.fixture(scope="module")
def cli_built_index(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real `cybernaut-mini build` through Kedro, reused by the tests below."""
    tmp_path = tmp_path_factory.mktemp("term_graph_build")
    corpus = _write_corpus(tmp_path / "corpus.jsonl", _varied_corpus())
    config = _write_config(tmp_path / "config.yaml", n_shards=12)
    index_dir = tmp_path / "idx"
    result = runner.invoke(
        app,
        [
            "build",
            "--input",
            str(corpus),
            "--index",
            str(index_dir),
            "--config",
            str(config),
            "--offline",
        ],
    )
    assert result.exit_code == 0, f"build failed ({result.exit_code}): {result.output}"
    return index_dir


def test_cli_build_produces_a_graph_per_shard_not_one_shared_graph(
    cli_built_index: Path,
) -> None:
    index = LoadedIndex.load(cli_built_index)
    try:
        graphs = {sid: m.term_graph for sid, m in index.manifests.items()}
        assert any(graphs.values()), "the build produced no term graph at all"

        # No two shards may carry the same graph. Identical graphs across shards is
        # the exact signature of the global-graph defect.
        seen: dict[str, int] = {}
        for shard_id, graph in sorted(graphs.items()):
            if not graph:
                continue
            fingerprint = json.dumps(graph, sort_keys=True)
            assert fingerprint not in seen, (
                f"shards {seen[fingerprint]} and {shard_id} carry an identical term "
                f"graph; the build has gone back to one global graph"
            )
            seen[fingerprint] = shard_id

        # And every term in a shard's graph must come from that shard's own documents.
        for shard_id, graph in graphs.items():
            local: set[str] = set()
            for doc_id in index.manifests[shard_id].document_ids:
                local.update(index.doc_tokens.get(doc_id, []))
            foreign = sorted(term for term in graph if term not in local)
            assert not foreign, f"shard {shard_id} graph has foreign terms {foreign[:5]}"
    finally:
        index.close()


def test_per_shard_manifest_stays_under_its_documented_bound(cli_built_index: Path) -> None:
    shard_files = sorted((cli_built_index / "shards").glob("shard_*.json"))
    assert shard_files, "no shard manifests were written"
    oversized = [
        (path.name, path.stat().st_size)
        for path in shard_files
        if path.stat().st_size > MANIFEST_BYTES_BOUND
    ]
    assert not oversized, (
        f"manifests exceed the {MANIFEST_BYTES_BOUND} byte bound: {oversized}. At 1,000 "
        f"shards that is what decides whether the index fits in memory."
    )


def test_cli_build_graphs_respect_the_cap(cli_built_index: Path) -> None:
    index = LoadedIndex.load(cli_built_index)
    try:
        for shard_id, manifest in index.manifests.items():
            graph = manifest.term_graph
            assert len(graph) <= TERM_GRAPH_MAX_TERMS, f"shard {shard_id} exceeds term cap"
            for term, edges in graph.items():
                assert len(edges) <= TERM_GRAPH_MAX_NEIGHBOURS, (
                    f"shard {shard_id} term {term!r} exceeds the neighbour cap"
                )
                # The manifest is round-tripped through canonical JSON, which rounds
                # the weights, so this is a JSON-precision tolerance, not a slack one.
                assert abs(sum(edges.values()) - 1.0) < 1e-6
    finally:
        index.close()


def test_cli_build_pins_shard_keywords_into_the_graph(cli_built_index: Path) -> None:
    """The keyword pin is wired up, not just available: it survives the real build."""
    index = LoadedIndex.load(cli_built_index)
    try:
        pinned_present = 0
        for manifest in index.manifests.values():
            graph_terms = set(manifest.term_graph)
            if not graph_terms:
                continue
            keyword_terms = {kw.term for kw in manifest.keywords}
            pinned_present += len(keyword_terms & graph_terms)
        assert pinned_present > 0, "no shard keyword appears in its own term graph"
    finally:
        index.close()
