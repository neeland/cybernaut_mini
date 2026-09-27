"""Streaming ingest: nearest-centroid routing, per-shard append, growth-curve evals.

Replays a slice of the committed real fixture corpus (MIRACL docs judged by the
committed qrels plus CC-News fillers) through
:mod:`cybernaut_mini.entities.ingest_stream`: a bootstrap clustered build, then
simulated days appended by nearest-frozen-centroid routing, with the entity loop's
sampled discovery and importance-weighted flushes riding the stream, and the
evaluation harness re-run at every corpus size. Routing geometry is unit-tested on
numeric arrays (pure math).

Blog ref: https://nosible.com/blog/introducing-cybernaut-1-agentic-search-with-mcts
    — quality stability "even as we continue expanding our web coverage (currently
    growing at ~20 million webpages per day)"; and
    https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize — "Every
    new document we index is written to the collection it belongs to and every so
    often those writes are flushed to disk." Local copies under
    ``docs/blog-archive/``.

Assumptions:
    - The judgment set must be fixed across increments, so the tests assert the
      filter (:func:`usable_judgments`) against the *planned* stream and that every
      day's metrics were computed over the same modes — the curve's only moving
      variable is corpus size.
    - Day sizes are scaled to the 460-document fixture (base 20, 15/day) — the gap
      analysis' 5-10k/day is the same code path with a larger constant.
    - Sampling at rate 1.0 in the entity-loop test makes the X% seam deterministic
      without patching the RNG; the rate, not the coin, is under test.

Alternatives rejected:
    - Asserting nDCG stays flat as the corpus grows: early days legitimately score
      lower before relevant documents arrive; the tests pin monotone corpus growth,
      metric presence and bounds, not a quality claim the fixture cannot support.
    - Disk round-trips through ``write_index``: the on-disk layout has its own
      tests; the in-memory ``LoadedIndex`` is what the curve measures.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cybernaut_mini.config import AppConfig
from cybernaut_mini.entities.flush import FlushBuffer
from cybernaut_mini.entities.ingest_stream import (
    DayReport,
    EntityStreamState,
    StreamingIndex,
    dump_curve,
    route_to_centroids,
    simulate_stream,
    usable_judgments,
)
from cybernaut_mini.entities.store import EntityStore
from cybernaut_mini.entities.tagger import CapitalizedSpanDiscovery
from cybernaut_mini.ingest import load_documents
from cybernaut_mini.models import Document, Judgment, canonical_dumps
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.text import TextProcessor

FIXTURE_DOCS = Path("data/01_raw/fixtures/documents.jsonl")
FIXTURE_JUDGMENTS = Path("data/01_raw/fixtures/judgments.jsonl")


@pytest.fixture(scope="module")
def fixture_documents() -> list[Document]:
    return load_documents(FIXTURE_DOCS)


@pytest.fixture(scope="module")
def fixture_judgments() -> list[Judgment]:
    return [
        Judgment.model_validate(json.loads(line))
        for line in FIXTURE_JUDGMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def stream_slice(
    fixture_documents: list[Document], fixture_judgments: list[Judgment]
) -> list[Document]:
    """The docs judged by the first two real qrels, plus 30 real CC-News fillers."""
    wanted: set[str] = set()
    for judgment in fixture_judgments[:2]:
        wanted |= set(judgment.relevant_document_ids)
    by_id = {document.id: document for document in fixture_documents}
    subset = [by_id[doc_id] for doc_id in sorted(wanted) if doc_id in by_id]
    subset += [
        document
        for document in fixture_documents
        if document.id.startswith("ccn-") and document.id not in wanted
    ][:30]
    return subset


# ------------------------------------------------------------------ #
# Routing geometry (pure math on numeric arrays)                      #
# ------------------------------------------------------------------ #


def test_route_to_centroids_picks_nearest_by_cosine() -> None:
    centroids = np.eye(3, dtype=np.float32)
    vectors = np.asarray(
        [
            [0.9, 0.1, 0.0],
            [0.0, 0.8, 0.2],
            [0.1, 0.0, 0.99],
        ],
        dtype=np.float32,
    )
    assert route_to_centroids(vectors, centroids) == [0, 1, 2]


def test_route_to_centroids_breaks_ties_to_lowest_shard_and_handles_empty() -> None:
    centroids = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
    assert route_to_centroids(vectors, centroids) == [0]
    empty = np.zeros((0, 2), dtype=np.float32)
    assert route_to_centroids(empty, centroids) == []


# ------------------------------------------------------------------ #
# StreamingIndex: bootstrap once, append thereafter                   #
# ------------------------------------------------------------------ #


def test_streaming_index_appends_without_recustering(stream_slice: list[Document]) -> None:
    embedder = HashEmbedder(dim=32)
    processor = TextProcessor(use_spacy=False)

    def embed(documents: list[Document]) -> np.ndarray:
        return embedder.embed_documents(
            [f"{document.title}\n{document.text}" for document in documents]
        )

    base, arrivals = stream_slice[:20], stream_slice[20:30]
    stream = StreamingIndex.bootstrap(
        base,
        embed(base),
        n_shards=4,
        seed=42,
        embedding_model=embedder.identifier,
        processor=processor,
    )
    centroids_before = stream.centroids.copy()
    assert stream.corpus_size == 20
    assert stream.dirty_shards() == (0, 1, 2, 3)

    index = stream.to_loaded_index()
    assert stream.dirty_shards() == ()  # building clears dirtiness
    assert index.meta.n_documents == 20
    assert len(index.manifests) == 4

    labels = stream.append(arrivals, embed(arrivals))
    assert len(labels) == 10
    assert stream.corpus_size == 30
    # Frozen centroids: appending must not move the routing geometry.
    assert np.array_equal(stream.centroids, centroids_before)
    # Only the shards that received documents are dirty.
    assert stream.dirty_shards() == tuple(sorted(set(labels)))

    grown = stream.to_loaded_index()
    assert grown.meta.n_documents == 30
    for document, label in zip(arrivals, labels, strict=True):
        assert document.id in grown.manifests[label].document_ids
        assert document.id in stream.shard_document_ids(label)


def test_streaming_index_rejects_mismatched_lengths(stream_slice: list[Document]) -> None:
    embedder = HashEmbedder(dim=32)
    with pytest.raises(ValueError, match="same length"):
        StreamingIndex(
            documents=stream_slice[:3],
            vectors=embedder.embed_documents(["only", "two"]),
            labels=[0, 0, 0],
            centroids=np.eye(2, dtype=np.float32),
            seed=42,
            embedding_model=embedder.identifier,
            processor=TextProcessor(use_spacy=False),
        )


# ------------------------------------------------------------------ #
# The fixed judgment set and the growth curve                         #
# ------------------------------------------------------------------ #


def test_usable_judgments_requires_a_positive_doc_in_the_planned_stream(
    fixture_judgments: list[Judgment], stream_slice: list[Document]
) -> None:
    planned = [document.id for document in stream_slice]
    usable = usable_judgments(fixture_judgments, planned)
    assert 0 < len(usable) < len(fixture_judgments)
    planned_set = set(planned)
    for judgment in usable:
        assert any(
            grade > 0 and doc_id in planned_set
            for doc_id, grade in judgment.relevant_document_ids.items()
        )
    assert usable_judgments(fixture_judgments, []) == []


def test_simulate_stream_produces_the_growth_curve(
    stream_slice: list[Document], fixture_judgments: list[Judgment]
) -> None:
    entity_state = EntityStreamState(
        store=EntityStore(),
        buffer=FlushBuffer(),
        discovery=CapitalizedSpanDiscovery(),
        sample_rate=1.0,  # deterministic: every chunk goes through discovery
    )
    reports = simulate_stream(
        stream_slice,
        fixture_judgments,
        provider=HashEmbedder(dim=32),
        processor=TextProcessor(use_spacy=False),
        config=AppConfig(seed=42),
        base_size=20,
        docs_per_day=15,
        n_shards=4,
        seed=42,
        days=2,
        modes=("hybrid",),
        entity_state=entity_state,
    )
    assert [report.day for report in reports] == [0, 1, 2]
    assert reports[0].corpus_size == 20
    assert reports[0].touched_shards == (0, 1, 2, 3)
    sizes = [report.corpus_size for report in reports]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3
    assert [report.added for report in reports] == [20, 15, 15]
    for report in reports:
        assert set(report.metrics) == {"hybrid"}
        assert 0.0 <= report.ndcg_at_10("hybrid") <= 1.0
        assert report.seconds >= 0.0
        # Sampling at 1.0 means every arriving chunk hit discovery.
        assert report.sampled_chunks == report.added
        assert report.flushed_collections >= 1
    later_touched = {shard for report in reports[1:] for shard in report.touched_shards}
    assert later_touched <= {0, 1, 2, 3}


def test_simulate_stream_validates_arguments(
    stream_slice: list[Document], fixture_judgments: list[Judgment]
) -> None:
    common = {
        "provider": HashEmbedder(dim=32),
        "processor": TextProcessor(use_spacy=False),
        "config": AppConfig(seed=42),
    }
    with pytest.raises(ValueError, match="base_size"):
        simulate_stream(
            stream_slice, fixture_judgments, base_size=2, n_shards=4, **common
        )
    with pytest.raises(ValueError, match="docs_per_day"):
        simulate_stream(
            stream_slice, fixture_judgments, base_size=20, docs_per_day=0, **common
        )


def test_simulate_ingest_script_writes_curve_and_prints_table(
    tmp_path: Path, fixture_judgments: list[Judgment]
) -> None:
    """The driver script end to end, offline: fixture corpus, two real judgments."""
    judgments_path = tmp_path / "judgments.jsonl"
    judgments_path.write_text(
        "\n".join(judgment.model_dump_json() for judgment in fixture_judgments[:2]) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "curve.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/simulate_ingest.py",
            "--judgments",
            str(judgments_path),
            "--base-size",
            "20",
            "--docs-per-day",
            "15",
            "--days",
            "1",
            "--n-shards",
            "4",
            "--dim",
            "32",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parent.parent,
    )
    assert "ndcg@10/hybrid" in result.stdout
    assert f"curve written to {out}" in result.stdout
    curve = json.loads(out.read_text(encoding="utf-8"))
    assert [point["day"] for point in curve] == [0, 1]
    assert curve[0]["corpus_size"] == 20
    assert curve[1]["corpus_size"] == 35


def test_dump_curve_is_canonical_json() -> None:
    report = DayReport(
        day=0,
        corpus_size=20,
        added=20,
        touched_shards=(0, 1),
        metrics={"hybrid": {"ndcg_at_10": 0.5}},
        sampled_chunks=3,
        flushed_collections=1,
        tags_written=0,
        seconds=0.01,
    )
    curve = dump_curve([report])
    assert curve == canonical_dumps([report.as_dict()])
    assert json.loads(curve)[0]["corpus_size"] == 20
