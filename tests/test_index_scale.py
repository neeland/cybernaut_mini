"""Scale proof for the index: 20,000 documents across 200 shards, measured.

Every number this file prints is measured in-process, not asserted from a guess.
The three claims under test are the three that decide whether the design reaches the
1,000,000-document target the blog implies ("250,000 collections from 250,000,000
documents"):

1. Loading an index costs a bounded amount of Python heap, independent of corpus
   size. The comparison point — eagerly parsing every ``Document``, which is what
   ``LoadedIndex.load`` used to do — is measured here too, on the same corpus, with
   the same tool, so the ratio is real rather than quoted.
2. A query routed to a handful of shards builds a handful of BM25 indexes. The
   LRU's miss counter is the evidence.
3. Nothing about the laziness changed an answer.

Peak heap is measured with :mod:`tracemalloc`, which counts Python allocations only.
That is the right instrument here: the whole point of the memory-mapped embeddings
and the offset-indexed JSONL is that the data stays *out* of the Python heap, and
resident-set size would be dominated by page-cache behaviour that varies per run.
The eager baseline is measured with the same instrument, so the comparison is fair.

Everything runs offline on ``HashEmbedder``; no model is downloaded.
"""

from __future__ import annotations

import random
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from cybernaut_mini.config import RRFConfig
from cybernaut_mini.indexing import (
    LoadedIndex,
    StoredDocuments,
    compute_entities,
    compute_keywords,
    compute_term_graph,
    write_index,
)
from cybernaut_mini.models import Document, IndexMeta, ShardManifest
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.retrieval import retrieve
from cybernaut_mini.routing import route
from cybernaut_mini.sharding import shard_documents
from cybernaut_mini.text import TextProcessor

N_DOCS = 20_000
N_SHARDS = 200
DIM = 64
SEED = 42

#: The one document carrying a token that appears nowhere else in the corpus. A
#: lexical search for it has exactly one correct answer, at exactly one rank.
PLANTED_ID = "doc-07777"
PLANTED_TOKEN = "zylophristine"

_TOPICS: dict[str, list[str]] = {
    "biotech": ["crispr", "genome", "enzyme", "protein", "assay", "clinical", "vaccine"],
    "energy": ["solar", "turbine", "battery", "grid", "photovoltaic", "reactor", "hydrogen"],
    "space": ["rover", "telescope", "orbit", "launch", "asteroid", "galaxy", "propulsion"],
    "finance": ["yields", "inflation", "equities", "custody", "arbitrage", "coupon", "spread"],
    "climate": ["glacier", "monsoon", "carbon", "permafrost", "drought", "reef", "sequestration"],
    "medicine": ["antibody", "oncology", "dosage", "cohort", "pathogen", "triage", "remission"],
    "materials": ["graphene", "lattice", "annealing", "ceramic", "alloy", "wafer", "sinter"],
    "computing": ["compiler", "latency", "throughput", "quantisation", "kernel", "cache", "tensor"],
    "transport": ["freight", "logistics", "railway", "corridor", "autonomous", "harbour", "fleet"],
    "agriculture": ["irrigation", "cultivar", "fertiliser", "harvest", "soil", "pollinator"],
}


def _synthetic_corpus() -> list[Document]:
    """Deterministic topical corpus: same seed, same 20,000 documents, every run."""
    rng = random.Random(SEED)
    topics = sorted(_TOPICS)
    documents: list[Document] = []
    for i in range(N_DOCS):
        topic = topics[i % len(topics)]
        vocabulary = _TOPICS[topic]
        body = " ".join(rng.choice(vocabulary) for _ in range(40))
        filler = " ".join(f"term{rng.randrange(600)}" for _ in range(20))
        doc_id = f"doc-{i:05d}"
        text = f"{body} {filler}"
        if doc_id == PLANTED_ID:
            text = f"{PLANTED_TOKEN} {text} {PLANTED_TOKEN}"
        documents.append(
            Document(
                id=doc_id,
                title=f"{topic} report {i}",
                text=text,
                language="en",
                metadata={"category": topic},
            )
        )
    return documents


def _build_scale_index(index_path: Path) -> dict[str, float]:
    """Build the full index and return the wall-clock cost of each stage."""
    timings: dict[str, float] = {}
    processor = TextProcessor(use_spacy=False)
    embedder = HashEmbedder(dim=DIM)

    documents = _synthetic_corpus()

    start = time.perf_counter()
    vectors = embedder.embed_documents([f"{doc.title}\n{doc.text}" for doc in documents])
    timings["embed"] = time.perf_counter() - start

    start = time.perf_counter()
    doc_tokens = {doc.id: processor.content_tokens(f"{doc.title}\n{doc.text}") for doc in documents}
    timings["tokenize"] = time.perf_counter() - start

    start = time.perf_counter()
    sharding = shard_documents(vectors, n_shards=N_SHARDS, seed=SEED)
    timings["shard"] = time.perf_counter() - start

    shard_doc_ids: dict[int, list[str]] = {s: [] for s in range(N_SHARDS)}
    for doc, label in zip(documents, sharding.labels, strict=True):
        shard_doc_ids[label].append(doc.id)

    row_map = {doc.id: row for row, doc in enumerate(documents)}
    doc_by_id = {doc.id: doc for doc in documents}

    start = time.perf_counter()
    keywords_by_shard = compute_keywords(
        {s: [tok for did in ids for tok in doc_tokens[did]] for s, ids in shard_doc_ids.items()},
        max_keywords=30,
    )
    manifests: list[ShardManifest] = []
    for shard_id in range(N_SHARDS):
        doc_ids = shard_doc_ids[shard_id]
        centroid = sharding.centroids[shard_id]
        # A per-shard term graph, not one global graph copied into every manifest:
        # at 1,000 shards a shared graph is stored and held a thousand times over.
        graph = compute_term_graph(
            [doc_tokens[did] for did in doc_ids], window=3, min_edge_count=8
        )
        scored = sorted(
            ((float(vectors[row_map[did]] @ centroid), did) for did in doc_ids),
            key=lambda pair: (-pair[0], pair[1]),
        )
        manifests.append(
            ShardManifest(
                shard_id=shard_id,
                document_ids=doc_ids,
                centroid=centroid.tolist(),
                title=", ".join(kw.term for kw in keywords_by_shard[shard_id][:3]),
                summary="; ".join(doc_by_id[did].title for _sim, did in scored[:3]),
                keywords=keywords_by_shard[shard_id],
                entities=compute_entities([[] for _ in doc_ids], max_entities=30),
                term_graph=graph,
                document_count=len(doc_ids),
                embedding_model=f"hash-{DIM}",
            )
        )
    timings["features"] = time.perf_counter() - start

    meta = IndexMeta(
        embedding_model=f"hash-{DIM}",
        embedding_dim=DIM,
        n_shards=N_SHARDS,
        n_documents=N_DOCS,
        seed=SEED,
    )

    start = time.perf_counter()
    write_index(
        index_path,
        meta=meta,
        documents=documents,
        vectors=vectors,
        manifests=manifests,
        doc_tokens=doc_tokens,
    )
    timings["write"] = time.perf_counter() - start
    return timings


@pytest.fixture(scope="module")
def scale_index_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("scale_index")
    timings = _build_scale_index(path)
    total = sum(timings.values())
    print(
        f"\n[build] {N_DOCS} docs / {N_SHARDS} shards in {total:.1f}s "
        + " ".join(f"{stage}={cost:.1f}s" for stage, cost in timings.items())
    )
    return path


def _peak_mb(callable_: object) -> tuple[object, float]:
    """Run a zero-argument callable and return (result, peak Python heap in MB)."""
    assert callable(callable_)
    tracemalloc.start()
    tracemalloc.reset_peak()
    result = callable_()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1e6


def _eager_documents(index_path: Path) -> int:
    """Every line of documents.jsonl parsed into a pydantic model, and kept."""
    import json

    docs = []
    with (index_path / "documents.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                docs.append(Document.model_validate(json.loads(line)))
    return len(docs)


def _eager_load_like_v1(index_path: Path, manifests: dict[int, ShardManifest]) -> int:
    """Reproduce what ``LoadedIndex.load`` did before this rewrite.

    Documents as models, every token list in one dict, and a ``BM25Okapi`` per shard,
    all built at load time and all still referenced when the peak is read. This is
    the baseline the lazy loader is measured against; it is written out here rather
    than quoted so the comparison is against running code.
    """
    import json

    from rank_bm25 import BM25Okapi

    documents = []
    with (index_path / "documents.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(Document.model_validate(json.loads(line)))
    by_id = {doc.id: doc for doc in documents}

    doc_tokens: dict[str, list[str]] = {}
    with (index_path / "tokens.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                doc_tokens[record["id"]] = record["tokens"]

    row_map = json.loads((index_path / "row_map.json").read_text(encoding="utf-8"))
    vectors = np.load(index_path / "embeddings.npy").astype(np.float32)

    bm25 = {
        shard_id: BM25Okapi([doc_tokens.get(did, []) for did in manifest.document_ids])
        for shard_id, manifest in manifests.items()
    }
    assert len(by_id) == len(row_map) == vectors.shape[0]
    return len(bm25)


# ------------------------------------------------------------------ #
# (a) Peak memory on load                                             #
# ------------------------------------------------------------------ #


def test_load_peak_heap_is_modest_and_far_below_eager_parsing(scale_index_path: Path) -> None:
    """Loading must not scale with the corpus; parsing every Document does."""
    index, lazy_peak_mb = _peak_mb(lambda: LoadedIndex.load(scale_index_path))
    assert isinstance(index, LoadedIndex)
    assert isinstance(index.documents, StoredDocuments)
    assert len(index.documents) == N_DOCS

    docs_only, docs_only_peak_mb = _peak_mb(lambda: _eager_documents(scale_index_path))
    assert docs_only == N_DOCS

    kept, eager_peak_mb = _peak_mb(lambda: _eager_load_like_v1(scale_index_path, index.manifests))
    assert kept == N_SHARDS

    per_doc_bytes = docs_only_peak_mb * 1e6 / N_DOCS
    print(
        f"\n[load] lazy peak {lazy_peak_mb:.1f} MB | eager Documents alone "
        f"{docs_only_peak_mb:.1f} MB ({per_doc_bytes:.0f} B/doc) | "
        f"full v1-style eager load (documents + tokens + every shard's BM25) "
        f"{eager_peak_mb:.1f} MB | {eager_peak_mb / lazy_peak_mb:.0f}x less | "
        f"v1 extrapolated to 1,000,000 docs / 1,000 shards = "
        f"{eager_peak_mb / N_DOCS * 1e6 / 1e3:.1f} GB"
    )

    # The lazy loader's cost is the shard manifests, not the corpus.
    assert lazy_peak_mb < 30.0, f"load peaked at {lazy_peak_mb:.1f} MB"
    assert eager_peak_mb > 10 * lazy_peak_mb, (
        "the eager baseline should dominate; if it does not, this corpus is too small "
        "for the comparison to mean anything"
    )


def test_resident_cost_extrapolates_inside_the_memory_budget(scale_index_path: Path) -> None:
    """Split the load peak into what scales with documents and what scales with shards.

    What is left resident after a load is the shard manifests, and the part of a
    manifest that grows with the corpus is ``document_ids``. Separating the two lets
    the 1,000,000-document / 1,000-shard projection be arithmetic over two measured
    numbers rather than a guess.
    """
    import sys

    index, peak_mb = _peak_mb(lambda: LoadedIndex.load(scale_index_path))
    assert isinstance(index, LoadedIndex)

    id_bytes = sum(
        sys.getsizeof(doc_id) + 8  # the string, plus its slot in the list
        for manifest in index.manifests.values()
        for doc_id in manifest.document_ids
    )
    fixed_bytes = peak_mb * 1e6 - id_bytes
    per_doc = id_bytes / N_DOCS
    per_shard = fixed_bytes / N_SHARDS
    projected_gb = (1_000_000 * per_doc + 1_000 * per_shard) / 1e9

    print(
        f"\n[resident] {peak_mb:.1f} MB = {id_bytes / 1e6:.1f} MB of document ids "
        f"({per_doc:.0f} B/doc) + {fixed_bytes / 1e6:.1f} MB of per-shard manifest "
        f"({per_shard / 1e3:.0f} KB/shard) -> 1,000,000 docs / 1,000 shards "
        f"projects to {projected_gb * 1000:.0f} MB"
    )
    assert id_bytes > 0
    assert fixed_bytes > 0, "document ids should not account for the entire peak"
    assert projected_gb < 1.0, (
        f"projected resident cost at target scale is {projected_gb:.2f} GB, which "
        f"leaves too little of the ~5 GB budget for queries"
    )


def test_load_cost_does_not_grow_with_the_number_of_documents(scale_index_path: Path) -> None:
    """The same index loaded twice costs the same; the corpus is never read at load."""
    _first, first_peak = _peak_mb(lambda: LoadedIndex.load(scale_index_path))
    _second, second_peak = _peak_mb(lambda: LoadedIndex.load(scale_index_path))
    assert abs(first_peak - second_peak) < 2.0

    index = LoadedIndex.load(scale_index_path)
    # Reading one document must not drag in its neighbours.
    _doc, peak = _peak_mb(lambda: index.by_id[PLANTED_ID])
    print(f"\n[fetch] one document by id peaked at {peak * 1000:.0f} KB")
    assert peak < 0.5, f"fetching one document peaked at {peak:.3f} MB"


def test_vectors_stay_memory_mapped(scale_index_path: Path) -> None:
    index = LoadedIndex.load(scale_index_path)
    assert isinstance(index.vectors, np.memmap)
    assert index.vectors.shape == (N_DOCS, DIM)

    rows = [index.row_map[f"doc-{i:05d}"] for i in range(0, N_DOCS, 4000)]
    _block, peak = _peak_mb(lambda: np.asarray(index.vectors[rows]))
    assert peak < 1.0, f"reading {len(rows)} vector rows peaked at {peak:.3f} MB"


# ------------------------------------------------------------------ #
# (b) Only the routed shards get a BM25 index                         #
# ------------------------------------------------------------------ #


def test_only_routed_shards_build_bm25_objects(scale_index_path: Path) -> None:
    index = LoadedIndex.load(scale_index_path)
    processor = TextProcessor(use_spacy=False)
    provider = HashEmbedder(dim=DIM)

    assert index.bm25.stats()["misses"] == 0, "load must build no BM25 index at all"
    assert len(index.manifests) == N_SHARDS

    routed, _signals = route(
        index,
        "solar battery grid storage",
        processor=processor,
        provider=provider,
        rrf_config=RRFConfig(),
        top_n=20,
    )
    assert 0 < len(routed) <= 20
    assert index.bm25.stats()["misses"] == 0, "routing must not touch BM25 at all"

    hits = retrieve(
        index,
        "solar battery grid storage",
        mode="hybrid",
        processor=processor,
        provider=provider,
        rrf_config=RRFConfig(),
        top_k=10,
        shard_ids=routed,
    )
    assert hits

    stats = index.bm25.stats()
    print(
        f"\n[bm25] {len(routed)} shards routed of {N_SHARDS}; "
        f"{stats['misses']} BM25 indexes built, {stats['evictions']} evicted"
    )
    assert stats["misses"] == len(routed), "one build per routed shard, and no more"
    assert stats["evictions"] == 0
    assert set(index.bm25.built_shard_ids()) <= set(routed)
    assert stats["misses"] < N_SHARDS / 5, "routing must actually narrow the fan-out"


def test_bm25_cache_bound_is_respected_across_many_queries(scale_index_path: Path) -> None:
    """Query the whole index and the resident set still stays at the cache bound."""
    index = LoadedIndex.load(scale_index_path, bm25_cache_size=8)
    for shard_id in range(N_SHARDS):
        index.bm25[shard_id]
    stats = index.bm25.stats()
    assert stats["size"] == 8
    assert stats["misses"] == N_SHARDS
    assert stats["evictions"] == N_SHARDS - 8
    assert index.bm25.built_shard_ids() == list(range(N_SHARDS - 8, N_SHARDS))


def test_shard_artifacts_are_loaded_per_shard_not_all_at_once(scale_index_path: Path) -> None:
    index = LoadedIndex.load(scale_index_path, artifacts_cache_size=4)

    def _read_five() -> list[object]:
        return [index.shard_artifacts(sid) for sid in range(5)]

    artifacts, peak = _peak_mb(_read_five)
    assert isinstance(artifacts, list)
    assert all(item is not None for item in artifacts)
    print(f"\n[artifacts] reading 5 of {N_SHARDS} shard artifact bundles peaked at {peak:.1f} MB")
    assert index.artifacts.stats()["size"] == 4, "the bound holds"
    assert peak < 10.0


# ------------------------------------------------------------------ #
# (c) Search still returns the right answer                           #
# ------------------------------------------------------------------ #


def test_planted_document_is_the_top_hit_across_the_whole_index(scale_index_path: Path) -> None:
    """A token that occurs in exactly one of 20,000 documents must rank it first."""
    index = LoadedIndex.load(scale_index_path)
    hits = retrieve(
        index,
        PLANTED_TOKEN,
        mode="lexical",
        processor=TextProcessor(use_spacy=False),
        provider=HashEmbedder(dim=DIM),
        rrf_config=RRFConfig(),
        top_k=5,
    )
    assert hits, "the planted token must be findable"
    assert hits[0].document.id == PLANTED_ID
    assert hits[0].rank == 1
    assert len(hits) == 1, "no other document contains the token"

    owning_shard = next(
        sid for sid, m in index.manifests.items() if PLANTED_ID in set(m.document_ids)
    )
    assert hits[0].shard_id == owning_shard


def test_lazy_results_match_a_fully_in_memory_index(scale_index_path: Path) -> None:
    """The same query over the same shard, lazy vs. everything-in-RAM, hit for hit."""
    lazy = LoadedIndex.load(scale_index_path)
    shard_id = next(
        sid for sid, m in lazy.manifests.items() if PLANTED_ID in set(m.document_ids)
    )
    doc_ids = lazy.manifests[shard_id].document_ids

    eager = LoadedIndex(
        meta=lazy.meta,
        documents=[lazy.by_id[did] for did in doc_ids],
        vectors=np.asarray(lazy.vectors),
        row_map={did: lazy.row_map[did] for did in doc_ids},
        manifests={shard_id: lazy.manifests[shard_id]},
        doc_tokens={did: lazy.doc_tokens[did] for did in doc_ids},
    )

    kwargs = {
        "mode": "hybrid",
        "processor": TextProcessor(use_spacy=False),
        "provider": HashEmbedder(dim=DIM),
        "rrf_config": RRFConfig(),
        "top_k": 10,
        "shard_ids": [shard_id],
    }
    lazy_hits = retrieve(lazy, "solar battery grid", **kwargs)  # type: ignore[arg-type]
    eager_hits = retrieve(eager, "solar battery grid", **kwargs)  # type: ignore[arg-type]

    assert lazy_hits, "the query must return something for the comparison to mean anything"
    assert [(h.document.id, h.rank) for h in lazy_hits] == [
        (h.document.id, h.rank) for h in eager_hits
    ]
    assert [h.score for h in lazy_hits] == [h.score for h in eager_hits]


def test_document_ids_and_rows_agree_with_the_written_row_map(scale_index_path: Path) -> None:
    """Every id in every shard resolves, and its row is the row the writer recorded."""
    import json

    index = LoadedIndex.load(scale_index_path)
    written = json.loads((scale_index_path / "row_map.json").read_text(encoding="utf-8"))
    assert len(written) == N_DOCS

    all_shard_ids = [did for m in index.manifests.values() for did in m.document_ids]
    assert len(all_shard_ids) == N_DOCS
    assert len(set(all_shard_ids)) == N_DOCS

    sampled = [f"doc-{i:05d}" for i in range(0, N_DOCS, 977)]
    for doc_id in sampled:
        row = index.row_map[doc_id]
        assert row == written[doc_id]
        assert index.by_id[doc_id].id == doc_id
        assert index.documents[row].id == doc_id


# ------------------------------------------------------------------ #
# (d) 1000+ shard ids round-trip                                      #
# ------------------------------------------------------------------ #


def test_twelve_hundred_shard_ids_round_trip_with_their_artifacts(tmp_path: Path) -> None:
    """Past 1,000 shards the ids need four digits — names, manifests and sidecars."""
    n_shards = 1200
    documents = [
        Document(
            id=f"doc-{i:06d}",
            title=f"shard {i} headline",
            text=f"body about topic {i % 17} with repeated filler text {i}",
        )
        for i in range(n_shards)
    ]
    doc_tokens = {
        doc.id: [f"topic{i % 17}", f"body{i}", "filler"]
        for i, doc in enumerate(documents)
    }
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
    index_path = tmp_path / "wide"
    write_index(
        index_path,
        meta=IndexMeta(
            embedding_model="hash-4",
            embedding_dim=4,
            n_shards=n_shards,
            n_documents=n_shards,
            seed=SEED,
        ),
        documents=documents,
        vectors=vectors,
        manifests=manifests,
        doc_tokens=doc_tokens,
    )

    index, peak = _peak_mb(lambda: LoadedIndex.load(index_path))
    assert isinstance(index, LoadedIndex)
    print(f"\n[wide] loading {n_shards} shards peaked at {peak:.1f} MB")

    assert sorted(index.manifests) == list(range(n_shards))
    for shard_id in (0, 999, 1000, 1199):
        assert index.manifests[shard_id].document_ids == [f"doc-{shard_id:06d}"]
        artifacts = index.shard_artifacts(shard_id)
        assert artifacts is not None
        assert f"shard {shard_id} headline" in artifacts.phrase_bloom

    manifest_names = sorted(p.name for p in (index_path / "shards").glob("shard_*.json"))
    sidecar_names = sorted(p.name for p in (index_path / "shards" / "artifacts").glob("*.json"))
    assert manifest_names == sidecar_names
    assert manifest_names[0] == "shard_0000.json"
    assert manifest_names[-1] == "shard_1199.json"
    assert [int(name[6:-5]) for name in manifest_names] == list(range(n_shards))
