from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from cybernaut_mini.indexing import LoadedIndex, write_index
from cybernaut_mini.models import Document, IndexMeta, ShardManifest
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.sharding import shard_documents
from cybernaut_mini.text import TextProcessor


@pytest.fixture()
def text_processor() -> TextProcessor:
    return TextProcessor(use_spacy=False)


@pytest.fixture()
def hash_embedder() -> HashEmbedder:
    return HashEmbedder(dim=64)


# ------------------------------------------------------------------ #
# Session-scoped built index for retrieval tests                     #
# ------------------------------------------------------------------ #

_CORPUS: list[Document] = [
    # Planted: lexical target — rare token "zylophristine" appears only here.
    Document(
        id="doc-lex",
        title="Zylophristine compound assay",
        text=(
            "Zylophristine compound assay methods are used to quantify trace amounts "
            "of zylophristine in biological samples using spectrophotometry."
        ),
        language="en",
        published_at=datetime(2024, 3, 1, tzinfo=UTC),
        metadata={"category": "chemistry"},
    ),
    # Planted: dense target — "genes"/"editing" share NO exact BM25 token with query "gene edit".
    Document(
        id="doc-para",
        title="Genes editing genomes precisely",
        text="Genes editing genomes precisely with molecular scissors technologies.",
        language="en",
        published_at=datetime(2024, 6, 15, tzinfo=UTC),
        metadata={"category": "biotech"},
    ),
    # Biotech docs with language "en".
    Document(
        id="doc-bt-01",
        title="CRISPR screening maps immune pathways",
        text="CRISPR screening maps immune pathways in T-cell activation studies.",
        language="en",
        published_at=datetime(2023, 2, 10, tzinfo=UTC),
        metadata={"category": "biotech"},
    ),
    Document(
        id="doc-bt-02",
        title="Gut microbiome shifts alter immune response",
        text="Gut microbiome shifts alter immune response through cytokine signalling.",
        language="en",
        published_at=datetime(2023, 8, 20, tzinfo=UTC),
        metadata={"category": "biotech"},
    ),
    # German document — must be excluded by language filter "en".
    Document(
        id="doc-de-01",
        title="Solarzellen Effizienz Rekord",
        text="Solarzellen erreichen neuen Effizienzrekord durch verbesserte Halbleitermaterialien.",
        language="de",
        published_at=datetime(2024, 1, 5, tzinfo=UTC),
        metadata={"category": "energy"},
    ),
    # Energy docs.
    Document(
        id="doc-en-01",
        title="Solar panel efficiency reaches new record",
        text="Solar panel efficiency reaches new record with advanced semiconductor coatings.",
        language="en",
        published_at=datetime(2023, 5, 12, tzinfo=UTC),
        metadata={"category": "energy"},
    ),
    Document(
        id="doc-en-02",
        title="Grid storage batteries stabilize renewable output",
        text="Grid storage batteries stabilize renewable output across peak demand periods.",
        language="en",
        published_at=datetime(2023, 11, 7, tzinfo=UTC),
        metadata={"category": "energy"},
    ),
    Document(
        id="doc-en-03",
        title="Offshore wind farms expand capacity",
        text="Offshore wind farms expand capacity with floating turbine platforms.",
        language="en",
        published_at=datetime(2024, 4, 22, tzinfo=UTC),
        metadata={"category": "energy"},
    ),
    # Space docs.
    Document(
        id="doc-sp-01",
        title="Rover samples suggest ancient river delta",
        text="Rover samples suggest ancient river delta formed on Mars three billion years ago.",
        language="en",
        published_at=datetime(2023, 3, 18, tzinfo=UTC),
        metadata={"category": "space"},
    ),
    Document(
        id="doc-sp-02",
        title="Telescope maps distant galaxy clusters",
        text="Telescope maps distant galaxy clusters at the edge of the observable universe.",
        language="en",
        published_at=datetime(2024, 9, 3, tzinfo=UTC),
        metadata={"category": "space"},
    ),
    Document(
        id="doc-sp-03",
        title="Reusable rockets cut launch costs",
        text="Reusable rockets cut launch costs significantly enabling commercial space access.",
        language="en",
        published_at=datetime(2025, 1, 14, tzinfo=UTC),
        metadata={"category": "space"},
    ),
    # Finance docs.
    Document(
        id="doc-fi-01",
        title="Central bank holds interest rates steady",
        text="Central bank holds interest rates steady amid mixed inflation signals.",
        language="en",
        published_at=datetime(2023, 7, 26, tzinfo=UTC),
        metadata={"category": "finance"},
    ),
    Document(
        id="doc-fi-02",
        title="Bond yields fall on inflation data",
        text="Bond yields fall on inflation data signalling potential rate cuts ahead.",
        language="en",
        published_at=datetime(2024, 2, 8, tzinfo=UTC),
        metadata={"category": "finance"},
    ),
    Document(
        id="doc-fi-03",
        title="Retail investors drive market volatility",
        text="Retail investors drive market volatility through coordinated social media activity.",
        language="en",
        published_at=datetime(2024, 7, 30, tzinfo=UTC),
        metadata={"category": "finance"},
    ),
    # Additional docs to reach ~24.
    Document(
        id="doc-med-01",
        title="mRNA vaccines demonstrate broad efficacy",
        text="mRNA vaccines demonstrate broad efficacy against multiple pathogen variants.",
        language="en",
        published_at=datetime(2023, 9, 14, tzinfo=UTC),
        metadata={"category": "medicine"},
    ),
    Document(
        id="doc-med-02",
        title="Cancer immunotherapy extends survival rates",
        text="Cancer immunotherapy extends survival rates in advanced melanoma clinical trials.",
        language="en",
        published_at=datetime(2024, 5, 11, tzinfo=UTC),
        metadata={"category": "medicine"},
    ),
    Document(
        id="doc-med-03",
        title="Antibiotic resistance patterns shift globally",
        text="Antibiotic resistance patterns shift globally requiring new treatment protocols.",
        language="en",
        published_at=datetime(2025, 2, 19, tzinfo=UTC),
        metadata={"category": "medicine"},
    ),
    Document(
        id="doc-ai-01",
        title="Large language models compress reasoning",
        text="Large language models compress reasoning chains into latent representations.",
        language="en",
        published_at=datetime(2023, 12, 1, tzinfo=UTC),
        metadata={"category": "ai"},
    ),
    Document(
        id="doc-ai-02",
        title="Neural scaling laws predict performance gains",
        text="Neural scaling laws predict performance gains from larger training datasets.",
        language="en",
        published_at=datetime(2024, 3, 25, tzinfo=UTC),
        metadata={"category": "ai"},
    ),
    Document(
        id="doc-ai-03",
        title="Reinforcement learning from human feedback refines outputs",
        text="Reinforcement learning from human feedback refines model outputs toward preference.",
        language="en",
        published_at=datetime(2024, 8, 6, tzinfo=UTC),
        metadata={"category": "ai"},
    ),
    Document(
        id="doc-cli-01",
        title="Climate tipping points accelerate feedback",
        text="Climate tipping points accelerate feedback loops in Arctic ice loss scenarios.",
        language="en",
        published_at=datetime(2023, 6, 3, tzinfo=UTC),
        metadata={"category": "climate"},
    ),
    Document(
        id="doc-cli-02",
        title="Carbon capture technologies scale up",
        text="Carbon capture technologies scale up with direct air capture pilot projects.",
        language="en",
        published_at=datetime(2024, 10, 17, tzinfo=UTC),
        metadata={"category": "climate"},
    ),
    Document(
        id="doc-cli-03",
        title="Ocean acidification harms coral reefs",
        text="Ocean acidification harms coral reefs through calcium carbonate dissolution.",
        language="en",
        published_at=datetime(2025, 3, 8, tzinfo=UTC),
        metadata={"category": "climate"},
    ),
    Document(
        id="doc-mat-01",
        title="Superconductors operate at room temperature",
        text="Superconductors operate at room temperature enabling lossless power transmission.",
        language="en",
        published_at=datetime(2024, 11, 22, tzinfo=UTC),
        metadata={"category": "materials"},
    ),
]


def _build_index_impl(index_path: Path) -> None:
    """Build a deterministic hash-embedded index from _CORPUS into index_path."""
    from cybernaut_mini.indexing import (
        _shard_summary,
        _shard_title,
        compute_entities,
        compute_keywords,
        compute_term_graph,
    )

    processor = TextProcessor(use_spacy=False)
    embedder = HashEmbedder(dim=64)

    docs = _CORPUS
    texts = [f"{doc.title}\n{doc.text}" for doc in docs]

    # Embed.
    vectors = embedder.embed_documents(texts)

    # Shard.
    n_shards = 4
    seed = 42
    shard_result = shard_documents(vectors, n_shards=n_shards, seed=seed)
    labels = shard_result.labels
    centroids = shard_result.centroids

    # Tokenize.
    doc_tokens: dict[str, list[str]] = {}
    doc_entities: dict[str, list[str]] = {}
    for doc in docs:
        combined = f"{doc.title}\n{doc.text}"
        doc_tokens[doc.id] = processor.content_tokens(combined)
        doc_entities[doc.id] = []  # regex backend returns no entities

    # Group by shard.
    shard_doc_ids: dict[int, list[str]] = {s: [] for s in range(n_shards)}
    for idx, doc in enumerate(docs):
        shard_doc_ids[labels[idx]].append(doc.id)

    row_map = {doc.id: idx for idx, doc in enumerate(docs)}

    shard_tokens: dict[int, list[str]] = {
        s: [tok for doc_id in ids for tok in doc_tokens.get(doc_id, [])]
        for s, ids in shard_doc_ids.items()
    }

    keywords_by_shard = compute_keywords(shard_tokens, max_keywords=30)
    global_term_graph = compute_term_graph(
        [doc_tokens.get(doc.id, []) for doc in docs],
        window=5,
        min_edge_count=2,
    )

    embedding_model = f"hash-{embedder.dim}"

    manifests: list[ShardManifest] = []
    for shard_id in range(n_shards):
        doc_ids = shard_doc_ids[shard_id]
        centroid = centroids[shard_id]
        shard_entity_lists = [doc_entities.get(doc_id, []) for doc_id in doc_ids]
        entities = compute_entities(shard_entity_lists, max_entities=30)
        keywords = keywords_by_shard[shard_id]
        doc_by_id = {doc.id: doc for doc in docs}
        summary = _shard_summary(shard_id, doc_ids, doc_by_id, vectors, row_map, centroid)
        title = _shard_title(shard_id, keywords)
        manifests.append(
            ShardManifest(
                shard_id=shard_id,
                document_ids=doc_ids,
                centroid=centroid.tolist(),
                title=title,
                summary=summary,
                keywords=keywords,
                entities=entities,
                term_graph=global_term_graph,
                document_count=len(doc_ids),
                embedding_model=embedding_model,
            )
        )

    meta = IndexMeta(
        embedding_model=embedding_model,
        embedding_dim=vectors.shape[1],
        n_shards=n_shards,
        n_documents=len(docs),
        seed=seed,
    )

    write_index(
        index_path,
        meta=meta,
        documents=docs,
        vectors=vectors,
        manifests=manifests,
        doc_tokens=doc_tokens,
    )


@pytest.fixture(scope="session")
def built_index_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the retrieval test index once per test session and return its path."""
    index_path = tmp_path_factory.mktemp("retrieval_index")
    _build_index_impl(index_path)
    return index_path


@pytest.fixture(scope="session")
def built_index(built_index_path: Path) -> LoadedIndex:
    """Return a LoadedIndex built from the session-scoped corpus."""
    return LoadedIndex.load(built_index_path)


@pytest.fixture()
def sample_documents() -> list[Document]:
    topics = {
        "biotech": [
            "Gene therapy trial shows durable response",
            "CRISPR screening maps immune pathways",
            "Gut microbiome shifts alter immune response",
        ],
        "energy": [
            "Solar panel efficiency reaches new record",
            "Grid storage batteries stabilize renewable output",
            "Offshore wind farms expand capacity",
        ],
        "space": [
            "Rover samples suggest ancient river delta",
            "Telescope maps distant galaxy clusters",
            "Reusable rockets cut launch costs",
        ],
        "finance": [
            "Central bank holds interest rates steady",
            "Bond yields fall on inflation data",
            "Retail investors drive market volatility",
        ],
    }
    documents: list[Document] = []
    index = 0
    for category, titles in topics.items():
        for title in titles:
            index += 1
            documents.append(
                Document(
                    id=f"doc-{index:03d}",
                    title=title,
                    text=f"{title}. Extended body text about {category} topic number {index}.",
                    language="en",
                    metadata={"category": category},
                )
            )
    return documents
