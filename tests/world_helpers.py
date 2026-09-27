"""Shared builder for WORLD-pillar tests: fixture corpus -> tagged event table.

Every ``tests/test_world_*.py`` file needs the same object — the WORLD event
table built from the committed CC-News fixture slice with its frozen hash-256
embeddings — so the build lives here once and is cached per process. Nothing in
this module invents data: documents, dates, URLs and embeddings are the
committed fixture bytes; clustering and tagging are the code under test.

Blog ref: https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world
    — "One real event is one record, no matter how many outlets repeat it"; the
    fixture's enterovirus-D68 syndication pair (two publishers, one event) is
    the real breadth-2 cluster the event-store tests pin. Local copy under
    ``docs/blog-archive/``.

Assumptions: the hash embedder at dim 256 is the frozen space the fixture
embeddings were built in, so ``FrozenEmbedder(HashEmbedder(dim=256))`` embeds
anchors and topics into the same space as the stored event vectors — cosines are
structurally meaningful (deterministic, comparable) even though a hash space has
no semantics.

Alternatives rejected: a conftest fixture (conftest is a shared file other
workstreams own); rebuilding per test file (the build is ~1s and every world
test wants the identical bytes).
"""

from __future__ import annotations

import datetime as dt
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

from cybernaut_mini.dedup import cluster_documents
from cybernaut_mini.models import Document
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.world import countries, events, ner, tickers
from cybernaut_mini.world.events import WorldEvent
from cybernaut_mini.world.vectors import FrozenEmbedder

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DOCS = REPO_ROOT / "data" / "01_raw" / "fixtures" / "documents.jsonl"
FIXTURE_EMBEDDINGS = REPO_ROOT / "artifacts" / "fixture" / "embeddings.npy"
FIXTURE_ROW_MAP = REPO_ROOT / "artifacts" / "fixture" / "row_map.json"

#: The real syndicated enterovirus-D68 story: two publishers, one event.
SYNDICATED_PAIR = ("ccn-1f588db2f4e51933", "ccn-2d3cb032d63fe96c")


def frozen_embedder() -> FrozenEmbedder:
    """The frozen space of the committed fixture embeddings (hash-256)."""
    return FrozenEmbedder(HashEmbedder(dim=256))


@lru_cache(maxsize=1)
def ccnews_documents() -> tuple[Document, ...]:
    rows = [json.loads(line) for line in FIXTURE_DOCS.read_text().splitlines()]
    docs = []
    for row in rows:
        if not str(row["id"]).startswith("ccn-"):
            continue
        published = row.get("published_at")
        docs.append(
            Document(
                id=str(row["id"]),
                title=str(row["title"]),
                text=str(row["text"]),
                language=row.get("language"),
                published_at=dt.datetime.fromisoformat(published) if published else None,
                url=row.get("url"),
                metadata=row.get("metadata") or {},
            )
        )
    return tuple(docs)


@lru_cache(maxsize=1)
def fixture_matrix() -> tuple[npt.NDArray[np.float32], dict[str, int]]:
    embeddings = np.load(FIXTURE_EMBEDDINGS).astype(np.float32)
    row_map: dict[str, int] = json.loads(FIXTURE_ROW_MAP.read_text())
    return embeddings, row_map


@lru_cache(maxsize=1)
def world_events() -> tuple[WorldEvent, ...]:
    """The tagged event table over the real fixture corpus (entities/tickers/countries)."""
    docs = list(ccnews_documents())
    embeddings, row_map = fixture_matrix()
    clusters = cluster_documents(docs, embeddings, row_map=row_map)
    table = events.build_events(clusters, docs, embeddings, row_map=row_map)
    table = ner.tag_events(table, docs)
    table = tickers.tag_events(table)
    table = countries.tag_events(table)
    return tuple(table)
