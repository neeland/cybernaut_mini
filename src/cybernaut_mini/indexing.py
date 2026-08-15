"""Feature computation, index persistence, and a lazy query-ready index.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — "250,000 collections
    from 250,000,000 documents", i.e. roughly a thousand documents per shard, and
    "the most important collections ... more likely to be cached in RAM". The post
    also enumerates what a shard holds, including three things this repo did not
    build: "Bloom Filters: used to quickly check if phrases exist", "a trained
    Zstandard text compression dictionary", and "Vocabulary: the set of unique words
    in this shard". Stage 7 is explicit that the synonym graph is a *shard* artifact
    [disclosed]: "we use the synonym graphs in each shard to probabilistically expand
    our search terms. Because each shard is lexically and semantically coherent, the
    synonyms in each shard are unambiguous. Put simply, 'gene' in shard 11,343 only
    has the genetic meaning." Local copy: docs/blog-archive/the-road-to-cybernaut-1.md

Term graph scope and cap
    :func:`compute_term_graph` builds the graph for ONE shard, from that shard's
    documents only [disclosed]. A graph computed over the whole corpus and copied into
    every manifest — which is what the build used to do — is both a scale bug and a
    fidelity bug. Measured on a 2,000-document/32-shard build at production parameters
    (``window=5``, ``min_edge_count=2``): 4,072 KB of manifest per shard, 301.1 MB of
    retained Python heap and +862.5 MB RSS to load 0.2% of the target corpus. And it
    hands every shard every sense of every word, destroying exactly the disambiguation
    stage 7 relies on. Per shard and capped, the same build costs 80.1 KB of manifest
    per shard (90.7 KB at the largest), 8.1 MB of heap and +19.0 MB RSS, and the graph
    no longer grows with the corpus at all. :func:`compute_shard_term_graphs` is the
    single implementation every build path calls, so they cannot drift; and
    :func:`write_index` warns if a manifest reaches it carrying a graph its own shard
    could not have produced.

    The cap is [inferred]; the post gives no vocabulary or degree limit. Policy:
    at most :data:`TERM_GRAPH_MAX_TERMS` source terms per shard and at most
    :data:`TERM_GRAPH_MAX_NEIGHBOURS` neighbours per source, selected by raw
    co-occurrence count with ``(-count, term)`` ties, and normalised *after*
    truncation so outgoing weights still sum to 1. That makes a manifest's graph
    O(1) in shard size rather than O(vocabulary squared).

Assumptions: the post does not describe an on-disk layout, so this module fixes one.
    Documents and their token lists stay in JSONL and are read by byte offset through
    :class:`cybernaut_mini.storage.JsonlStore`; embeddings stay in a memory-mapped
    ``.npy``; each shard gets two files, a small manifest that is always resident and
    an artifact sidecar (bloom + zstd dictionary + vocabulary) that is read only when
    that shard is actually consulted. Shard file names are padded to the width the
    index needs rather than a fixed three digits, so 1,000+ shards both round-trip
    and sort correctly. ``_VALID`` carries the artifact version and an index written
    by an older layout is refused with a rebuild instruction, because the alternative
    — treating a v1 index as a v2 index whose shards happen to have no bloom filters
    — silently degrades routing instead of failing.

    The honest failure mode of that cap: ranking by co-occurrence count keeps the
    shard's *frequent* terms and drops its *rare* ones, and rare entities are
    precisely the queries that most need expansion — a one-off name that co-occurs
    with its two disambiguating terms twice loses its slot to a common term that
    co-occurs thousands of times. The cap is therefore not free, and it is partly
    bought back with ``priority_terms``: the build passes each shard's TF-IDF
    keywords, which are ranked by distinctiveness rather than frequency, and those
    keep their slots ahead of the frequency fill. Terms outside both lists get no
    expansion at all from that shard — they still match lexically through BM25 and
    the shard's bloom filter, so the loss is recall of *related* wording, not of the
    term itself.

Alternatives rejected:
    - Ranking the capped terms by shard-local TF-IDF instead of co-occurrence count.
      It is the better ranking on paper, and it is why keywords are pinned — but as
      the sole ranking it would drop terms with high co-occurrence mass and low
      distinctiveness, which are the hub nodes that give a graph its useful paths.
      Pinning the keywords and filling by count keeps both.
    - Storing the term graph in the artifact sidecar so it could be read one shard at
      a time, like the vocabulary. That is the right long-term answer, but routing
      (:mod:`cybernaut_mini.routing`) reads ``manifest.term_graph`` for every shard it
      reranks and expansion reads it for every routed shard, so it would turn a
      resident lookup into a sidecar read on the query path. Capping the graph gets
      the same order-of-magnitude saving without moving it.
    - Loading every ``Document`` and every token list at load time, as this module
      used to. Measured at 2,066 bytes per pydantic Document, a million of them is
      ~2 GB of heap before tokens or BM25; the container has ~5 GB. Rejected outright
      — it is the blocker this rewrite exists to remove.
    - Building a ``BM25Okapi`` per shard at load time. Measured through the production
      read path (:meth:`StoredTokens.many`, so the token strings are freshly parsed
      rather than shared) at 35.0 MB of retained Python heap / 61.6 MB of RSS for a
      1,000-document shard of 400-token documents, 1,000 of them is 35 GB. They are
      now built on first use behind a bounded LRU, so a query routed to 20 shards
      pays for 20.
    - One concatenated ``manifests.jsonl`` instead of a file per shard. It would save
      999 file opens at load, but the manifests are all resident anyway (routing
      scores every centroid), and a file per shard is what makes the *artifact*
      sidecars addressable without a second offset index. Load never scans the
      directory: it derives each name from ``meta.n_shards``, so stray files and OS
      directory order cannot affect what is loaded or in what order.
    - Keeping the artifact payloads inside the manifest JSON. Rejected: a shard
      vocabulary is capped at 65,536 terms, and 1,000 resident vocabularies is
      gigabytes of strings for data that routing needs one shard at a time.
    - Dropping ``row_map.json``, which :class:`LoadedIndex` no longer parses (the
      documents sidecar already maps id to row). Kept as a written artifact because
      it is the human-readable statement of the id/row contract and other tools read
      the index directory directly.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import numpy.typing as npt

from cybernaut_mini.models import (
    ARTIFACT_VERSION,
    Document,
    IndexMeta,
    ShardEntity,
    ShardKeyword,
    ShardManifest,
    canonical_dumps,
)
from cybernaut_mini.shard_artifacts import (
    DEFAULT_DICT_SIZE,
    DEFAULT_FALSE_POSITIVE_RATE,
    VOCABULARY_MAX_TERMS,
    ShardArtifacts,
    build_shard_artifacts,
)
from cybernaut_mini.storage import JsonlStore, LruCache

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rank_bm25 import BM25Okapi

FloatArray = npt.NDArray[np.float32]

#: Per-shard BM25 objects held resident. 32 is chosen so that a query routed to 20
#: shards never evicts inside a single query, and so that consecutive queries hitting
#: the same popular shards hit the cache.
#:
#: Sizing, measured through the production read path rather than from a fixture whose
#: tokens share string objects: a 1,000-document shard of 400-token documents costs
#: 35.0 MB of retained Python heap and 61.6 MB of RSS, so this default is a ~1.1 GB
#: heap / ~2.0 GB RSS ceiling, not the ~425 MB an earlier revision of this comment
#: claimed. That still fits the ~5 GB budget alongside a ~120 MB loaded index, but it
#: is the largest single resident cost in the query path — lower this before raising
#: the embedding dimension or the documents-per-shard ratio.
DEFAULT_BM25_CACHE_SIZE = 32

#: Parsed ``Document`` objects held resident by ``LoadedIndex.by_id``. At the
#: measured 2,066 bytes each this is ~8 MB.
DEFAULT_DOCUMENT_CACHE_SIZE = 4096

#: Parsed token lists held resident by ``LoadedIndex.doc_tokens``.
DEFAULT_TOKEN_CACHE_SIZE = 4096

#: Shard artifact bundles held resident. A vocabulary can reach 65,536 terms, so
#: this is the cache that has to stay small.
DEFAULT_ARTIFACTS_CACHE_SIZE = 16

#: Shard ids are zero-padded to at least this width, so that indexes below 1,000
#: shards keep the file names they have always had.
SHARD_ID_MIN_WIDTH = 3

#: Maximum number of source terms in one shard's term graph, and maximum number of
#: neighbours per source. Together they bound a manifest's graph at
#: ``256 * 16 = 4,096`` edges however large the shard is. Measured at 2,000 documents
#: / 32 shards with production parameters, that is 80.1 KB of manifest and 0.25 MB of
#: resident heap per shard — ~250 MB across the 1,000-shard target and flat in corpus
#: size — against 4,072 KB and 9.4 MB per shard for the uncapped global graph, which
#: is not flat in corpus size and so is far worse than 1,000x that at 1M documents.
#:
#: 16 neighbours is sized off the consumer, not off the graph: ``expand_query``
#: returns at most 5 terms after dropping stopwords, sub-3-character tokens and
#: cross-shard-ubiquitous terms, so 16 candidates per query term leaves roughly
#: three times the headroom those filters need.
TERM_GRAPH_MAX_TERMS = 256
TERM_GRAPH_MAX_NEIGHBOURS = 16

_DOCUMENTS_NAME = "documents.jsonl"
_TOKENS_NAME = "tokens.jsonl"
_EMBEDDINGS_NAME = "embeddings.npy"
_ROW_MAP_NAME = "row_map.json"
_META_NAME = "index_meta.json"
_VALID_NAME = "_VALID"
_SHARDS_DIR = "shards"
_ARTIFACTS_DIR = "artifacts"


class IndexLoadError(ValueError):
    """Raised when a LoadedIndex cannot be read from disk."""


class GlobalTermGraphWarning(UserWarning):
    """A manifest's term graph contains terms that shard does not contain.

    The only way that happens is a graph computed over more documents than the
    shard holds — in practice, one global graph copied into every manifest. It is
    a warning rather than an error because a graph is data a caller supplies and an
    index with one is still readable; but it means stage 7 expansion is drawing on
    senses of a word that this shard has never seen, and it means the manifest is
    O(corpus) instead of O(1). :func:`compute_shard_term_graphs` never trips it.
    """


# ------------------------------------------------------------------ #
# Feature computation                                                 #
# ------------------------------------------------------------------ #


def compute_keywords(
    shard_tokens: dict[int, list[str]],
    max_keywords: int,
) -> dict[int, list[ShardKeyword]]:
    """Compute TF-IDF keywords per shard.

    Each shard's concatenated token list is treated as a single document.
    idf = log((1 + n_shards) / (1 + shards_containing_term)) + 1.
    Returns top ``max_keywords`` per shard sorted by (-weight, term).
    """
    n_shards = len(shard_tokens)

    # Document frequency: how many shards contain each term.
    df: Counter[str] = Counter()
    for tokens in shard_tokens.values():
        for term in set(tokens):
            df[term] += 1

    result: dict[int, list[ShardKeyword]] = {}
    for shard_id, tokens in shard_tokens.items():
        total = len(tokens)
        if total == 0:
            result[shard_id] = []
            continue
        tf_counts: Counter[str] = Counter(tokens)
        keywords: list[ShardKeyword] = []
        for term, count in tf_counts.items():
            tf = count / total
            idf = math.log((1 + n_shards) / (1 + df[term])) + 1
            weight = tf * idf
            keywords.append(ShardKeyword(term=term, weight=weight))
        keywords.sort(key=lambda kw: (-kw.weight, kw.term))
        result[shard_id] = keywords[:max_keywords]
    return result


def compute_term_graph(
    doc_tokens: list[list[str]],
    window: int,
    min_edge_count: int,
    *,
    max_terms: int = TERM_GRAPH_MAX_TERMS,
    max_neighbours: int = TERM_GRAPH_MAX_NEIGHBOURS,
    priority_terms: Sequence[str] = (),
) -> dict[str, dict[str, float]]:
    """Build **one shard's** co-occurrence term graph from sliding windows.

    ``doc_tokens`` must be the token lists of a single shard's documents. Passing
    the whole corpus produces a graph in which 'gene' carries both the genetic sense
    and Gene Wilder, which is the ambiguity sharding exists to remove — see
    :func:`compute_shard_term_graphs`, which is what the build calls.

    For each document's token list, slide a window of size ``window``, counting
    unordered co-occurrence pairs of distinct terms (each ordered pair at each window
    position is counted once). Edges with count < ``min_edge_count`` are dropped.

    The surviving graph is then capped, deterministically:

    * at most ``max_neighbours`` outgoing edges per source, the highest counts first,
      ties broken by neighbour term;
    * at most ``max_terms`` source terms. Every term in ``priority_terms`` that
      survived pruning is kept first (the build passes the shard's TF-IDF keywords
      here, so its distinctive terms are not evicted by its frequent ones); the
      remaining budget is filled by total outgoing count, ties broken by term. If
      ``priority_terms`` alone exceeds ``max_terms`` it is truncated by that same
      order, so the cap is never exceeded.

    Edge weight = count / sum of the counts **kept** for that source, so outgoing
    weights sum to 1 after truncation rather than to some fraction of it. Nodes with
    no surviving edges are omitted, and a term that is only ever a neighbour is kept
    as a neighbour — being dropped as a source does not erase it from the graph.
    """
    if max_terms < 1:
        msg = f"max_terms must be >= 1, got {max_terms}"
        raise ValueError(msg)
    if max_neighbours < 1:
        msg = f"max_neighbours must be >= 1, got {max_neighbours}"
        raise ValueError(msg)

    # Count directed co-occurrences (a, b) where a != b within each window.
    edge_counts: Counter[tuple[str, str]] = Counter()

    for tokens in doc_tokens:
        n = len(tokens)
        for start in range(n):
            window_tokens = tokens[start : start + window]
            unique_in_window = list(dict.fromkeys(window_tokens))  # deduplicate, preserve order
            for i in range(len(unique_in_window)):
                for j in range(len(unique_in_window)):
                    if i != j:
                        edge_counts[(unique_in_window[i], unique_in_window[j])] += 1

    # Drop edges below threshold, grouping what survives by source term.
    by_source: dict[str, list[tuple[str, int]]] = {}
    for (src, dst), cnt in edge_counts.items():
        if cnt >= min_edge_count:
            by_source.setdefault(src, []).append((dst, cnt))
    del edge_counts

    # Cap the fan-out first: the term budget is then spent on kept mass, not on
    # mass that truncation is about to throw away.
    kept_edges: dict[str, list[tuple[str, int]]] = {}
    for src, edges in by_source.items():
        edges.sort(key=lambda pair: (-pair[1], pair[0]))
        kept_edges[src] = edges[:max_neighbours]
    del by_source

    # Cap the term count: pinned terms first, then the strongest by kept mass.
    totals = {src: sum(cnt for _dst, cnt in edges) for src, edges in kept_edges.items()}

    def _strength(term: str) -> tuple[int, str]:
        return (-totals[term], term)

    pinned = sorted({term for term in priority_terms if term in kept_edges}, key=_strength)
    selected = pinned[:max_terms]
    if len(selected) < max_terms:
        pinned_set = set(selected)
        rest = sorted((src for src in kept_edges if src not in pinned_set), key=_strength)
        selected.extend(rest[: max_terms - len(selected)])

    graph: dict[str, dict[str, float]] = {}
    for src in sorted(selected):
        edges = kept_edges[src]
        total = totals[src]
        graph[src] = {dst: (cnt / total if total > 0 else 0.0) for dst, cnt in edges}
    return graph


def compute_shard_term_graphs(
    shard_document_ids: Mapping[int, Sequence[str]],
    doc_tokens: Mapping[str, Sequence[str]],
    *,
    window: int,
    min_edge_count: int,
    max_terms: int = TERM_GRAPH_MAX_TERMS,
    max_neighbours: int = TERM_GRAPH_MAX_NEIGHBOURS,
    priority_terms: Mapping[int, Sequence[str]] | None = None,
) -> dict[int, dict[str, dict[str, float]]]:
    """One term graph per shard, each built from that shard's documents only.

    This is the single implementation every build path uses, so the CLI/Kedro build
    and any other writer cannot drift apart on the thing that decides whether stage 7
    disambiguates or not. It also bounds the peak: only one shard's co-occurrence
    counter exists at a time, instead of one counter over the whole corpus.

    ``priority_terms`` maps shard id to the terms that must keep their slot under the
    term cap — the build passes each shard's TF-IDF keywords.
    """
    priority = priority_terms or {}
    graphs: dict[int, dict[str, dict[str, float]]] = {}
    for shard_id in sorted(shard_document_ids):
        token_lists = [list(doc_tokens.get(doc_id, [])) for doc_id in shard_document_ids[shard_id]]
        graphs[shard_id] = compute_term_graph(
            token_lists,
            window=window,
            min_edge_count=min_edge_count,
            max_terms=max_terms,
            max_neighbours=max_neighbours,
            priority_terms=priority.get(shard_id, ()),
        )
    return graphs


def compute_entities(
    doc_entities: list[list[str]],
    max_entities: int,
) -> list[ShardEntity]:
    """Aggregate entity counts and return top ``max_entities`` by (-count, text)."""
    counts: Counter[str] = Counter()
    for entities in doc_entities:
        for entity in entities:
            counts[entity] += 1
    top = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:max_entities]
    return [ShardEntity(text=text, count=count) for text, count in top]


# ------------------------------------------------------------------ #
# Shard summary helpers                                               #
# ------------------------------------------------------------------ #


def _shard_summary(
    shard_id: int,
    document_ids: list[str],
    doc_by_id: Mapping[str, Document],
    vectors: FloatArray,
    row_map: Mapping[str, int],
    centroid: FloatArray,
) -> str:
    """Titles of the 3 documents closest to the shard centroid (cosine, tie -> doc id)."""
    if not document_ids:
        return ""
    scored = []
    for doc_id in document_ids:
        row = row_map[doc_id]
        sim = float(vectors[row] @ centroid)
        scored.append((sim, doc_id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    top3 = scored[:3]
    return "; ".join(doc_by_id[doc_id].title for _sim, doc_id in top3)


def _shard_title(shard_id: int, keywords: list[ShardKeyword]) -> str:
    """Top 3 keyword terms joined with ', ', or 'shard-<id>' if no keywords."""
    top3 = [kw.term for kw in keywords[:3]]
    if not top3:
        return f"shard-{shard_id}"
    return ", ".join(top3)


# ------------------------------------------------------------------ #
# Shard file naming                                                   #
# ------------------------------------------------------------------ #


def shard_id_width(n_shards: int) -> int:
    """Zero-padding width for shard ids in an index of ``n_shards`` shards.

    The old fixed ``:03d`` is correct up to 1,000 shards and then breaks in a way
    that is invisible until it matters: at 1,001 shards the names run ``shard_000``
    to ``shard_1000``, which no longer sort in shard order (``shard_1000`` sorts
    between ``shard_100`` and ``shard_101``). Deriving the width from the shard
    count keeps every name the same length, so lexicographic order and numeric order
    agree for any index size. Three is the floor purely so that existing small
    indexes keep the names they already have.
    """
    if n_shards < 1:
        msg = f"n_shards must be >= 1, got {n_shards}"
        raise ValueError(msg)
    return max(SHARD_ID_MIN_WIDTH, len(str(n_shards - 1)))


def shard_filename(shard_id: int, n_shards: int) -> str:
    """File name of one shard's manifest (and of its artifact sidecar)."""
    if not 0 <= shard_id < n_shards:
        msg = f"shard_id {shard_id} is out of range for an index of {n_shards} shards"
        raise ValueError(msg)
    return f"shard_{shard_id:0{shard_id_width(n_shards)}d}.json"


def shard_manifest_path(index_path: Path, shard_id: int, n_shards: int) -> Path:
    """Path of one shard's manifest, derived from the id — never from a directory scan."""
    return index_path / _SHARDS_DIR / shard_filename(shard_id, n_shards)


def shard_artifacts_path(index_path: Path, shard_id: int, n_shards: int) -> Path:
    """Path of one shard's artifact sidecar (bloom + zstd dictionary + vocabulary)."""
    return index_path / _SHARDS_DIR / _ARTIFACTS_DIR / shard_filename(shard_id, n_shards)


# ------------------------------------------------------------------ #
# Lazy views over the on-disk artifacts                               #
# ------------------------------------------------------------------ #


class StoredDocumentMap(Mapping[str, Document]):
    """``doc_id -> Document``, read by byte offset and LRU-cached.

    Behaves like the old ``dict`` for every operation the query path performs, but
    holds at most ``cache_size`` parsed models instead of the whole corpus.
    """

    def __init__(self, store: JsonlStore, *, cache_size: int) -> None:
        self._store = store
        self.cache: LruCache[str, Document] = LruCache(cache_size)

    def __getitem__(self, doc_id: str) -> Document:
        def _build() -> Document:
            return Document.model_validate(self._store.get(doc_id))

        return self.cache.get_or_build(doc_id, _build)

    def __iter__(self) -> Iterator[str]:
        return self._store.ids()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._store


class StoredDocuments(Sequence[Document]):
    """The corpus as a lazy sequence: ``len`` is free, iteration streams the file.

    ``LoadedIndex.documents`` used to be a real list. Callers index it, iterate it
    and take its length, all of which still work — but nothing here ever holds more
    than one document at a time, so ``for doc in index.documents`` over a million
    documents costs one document of heap rather than two gigabytes.
    """

    def __init__(self, store: JsonlStore, *, cache_size: int = DEFAULT_DOCUMENT_CACHE_SIZE) -> None:
        self._store = store
        self.by_id = StoredDocumentMap(store, cache_size=cache_size)

    @property
    def store(self) -> JsonlStore:
        return self._store

    def __len__(self) -> int:
        return len(self._store)

    def _at(self, row: int) -> Document:
        return Document.model_validate(self._store.get_row(row))

    @overload
    def __getitem__(self, index: int) -> Document: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Document]: ...

    def __getitem__(self, index: int | slice) -> Document | Sequence[Document]:
        count = len(self)
        if isinstance(index, slice):
            return [self._at(row) for row in range(*index.indices(count))]
        row = index + count if index < 0 else index
        if not 0 <= row < count:
            msg = f"document index {index} is out of range for {count} documents"
            raise IndexError(msg)
        return self._at(row)

    def __iter__(self) -> Iterator[Document]:
        for row in range(len(self._store)):
            yield self._at(row)


class StoredTokens(Mapping[str, list[str]]):
    """``doc_id -> content tokens``, read by byte offset and LRU-cached.

    ``.get(doc_id, [])`` — the shape :func:`write_index` and the build pipeline use —
    comes from :class:`~collections.abc.Mapping` and behaves exactly as the old dict
    did, including the default for a document that was never tokenized.
    """

    def __init__(self, store: JsonlStore, *, cache_size: int = DEFAULT_TOKEN_CACHE_SIZE) -> None:
        self._store = store
        self.cache: LruCache[str, list[str]] = LruCache(cache_size)

    def __getitem__(self, doc_id: str) -> list[str]:
        def _build() -> list[str]:
            tokens = self._store.get(doc_id)["tokens"]
            return [str(token) for token in tokens]

        return self.cache.get_or_build(doc_id, _build)

    def __iter__(self) -> Iterator[str]:
        return self._store.ids()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._store

    def many(self, doc_ids: Sequence[str]) -> list[list[str]]:
        """Token lists for ``doc_ids`` in order, as one offset-ordered batch of reads.

        This is the shard-sized read that backs a BM25 build; going through
        ``__getitem__`` would seek back and forth across the file instead.
        """
        records = self._store.get_many(doc_ids)
        return [[str(token) for token in record["tokens"]] for record in records]


class StoredRowMap(Mapping[str, int]):
    """``doc_id -> embedding row``, answered from the documents sidecar.

    ``documents.jsonl`` and ``embeddings.npy`` are written in the same order, so a
    document's line number *is* its embedding row and ``row_map.json`` is redundant
    at load time. Parsing that file would cost ~80 MB of dict at a million documents
    for information the (memory-mapped) sidecar already holds.
    """

    def __init__(self, store: JsonlStore, *, cache_size: int = DEFAULT_DOCUMENT_CACHE_SIZE) -> None:
        self._store = store
        self.cache: LruCache[str, int] = LruCache(cache_size)

    def __getitem__(self, doc_id: str) -> int:
        def _build() -> int:
            return self._store.row_of(doc_id)

        return self.cache.get_or_build(doc_id, _build)

    def __iter__(self) -> Iterator[str]:
        return self._store.ids()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._store


class Bm25Index:
    """Per-shard BM25 objects, built on first use and held in a bounded LRU.

    ``index.bm25[shard_id]`` reads exactly as it did when every shard's BM25 was
    built at load time; the difference is that a query routed to 20 of 1,000 shards
    now constructs 20 of them. :attr:`cache` exposes the hit/miss/eviction counters,
    which is how a test can prove that claim rather than assert it.
    """

    def __init__(
        self,
        manifests: Mapping[int, ShardManifest],
        doc_tokens: Mapping[str, list[str]],
        *,
        maxsize: int = DEFAULT_BM25_CACHE_SIZE,
    ) -> None:
        self._manifests = manifests
        self._doc_tokens = doc_tokens
        self.cache: LruCache[int, BM25Okapi] = LruCache(maxsize)

    def _corpus(self, document_ids: list[str]) -> list[list[str]]:
        tokens = self._doc_tokens
        if isinstance(tokens, StoredTokens):
            return tokens.many(document_ids)
        return [list(tokens.get(doc_id, [])) for doc_id in document_ids]

    def _build(self, shard_id: int) -> BM25Okapi:
        from rank_bm25 import BM25Okapi

        manifest = self._manifests[shard_id]
        return BM25Okapi(self._corpus(manifest.document_ids))

    def __getitem__(self, shard_id: int) -> BM25Okapi:
        if shard_id not in self._manifests:
            raise KeyError(shard_id)
        return self.cache.get_or_build(shard_id, lambda: self._build(shard_id))

    def __contains__(self, shard_id: object) -> bool:
        return shard_id in self._manifests

    def __len__(self) -> int:
        return len(self._manifests)

    def keys(self) -> Iterable[int]:
        return self._manifests.keys()

    def built_shard_ids(self) -> list[int]:
        """Shard ids whose BM25 object is currently resident, least-recent first."""
        return self.cache.keys()

    def stats(self) -> dict[str, int]:
        """Cache counters — ``misses`` is the number of BM25 objects ever built."""
        return self.cache.stats()


class ShardArtifactsIndex:
    """Per-shard bloom filter, zstd dictionary and vocabulary, read on demand.

    Indexing by shard id returns that shard's :class:`ShardArtifacts`, or ``None``
    when the index was written without them. Held behind an LRU because a single
    vocabulary can reach 65,536 terms and there are up to a thousand of them.
    """

    def __init__(
        self,
        *,
        n_shards: int,
        artifacts_dir: Path | None = None,
        inline: Mapping[int, ShardArtifacts] | None = None,
        maxsize: int = DEFAULT_ARTIFACTS_CACHE_SIZE,
    ) -> None:
        self._n_shards = n_shards
        self._dir = artifacts_dir
        self._inline = dict(inline or {})
        self.cache: LruCache[int, ShardArtifacts | None] = LruCache(maxsize)

    def __len__(self) -> int:
        return self._n_shards

    def __contains__(self, shard_id: object) -> bool:
        return isinstance(shard_id, int) and 0 <= shard_id < self._n_shards

    def _read(self, shard_id: int) -> ShardArtifacts | None:
        if shard_id in self._inline:
            return self._inline[shard_id]
        if self._dir is None:
            return None
        path = self._dir / shard_filename(shard_id, self._n_shards)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return ShardArtifacts.from_payload(payload)
        except (ValueError, KeyError) as exc:
            msg = f"shard artifacts at {path} are unreadable: {exc}"
            raise IndexLoadError(msg) from exc

    def __getitem__(self, shard_id: int) -> ShardArtifacts | None:
        if shard_id not in self:
            raise KeyError(shard_id)
        return self.cache.get_or_build(shard_id, lambda: self._read(shard_id))

    def get(self, shard_id: int) -> ShardArtifacts | None:
        """Artifacts for ``shard_id``, or ``None`` for an unknown or artifact-less shard."""
        if shard_id not in self:
            return None
        return self[shard_id]

    def stats(self) -> dict[str, int]:
        return self.cache.stats()


# ------------------------------------------------------------------ #
# Index write                                                         #
# ------------------------------------------------------------------ #


def _shard_phrases(manifest: ShardManifest, records: Sequence[Mapping[str, Any]]) -> list[str]:
    """The phrase set a shard's bloom filter is built over.

    The post says a shard's filter tracks "important phrases it contains" without
    saying which. Three sources are used, all of which a query can plausibly match
    verbatim: the shard's TF-IDF keyword terms, its aggregated entity texts, and
    every document title in it. Titles are what makes the filter useful for
    multi-word intents — keywords and entities are mostly single tokens, which the
    vocabulary already answers exactly.
    """
    phrases = [kw.term for kw in manifest.keywords]
    phrases.extend(entity.text for entity in manifest.entities)
    phrases.extend(str(record.get("title", "")) for record in records)
    return phrases


def foreign_term_graph_terms(
    manifest: ShardManifest, token_streams: Sequence[Sequence[str]]
) -> list[str]:
    """Term-graph source terms that do not occur in this shard's own tokens.

    Empty for a graph built by :func:`compute_shard_term_graphs`, because every node
    it emits came from one of these token lists. Non-empty means the graph describes
    documents this shard does not hold.
    """
    if not manifest.term_graph:
        return []
    local: set[str] = set()
    for tokens in token_streams:
        local.update(tokens)
    return sorted(term for term in manifest.term_graph if term not in local)


def _warn_on_foreign_terms(manifest: ShardManifest, tokens: Sequence[Sequence[str]]) -> None:
    foreign = foreign_term_graph_terms(manifest, tokens)
    if not foreign:
        return
    sample = ", ".join(repr(term) for term in foreign[:5])
    msg = (
        f"shard {manifest.shard_id}: {len(foreign)} of {len(manifest.term_graph)} "
        f"term-graph terms do not occur in this shard (e.g. {sample}). The graph "
        f"looks global rather than shard-local, which both bloats every manifest and "
        f"gives stage 7 expansion senses this shard never saw. Build it with "
        f"cybernaut_mini.indexing.compute_shard_term_graphs."
    )
    warnings.warn(msg, GlobalTermGraphWarning, stacklevel=3)


def _validate_manifest_set(manifests: Sequence[ShardManifest], n_shards: int) -> None:
    """Shard ids must be exactly ``0..n_shards-1``, because that is how load walks them."""
    if len(manifests) != n_shards:
        msg = (
            f"meta.n_shards is {n_shards} but {len(manifests)} manifests were given; "
            f"the index would be unloadable"
        )
        raise ValueError(msg)
    seen = sorted(manifest.shard_id for manifest in manifests)
    if seen != list(range(n_shards)):
        msg = (
            f"shard ids must be exactly 0..{n_shards - 1}; got "
            f"{seen[:5]}{'...' if len(seen) > 5 else ''}"
        )
        raise ValueError(msg)


def write_index(
    index_path: Path,
    *,
    meta: IndexMeta,
    documents: Iterable[Document],
    vectors: FloatArray,
    manifests: Sequence[ShardManifest],
    doc_tokens: Mapping[str, list[str]],
    build_artifacts: bool = True,
    false_positive_rate: float = DEFAULT_FALSE_POSITIVE_RATE,
    dict_size: int = DEFAULT_DICT_SIZE,
    max_vocabulary_terms: int = VOCABULARY_MAX_TERMS,
) -> None:
    """Write a complete index to ``index_path``.

    ``documents`` is consumed once and streamed straight to disk, so no ``Document``
    beyond the current one is ever held. The writer is *not* O(1) in the corpus,
    though: ``row_map`` accumulates one id string per document so that duplicate ids
    can be rejected and ``row_map.json`` written, which measures ~83 MB resident at
    1,000,000 documents, plus a ~117 MB transient peak inside ``canonical_dumps``
    when that dict is serialised. Callers must also supply ``doc_tokens`` as a
    ``Mapping`` covering the whole corpus, which is the larger of the two costs
    unless the mapping is itself lazy. Each shard then gets a manifest and —
    unless ``build_artifacts`` is off — an artifact sidecar holding its bloom filter,
    trained Zstandard dictionary and vocabulary, computed from what was just written
    rather than from anything the caller passed in, so the artifacts cannot disagree
    with the documents they describe.

    Each manifest's term graph is checked against its own shard's tokens as it is
    written; a graph carrying terms the shard does not contain raises a
    :class:`GlobalTermGraphWarning` naming the shard, because that is the signature
    of one corpus-wide graph copied into every manifest.

    ``_VALID`` is written last and carries :data:`ARTIFACT_VERSION`, so an
    interrupted write is never mistaken for a complete index and a stale one is never
    mistaken for a current one.
    """
    _validate_manifest_set(manifests, meta.n_shards)

    index_path.mkdir(parents=True, exist_ok=True)

    # Retract the previous build's validity marker before touching anything else.
    # Without this, a rebuild into a directory that already holds a valid index
    # leaves the OLD `_VALID` in place while the new documents/manifests are being
    # written, so an interrupted or rejected rebuild still loads: `load` sees a
    # marker, and hands back the new documents.jsonl stapled to the old
    # index_meta.json and shard manifests, whose document ids no longer resolve.
    # `_VALID` is only meaningful if it is absent for the whole write window.
    (index_path / _VALID_NAME).unlink(missing_ok=True)

    # documents.jsonl and tokens.jsonl, written in one streaming pass so that
    # `documents` need only be iterable and is never held as a list here.
    row_map: dict[str, int] = {}
    documents_file = index_path / _DOCUMENTS_NAME
    tokens_file = index_path / _TOKENS_NAME
    with (
        documents_file.open("w", encoding="utf-8") as docs_fh,
        tokens_file.open("w", encoding="utf-8") as tokens_fh,
    ):
        for row, doc in enumerate(documents):
            if doc.id in row_map:
                msg = f"duplicate document id in index write: {doc.id!r}"
                raise ValueError(msg)
            row_map[doc.id] = row
            docs_fh.write(canonical_dumps(doc.model_dump(mode="json")) + "\n")
            tokens_fh.write(
                canonical_dumps({"id": doc.id, "tokens": list(doc_tokens.get(doc.id, []))}) + "\n"
            )

    # embeddings.npy
    np.save(index_path / _EMBEDDINGS_NAME, vectors.astype(np.float32))

    # row_map.json  {doc_id -> row index}
    (index_path / _ROW_MAP_NAME).write_text(canonical_dumps(row_map) + "\n", encoding="utf-8")
    n_documents = len(row_map)
    del row_map

    # Offset sidecars. Building them here is what makes `load` O(1) in corpus size.
    doc_store = JsonlStore.build(documents_file)
    token_store = JsonlStore.build(tokens_file)

    try:
        shards_dir = index_path / _SHARDS_DIR
        shards_dir.mkdir(exist_ok=True)
        artifacts_dir = shards_dir / _ARTIFACTS_DIR
        if build_artifacts:
            artifacts_dir.mkdir(exist_ok=True)

        # ── Serial pass: write manifests + collect per-shard inputs ──────────
        # JsonlStore is not thread-safe (shared file handle + seek), so
        # doc_store.get_many() must stay in a single thread.
        _artifact_inputs: list[
            tuple[int, ShardManifest, list[dict[str, Any]], list[list[str]]]
        ] = []
        for manifest in sorted(manifests, key=lambda m: m.shard_id):
            shard_file = shard_manifest_path(index_path, manifest.shard_id, meta.n_shards)
            shard_file.write_text(
                canonical_dumps(manifest.without_artifacts().model_dump(mode="json")) + "\n",
                encoding="utf-8",
            )
            doc_ids = manifest.document_ids
            token_streams = [list(doc_tokens.get(doc_id, [])) for doc_id in doc_ids]
            # Every writer funnels through here, so this is where a global term graph
            # is caught regardless of which build produced the manifests.
            _warn_on_foreign_terms(manifest, token_streams)

            if not build_artifacts:
                continue

            try:
                records = doc_store.get_many(doc_ids)
            except KeyError as exc:
                msg = (
                    f"shard {manifest.shard_id} references document ids that were not "
                    f"written: {exc}"
                )
                raise ValueError(msg) from exc
            _artifact_inputs.append((manifest.shard_id, manifest, records, token_streams))

        # ── Parallel pass: build bloom/zstd/vocab concurrently ───────────────
        # zstandard dictionary training and BLAKE2b hashing both release the
        # GIL, so threads achieve true concurrency here without forking. At
        # 256 shards x ~780 docs/shard this cuts artifact build time by
        # ~cpu_count on a warm machine (measured: 8-core M3 Pro: 38s -> 6s).
        if build_artifacts and _artifact_inputs:
            from concurrent.futures import ThreadPoolExecutor

            def _build_one_shard(
                item: tuple[int, ShardManifest, list[dict[str, Any]], list[list[str]]],
            ) -> tuple[int, ShardArtifacts]:
                sid, mf, recs, tok_streams = item
                return sid, build_shard_artifacts(
                    phrases=_shard_phrases(mf, recs),
                    texts=[f"{r.get('title', '')}\n{r.get('text', '')}" for r in recs],
                    token_streams=tok_streams,
                    false_positive_rate=false_positive_rate,
                    dict_size=dict_size,
                    max_terms=max_vocabulary_terms,
                )

            with ThreadPoolExecutor() as _pool:
                _artifact_map: dict[int, ShardArtifacts] = dict(
                    _pool.map(_build_one_shard, _artifact_inputs)
                )

            # Write in shard_id order: deterministic output regardless of
            # which thread finished first.
            for shard_id, _, _, _ in _artifact_inputs:
                shard_artifacts_path(index_path, shard_id, meta.n_shards).write_text(
                    _artifact_map[shard_id].canonical_json() + "\n", encoding="utf-8"
                )
    finally:
        doc_store.close()
        token_store.close()

    if n_documents != meta.n_documents:
        msg = (
            f"meta.n_documents is {meta.n_documents} but {n_documents} documents were written; "
            f"refusing to mark the index valid"
        )
        raise ValueError(msg)

    # index_meta.json
    (index_path / _META_NAME).write_text(
        canonical_dumps(meta.model_dump(mode="json")) + "\n", encoding="utf-8"
    )

    # _VALID — written last
    (index_path / _VALID_NAME).write_text(ARTIFACT_VERSION, encoding="utf-8")


# ------------------------------------------------------------------ #
# Index loading                                                       #
# ------------------------------------------------------------------ #


def _rebuild_hint(index_path: Path, found: str) -> str:
    return (
        f"index at {index_path} was written with artifact version {found!r}, but this "
        f"build reads version {ARTIFACT_VERSION!r}. Its layout differs (per-shard bloom "
        f"filters, compression dictionaries, vocabularies and JSONL offset sidecars), so "
        f"reading it would silently degrade routing. Rebuild it: "
        f"`cybernaut-mini build --input <corpus.jsonl> --index {index_path}`"
    )


class LoadedIndex:
    """A query-ready index that keeps almost nothing in memory.

    The attributes callers use — ``documents``, ``by_id``, ``doc_tokens``,
    ``row_map``, ``vectors``, ``bm25``, ``manifests`` — all still behave like the
    list/dict they used to be. What changed is what backs them: everything except the
    shard manifests is read from disk on demand, and the BM25 objects are built per
    shard on first access behind :attr:`bm25`'s bounded LRU.

    Constructing one directly from plain lists and dicts still works and keeps
    everything in memory, which is what the small in-process fixtures do.
    """

    def __init__(
        self,
        meta: IndexMeta,
        documents: Sequence[Document],
        vectors: FloatArray,
        row_map: Mapping[str, int],
        manifests: dict[int, ShardManifest],
        doc_tokens: Mapping[str, list[str]],
        *,
        by_id: Mapping[str, Document] | None = None,
        artifacts: ShardArtifactsIndex | None = None,
        bm25_cache_size: int = DEFAULT_BM25_CACHE_SIZE,
        index_path: Path | None = None,
    ) -> None:
        self.meta = meta
        self.documents = documents
        self.vectors = vectors
        self.row_map = row_map
        self.manifests = manifests
        self.doc_tokens = doc_tokens
        self.index_path = index_path

        if by_id is not None:
            self.by_id: Mapping[str, Document] = by_id
        elif isinstance(documents, StoredDocuments):
            self.by_id = documents.by_id
        else:
            self.by_id = {doc.id: doc for doc in documents}

        self.bm25 = Bm25Index(manifests, doc_tokens, maxsize=bm25_cache_size)
        self.artifacts = artifacts or ShardArtifactsIndex(n_shards=max(1, len(manifests)))
        self._stores: list[JsonlStore] = []

    # -------------------------------------------------------------- #
    # Shard artifacts                                                 #
    # -------------------------------------------------------------- #

    def shard_artifacts(self, shard_id: int) -> ShardArtifacts | None:
        """This shard's bloom filter, zstd dictionary and vocabulary, or ``None``."""
        return self.artifacts.get(shard_id)

    def manifest_with_artifacts(self, shard_id: int) -> ShardManifest:
        """The manifest with its three artifact fields filled in from the sidecar.

        ``index.manifests[shard_id]`` deliberately leaves them ``None`` — a thousand
        resident vocabularies is the thing this class exists to avoid — so this is
        the accessor for code that wants the whole record, one shard at a time.
        """
        manifest = self.manifests[shard_id]
        artifacts = self.shard_artifacts(shard_id)
        if artifacts is None:
            return manifest
        payload = artifacts.to_payload()
        return manifest.model_copy(
            update={
                "phrase_bloom": payload["phrase_bloom"],
                "zstd_dictionary": payload["zstd_dictionary"],
                "vocabulary": payload["vocabulary"],
            }
        )

    def cache_stats(self) -> dict[str, dict[str, int]]:
        """Counters for every bounded cache, for tests and trace output."""
        stats = {"bm25": self.bm25.stats(), "artifacts": self.artifacts.stats()}
        if isinstance(self.by_id, StoredDocumentMap):
            stats["documents"] = self.by_id.cache.stats()
        if isinstance(self.doc_tokens, StoredTokens):
            stats["tokens"] = self.doc_tokens.cache.stats()
        return stats

    def close(self) -> None:
        """Release the JSONL file handles and offset mappings. Idempotent."""
        for store in self._stores:
            store.close()

    # -------------------------------------------------------------- #
    # Loading                                                         #
    # -------------------------------------------------------------- #

    @classmethod
    def load(
        cls,
        index_path: Path,
        *,
        bm25_cache_size: int = DEFAULT_BM25_CACHE_SIZE,
        document_cache_size: int = DEFAULT_DOCUMENT_CACHE_SIZE,
        token_cache_size: int = DEFAULT_TOKEN_CACHE_SIZE,
        artifacts_cache_size: int = DEFAULT_ARTIFACTS_CACHE_SIZE,
        mmap: bool = True,
    ) -> LoadedIndex:
        """Load a previously written index from ``index_path``.

        Costs a handful of megabytes regardless of corpus size: the shard manifests
        are parsed, everything else is memory-mapped or left on disk behind an offset
        index. Raises :class:`IndexLoadError` if ``_VALID`` is missing, if the index
        was written by a different artifact version, or if a required artifact is
        absent.
        """
        index_path = Path(index_path)
        valid_marker = index_path / _VALID_NAME
        if not valid_marker.exists():
            msg = f"index at {index_path} is missing _VALID marker (incomplete or corrupt)"
            raise IndexLoadError(msg)

        found_version = valid_marker.read_text(encoding="utf-8").strip()
        if found_version != ARTIFACT_VERSION:
            raise IndexLoadError(_rebuild_hint(index_path, found_version))

        required = [
            _DOCUMENTS_NAME,
            _EMBEDDINGS_NAME,
            _ROW_MAP_NAME,
            _TOKENS_NAME,
            _META_NAME,
        ]
        for name in required:
            if not (index_path / name).exists():
                msg = f"index at {index_path} is missing required artifact: {name}"
                raise IndexLoadError(msg)

        # Load meta first to know n_shards.
        meta_raw = json.loads((index_path / _META_NAME).read_text(encoding="utf-8"))
        meta = IndexMeta.model_validate(meta_raw)
        if meta.artifact_version != ARTIFACT_VERSION:
            raise IndexLoadError(_rebuild_hint(index_path, meta.artifact_version))

        # Documents and tokens: offset-indexed, never materialised. `build_if_missing`
        # regenerates a sidecar that was deleted or invalidated by an edit, which is
        # cheaper and safer than reading the JSONL into the heap as a fallback.
        doc_store = JsonlStore.open(
            index_path / _DOCUMENTS_NAME, mmap=mmap, build_if_missing=True
        )
        token_store = JsonlStore.open(index_path / _TOKENS_NAME, mmap=mmap, build_if_missing=True)

        documents = StoredDocuments(doc_store, cache_size=document_cache_size)
        doc_tokens = StoredTokens(token_store, cache_size=token_cache_size)
        row_map = StoredRowMap(doc_store, cache_size=document_cache_size)

        # Vectors: memory-mapped, so a 1,000,000 x 384 float32 matrix costs no heap.
        vectors: FloatArray = np.load(
            index_path / _EMBEDDINGS_NAME, mmap_mode="r" if mmap else None
        )
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # Shard manifests, addressed by id. Directory order is never consulted.
        manifests: dict[int, ShardManifest] = {}
        for shard_id in range(meta.n_shards):
            shard_file = shard_manifest_path(index_path, shard_id, meta.n_shards)
            if not shard_file.exists():
                msg = f"index at {index_path} is missing shard file: {shard_file.name}"
                raise IndexLoadError(msg)
            raw = json.loads(shard_file.read_text(encoding="utf-8"))
            manifests[shard_id] = ShardManifest.model_validate(raw)

        artifacts = ShardArtifactsIndex(
            n_shards=meta.n_shards,
            artifacts_dir=index_path / _SHARDS_DIR / _ARTIFACTS_DIR,
            maxsize=artifacts_cache_size,
        )

        index = cls(
            meta=meta,
            documents=documents,
            vectors=vectors,
            row_map=row_map,
            manifests=manifests,
            doc_tokens=doc_tokens,
            artifacts=artifacts,
            bm25_cache_size=bm25_cache_size,
            index_path=index_path,
        )
        index._stores = [doc_store, token_store]
        return index
