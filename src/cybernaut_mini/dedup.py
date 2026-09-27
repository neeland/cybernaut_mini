"""Near-duplicate clustering with apex-story election and publisher-breadth counting.

Syndicated news is the same story wearing many URLs. This module folds those copies
into one :class:`EventCluster` per underlying event: connected components over a
cosine threshold, an elected "apex" representative, a coverage-peak date, and the
publisher breadth (``total_netlocs``) that the WORLD pillar uses as an event's
weight. The output schema documented on :class:`EventCluster` is the input contract
for the event store — everything downstream (GPR/TPU/EPU indices, media-coverage
gates, RAG dedup) consumes these records, so the function is pure and deterministic:
same ``(doc_ids, embeddings, urls, dates)`` in, byte-identical canonical JSON out.

Blog ref: https://nosible.com/blog/using-vector-search-to-see-signals-in-company-news
    — "Every time we create a new index, we de-duplicate all news and find the 'apex'
    story for each cluster" and "duplicate stories are naturally clustered by time and
    by metadata. Instead of deduplicating across all 55-million embeddings we
    deduplicate shards of the data". Breadth and dating come from
    https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world
    — "``breadth(e)`` [is] the number of distinct publishers that covered an event
    ``e``, which is the ``total_netlocs`` field carried on every event" — and
    https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty
    — "Ours dates each event by the day its coverage peaked." Local copies under
    ``docs/blog-archive/``.

Assumptions:
    - Cosine threshold 0.9 on L2-normalised embeddings defines "near-duplicate", and
      clusters are the *connected components* of that pair graph: if A~B and B~C then
      A, B, C are one event even when cos(A, C) dips below the threshold. Components
      match how syndication actually chains (wire copy -> trimmed copy -> rewritten
      copy) at the cost of rare over-merges; the threshold is a parameter.
    - Publisher breadth counts distinct *registered* domains, not raw netlocs:
      ``www.dailyrecord.co.uk`` and ``dailyrecord.co.uk`` are one publisher. Without
      a public-suffix dependency, :func:`registered_domain` keeps the last two labels
      plus a third when the second-level label is a known country-code second level
      (``co.uk``, ``com.au``, ...). The posts' own field name says ``netlocs``, so
      the fields keep that name even though the values are registered domains.
    - The apex is the longest member (most complete syndicated copy), tie-broken by
      earliest date and then lexicographic id. When callers cannot supply text
      lengths, earliest-then-id is used alone; either way election is deterministic.
    - The event date is the day coverage peaked — the calendar day contributing the
      most members, tie-broken toward the earliest day. A breadth-1 cluster's peak
      day *is* its publish day, so the "degrade to publish date" mode is the same
      code path. Members without dates abstain; a fully undated cluster has
      ``date=None`` rather than an invented one.
    - Event ids are content-addressed (SHA-256 over the sorted member ids), so the
      same cluster gets the same id in every run and across overlapping time shards.

Alternatives rejected:
    - MinHash over title shingles as the primary signal: cheap and byte-exact, but it
      misses rewritten syndication with identical meaning and different tokens — the
      case embeddings exist to catch. Kept in the posts as a fallback, not needed at
      this corpus scale.
    - Centroid/agglomerative clustering with a merge radius: needs a linkage-order
      tie-break policy to stay deterministic and re-opens the "which copy is
      canonical" question; union-find over a fixed pair set has neither problem.
    - Always-on LSH bucketing: at fixture scale the exact upper-triangle cosine is
      microseconds; the integer prefilter (``lsh_bits``) exists for larger shards but
      defaults off so the default path has zero coupling to the hasher.
    - ``urlparse(url).netloc`` verbatim (no registered-domain fold): what the post's
      field name implies, but it double-counts a publisher that syndicates to its own
      subdomains, and breadth is the one number every downstream index weights by.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from cybernaut_mini.lsh import (
    DEFAULT_SIMILARITY_FLOOR_FRACTION,
    RandomHyperplanesLSH,
)
from cybernaut_mini.models import Document, canonical_dumps

__all__ = [
    "DEFAULT_COSINE_THRESHOLD",
    "EventCluster",
    "cluster_documents",
    "cluster_near_duplicates",
    "registered_domain",
    "write_clusters",
]

FloatArray = npt.NDArray[np.float32]

#: Cosine similarity at or above which two documents are near-duplicates.
DEFAULT_COSINE_THRESHOLD = 0.9

#: Country-code second-level labels: ``example.co.uk`` registers three labels deep.
_SECOND_LEVEL_LABELS = frozenset(
    {"ac", "co", "com", "edu", "go", "gov", "mil", "ne", "net", "or", "org"}
)


class EventCluster(BaseModel):
    """One de-duplicated event — the WORLD pillar's input contract.

    Output schema (one record per cluster, singletons included):

    - ``event_id``: content-addressed id, ``evt-`` + first 16 hex chars of the
      SHA-256 of the canonical JSON list of sorted member ids.
    - ``apex_doc_id``: the elected representative (longest, then earliest, then
      lexicographically-first member).
    - ``member_doc_ids``: all member document ids, sorted ascending.
    - ``date``: the day coverage peaked, or ``None`` when no member is dated.
    - ``total_netlocs``: publisher breadth — ``len(netlocs)`` (0 when no member
      has a parseable URL).
    - ``netlocs``: sorted distinct registered domains of the member URLs.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    apex_doc_id: str = Field(min_length=1)
    member_doc_ids: list[str] = Field(min_length=1)
    date: dt.date | None = None
    total_netlocs: int = Field(ge=0)
    netlocs: list[str] = Field(default_factory=list)


def registered_domain(url: str) -> str:
    """Registered domain of a URL (or bare host), lowercased; ``""`` if unparseable.

    ``http://www.dailyrecord.co.uk/news/...`` -> ``dailyrecord.co.uk``;
    ``https://news.denverpost.com/x`` -> ``denverpost.com``. A string without a
    scheme is treated as a bare host if it looks like one.
    """
    if not url:
        return ""
    netloc = urlparse(url).netloc
    if not netloc and "/" not in url and "." in url:
        netloc = url
    host = netloc.rsplit("@", 1)[-1].split(":", 1)[0].strip().lower().rstrip(".")
    if not host or "." not in host:
        return ""
    labels = [label for label in host.split(".") if label]
    if len(labels) <= 2:
        return ".".join(labels)
    if labels[-2] in _SECOND_LEVEL_LABELS and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            # Deterministic: the smaller index is always the root.
            if root_left < root_right:
                self._parent[root_right] = root_left
            else:
                self._parent[root_left] = root_right


def _l2_normalize_rows(matrix: FloatArray) -> FloatArray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


def _near_duplicate_pairs(
    embeddings: FloatArray, threshold: float, lsh_bits: int | None, seed: int
) -> npt.NDArray[np.int64]:
    """``(k, 2)`` array of ``i < j`` index pairs with cosine >= ``threshold``."""
    n_rows = embeddings.shape[0]
    if lsh_bits is None:
        sims = embeddings @ embeddings.T
        upper = np.triu(sims >= threshold, k=1)
        return np.argwhere(upper).astype(np.int64)

    # Integer prefilter: only pairs above the 0.70 * max_codes matching-code floor
    # are checked with an exact cosine. At cosine 0.9 the expected matching count is
    # far above the floor, so the prefilter is (probabilistically) recall-safe.
    hasher = RandomHyperplanesLSH.fit(embeddings.shape[1], n_bits=lsh_bits, seed=seed)
    codes = hasher.encode(embeddings)
    floor = hasher.n_bits * DEFAULT_SIMILARITY_FLOOR_FRACTION
    pairs: list[tuple[int, int]] = []
    for i in range(n_rows - 1):
        xor = np.bitwise_xor(codes[i + 1 :], codes[i][None, :])
        matching = hasher.n_bits - np.bitwise_count(xor).sum(axis=1, dtype=np.int64)
        candidates = np.nonzero(matching >= floor)[0]
        if candidates.size == 0:
            continue
        cosines = embeddings[i + 1 :][candidates] @ embeddings[i]
        for offset, cosine in zip(candidates, cosines, strict=True):
            if float(cosine) >= threshold:
                pairs.append((i, i + 1 + int(offset)))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _coverage_peak_date(dates: Sequence[dt.datetime | dt.date | None]) -> dt.date | None:
    """The calendar day contributing the most members; ties go to the earliest day."""
    days = [d.date() if isinstance(d, dt.datetime) else d for d in dates if d is not None]
    if not days:
        return None
    counts = Counter(days)
    return min(counts, key=lambda day: (-counts[day], day))


def _elect_apex(
    member_indices: Sequence[int],
    doc_ids: Sequence[str],
    dates: Sequence[dt.datetime | dt.date | None],
    text_lengths: Sequence[int] | None,
) -> str:
    """Longest member, then earliest, then lexicographically-first id."""

    def sort_key(index: int) -> tuple[int, tuple[int, str], str]:
        length = -(text_lengths[index] if text_lengths is not None else 0)
        when = dates[index]
        day = when.date() if isinstance(when, dt.datetime) else when
        dated = (0, day.isoformat()) if day is not None else (1, "")
        return (length, dated, doc_ids[index])

    return doc_ids[min(member_indices, key=sort_key)]


def _event_id(member_doc_ids: Sequence[str]) -> str:
    digest = hashlib.sha256(canonical_dumps(sorted(member_doc_ids)).encode("utf-8")).hexdigest()
    return f"evt-{digest[:16]}"


def cluster_near_duplicates(
    doc_ids: Sequence[str],
    embeddings: FloatArray,
    urls: Sequence[str | None],
    dates: Sequence[dt.datetime | dt.date | None],
    *,
    threshold: float = DEFAULT_COSINE_THRESHOLD,
    text_lengths: Sequence[int] | None = None,
    lsh_bits: int | None = None,
    seed: int = 42,
) -> list[EventCluster]:
    """Pure function: ``(doc_ids, embeddings, urls, dates)`` -> event clusters.

    All four sequences are row-aligned. Every document lands in exactly one cluster;
    singletons are clusters of one. Set ``lsh_bits`` (e.g. 256) to prefilter pairs
    with the integer Hamming floor before exact cosine — same output, fewer float
    comparisons — on shards too big for the dense similarity matrix.

    Returned clusters are sorted by (undated-last, date, event_id).
    """
    n_rows = len(doc_ids)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != n_rows:
        msg = f"embeddings must be ({n_rows}, dim), got {matrix.shape}"
        raise ValueError(msg)
    for name, seq in (("urls", urls), ("dates", dates)):
        if len(seq) != n_rows:
            msg = f"{name} has {len(seq)} rows, expected {n_rows}"
            raise ValueError(msg)
    if text_lengths is not None and len(text_lengths) != n_rows:
        msg = f"text_lengths has {len(text_lengths)} rows, expected {n_rows}"
        raise ValueError(msg)
    if len(set(doc_ids)) != n_rows:
        msg = "doc_ids must be unique"
        raise ValueError(msg)
    if n_rows == 0:
        return []

    matrix = _l2_normalize_rows(matrix)
    components = _UnionFind(n_rows)
    for i, j in _near_duplicate_pairs(matrix, threshold, lsh_bits, seed):
        components.union(int(i), int(j))

    members_by_root: dict[int, list[int]] = {}
    for index in range(n_rows):
        members_by_root.setdefault(components.find(index), []).append(index)

    clusters: list[EventCluster] = []
    for member_indices in members_by_root.values():
        member_ids = sorted(doc_ids[index] for index in member_indices)
        domains = sorted(
            {
                domain
                for index in member_indices
                if (domain := registered_domain(urls[index] or ""))
            }
        )
        clusters.append(
            EventCluster(
                event_id=_event_id(member_ids),
                apex_doc_id=_elect_apex(member_indices, doc_ids, dates, text_lengths),
                member_doc_ids=member_ids,
                date=_coverage_peak_date([dates[index] for index in member_indices]),
                total_netlocs=len(domains),
                netlocs=domains,
            )
        )
    clusters.sort(key=lambda c: (c.date is None, c.date.isoformat() if c.date else "", c.event_id))
    return clusters


def cluster_documents(
    documents: Sequence[Document],
    embeddings: FloatArray,
    *,
    row_map: Mapping[str, int] | None = None,
    threshold: float = DEFAULT_COSINE_THRESHOLD,
    lsh_bits: int | None = None,
    seed: int = 42,
) -> list[EventCluster]:
    """Convenience wrapper over :func:`cluster_near_duplicates` for ``Document`` rows.

    ``row_map`` maps document id -> row in ``embeddings`` (the index's
    ``row_map.json``); when ``None``, rows are assumed aligned with ``documents``.
    Text lengths for apex election come from ``len(doc.text)``.
    """
    matrix = np.asarray(embeddings, dtype=np.float32)
    if row_map is not None:
        rows = [row_map[doc.id] for doc in documents]
        matrix = matrix[np.asarray(rows, dtype=np.int64)]
    return cluster_near_duplicates(
        [doc.id for doc in documents],
        matrix,
        [doc.url for doc in documents],
        [doc.published_at for doc in documents],
        threshold=threshold,
        text_lengths=[len(doc.text) for doc in documents],
        lsh_bits=lsh_bits,
        seed=seed,
    )


def write_clusters(path: Path, clusters: Sequence[EventCluster]) -> None:
    """Persist clusters as one canonical-JSON array — byte-identical across runs."""
    payload = [cluster.model_dump(mode="json") for cluster in clusters]
    path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")
