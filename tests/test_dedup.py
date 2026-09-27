"""Near-duplicate dedup — real CC-News syndication from the committed fixture.

The load-bearing assertions run over the 200 real CC-News documents in
``data/01_raw/fixtures/documents.jsonl`` with their real hash-256 embeddings from
``artifacts/fixture/embeddings.npy``. That slice genuinely contains syndicated
near-duplicates: the enterovirus D68 story of 2016-10-17/18 appears once on
``coventrytelegraph.net`` and once on ``newsletter.co.uk`` at cosine ~0.915, and a
Coventry City FC follow-up pair sits at ~0.902 on one publisher. Those real clusters
pin apex election, coverage-peak dating, and publisher breadth. Clustering *geometry*
(transitivity, thresholds) is additionally tested on seeded unit vectors, which is
pure math, not a fabricated corpus.

Blog ref: https://nosible.com/blog/using-vector-search-to-see-signals-in-company-news
    — "we de-duplicate all news and find the 'apex' story for each cluster";
    https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world
    — breadth as ``total_netlocs``. Local copies under ``docs/blog-archive/``.

Assumptions:
    - The two real duplicate pairs are properties of committed bytes (fixture
      embeddings + fixture documents), so asserting their ids, dates, and domains is
      pinning frozen data, not fitting the test to the code.
    - The LSH-prefiltered path must produce *identical* clusters to the exact path
      on this corpus; at cosine >= 0.9 the 0.70 matching-code floor is ~6 sigma
      below the expected matching count, so equality is deterministic in practice
      for the frozen seed.

Alternatives rejected:
    - Synthesising a corpus with planted duplicates: forbidden (real data only) and
      unnecessary — CC-News syndication provides the ground truth for free.
    - Asserting cluster *count* on the real slice: brittle under threshold changes;
      the tests assert the partition property and the known pairs instead.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from cybernaut_mini.dedup import (
    EventCluster,
    cluster_documents,
    cluster_near_duplicates,
    registered_domain,
    write_clusters,
)
from cybernaut_mini.models import Document

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DOCS = REPO_ROOT / "data" / "01_raw" / "fixtures" / "documents.jsonl"
FIXTURE_EMBEDDINGS = REPO_ROOT / "artifacts" / "fixture" / "embeddings.npy"
FIXTURE_ROW_MAP = REPO_ROOT / "artifacts" / "fixture" / "row_map.json"

#: The real syndicated enterovirus D68 story: two publishers, one event.
SYNDICATED_PAIR = ("ccn-1f588db2f4e51933", "ccn-2d3cb032d63fe96c")
#: The real Coventry City FC pair: same publisher twice.
SAME_PUBLISHER_PAIR = ("ccn-0b3dffccaeb036dc", "ccn-0bbabb6be34c07f7")


@pytest.fixture(scope="module")
def ccnews_rows() -> list[dict[str, object]]:
    rows = [json.loads(line) for line in FIXTURE_DOCS.read_text().splitlines()]
    return [row for row in rows if str(row["id"]).startswith("ccn-")]


@pytest.fixture(scope="module")
def ccnews_inputs(
    ccnews_rows: list[dict[str, object]],
) -> tuple[list[str], npt.NDArray[np.float32], list[str], list[dt.datetime | None], list[int]]:
    row_map: dict[str, int] = json.loads(FIXTURE_ROW_MAP.read_text())
    embeddings = np.load(FIXTURE_EMBEDDINGS).astype(np.float32)
    ids = [str(row["id"]) for row in ccnews_rows]
    matrix = embeddings[np.asarray([row_map[i] for i in ids], dtype=np.int64)]
    urls = [str(row["url"]) for row in ccnews_rows]
    dates = [
        dt.datetime.fromisoformat(str(row["published_at"])) if row["published_at"] else None
        for row in ccnews_rows
    ]
    lengths = [len(str(row["text"])) for row in ccnews_rows]
    return ids, matrix, urls, dates, lengths


@pytest.fixture(scope="module")
def ccnews_clusters(
    ccnews_inputs: tuple[
        list[str], npt.NDArray[np.float32], list[str], list[dt.datetime | None], list[int]
    ],
) -> list[EventCluster]:
    ids, matrix, urls, dates, lengths = ccnews_inputs
    return cluster_near_duplicates(ids, matrix, urls, dates, text_lengths=lengths)


def _cluster_of(clusters: list[EventCluster], doc_id: str) -> EventCluster:
    for cluster in clusters:
        if doc_id in cluster.member_doc_ids:
            return cluster
    msg = f"{doc_id} not in any cluster"
    raise AssertionError(msg)


# ------------------------------------------------------------------ #
# registered_domain                                                  #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://www.dailyrecord.co.uk/news/local-news/x-9079348", "dailyrecord.co.uk"),
        ("http://www.coventrytelegraph.net/news/coventry-news/y", "coventrytelegraph.net"),
        ("https://news.denverpost.com/2016/story", "denverpost.com"),
        ("http://www.newsletter.co.uk/news/health/z", "newsletter.co.uk"),
        ("https://user@www.example.com:8080/path", "example.com"),
        ("www.manilatimes.net", "manilatimes.net"),
        ("1572240#0", ""),  # A MIRACL pseudo-URL has no host at all.
        ("", ""),
        ("http:///nohost", ""),
    ],
)
def test_registered_domain(url: str, expected: str) -> None:
    assert registered_domain(url) == expected


# ------------------------------------------------------------------ #
# Clustering geometry (pure math, seeded unit vectors)               #
# ------------------------------------------------------------------ #


def _unit(angle_degrees: float) -> list[float]:
    radians = np.deg2rad(angle_degrees)
    return [float(np.cos(radians)), float(np.sin(radians))]


def test_transitive_chain_is_one_component() -> None:
    """A~B and B~C merge even when cos(A, C) is below the threshold."""
    vectors = np.asarray(
        [_unit(0.0), _unit(25.0), _unit(50.0), _unit(180.0)], dtype=np.float32
    )
    # cos(0,25)=0.906, cos(25,50)=0.906 — above 0.9; cos(0,50)=0.643 — below.
    clusters = cluster_near_duplicates(
        ["a", "b", "c", "d"], vectors, [None] * 4, [None] * 4, threshold=0.9
    )
    by_size = sorted(len(c.member_doc_ids) for c in clusters)
    assert by_size == [1, 3]
    chained = next(c for c in clusters if len(c.member_doc_ids) == 3)
    assert chained.member_doc_ids == ["a", "b", "c"]
    assert chained.date is None
    assert chained.total_netlocs == 0


def test_coverage_peak_day_wins_and_ties_go_earliest() -> None:
    vectors = np.asarray([_unit(0.0)] * 3, dtype=np.float32)
    day1 = dt.datetime(2016, 10, 17)
    day2 = dt.datetime(2016, 10, 18)
    peaked = cluster_near_duplicates(["a", "b", "c"], vectors, [None] * 3, [day1, day2, day2])
    assert peaked[0].date == dt.date(2016, 10, 18)  # Two members on the 18th.
    tied = cluster_near_duplicates(
        ["a", "b"], vectors[:2], [None] * 2, [day2, day1]
    )
    assert tied[0].date == dt.date(2016, 10, 17)  # One each: earliest day wins.


def test_apex_without_lengths_falls_back_to_earliest_then_id() -> None:
    vectors = np.asarray([_unit(0.0)] * 3, dtype=np.float32)
    dates = [dt.datetime(2016, 10, 18), dt.datetime(2016, 10, 17), dt.datetime(2016, 10, 17)]
    clusters = cluster_near_duplicates(["z", "b", "a"], vectors, [None] * 3, dates)
    assert clusters[0].apex_doc_id == "a"  # Earliest day, then lexicographic id.


def test_input_validation() -> None:
    vectors = np.asarray([_unit(0.0), _unit(90.0)], dtype=np.float32)
    with pytest.raises(ValueError, match="urls"):
        cluster_near_duplicates(["a", "b"], vectors, [None], [None, None])
    with pytest.raises(ValueError, match="unique"):
        cluster_near_duplicates(["a", "a"], vectors, [None] * 2, [None] * 2)
    with pytest.raises(ValueError, match="embeddings"):
        cluster_near_duplicates(["a", "b"], vectors[:1], [None] * 2, [None] * 2)
    with pytest.raises(ValueError, match="text_lengths"):
        cluster_near_duplicates(["a", "b"], vectors, [None] * 2, [None] * 2, text_lengths=[1])
    assert cluster_near_duplicates([], np.zeros((0, 2), np.float32), [], []) == []


# ------------------------------------------------------------------ #
# The real CC-News slice                                             #
# ------------------------------------------------------------------ #


def test_every_document_lands_in_exactly_one_cluster(
    ccnews_clusters: list[EventCluster],
    ccnews_inputs: tuple[
        list[str], npt.NDArray[np.float32], list[str], list[dt.datetime | None], list[int]
    ],
) -> None:
    ids = ccnews_inputs[0]
    members = [doc_id for cluster in ccnews_clusters for doc_id in cluster.member_doc_ids]
    assert sorted(members) == sorted(ids)
    assert len(members) == len(set(members))
    for cluster in ccnews_clusters:
        assert cluster.apex_doc_id in cluster.member_doc_ids
        assert cluster.member_doc_ids == sorted(cluster.member_doc_ids)
        assert cluster.total_netlocs == len(cluster.netlocs)


def test_real_syndicated_pair_becomes_one_event(ccnews_clusters: list[EventCluster]) -> None:
    """Two publishers ran the same enterovirus story a day apart: one event."""
    cluster = _cluster_of(ccnews_clusters, SYNDICATED_PAIR[0])
    assert cluster.member_doc_ids == sorted(SYNDICATED_PAIR)
    assert cluster.netlocs == ["coventrytelegraph.net", "newsletter.co.uk"]
    assert cluster.total_netlocs == 2
    # One member per day: the tie degrades to the earliest (first-publish) day.
    assert cluster.date == dt.date(2016, 10, 17)
    # Apex is the longer copy (2964 vs 2917 chars): the 10-18 rewrite.
    assert cluster.apex_doc_id == "ccn-2d3cb032d63fe96c"


def test_real_same_publisher_pair_counts_one_netloc(
    ccnews_clusters: list[EventCluster],
) -> None:
    cluster = _cluster_of(ccnews_clusters, SAME_PUBLISHER_PAIR[0])
    assert cluster.member_doc_ids == sorted(SAME_PUBLISHER_PAIR)
    assert cluster.netlocs == ["coventrytelegraph.net"]
    assert cluster.total_netlocs == 1  # Breadth counts publishers, not copies.
    assert cluster.apex_doc_id == "ccn-0bbabb6be34c07f7"  # The 6546-char original.


def test_lsh_prefiltered_path_matches_exact(
    ccnews_inputs: tuple[
        list[str], npt.NDArray[np.float32], list[str], list[dt.datetime | None], list[int]
    ],
    ccnews_clusters: list[EventCluster],
) -> None:
    ids, matrix, urls, dates, lengths = ccnews_inputs
    accelerated = cluster_near_duplicates(
        ids, matrix, urls, dates, text_lengths=lengths, lsh_bits=256, seed=42
    )
    assert [c.model_dump(mode="json") for c in accelerated] == [
        c.model_dump(mode="json") for c in ccnews_clusters
    ]


# ------------------------------------------------------------------ #
# Determinism and serialization                                      #
# ------------------------------------------------------------------ #


def test_event_ids_are_content_addressed(ccnews_clusters: list[EventCluster]) -> None:
    cluster = _cluster_of(ccnews_clusters, SYNDICATED_PAIR[0])
    assert cluster.event_id.startswith("evt-")
    assert len(cluster.event_id) == 4 + 16
    # Same members in a permuted input order -> same event id.
    vectors = np.asarray([_unit(0.0)] * 2, dtype=np.float32)
    forward = cluster_near_duplicates(list(SYNDICATED_PAIR), vectors, [None] * 2, [None] * 2)
    backward = cluster_near_duplicates(
        list(reversed(SYNDICATED_PAIR)), vectors, [None] * 2, [None] * 2
    )
    assert forward[0].event_id == backward[0].event_id == cluster.event_id


def test_write_clusters_is_byte_deterministic(
    ccnews_clusters: list[EventCluster], tmp_path: Path
) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    write_clusters(first, ccnews_clusters)
    write_clusters(second, ccnews_clusters)
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text())
    assert len(payload) == len(ccnews_clusters)
    reloaded = [EventCluster.model_validate(record) for record in payload]
    assert reloaded == ccnews_clusters


# ------------------------------------------------------------------ #
# Document wrapper                                                   #
# ------------------------------------------------------------------ #


def test_cluster_documents_with_row_map(ccnews_rows: list[dict[str, object]]) -> None:
    wanted = set(SYNDICATED_PAIR) | set(SAME_PUBLISHER_PAIR)
    docs = [
        Document.model_validate(row) for row in ccnews_rows if str(row["id"]) in wanted
    ]
    row_map: dict[str, int] = json.loads(FIXTURE_ROW_MAP.read_text())
    embeddings = np.load(FIXTURE_EMBEDDINGS).astype(np.float32)
    clusters = cluster_documents(docs, embeddings, row_map=row_map)
    assert sorted(len(c.member_doc_ids) for c in clusters) == [2, 2]
    syndicated = _cluster_of(clusters, SYNDICATED_PAIR[0])
    assert syndicated.apex_doc_id == "ccn-2d3cb032d63fe96c"
    assert syndicated.total_netlocs == 2
