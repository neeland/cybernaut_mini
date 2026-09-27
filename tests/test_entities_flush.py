"""Flush over real corpus text: stale-only rescans, version stamps, importance order.

Chunk texts are real CC-News/MIRACL documents from
``data/01_raw/fixtures/documents.jsonl`` and the patterns are drawn from the post's
own JPMorgan example ("chase", "chase field"), so the rescan counts are pinned
against genuine text, not synthetic sentences.

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "Every new document we index is written to the collection it belongs to and every
    so often those writes are flushed to disk. Larger or most important collections
    are flushed more frequently … when the collection is flushed it will check for
    new patterns. If new patterns are found it will tag all untagged chunks and
    update the patterns associated with that collection. This happens in seconds."
    Local copy: ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions:
    - "Only stale chunks" is the load-bearing claim, so the tests pin
      ``chunks_scanned`` exactly: 0 when nothing changed, only-the-new-chunk when a
      write arrives without new patterns, and everything stale after a version bump.
    - The scheduler formula under test is the repo's inferred
      ``w1 * pending + w2 * recent_queries`` with the default 1.0/0.5 weights; the
      priority arithmetic is asserted numerically.

Alternatives rejected:
    - Timing assertions on the post's "seconds" claim: wall-clock bounds flake on
      shared machines; the report carries ``seconds`` and the test asserts it is
      measured (non-negative), while the *reason* it stays small — stale-only
      scanning — is what gets pinned.
"""

from __future__ import annotations

import json
from pathlib import Path

from cybernaut_mini.entities.flush import (
    FlushBuffer,
    FlushScheduler,
    flush_collection,
    run_flush_cycle,
    texts_from,
)
from cybernaut_mini.entities.store import EntityStore

FIXTURE_DOCS = Path("data/01_raw/fixtures/documents.jsonl")

#: Real fixture documents: two match the standalone unigram "chase", one of them is
#: the Phoenix doc that also contains the bigram "chase field"; one contains only
#: "purchased"/"chased" and must never match.
CHASE_ARRIETA = "ccn-4fff8c81dee64495"
CHASE_FIELD = "ccn-ecd900247927d44d"
NO_MATCH = "ccn-0a7e4496a9cac29e"


def _texts() -> dict[str, str]:
    rows = [json.loads(line) for line in FIXTURE_DOCS.read_text(encoding="utf-8").splitlines()]
    wanted = {CHASE_ARRIETA, CHASE_FIELD, NO_MATCH}
    return {row["id"]: row["text"] for row in rows if row["id"] in wanted}


def test_flush_scans_new_chunks_and_stamps_versions() -> None:
    texts = _texts()
    with EntityStore() as store:
        buffer = FlushBuffer()
        store.add_patterns("news", "Q192314", ["chase"])  # version 1
        for chunk_id, text in sorted(texts.items()):
            buffer.add("news", chunk_id, text)
        report = flush_collection(store, buffer, "news", texts_from(texts))
        assert report.version == 1
        assert report.new_chunks == 3
        assert report.chunks_scanned == 3
        assert report.tags_written == 2  # the two real "chase" documents
        assert report.seconds >= 0.0
        assert {c for c, _, _ in store.tags_for("news")} == {CHASE_ARRIETA, CHASE_FIELD}
        # Every chunk is stamped with the version it was scanned against.
        assert store.stale_chunks("news") == []


def test_flush_rescans_only_stale_chunks() -> None:
    texts = _texts()
    with EntityStore() as store:
        buffer = FlushBuffer()
        store.add_patterns("news", "Q192314", ["chase"])
        buffer.add("news", CHASE_ARRIETA, texts[CHASE_ARRIETA])
        buffer.add("news", NO_MATCH, texts[NO_MATCH])
        flush_collection(store, buffer, "news", texts_from(texts))

        # Nothing pending, no new patterns: a flush is a no-op.
        report = flush_collection(store, buffer, "news", texts_from(texts))
        assert (report.new_chunks, report.chunks_scanned, report.tags_written) == (0, 0, 0)

        # A new write without new patterns rescans exactly that one chunk.
        buffer.add("news", CHASE_FIELD, texts[CHASE_FIELD])
        report = flush_collection(store, buffer, "news", texts_from(texts))
        assert report.new_chunks == 1
        assert report.stale_chunks == 0
        assert report.chunks_scanned == 1
        assert report.tags_written == 1

        # New patterns bump the version, so every chunk is stale again; the rescan
        # pulls stale texts through the read-only chunk_text provider and the
        # replace semantics pick up the new pattern retroactively (back-tagging).
        store.add_patterns("news", "Q192314", ["chase field"])
        report = flush_collection(store, buffer, "news", texts_from(texts))
        assert report.version == 2
        assert (report.new_chunks, report.stale_chunks, report.chunks_scanned) == (0, 3, 3)
        assert ("chase field" in {p for _, _, p in store.tags_for("news", CHASE_FIELD)})


def test_cold_start_chunks_wait_untagged_until_patterns_exist() -> None:
    texts = _texts()
    with EntityStore() as store:
        buffer = FlushBuffer()
        buffer.add("news", CHASE_ARRIETA, texts[CHASE_ARRIETA])
        report = flush_collection(store, buffer, "news", texts_from(texts))
        # Version 0: registered but never scanned, so no tags and no version stamp.
        assert (report.version, report.chunks_scanned, report.tags_written) == (0, 0, 0)
        store.add_patterns("news", "Q192314", ["chase"])
        report = flush_collection(store, buffer, "news", texts_from(texts))
        assert (report.chunks_scanned, report.tags_written) == (1, 1)


def test_scheduler_weighs_pending_writes_over_queries() -> None:
    buffer = FlushBuffer()
    scheduler = FlushScheduler()  # w_pending=1.0, w_queries=0.5
    buffer.add("big", "doc-1", "text")
    buffer.add("big", "doc-2", "text")
    buffer.add("hot", "doc-3", "text")
    for _ in range(3):
        buffer.record_query("hot")
    assert scheduler.priority(buffer, "big") == 2.0
    assert scheduler.priority(buffer, "hot") == 2.5
    assert scheduler.order(buffer) == ["hot", "big"]
    assert scheduler.next_collection(buffer) == "hot"
    # Draining resets both pressure signals; an idle collection never flushes.
    buffer.drain("hot")
    buffer.drain("big")
    assert scheduler.next_collection(buffer) is None


def test_run_flush_cycle_respects_the_budget() -> None:
    texts = _texts()
    with EntityStore() as store:
        buffer = FlushBuffer()
        for collection_id in ("alpha", "beta"):
            store.add_patterns(collection_id, "Q192314", ["chase"])
        buffer.add("alpha", CHASE_ARRIETA, texts[CHASE_ARRIETA])
        buffer.add("beta", CHASE_FIELD, texts[CHASE_FIELD])
        buffer.add("beta", NO_MATCH, texts[NO_MATCH])
        reports = run_flush_cycle(
            store, buffer, FlushScheduler(), texts_from(texts), budget=1
        )
        # Budget 1 flushes only the most important collection (beta: 2 pending).
        assert [r.collection_id for r in reports] == ["beta"]
        assert buffer.pending_count("alpha") == 1
        reports = run_flush_cycle(
            store, buffer, FlushScheduler(), texts_from(texts), budget=5
        )
        assert [r.collection_id for r in reports] == ["alpha"]
