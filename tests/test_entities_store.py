"""Entity-loop state: the suggestion UPSERT, versioned patterns, staleness, promotion.

Everything here runs against an in-memory SQLite store (plus one on-disk round trip),
so the tests are exact: counts, versions and statuses are pinned, not ranged.

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "Those suggestions are accumulated in a small database and, once a certain
    threshold is met, the suggestion is sent to the Resolver Agent." Local copy:
    ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions:
    - The promotion threshold under test is 3, the repo's reading of the post's
      "certain threshold"; the store itself takes it as a parameter, and the test
      pins the boundary (2 sightings stay pending, the 3rd promotes).
    - Version arithmetic is the load-bearing behaviour: adding known patterns must
      NOT bump the version, because a bump marks every chunk in the collection
      stale and the whole flush-cost story rests on that not happening spuriously.

Alternatives rejected:
    - Mocking sqlite3: the store *is* SQL; the UPSERT and RETURNING clauses are the
      implementation, and a mock would test a paraphrase of them.
"""

from __future__ import annotations

from pathlib import Path

from cybernaut_mini.entities.store import (
    SUGGESTION_PENDING,
    SUGGESTION_READY,
    SUGGESTION_RESOLVED,
    EntityStore,
)
from cybernaut_mini.entities.suggest import PROMOTION_THRESHOLD, accumulate
from cybernaut_mini.entities.tagger import CollectionTagger


def test_upsert_suggestion_counts_up() -> None:
    with EntityStore() as store:
        assert store.upsert_suggestion("c1", "acme corp") == 1
        assert store.upsert_suggestion("c1", "acme corp") == 2
        # Same surface in another collection is a separate accumulator.
        assert store.upsert_suggestion("c2", "acme corp") == 1
        (row,) = store.suggestions("c2")
        assert (row.surface_norm, row.count, row.status) == ("acme corp", 1, SUGGESTION_PENDING)


def test_promotion_happens_exactly_at_threshold() -> None:
    with EntityStore() as store:
        store.upsert_suggestion("c1", "acme corp")
        store.upsert_suggestion("c1", "acme corp")
        assert store.promote_ready("c1", PROMOTION_THRESHOLD) == []
        store.upsert_suggestion("c1", "acme corp")
        assert store.promote_ready("c1", PROMOTION_THRESHOLD) == ["acme corp"]
        # Already promoted: a second sweep returns nothing, whatever the count does.
        store.upsert_suggestion("c1", "acme corp")
        assert store.promote_ready("c1", PROMOTION_THRESHOLD) == []
        (row,) = store.suggestions("c1")
        assert row.status == SUGGESTION_READY


def test_suggestion_status_lifecycle() -> None:
    with EntityStore() as store:
        store.upsert_suggestion("c1", "acme corp")
        store.set_suggestion_status("c1", "acme corp", SUGGESTION_RESOLVED)
        (row,) = store.suggestions("c1")
        assert row.status == SUGGESTION_RESOLVED
        assert store.suggestions("c1", status=SUGGESTION_PENDING) == []


def test_add_patterns_versions_only_on_novelty() -> None:
    with EntityStore() as store:
        assert store.pattern_version("c1") == 0
        assert store.add_patterns("c1", "Q1", ["acme", "acme corp"]) == 1
        # Nothing new: version must not move.
        assert store.add_patterns("c1", "Q1", ["acme"]) == 1
        # A new pattern for the same entity bumps once.
        assert store.add_patterns("c1", "Q1", ["acme corp", "acme inc"]) == 2
        # The same surface for a DIFFERENT entity is a new (pattern, entity) row.
        assert store.add_patterns("c1", "Q2", ["acme"]) == 3
        assert store.active_patterns("c1") == [
            ("acme", "Q1"),
            ("acme", "Q2"),
            ("acme corp", "Q1"),
            ("acme inc", "Q1"),
        ]


def test_stale_chunks_track_pattern_version() -> None:
    with EntityStore() as store:
        store.register_chunk("c1", "doc-1")
        store.register_chunk("c1", "doc-2")
        # No patterns yet: version 0, nothing is stale.
        assert store.stale_chunks("c1") == []
        store.add_patterns("c1", "Q1", ["acme"])
        assert store.stale_chunks("c1") == ["doc-1", "doc-2"]
        store.mark_chunks_tagged("c1", ["doc-1", "doc-2"], 1)
        assert store.stale_chunks("c1") == []
        store.add_patterns("c1", "Q1", ["acme corp"])
        assert store.stale_chunks("c1") == ["doc-1", "doc-2"]


def test_tags_replace_and_report() -> None:
    with EntityStore() as store:
        assert store.write_tags("c1", "doc-1", [("Q1", "acme")]) == 1
        assert store.write_tags("c1", "doc-1", [("Q1", "acme"), ("Q1", "acme corp")]) == 2
        assert store.tags_for("c1", "doc-1") == [
            ("doc-1", "Q1", "acme"),
            ("doc-1", "Q1", "acme corp"),
        ]
        assert store.occupancy_cells() == [("Q1", "c1")]


def test_entities_round_trip_canonical_json(tmp_path: Path) -> None:
    path = tmp_path / "state" / "entities.sqlite"
    with EntityStore(path) as store:
        store.put_entity("Q1", {"name_orig": "Acme", "is_company": True}, "resolver-x", ["a", "b"])
    with EntityStore(path) as store:
        entity = store.get_entity("Q1")
        assert entity is not None
        assert entity.record == {"is_company": True, "name_orig": "Acme"}
        assert entity.sources == ("a", "b")
        assert store.get_entity("Q2") is None


def test_accumulate_skips_covered_and_promotes() -> None:
    tagger = CollectionTagger([("jpmorgan chase", "Q192314")])
    with EntityStore() as store:
        batches = [
            ["JPMorgan Chase & Co.", "Acme Robotics", "xx"],  # covered / new / too short
            ["Acme Robotics"],
            ["acme robotics"],
        ]
        updates = [accumulate(store, "c1", batch, tagger) for batch in batches]
        assert updates[0].skipped_covered == ("jpmorgan chase & co.",)
        assert updates[0].accumulated == ("acme robotics",)
        assert updates[0].promoted == ()
        assert updates[1].promoted == ()
        # Third sighting (case-folded into the same row) crosses the threshold.
        assert updates[2].promoted == ("acme robotics",)
