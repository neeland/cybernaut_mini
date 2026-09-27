"""Sparsity and scoped-vs-global precision, measured on the real fixture corpus.

The ambiguous unigram is the post's own example, "chase". The 460 committed
CC-News/MIRACL documents genuinely contain it as a police-pursuit noun, a baseball
verb ("to chase Arrieta") and the venue "Chase Field", so the precision numbers
below are measured, not staged. Collections and relevance are both derived
mechanically from the text itself — no invented judgments.

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "Not all entities exist in every collection … the distribution of entities is
    extremely sparse in the average case. So, learning collection-specific models
    saves a tonne of compute cost and minimizes spurious matches." Local copy:
    ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions:
    - The mechanical collection rule scopes the venue entity to documents that fold
      to contain "phoenix" (Chase Field is in Phoenix), and the mechanical relevance
      oracle is the unambiguous bigram "chase field" — a longer pattern of the same
      entity, which is exactly the evidence a coherent collection would carry.
    - The pinned counts (5 global matches, 1 relevant) are properties of the
      committed corpus; if the fixtures change, these numbers should change with
      them and the test should be re-derived, not loosened.

Alternatives rejected:
    - A synthetic occupancy matrix for the sparsity test: the store-backed path is
      three inserts away from real tags, so the test uses the tags table itself with
      real fixture chunk ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cybernaut_mini.entities.metrics import ambiguous_term_precision, occupancy
from cybernaut_mini.entities.store import EntityStore
from cybernaut_mini.query.s8_retrieve.intent_scan import fold

FIXTURE_DOCS = Path("data/01_raw/fixtures/documents.jsonl")


def _corpus() -> list[tuple[str, str, str]]:
    """``(collection_id, chunk_id, text)`` with a mechanical collection rule."""
    triples: list[tuple[str, str, str]] = []
    for line in FIXTURE_DOCS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        collection_id = "phoenix" if "phoenix" in fold(row["text"]) else "general"
        triples.append((collection_id, row["id"], row["text"]))
    return triples


def test_scoped_precision_beats_global_on_the_ambiguous_unigram() -> None:
    report = ambiguous_term_precision(
        "chase",
        _corpus(),
        scoped_collections={"phoenix"},
        is_relevant=lambda _collection, _chunk, text: "chase field" in fold(text),
    )
    # Globally, "chase" fires on five real documents and only the Chase Field one
    # concerns the entity: precision 1/5.
    assert report.global_matches == 5
    assert report.global_relevant == 1
    assert report.global_precision == pytest.approx(0.2)
    # Scoped to the coherent collection, the same pattern is perfectly precise —
    # the spurious matches live in collections where the pattern would not exist.
    assert (report.scoped_matches, report.scoped_relevant) == (1, 1)
    assert report.scoped_precision == 1.0
    assert report.scoped_precision > report.global_precision


def test_zero_matches_report_zero_not_nan() -> None:
    report = ambiguous_term_precision(
        "zzyzzx",
        _corpus(),
        scoped_collections={"phoenix"},
        is_relevant=lambda *_: True,
    )
    assert report.global_matches == 0
    assert report.global_precision == 0.0
    assert report.scoped_precision == 0.0


def test_occupancy_measures_sparsity_over_the_tags_table() -> None:
    with EntityStore() as store:
        # Three collections, two entities, three occupied cells out of six: the
        # venue exists only where its documents are, which is the whole argument.
        store.write_tags("phoenix", "ccn-ecd900247927d44d", [("Q683659", "chase field")])
        store.write_tags("general", "ccn-4fff8c81dee64495", [("Q192314", "chase")])
        store.write_tags("markets", "ccn-0b3dffccaeb036dc", [("Q192314", "jpmorgan")])
        report = occupancy(store)
    assert (report.n_entities, report.n_collections) == (2, 3)
    assert report.occupied_cells == 3
    assert report.total_cells == 6
    assert report.density == pytest.approx(0.5)
    assert report.sparsity == pytest.approx(0.5)


def test_occupancy_of_an_empty_loop_is_degenerate_but_defined() -> None:
    with EntityStore() as store:
        report = occupancy(store)
    assert report.total_cells == 0
    assert report.density == 0.0
    assert report.sparsity == 1.0
