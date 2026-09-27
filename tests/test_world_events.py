"""WORLD event store + tag layers over the real CC-News fixture slice.

The end-to-end assertions run over the 200 committed CC-News documents and their
frozen hash-256 embeddings: the real enterovirus-D68 syndication pair must become
one breadth-2 event, the store must round-trip byte-identically through JSONL and
parquet, and the tag layers must fill entities, tickers (real META/AMZN
resolutions present in the slice), and countries. Point-in-time safety is pinned
as the prefix property: building from a date-prefix of the corpus yields
byte-identical records for the shared events.

Blog ref: https://nosible.com/blog/point-in-time-knowledge-graphs-over-named-entities
    — the nested record shape; https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world
    — breadth as ``total_netlocs``;
    https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty
    — coverage-peak dating. Local copies under ``docs/blog-archive/``.

Assumptions:
    - The syndicated pair's ids, and the META/AMZN ticker hits, are properties of
      committed fixture bytes — pinning frozen data, not fitting tests to code.
    - Country-resolver unit rows ("Indiana" never resolves, "Georgia" drops,
      "U.S." resolves, native scripts resolve, "TV" never resolves) are the
      post's own examples plus the code-ambiguity rule.

Alternatives rejected: synthetic documents (forbidden — the fixture provides real
syndication); asserting exact event counts (brittle under threshold changes; the
partition and prefix properties are the invariants).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from cybernaut_mini.world import countries, events, ner, tickers
from world_helpers import (
    SYNDICATED_PAIR,
    ccnews_documents,
    fixture_matrix,
    world_events,
)

# ------------------------------------------------------------------ #
# Event store over the real corpus                                   #
# ------------------------------------------------------------------ #


def test_every_document_lands_in_exactly_one_event() -> None:
    table = world_events()
    members = [doc_id for event in table for doc_id in event.member_doc_ids]
    assert sorted(members) == sorted(doc.id for doc in ccnews_documents())


def test_syndicated_pair_is_one_event_with_breadth_two() -> None:
    table = world_events()
    (event,) = [e for e in table if set(SYNDICATED_PAIR) <= set(e.member_doc_ids)]
    assert event.breadth == 2
    assert event.coverage.total_coverage >= 2
    assert event.apex_doc_id in event.member_doc_ids
    assert event.event.date is not None  # the coverage-peak day exists


def test_embeddings_are_unit_norm_centroids() -> None:
    matrix = events.embedding_matrix(list(world_events()))
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_materiality_is_normalized_by_expanding_max_only() -> None:
    table = world_events()
    running_max = 0
    for event in table:
        running_max = max(running_max, event.breadth)
        expected = float(np.log1p(event.breadth) / np.log1p(running_max)) if running_max else 0.0
        assert event.signals.materiality_score == expected


def test_prefix_build_is_byte_identical_for_shared_events() -> None:
    """Point-in-time rule: later data must not change earlier records."""
    docs = list(ccnews_documents())
    embeddings, row_map = fixture_matrix()
    cutoff = dt.date(2016, 10, 18)
    prefix_docs = [
        doc for doc in docs if doc.published_at and doc.published_at.date() <= cutoff
    ]
    from cybernaut_mini.dedup import cluster_documents

    full = events.build_events(
        cluster_documents(docs, embeddings, row_map=row_map), docs, embeddings, row_map=row_map
    )
    prefix = events.build_events(
        cluster_documents(prefix_docs, embeddings, row_map=row_map),
        prefix_docs,
        embeddings,
        row_map=row_map,
    )
    full_by_id = {event.event_id: event for event in full}
    shared = [event for event in prefix if event.event_id in full_by_id]
    assert shared, "the prefix must share events with the full build"
    for event in shared:
        assert event.model_dump(mode="json") == full_by_id[event.event_id].model_dump(mode="json")


def test_jsonl_roundtrip_is_byte_identical(tmp_path: Path) -> None:
    table = list(world_events())
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    events.write_events_jsonl(first, table)
    events.write_events_jsonl(second, events.read_events_jsonl(first))
    assert first.read_bytes() == second.read_bytes()


def test_parquet_roundtrip_preserves_every_field(tmp_path: Path) -> None:
    table = list(world_events())
    path = tmp_path / "events.parquet"
    events.write_events_parquet(path, table)
    restored = events.read_events_parquet(path)
    assert [e.model_dump(mode="json") for e in restored] == [
        e.model_dump(mode="json") for e in table
    ]


# ------------------------------------------------------------------ #
# Tag layers                                                         #
# ------------------------------------------------------------------ #


def test_entities_are_typed_mention_counts() -> None:
    tagged = [event for event in world_events() if event.entities]
    assert tagged, "the real slice must yield entities"
    for event in tagged:
        for type_name, counts in event.entities.items():
            assert type_name.isupper()
            assert all(count >= 1 for count in counts.values())
        assert sorted(event.entities.get("GPE", {})) == event.ent_gpe


def test_real_ticker_resolutions_from_the_slice() -> None:
    resolved = {ref.ticker_eodhd for event in world_events() for ref in event.tickers}
    # Facebook/Meta and Amazon coverage exists in the committed CC-News slice.
    assert {"META.US", "AMZN.US"} <= resolved


def test_ticker_resolution_is_exact_never_fuzzy() -> None:
    assert tickers.resolve_org("Apple") is not None
    assert tickers.resolve_org("Apple Inc.") is not None
    assert tickers.resolve_org("Apple Daily") is None  # substring must not match
    assert tickers.resolve_org("Pineapple") is None


def test_surname_folding_and_suffix_merging_rules() -> None:
    merged = ner.merge_corporate_suffixes({"Apple": 3, "Apple Inc.": 2, "apple": 1})
    assert merged == {"Apple": 6}
    folded = ner.fold_surnames({"Theresa May": 2, "May": 1})
    assert folded == {"Theresa May": 3}
    # Ambiguous surname stays its own node.
    kept = ner.fold_surnames({"Theresa May": 1, "James May": 1, "May": 1})
    assert kept["May"] == 1


def test_country_resolver_published_examples() -> None:
    assert countries.resolve("Indiana") is None  # never a partial match to India
    assert countries.resolve("Georgia") is None  # the post's own dropped alias
    assert countries.resolve("U.S.") == "US"
    assert countries.resolve("US") == "US"
    assert countries.resolve("UK") == "GB"
    assert countries.resolve("USA") == "US"
    assert countries.resolve("China") == "CN"
    assert countries.resolve("中国") == "CN"  # native script resolves
    assert countries.resolve("россия") == "RU"
    assert countries.resolve("Americans") == "US"  # demonym plural
    assert countries.resolve("TV") is None  # ambiguous alpha-2 codes never match
    assert countries.resolve("IN") is None
    assert countries.resolve("Thai") is None  # demonyms under four letters never ship


def test_attribution_is_main_country_union_resolved_gpes() -> None:
    table = world_events()
    attributions = countries.attributions(table)
    assert any(attributions), "the real slice must attribute some events"
    for event, attribution in zip(table, attributions, strict=True):
        if event.event.country:
            main = countries.resolve(event.event.country)
            assert main is not None and main in attribution
        assert len(attribution) <= 15
        assert list(attribution) == sorted(attribution)
