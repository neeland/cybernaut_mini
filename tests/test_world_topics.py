"""IPTC-ish topic classification and the exact-code geopolitical filter.

The filter tests transcribe the post's truth table: level-1 ``conflict, war and
peace`` passes, level-2 ``international relations`` passes, the four level-3
leaves pass, and *label words never leak* — a sport event whose real headline
contains "tug-of-war" (or a weather warning) must not pass, because codes are
compared exactly, never the words in the labels. Classification tests run the
embedding-nearest-topic path over the real fixture events in the frozen hash
space, asserting the single-label constraint and determinism.

Blog ref: https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world
    — the exact boolean and "so 'weather warning' or 'tug-of-war' never leak
    in". Local copy under ``docs/blog-archive/``.

Assumptions: the filter tests need events carrying specific topic codes; those
are real fixture events with their topic *fields* set by the classifier under
test or copied via ``model_copy`` — no invented event records.

Alternatives rejected: asserting which topic any real event lands on (hash-space
cosines carry no semantics to justify such a pin).
"""

from __future__ import annotations

from cybernaut_mini.world import topics
from cybernaut_mini.world.events import embedding_matrix
from world_helpers import frozen_embedder, world_events


def test_topic_table_loads_the_committed_ontology_subset() -> None:
    table = topics.load_topics()
    level_1 = {row.level_1 for row in table.rows}
    assert topics.GEOPOLITICAL_LEVEL_1 in level_1
    assert len(level_1) == 17  # the 17 top-level IPTC Media Topics
    leaves = {row.level_3 for row in table.rows if row.level_3}
    assert leaves == set(topics.GEOPOLITICAL_LEVEL_3)
    assert any(row.level_2 == topics.GEOPOLITICAL_LEVEL_2 for row in table.rows)
    assert all(row.description for row in table.rows)


def test_single_label_per_event_and_determinism() -> None:
    table = list(world_events())
    embedder = frozen_embedder()
    first = topics.tag_events(table, embedder)
    second = topics.tag_events(table, embedder)
    ontology = topics.load_topics()
    valid_paths = {(row.level_1, row.level_2, row.level_3) for row in ontology.rows}
    for one, two in zip(first, second, strict=True):
        assert (one.iptc_level_1, one.iptc_level_2, one.iptc_level_3) in valid_paths
        assert one.iptc_level_1 == two.iptc_level_1
        assert one.iptc_level_2 == two.iptc_level_2
        assert one.iptc_level_3 == two.iptc_level_3


def test_geopolitical_truth_table_exact_codes_only() -> None:
    base = world_events()[0]
    conflict = base.model_copy(update={"iptc_level_1": "conflict, war and peace"})
    assert topics.geopolitical(conflict)
    relations = base.model_copy(
        update={"iptc_level_1": "politics", "iptc_level_2": "international relations"}
    )
    assert topics.geopolitical(relations)
    for leaf in ("war crime", "genocide", "terrorism", "nuclear policy"):
        event = base.model_copy(
            update={"iptc_level_1": "crime, law and justice", "iptc_level_3": leaf}
        )
        assert topics.geopolitical(event)
    economy = base.model_copy(update={"iptc_level_1": "economy, business and finance"})
    assert not topics.geopolitical(economy)
    untagged = base.model_copy(
        update={"iptc_level_1": None, "iptc_level_2": None, "iptc_level_3": None}
    )
    assert not topics.geopolitical(untagged)


def test_label_words_never_leak_tug_of_war_and_weather_warning() -> None:
    """A sport event with 'tug-of-war' in its (real) title must not pass the filter."""
    base = world_events()[0]
    # Codes are compared exactly: 'sport' and 'weather' stay out no matter what
    # words the headline or the label share with the conflict bucket.
    tug_of_war = base.model_copy(update={"iptc_level_1": "sport"})
    assert "war" in "tug-of-war"  # the word-level trap the post calls out
    assert not topics.geopolitical(tug_of_war)
    weather_warning = base.model_copy(update={"iptc_level_1": "weather"})
    assert not topics.geopolitical(weather_warning)


def test_geopolitical_mask_matches_scalar_filter() -> None:
    embedder = frozen_embedder()
    tagged = topics.tag_events(list(world_events()), embedder)
    mask = topics.geopolitical_mask(tagged)
    assert mask.tolist() == [topics.geopolitical(event) for event in tagged]
    assert mask.shape[0] == len(tagged)


def test_nearest_topics_is_one_matmul_argmax() -> None:
    table = topics.load_topics()
    embedder = frozen_embedder()
    topic_matrix = embedder.embed_documents(table.descriptions())
    events_matrix = embedding_matrix(list(world_events()))
    rows = topics.nearest_topics(events_matrix, topic_matrix, table)
    assert len(rows) == events_matrix.shape[0]
    scores = events_matrix @ topic_matrix.T
    for index, row in enumerate(rows):
        best = table.rows[int(scores[index].argmax())]
        assert row == best
