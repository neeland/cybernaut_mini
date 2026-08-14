"""Tests for blog stage 7: shard-based question expansion.

The blog's worked example expands nine original stems with twenty-one new ones. Its exact
terms ("mojica", "raffinos", "bacto") are properties of NOSIBLE's 250M-document corpus and
cannot be reproduced from this repo's fixtures, so what is asserted here is the *shape*
the example demonstrates and the claims the post makes in prose:

* originals are preserved and marked, additions are marked ``[NEW]``;
* the number of additions is bounded relative to the number of originals;
* the same token expands differently in shards with different senses;
* terms that occur in nearly every shard are dropped;
* the same input gives the same output, in both selection modes;
* an absent or empty graph degrades to a no-op.
"""

from __future__ import annotations

import numpy as np
import pytest

from cybernaut_mini.indexing import LoadedIndex
from cybernaut_mini.models import Document, IndexMeta, ShardKeyword, ShardManifest
from cybernaut_mini.query.s7_expand import (
    Candidate,
    ExpandedTerm,
    Expansion,
    ExpansionConfig,
    ShardGraph,
    TermFilter,
    UbiquityFilter,
    activate,
    expand_query,
    expand_question,
    expand_terms,
    expansion_cap,
    filter_candidates,
    neighbour_distribution,
    question_terms,
    rank_weights,
    select,
    shard_graphs_from_index,
)

# --------------------------------------------------------------------------- #
# Fixtures: two shards that disagree about what "gene" means.                  #
# --------------------------------------------------------------------------- #

#: Shard 11,343 in the post: "gene" only has the genetic meaning.
GENETICS_GRAPH: dict[str, dict[str, float]] = {
    "gene": {"crispr": 8.0, "genom": 6.0, "prokaryot": 3.0, "chromosom": 3.0, "the": 9.0},
    "bacteria": {"archaea": 5.0, "bacteriophag": 4.0, "pathogen": 3.0, "phage": 2.0},
    "yeast": {"galactos": 4.0, "dextros": 3.0, "raffinos": 2.0},
}

#: Some other shard, where "Gene" is a name that co-occurs with "Willy Wonka".
CHOCOLATE_GRAPH: dict[str, dict[str, float]] = {
    "gene": {"wonka": 9.0, "wilder": 5.0, "chocolat": 4.0, "oompa": 2.0},
    "bacteria": {"hygien": 1.0},
}

ORIGINALS = ("gene", "bacteria", "yeast")


def _genetics(weight: float = 1.0) -> ShardGraph:
    return ShardGraph(shard_id=11343, weight=weight, graph=GENETICS_GRAPH)


def _chocolate(weight: float = 1.0) -> ShardGraph:
    return ShardGraph(shard_id=90210, weight=weight, graph=CHOCOLATE_GRAPH)


def _manifest(
    shard_id: int,
    *,
    term_graph: dict[str, dict[str, float]],
    keyword_terms: tuple[str, ...] = (),
) -> ShardManifest:
    return ShardManifest(
        shard_id=shard_id,
        document_ids=[f"doc-{shard_id}"],
        centroid=[0.0] * 4,
        title=f"shard-{shard_id}",
        summary=f"summary for shard {shard_id}",
        keywords=[ShardKeyword(term=term, weight=0.5) for term in keyword_terms],
        entities=[],
        term_graph=term_graph,
        document_count=1,
        embedding_model="hash-4",
    )


def _index(manifests: list[ShardManifest]) -> LoadedIndex:
    """A LoadedIndex shell carrying the given manifests and nothing else of substance."""
    docs = [
        Document(id=f"doc-{m.shard_id}", title=f"T{m.shard_id}", text=f"text {m.shard_id}")
        for m in manifests
    ]
    meta = IndexMeta(
        embedding_model="hash-4",
        embedding_dim=4,
        n_shards=len(manifests),
        n_documents=len(docs),
        seed=0,
    )
    return LoadedIndex(
        meta=meta,
        documents=docs,
        vectors=np.zeros((len(docs), 4), dtype=np.float32),
        row_map={doc.id: i for i, doc in enumerate(docs)},
        manifests={m.shard_id: m for m in manifests},
        doc_tokens={doc.id: ["placeholder"] for doc in docs},
    )


@pytest.fixture()
def two_sense_index() -> LoadedIndex:
    """Two shards whose graphs give "gene" incompatible senses, plus a filler shard."""
    return _index(
        [
            _manifest(11343, term_graph=GENETICS_GRAPH, keyword_terms=("gene", "bacteria")),
            _manifest(90210, term_graph=CHOCOLATE_GRAPH, keyword_terms=("gene", "wonka")),
            _manifest(7, term_graph={"solar": {"panel": 2.0}}, keyword_terms=("solar",)),
        ]
    )


# --------------------------------------------------------------------------- #
# graph.py                                                                     #
# --------------------------------------------------------------------------- #


def test_neighbour_distribution_is_l1_normalised() -> None:
    distribution = neighbour_distribution(GENETICS_GRAPH, "bacteria")
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert distribution["archaea"] == pytest.approx(5.0 / 14.0)


def test_neighbour_distribution_drops_non_positive_edges() -> None:
    graph = {"gene": {"crispr": 3.0, "noise": 0.0, "anti": -1.0}}
    assert neighbour_distribution(graph, "gene") == {"crispr": 1.0}


def test_neighbour_distribution_of_absent_or_dead_node_is_empty() -> None:
    assert neighbour_distribution(GENETICS_GRAPH, "solar") == {}
    assert neighbour_distribution({"gene": {}}, "gene") == {}
    assert neighbour_distribution({"gene": {"x": 0.0}}, "gene") == {}


def test_rank_weights_decay_and_dedupe() -> None:
    weights = rank_weights([5, 2, 9, 5])
    assert weights == {5: 1.0, 2: 0.5, 9: pytest.approx(1 / 3)}
    flat = rank_weights([5, 2, 9], k=1000.0)
    assert flat[5] > flat[2] > flat[9]
    assert flat[9] > 0.99  # large k approaches uniform


def test_rank_weights_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        rank_weights([1], k=0.0)


def test_shard_graph_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ShardGraph(shard_id=1, weight=-0.5, graph={})


def test_shard_graphs_from_index_preserves_order_and_skips_unknown(
    two_sense_index: LoadedIndex,
) -> None:
    graphs = shard_graphs_from_index(
        two_sense_index, [90210, 11343, 4242, 90210], shard_weights={90210: 0.25}
    )
    assert [g.shard_id for g in graphs] == [90210, 11343]
    assert [g.weight for g in graphs] == [0.25, 1.0]
    assert graphs[0].graph is two_sense_index.manifests[90210].term_graph


# --------------------------------------------------------------------------- #
# selection.py                                                                 #
# --------------------------------------------------------------------------- #


def test_activation_is_a_probability_distribution() -> None:
    candidates = activate(ORIGINALS, [_genetics(0.7), _chocolate(0.3)])
    assert sum(c.score for c in candidates.values()) == pytest.approx(1.0)
    assert all(c.score > 0.0 for c in candidates.values())


def test_activation_records_sources_and_shards() -> None:
    candidates = activate(["gene", "bacteria"], [_genetics(), _chocolate()])
    assert candidates["wonka"].sources == ("gene",)
    assert candidates["wonka"].shards == (90210,)
    assert candidates["archaea"].shards == (11343,)
    assert candidates["archaea"].support == 1


def test_activation_is_independent_of_original_ordering() -> None:
    forwards = activate(["gene", "bacteria", "yeast"], [_genetics(0.6), _chocolate(0.4)])
    backwards = activate(["yeast", "bacteria", "gene"], [_genetics(0.6), _chocolate(0.4)])
    assert {t: c.score for t, c in forwards.items()} == {t: c.score for t, c in backwards.items()}


def test_activation_ignores_zero_weight_shards() -> None:
    candidates = activate(ORIGINALS, [_genetics(1.0), _chocolate(0.0)])
    assert "wonka" not in candidates
    assert sum(c.score for c in candidates.values()) == pytest.approx(1.0)


def test_shard_weight_shifts_the_ranking() -> None:
    genetics_first = expand_terms(["gene"], [_genetics(0.9), _chocolate(0.1)])
    chocolate_first = expand_terms(["gene"], [_genetics(0.1), _chocolate(0.9)])
    assert genetics_first.new_terms[0] == "crispr"
    assert chocolate_first.new_terms[0] == "wonka"


def test_expansion_cap_arithmetic() -> None:
    assert expansion_cap(9) == 18
    assert expansion_cap(9, max_new_per_original=2.5) == 22
    assert expansion_cap(9, max_new_per_original=2.5, hard_cap=20) == 20
    assert expansion_cap(0) == 0
    assert expansion_cap(1, max_new_per_original=0.5) == 0
    with pytest.raises(ValueError, match="max_new_per_original"):
        expansion_cap(3, max_new_per_original=-1.0)


def test_select_top_k_takes_the_highest_scores() -> None:
    candidates = [
        Candidate(term="a", score=0.5, sources=(), shards=()),
        Candidate(term="b", score=0.3, sources=(), shards=()),
        Candidate(term="c", score=0.2, sources=(), shards=()),
    ]
    assert [c.term for c in select(candidates, 2)] == ["a", "b"]
    assert select(candidates, 0) == []
    assert [c.term for c in select(candidates, 99)] == ["a", "b", "c"]


def test_select_breaks_score_ties_alphabetically() -> None:
    candidates = [
        Candidate(term="zeta", score=0.5, sources=(), shards=()),
        Candidate(term="alpha", score=0.5, sources=(), shards=()),
    ]
    assert [c.term for c in select(candidates, 1)] == ["alpha"]


def test_sampled_selection_is_seeded_and_score_ordered() -> None:
    candidates = filter_candidates(activate(ORIGINALS, [_genetics(), _chocolate()]), TermFilter())
    first = [c.term for c in select(candidates, 4, mode="sampled", seed=17)]
    again = [c.term for c in select(candidates, 4, mode="sampled", seed=17)]
    assert first == again
    assert len(first) == 4
    assert len(set(first)) == 4  # sampling is without replacement
    scores = [c.score for c in select(candidates, 4, mode="sampled", seed=17)]
    assert scores == sorted(scores, reverse=True)


def test_sampled_selection_can_differ_from_top_k() -> None:
    """The serendipity the post mentions: some seed must reach past the top-k head."""
    candidates = filter_candidates(activate(ORIGINALS, [_genetics(), _chocolate()]), TermFilter())
    head = {c.term for c in select(candidates, 4)}
    assert any(
        {c.term for c in select(candidates, 4, mode="sampled", seed=seed)} != head
        for seed in range(20)
    )


def test_sampled_selection_survives_zero_scored_candidates() -> None:
    candidates = [
        Candidate(term="a", score=0.9, sources=(), shards=()),
        Candidate(term="b", score=0.1, sources=(), shards=()),
        Candidate(term="dead", score=0.0, sources=(), shards=()),
    ]
    picked = [c.term for c in select(candidates, 2, mode="sampled", seed=3)]
    assert "dead" not in picked
    assert len(picked) == 2


# --------------------------------------------------------------------------- #
# filters.py                                                                   #
# --------------------------------------------------------------------------- #


def test_term_filter_reasons() -> None:
    term_filter = TermFilter(blocked=frozenset({"gene"}))
    assert term_filter.rejection("gene") == "original"
    assert term_filter.rejection("ab") == "too-short"
    assert term_filter.rejection("the") == "stopword"
    assert term_filter.rejection("2024") == "numeric"
    assert term_filter.rejection("crispr") is None
    assert TermFilter(allow_numeric=True).accepts("2024")


def test_ubiquity_filter_from_index_counts_shards(two_sense_index: LoadedIndex) -> None:
    ubiquity = UbiquityFilter.from_index(two_sense_index, threshold=0.5)
    assert ubiquity.n_shards == 3
    # "gene" is a node in two of the three shard graphs.
    assert ubiquity.document_frequency["gene"] == 2
    assert ubiquity.fraction("gene") == pytest.approx(2 / 3)
    assert ubiquity.is_ubiquitous("gene")
    assert not ubiquity.is_ubiquitous("crispr")
    assert ubiquity.fraction("nowhere") == 0.0


def test_ubiquity_filter_keyword_source_differs_from_graph_source(
    two_sense_index: LoadedIndex,
) -> None:
    by_graph = UbiquityFilter.from_index(two_sense_index, source="graph")
    by_keyword = UbiquityFilter.from_index(two_sense_index, source="keywords")
    both = UbiquityFilter.from_index(two_sense_index, source="both")
    assert by_graph.document_frequency.get("crispr") == 1
    assert "crispr" not in by_keyword.document_frequency
    assert both.document_frequency["gene"] == 2


def test_empty_ubiquity_filter_never_rejects() -> None:
    assert not UbiquityFilter().is_ubiquitous("anything")
    assert UbiquityFilter().fraction("anything") == 0.0


def test_ubiquity_filter_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="threshold"):
        UbiquityFilter(threshold=1.5)
    with pytest.raises(ValueError, match="n_shards"):
        UbiquityFilter(n_shards=-1)


def test_ubiquitous_terms_are_filtered_out() -> None:
    """A synonym that every shard offers carries no signal, so it must not be added."""
    shards = [
        ShardGraph(
            shard_id=sid,
            weight=1.0,
            graph={"gene": {"ubiquit": 10.0, f"local{sid}": 1.0}},
        )
        for sid in range(10)
    ]
    without = expand_terms(["gene"], shards)
    assert "ubiquit" in without.new_terms

    ubiquity = UbiquityFilter.from_graphs(shards, threshold=0.8)
    assert ubiquity.is_ubiquitous("ubiquit")
    with_filter = expand_terms(["gene"], shards, ubiquity=ubiquity)
    assert "ubiquit" not in with_filter.new_terms
    assert set(with_filter.new_terms) <= {f"local{sid}" for sid in range(10)}


def test_stopwords_and_short_terms_never_become_expansions() -> None:
    expansion = expand_terms(
        ["gene"], [_genetics()], config=ExpansionConfig(max_new_per_original=99.0)
    )
    assert "the" not in expansion.new_terms  # "the" is the heaviest edge in the fixture
    assert all(len(term) >= 3 for term in expansion.new_terms)


def test_filter_candidates_orders_by_score_then_term() -> None:
    candidates = {
        "beta": Candidate(term="beta", score=0.5, sources=(), shards=()),
        "alpha": Candidate(term="alpha", score=0.5, sources=(), shards=()),
        "gamma": Candidate(term="gamma", score=0.9, sources=(), shards=()),
        "zero": Candidate(term="zero", score=0.0, sources=(), shards=()),
        "no": Candidate(term="no", score=0.7, sources=(), shards=()),
    }
    ordered = [c.term for c in filter_candidates(candidates, TermFilter())]
    assert ordered == ["gamma", "alpha", "beta"]  # "no" too short, "zero" zero-scored


# --------------------------------------------------------------------------- #
# The post's claims                                                            #
# --------------------------------------------------------------------------- #


def test_originals_are_always_preserved_and_marked() -> None:
    expansion = expand_terms(ORIGINALS, [_genetics(), _chocolate()])
    assert expansion.originals == ORIGINALS
    assert expansion.all_terms[: len(ORIGINALS)] == ORIGINALS
    assert all(not term.is_new for term in expansion.terms[: len(ORIGINALS)])
    assert all(term.is_new for term in expansion.terms[len(ORIGINALS) :])
    assert set(expansion.new_terms).isdisjoint(expansion.originals)


def test_originals_survive_even_when_they_would_be_filtered() -> None:
    """The user's own words are never subject to the stopword or length filters."""
    expansion = expand_terms(["the", "ai"], [ShardGraph(shard_id=1, weight=1.0, graph={})])
    assert expansion.originals == ("the", "ai")
    assert expansion.new_terms == ()


def test_duplicate_originals_are_collapsed_once() -> None:
    expansion = expand_terms(["gene", "gene", "bacteria"], [_genetics()])
    assert expansion.originals == ("gene", "bacteria")
    assert expansion.cap == 4  # two originals, not three


def test_expansion_respects_the_cap() -> None:
    shard = ShardGraph(
        shard_id=1,
        weight=1.0,
        graph={"gene": {f"syn{i:02d}": float(50 - i) for i in range(50)}},
    )
    default = expand_terms(["gene"], [shard])
    assert len(default.new_terms) == 2 == default.cap

    ratio = expand_terms(["gene"], [shard], config=ExpansionConfig(max_new_per_original=5.0))
    assert len(ratio.new_terms) == 5

    hard = expand_terms(
        ["gene"], [shard], config=ExpansionConfig(max_new_per_original=99.0, hard_cap=7)
    )
    assert len(hard.new_terms) == 7


def test_cap_scales_with_the_number_of_originals() -> None:
    """ "Not overwhelm" is relative: nine originals may gain more than one may."""
    shard = ShardGraph(
        shard_id=1,
        weight=1.0,
        graph={f"orig{i}": {f"syn{i}{j}": float(10 - j) for j in range(9)} for i in range(9)},
    )
    originals = [f"orig{i}" for i in range(9)]
    nine = expand_terms(originals, [shard])
    one = expand_terms(originals[:1], [shard])
    assert nine.cap == 18
    assert one.cap == 2
    assert len(nine.new_terms) == 18
    assert len(one.new_terms) == 2
    # The blog's own ratio, 21 new for 9 originals, is reachable by configuration.
    blog = expand_terms(originals, [shard], config=ExpansionConfig(max_new_per_original=21 / 9))
    assert len(blog.new_terms) == 21


def test_expansion_never_outnumbers_originals_by_more_than_the_ratio() -> None:
    shard = ShardGraph(
        shard_id=1,
        weight=1.0,
        graph={"gene": {f"syn{i:02d}": float(50 - i) for i in range(50)}},
    )
    for n_originals in range(1, 8):
        originals = ["gene", *[f"filler{i}" for i in range(n_originals - 1)]]
        expansion = expand_terms(originals, [shard])
        assert len(expansion.new_terms) <= 2 * n_originals


def test_same_token_expands_differently_in_shards_with_different_senses() -> None:
    """The post's central claim for this stage, stated as a test.

    "gene" in the genetics shard only has the genetic meaning; in the shard where Gene is
    a name it reaches Willy Wonka. Pooling the two graphs is what per-shard expansion
    exists to avoid, so the two expansions must not agree.
    """
    config = ExpansionConfig(max_new_per_original=4.0)
    genetic = expand_terms(["gene"], [_genetics()], config=config)
    nominal = expand_terms(["gene"], [_chocolate()], config=config)

    assert set(genetic.new_terms) == {"crispr", "genom", "prokaryot", "chromosom"}
    assert set(nominal.new_terms) == {"wonka", "wilder", "chocolat", "oompa"}
    assert set(genetic.new_terms).isdisjoint(nominal.new_terms)


def test_per_shard_senses_via_the_index(two_sense_index: LoadedIndex) -> None:
    genetic = expand_query(two_sense_index, ["gene"], shard_ids=[11343])
    nominal = expand_query(two_sense_index, ["gene"], shard_ids=[90210])
    assert set(genetic.new_terms).isdisjoint(nominal.new_terms)
    assert genetic.new_terms[0] == "crispr"
    assert nominal.new_terms[0] == "wonka"


def test_provenance_records_which_shard_voted(two_sense_index: LoadedIndex) -> None:
    expansion = expand_query(
        two_sense_index,
        ["gene", "bacteria"],
        shard_ids=[11343, 90210],
        shard_weights=rank_weights([11343, 90210]),
        config=ExpansionConfig(max_new_per_original=99.0),
    )
    by_term = {term.term: term for term in expansion.terms}
    assert by_term["crispr"].shards == (11343,)
    assert by_term["wonka"].shards == (90210,)
    assert by_term["crispr"].sources == ("gene",)
    assert by_term["crispr"].score > by_term["wonka"].score  # rank 0 outweighs rank 1


def test_render_matches_the_posts_notation() -> None:
    expansion = expand_terms(["gene"], [_genetics()], config=ExpansionConfig(hard_cap=2))
    assert expansion.render() == '* "gene"\n* "crispr" [NEW]\n* "genom" [NEW]'
    assert ExpandedTerm(term="gene", is_new=False).render() == '"gene"'


def test_dropped_records_what_lost_to_the_cap() -> None:
    expansion = expand_terms(["gene"], [_genetics()], config=ExpansionConfig(hard_cap=1))
    assert expansion.new_terms == ("crispr",)
    assert "genom" in expansion.dropped
    assert "the" not in expansion.dropped  # filtered, not dropped
    full = expand_terms(["gene"], [_genetics()], config=ExpansionConfig(max_new_per_original=99.0))
    assert full.dropped == ()


def test_weights_keep_originals_ahead_of_expansions() -> None:
    expansion = expand_terms(ORIGINALS, [_genetics()], config=ExpansionConfig(hard_cap=6))
    weights = expansion.weights(expansion_weight=0.3)
    assert all(weights[term] == 1.0 for term in expansion.originals)
    assert max(weights[term] for term in expansion.new_terms) == pytest.approx(0.3)
    assert all(weights[term] <= 0.3 for term in expansion.new_terms)


def test_weights_of_a_no_op_expansion_are_all_original() -> None:
    expansion = expand_terms(["gene"], [])
    assert expansion.weights() == {"gene": 1.0}


# --------------------------------------------------------------------------- #
# Determinism                                                                  #
# --------------------------------------------------------------------------- #


def test_determinism_across_runs_in_both_modes(two_sense_index: LoadedIndex) -> None:
    for mode in ("top_k", "sampled"):
        config = ExpansionConfig(mode=mode, seed=11, hard_cap=6)  # type: ignore[arg-type]
        runs = [
            expand_query(
                two_sense_index,
                ORIGINALS,
                shard_ids=[11343, 90210],
                shard_weights=rank_weights([11343, 90210]),
                config=config,
                ubiquity=UbiquityFilter.from_index(two_sense_index),
            )
            for _ in range(5)
        ]
        assert all(run == runs[0] for run in runs)


def test_determinism_is_independent_of_graph_insertion_order() -> None:
    shuffled = {
        term: dict(reversed(list(neighbours.items())))
        for term, neighbours in reversed(list(GENETICS_GRAPH.items()))
    }
    baseline = expand_terms(ORIGINALS, [_genetics()], config=ExpansionConfig(hard_cap=6))
    reordered = expand_terms(
        ORIGINALS,
        [ShardGraph(shard_id=11343, weight=1.0, graph=shuffled)],
        config=ExpansionConfig(hard_cap=6),
    )
    assert baseline == reordered


def test_expansion_is_a_frozen_value() -> None:
    expansion = expand_terms(["gene"], [_genetics()])
    with pytest.raises(AttributeError):
        expansion.terms = ()  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Degradation                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shards",
    [
        pytest.param([], id="no-shards"),
        pytest.param([ShardGraph(shard_id=1, weight=1.0, graph={})], id="empty-graph"),
        pytest.param([_genetics(0.0)], id="zero-weight-shard"),
        pytest.param(
            [ShardGraph(shard_id=1, weight=1.0, graph={"solar": {"panel": 1.0}})],
            id="no-overlap",
        ),
        pytest.param(
            [ShardGraph(shard_id=1, weight=1.0, graph={"gene": {"the": 1.0, "ab": 1.0}})],
            id="every-candidate-filtered",
        ),
    ],
)
def test_absent_or_useless_graphs_degrade_to_a_no_op(shards: list[ShardGraph]) -> None:
    expansion = expand_terms(ORIGINALS, shards)
    assert expansion.originals == ORIGINALS
    assert expansion.new_terms == ()
    assert expansion.all_terms == ORIGINALS
    assert len(expansion) == len(ORIGINALS)


def test_no_originals_means_no_expansion() -> None:
    expansion = expand_terms([], [_genetics()])
    assert expansion == Expansion()
    assert expansion.all_terms == ()


def test_expand_query_tolerates_unknown_shard_ids(two_sense_index: LoadedIndex) -> None:
    expansion = expand_query(two_sense_index, ["gene"], shard_ids=[999_999])
    assert expansion.new_terms == ()
    assert expansion.originals == ("gene",)


def test_expand_query_reads_no_documents(built_index: LoadedIndex) -> None:
    """Stage 7 must stay manifest-only: touching documents breaks the 1M-doc target.

    Run against the disk-backed session index, whose ``cache_stats`` carries document,
    token and artifact counters that move the instant anything but a manifest is read.
    """
    shard_ids = sorted(built_index.manifests)[:2]
    before = built_index.cache_stats()
    assert {"documents", "tokens", "artifacts"} <= set(before)
    expand_query(
        built_index,
        ["gene", "editing"],
        shard_ids=shard_ids,
        ubiquity=UbiquityFilter.from_index(built_index),
    )
    assert built_index.cache_stats() == before


# --------------------------------------------------------------------------- #
# Config validation                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_new_per_original": -1.0}, "max_new_per_original"),
        ({"hard_cap": -1}, "hard_cap"),
        ({"min_length": 0}, "min_length"),
        ({"mode": "montecarlo"}, "mode"),
    ],
)
def test_config_validates(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ExpansionConfig(**kwargs)  # type: ignore[arg-type]


def test_the_threshold_belongs_to_the_ubiquity_filter(two_sense_index: LoadedIndex) -> None:
    """One place to set it: the config never rewrites the table's own threshold."""
    permissive = ExpansionConfig().term_filter(
        UbiquityFilter.from_index(two_sense_index, threshold=0.99)
    )
    strict = ExpansionConfig().term_filter(
        UbiquityFilter.from_index(two_sense_index, threshold=0.5)
    )
    assert permissive.ubiquity is not None
    assert strict.ubiquity is not None
    assert permissive.ubiquity.threshold == 0.99
    assert not permissive.ubiquity.is_ubiquitous("gene")
    assert strict.ubiquity.is_ubiquitous("gene")
    assert ExpansionConfig().term_filter(None).ubiquity is None


def test_hard_cap_only_lowers_the_ratio_cap() -> None:
    shard = ShardGraph(
        shard_id=1, weight=1.0, graph={"gene": {f"syn{i:02d}": float(20 - i) for i in range(20)}}
    )
    assert len(expand_terms(["gene"], [shard], config=ExpansionConfig(hard_cap=99)).new_terms) == 2
    assert len(expand_terms(["gene"], [shard], config=ExpansionConfig(hard_cap=1)).new_terms) == 1


# --------------------------------------------------------------------------- #
# The stage-2 bridge                                                           #
# --------------------------------------------------------------------------- #


def test_question_terms_are_the_posts_original_search_words() -> None:
    """The post's stage-7 example lists exactly these nine stems as the originals."""
    terms = question_terms(
        "What lessons from bacteria and yeast actually translate into safer gene-editing medicines?"
    )
    assert set(terms) == {
        "yeast",
        "bacteria",
        "safer",
        "lesson",
        "translat",
        "gene",
        "medicin",
        "edit",
        "actual",
    }
    assert len(terms) == len(set(terms))


def test_expand_question_expands_the_posts_question(two_sense_index: LoadedIndex) -> None:
    expansion = expand_question(
        two_sense_index,
        "What lessons from bacteria and yeast actually translate into "
        "safer gene-editing medicines?",
        shard_ids=[11343],
        shard_weights=rank_weights([11343]),
    )
    assert len(expansion.originals) == 9
    assert expansion.cap == 18
    # The shape of the post's example: originals preserved, additions marked, bounded.
    assert set(expansion.originals) >= {"gene", "bacteria", "yeast"}
    assert set(expansion.new_terms) == {
        "crispr",
        "genom",
        "prokaryot",
        "chromosom",
        "archaea",
        "bacteriophag",
        "pathogen",
        "phage",
        "galactos",
        "dextros",
        "raffinos",
    }
    assert "[NEW]" in expansion.render()
    assert expansion.render().count("[NEW]") == len(expansion.new_terms)


def test_expand_question_on_a_built_index_is_a_no_op_or_bounded(
    built_index: LoadedIndex,
) -> None:
    """Against the real session index: never fewer terms than the question, never more
    than the cap allows, and manifest-only regardless of what the build produced."""
    shard_ids = sorted(built_index.manifests)[:3]
    expansion = expand_question(
        built_index,
        "gene editing medicine",
        shard_ids=shard_ids,
        shard_weights=rank_weights(shard_ids),
        ubiquity=UbiquityFilter.from_index(built_index),
    )
    assert set(expansion.originals) <= set(expansion.all_terms)
    assert len(expansion.new_terms) <= expansion.cap
    assert set(expansion.new_terms).isdisjoint(expansion.originals)
    assert expansion == expand_question(
        built_index,
        "gene editing medicine",
        shard_ids=shard_ids,
        shard_weights=rank_weights(shard_ids),
        ubiquity=UbiquityFilter.from_index(built_index),
    )
