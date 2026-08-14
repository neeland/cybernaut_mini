"""Tests for blog stage 3 — search intent prediction.

The headline test is the post's own worked example: the question about bacteria,
yeast and gene-editing medicines must yield exactly the six intents the post
prints. Everything else guards the three rules that produce them (proximity,
function-word exclusion, harmonic TF-IDF) and the determinism the pipeline needs.

Offline by construction: no API key, no network, no optional dependency. spaCy is
absent from the default environment, so the closed-class word list is the path
under test; the tag-driven detector is exercised with synthetic POS tags.
"""

from __future__ import annotations

import math
from statistics import harmonic_mean
from typing import Any

import pytest

from cybernaut_mini.query.s3_intents import (
    ClosedClassFunctionWords,
    PosFunctionWords,
    QueryToken,
    SearchIntent,
    compute_idf,
    extract_intents,
    harmonic_tfidf,
    predict_intents,
    query_tfidf,
    resolve_idf,
    singularize,
    tokenize_query,
)

# The post's worked example, verbatim.
WORKED_QUESTION = (
    "What lessons from bacteria and yeast actually translate into safer gene-editing medicines?"
)

# The six intents the post prints for it, verbatim.
WORKED_INTENTS = frozenset(
    {
        "editing medicine",
        "gene-editing",
        "gene-editing medicine",
        "safer gene",
        "safer gene-editing",
        "safer gene-editing medicine",
    }
)


@pytest.fixture
def idf() -> dict[str, float]:
    """A small IDF table with deliberately distinct values.

    Distinct IDFs mean the ranking carries information instead of collapsing into
    floating-point tie-break noise. "safer" is the rarest term, "editing" the
    most common of the four that matter.
    """
    return {
        "safer": 4.0,
        "medicine": 3.0,
        "gene": 2.0,
        "editing": 1.0,
        "lesson": 3.5,
        "bacteria": 3.5,
        "yeast": 3.5,
        "translate": 1.5,
    }


# --------------------------------------------------------------------------
# harmonic_tfidf — the aggregator the post names
# --------------------------------------------------------------------------


def test_harmonic_tfidf_matches_the_textbook_definition() -> None:
    assert harmonic_tfidf([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    # 2 / (1/1 + 1/3) = 1.5
    assert harmonic_tfidf([1.0, 3.0]) == pytest.approx(1.5)
    assert harmonic_tfidf([4.0]) == pytest.approx(4.0)
    assert harmonic_tfidf([1.0, 2.0, 4.0]) == pytest.approx(harmonic_mean([1.0, 2.0, 4.0]))


def test_a_single_zero_makes_the_whole_ngram_worthless() -> None:
    """The reason the post picks the harmonic mean over the arithmetic mean."""
    assert harmonic_tfidf([9.0, 9.0, 0.0]) == 0.0
    # The arithmetic mean would have called that n-gram excellent.
    assert sum([9.0, 9.0, 0.0]) / 3 == pytest.approx(6.0)


def test_harmonic_tfidf_rejects_degenerate_input() -> None:
    assert harmonic_tfidf([]) == 0.0
    assert harmonic_tfidf([2.0, -1.0]) == 0.0
    assert harmonic_tfidf([2.0, math.nan]) == 0.0
    assert harmonic_tfidf([2.0, math.inf]) == 0.0


def test_harmonic_mean_punishes_a_weak_token_harder_than_the_arithmetic_mean() -> None:
    strong = [5.0, 5.0]
    mixed = [9.9, 0.5]
    assert sum(mixed) / 2 > sum(strong) / 2  # arithmetic prefers the mixed pair
    assert harmonic_tfidf(mixed) < harmonic_tfidf(strong)  # harmonic does not


# --------------------------------------------------------------------------
# IDF plumbing
# --------------------------------------------------------------------------


def test_compute_idf_smoothed_is_always_positive() -> None:
    table = compute_idf([["gene", "gene", "editing"], ["gene"], ["gene", "yeast"]])
    assert table["gene"] == pytest.approx(math.log(4 / 4) + 1.0)  # in every document
    assert table["yeast"] == pytest.approx(math.log(4 / 2) + 1.0)
    assert table["editing"] > table["gene"]
    assert all(value > 0.0 for value in table.values())
    assert list(table) == sorted(table)  # key-sorted for reproducible iteration


def test_compute_idf_unsmoothed_zeroes_a_universal_term() -> None:
    table = compute_idf([["gene", "editing"], ["gene", "yeast"]], smooth=False)
    assert table["gene"] == 0.0
    assert table["yeast"] == pytest.approx(math.log(2))


def test_compute_idf_of_an_empty_corpus_is_empty() -> None:
    assert compute_idf([]) == {}


def test_resolve_idf_treats_an_unseen_term_as_the_rarest() -> None:
    lookup = resolve_idf({"gene": 2.0, "safer": 4.0})
    assert lookup("gene") == 2.0
    assert lookup("crispr") == 4.0  # unseen -> the table's maximum
    assert resolve_idf({"gene": 2.0}, 0.25)("crispr") == 0.25
    assert resolve_idf({})("anything") == 1.0
    assert resolve_idf(lambda term: len(term) * 1.0)("gene") == 4.0


def test_query_tfidf_uses_sublinear_term_frequency() -> None:
    scores = query_tfidf(["gene", "gene", "safer"], {"gene": 2.0, "safer": 4.0})
    assert scores["safer"] == pytest.approx(4.0)  # count 1 -> tf 1.0
    assert scores["gene"] == pytest.approx((1.0 + math.log(2)) * 2.0)
    raw = query_tfidf(["gene", "gene"], {"gene": 2.0}, sublinear_tf=False)
    assert raw["gene"] == pytest.approx(4.0)


# --------------------------------------------------------------------------
# Tokenization: positions and glue
# --------------------------------------------------------------------------


def test_tokenize_query_records_positions_and_separators() -> None:
    tokens = tokenize_query("safer gene-editing medicines?")
    assert [token.text for token in tokens] == ["safer", "gene", "editing", "medicine"]
    assert [token.index for token in tokens] == [0, 1, 2, 3]
    assert tokens[2].separator_before == "-"
    assert tokens[2].glue == "-"
    assert tokens[1].glue == " "
    assert all(token.joins_previous for token in tokens)
    assert tokens[0].raw == "safer"


def test_clause_punctuation_breaks_a_sequence() -> None:
    tokens = tokenize_query("bacteria, yeast")
    assert tokens[1].joins_previous is False
    assert tokens[1].glue == " "


def test_singularize_is_deliberately_minimal() -> None:
    assert singularize("medicines") == "medicine"
    assert singularize("lessons") == "lesson"
    assert singularize("therapies") == "therapy"
    assert singularize("batches") == "batch"
    # Left alone: not plurals, and the words the post keeps in its intents.
    for word in ("safer", "editing", "gene", "bacteria", "bias", "analysis", "virus", "gas"):
        assert singularize(word) == word


def test_singularize_decides_case_insensitively() -> None:
    """The guard lists are lowercase; a capitalised surface must still hit them.

    Regression: the guards used to be tested against the raw surface, so
    "News" missed the invariant-noun list and came back as "New".
    """
    assert singularize("News") == "News"
    assert singularize("Series") == "Series"
    assert singularize("Bias") == "Bias"
    assert singularize("MEDICINES") == "MEDICINE"
    assert singularize("Medicines") == "Medicine"
    assert singularize("Therapies") == "Therapy"


# --------------------------------------------------------------------------
# Function-word exclusion
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "word",
    ["what", "the", "and", "from", "into", "actually", "they", "is", "however", "very"],
)
def test_closed_class_detector_flags_function_words(word: str) -> None:
    assert ClosedClassFunctionWords()(QueryToken(text=word, index=0)) is True


@pytest.mark.parametrize("word", ["gene", "editing", "medicine", "safer", "bacteria", "yeast"])
def test_closed_class_detector_keeps_content_words(word: str) -> None:
    assert ClosedClassFunctionWords()(QueryToken(text=word, index=0)) is False


def test_closed_class_allow_list_overrides_the_built_in_list() -> None:
    detector = ClosedClassFunctionWords(allow=frozenset({"back"}), extra=frozenset({"gene"}))
    assert detector(QueryToken(text="back", index=0)) is False
    assert detector(QueryToken(text="gene", index=1)) is True


def test_pos_detector_uses_tags_and_falls_back_when_they_are_missing() -> None:
    detector = PosFunctionWords()
    assert detector(QueryToken(text="editing", index=0, pos="VERB")) is False
    assert detector(QueryToken(text="safer", index=1, pos="ADJ")) is False
    assert detector(QueryToken(text="actually", index=2, pos="ADV")) is True
    assert detector(QueryToken(text="and", index=3, pos="CCONJ")) is True
    # A tagger that skipped this token must not disable the filter.
    assert detector(QueryToken(text="and", index=4, pos=None)) is True
    # A tag beats the word list: "back" tagged NOUN survives.
    assert detector(QueryToken(text="back", index=5, pos="NOUN")) is False


def test_no_extracted_intent_contains_a_function_word(idf: dict[str, float]) -> None:
    intents = predict_intents(WORKED_QUESTION, idf, top_k=None)
    detector = ClosedClassFunctionWords()
    for intent in intents:
        for token in intent.tokens:
            assert detector(QueryToken(text=token, index=0)) is False


# --------------------------------------------------------------------------
# The worked example — the acceptance test
# --------------------------------------------------------------------------


def test_worked_example_reproduces_the_posts_six_intents(idf: dict[str, float]) -> None:
    """Exact parity with the post's stage-3 worked example.

    "What lessons from bacteria and yeast actually translate into safer
    gene-editing medicines?" -> exactly six intents. They fall out of the three
    rules with nothing left over: the only run of proximal content tokens in the
    question is "safer gene editing medicine" (every other content word is walled
    off by "from", "and", "actually" or "into"), and its contiguous n-grams for
    n = 2, 3, 4 number 3 + 2 + 1 = 6.
    """
    intents = predict_intents(WORKED_QUESTION, idf, top_k=None)
    assert {intent.text for intent in intents} == set(WORKED_INTENTS)


def test_worked_example_intents_carry_tokens_scores_and_spans(idf: dict[str, float]) -> None:
    by_text = {intent.text: intent for intent in predict_intents(WORKED_QUESTION, idf, top_k=None)}
    longest = by_text["safer gene-editing medicine"]
    assert longest.tokens == ("safer", "gene", "editing", "medicine")
    assert longest.span == (9, 13)  # raw-token indices, function words included
    assert len(longest) == 4
    assert longest.score == pytest.approx(harmonic_mean([4.0, 2.0, 1.0, 3.0]))
    # The hyphen of "gene-editing" is the query's own, preserved by the renderer.
    assert by_text["gene-editing"].tokens == ("gene", "editing")
    assert by_text["safer gene"].score == pytest.approx(harmonic_mean([4.0, 2.0]))


def test_worked_example_is_ranked_by_harmonic_tfidf(idf: dict[str, float]) -> None:
    intents = predict_intents(WORKED_QUESTION, idf, top_k=None)
    scores = [intent.score for intent in intents]
    assert scores == sorted(scores, reverse=True)
    # "safer gene" (4.0, 2.0) wins: it is the only candidate free of the low-IDF
    # "editing", which drags down every n-gram it appears in — the bigram
    # "gene-editing" (2.0, 1.0) lands last precisely because of it.
    assert intents[0].text == "safer gene"
    assert intents[0].score == pytest.approx(harmonic_mean([4.0, 2.0]))
    assert intents[-1].text == "gene-editing"
    assert intents[-1].score == pytest.approx(harmonic_mean([2.0, 1.0]))


def test_worked_example_top_k_truncates_without_reordering(idf: dict[str, float]) -> None:
    everything = predict_intents(WORKED_QUESTION, idf, top_k=None)
    assert len(everything) == 6
    assert predict_intents(WORKED_QUESTION, idf, top_k=3) == everything[:3]
    assert predict_intents(WORKED_QUESTION, idf, top_k=0) == []


def test_worked_example_survives_a_missing_idf_table(idf: dict[str, float]) -> None:
    """An empty table means every term is unseen — the intents must still appear."""
    intents = predict_intents(WORKED_QUESTION, {})
    assert {intent.text for intent in intents} == set(WORKED_INTENTS)


def test_worked_example_with_the_spacy_switch_degrades_to_the_same_answer(
    idf: dict[str, float],
) -> None:
    """``use_spacy=True`` must never crash when spaCy is not installed.

    With the model present the POS tagger drives the filter; without it the tags
    are ``None`` and :class:`PosFunctionWords` defers to the word list. Either way
    this question yields the post's six intents.
    """
    intents = predict_intents(WORKED_QUESTION, idf, use_spacy=True, top_k=None)
    assert {intent.text for intent in intents} == set(WORKED_INTENTS)


# --------------------------------------------------------------------------
# Proximity
# --------------------------------------------------------------------------


def test_a_conjunction_between_two_nouns_blocks_an_intent(idf: dict[str, float]) -> None:
    intents = {intent.text for intent in predict_intents(WORKED_QUESTION, idf, top_k=None)}
    assert "bacteria yeast" not in intents
    assert "lesson bacteria" not in intents
    assert "translate safer" not in intents


def test_max_gap_loosens_proximity_on_request(idf: dict[str, float]) -> None:
    intents = {
        intent.text for intent in predict_intents(WORKED_QUESTION, idf, max_gap=1, top_k=None)
    }
    assert "bacteria yeast" in intents  # only "and" stood between them
    assert intents >= WORKED_INTENTS  # the post's six are still there
    # One function word between them is now tolerated, so "lessons from bacteria"
    # does become an intent — that is the whole point of the knob.
    assert "lesson bacteria" in intents
    # Loosening proximity still does not make an intent out of non-adjacent members
    # of a run: candidates remain contiguous *windows*, so "lesson yeast" (which
    # would have to skip over "bacteria") is never emitted.
    assert "lesson yeast" not in intents


def test_clause_punctuation_survives_a_dropped_function_word(idf: dict[str, float]) -> None:
    """A comma ends a run even when it attaches to the function word that follows.

    Regression: the run-break test only looked at the *content* token's own
    separator, so at ``max_gap=1`` the comma in "bacteria, and yeast" rode along
    with the dropped "and" and the two nouns fused into one run.
    """
    across_a_comma = {
        intent.text for intent in predict_intents("bacteria, and yeast", idf, max_gap=1, top_k=None)
    }
    assert across_a_comma == set()
    # Without the comma the same sentence shape does fuse — the knob still works.
    without_a_comma = {
        intent.text for intent in predict_intents("bacteria and yeast", idf, max_gap=1, top_k=None)
    }
    assert "bacteria yeast" in without_a_comma


def test_punctuation_ends_a_run(idf: dict[str, float]) -> None:
    intents = {
        intent.text
        for intent in predict_intents("safer gene-editing medicines, yeast lessons", idf)
    }
    assert "medicine yeast" not in intents
    assert "yeast lesson" in intents


def test_max_n_caps_intent_length(idf: dict[str, float]) -> None:
    intents = predict_intents(WORKED_QUESTION, idf, max_n=2, top_k=None)
    assert {intent.text for intent in intents} == {
        "safer gene",
        "gene-editing",
        "editing medicine",
    }
    assert all(len(intent) == 2 for intent in intents)


def test_min_n_of_one_admits_unigram_intents(idf: dict[str, float]) -> None:
    intents = {intent.text for intent in predict_intents(WORKED_QUESTION, idf, min_n=1, top_k=None)}
    assert "translate" in intents  # a lone content word, walled off by function words
    assert intents >= WORKED_INTENTS


# --------------------------------------------------------------------------
# Scoring rules inside extraction
# --------------------------------------------------------------------------


def test_a_zero_idf_token_removes_every_intent_it_appears_in() -> None:
    """Unsmoothed IDF of a term in every document is 0, and 0 poisons its n-grams."""
    table = compute_idf([["gene", "editing"], ["gene", "yeast"]], smooth=False)
    assert table["gene"] == 0.0
    intents = {intent.text for intent in predict_intents("safer gene-editing medicines", table)}
    assert intents == {"editing medicine"}  # every "gene" n-gram scored 0 and was dropped


def test_min_score_raises_the_bar(idf: dict[str, float]) -> None:
    lenient = predict_intents(WORKED_QUESTION, idf, top_k=None)
    strict = predict_intents(WORKED_QUESTION, idf, top_k=None, min_score=2.0)
    assert {intent.text for intent in strict} == {"safer gene"}
    assert len(strict) < len(lenient)


def test_a_callable_idf_lookup_is_accepted() -> None:
    intents = predict_intents(WORKED_QUESTION, lambda term: 1.0 + len(term), top_k=None)
    assert {intent.text for intent in intents} == set(WORKED_INTENTS)


def test_repeated_tokens_are_deduplicated_keeping_the_best_occurrence() -> None:
    text = "gene editing rules and gene editing"
    intents = predict_intents(text, {"gene": 2.0, "editing": 2.0, "rule": 1.0})
    texts = [intent.text for intent in intents]
    assert texts.count("gene editing") == 1
    first = next(intent for intent in intents if intent.text == "gene editing")
    assert first.span == (0, 2)  # the earliest of the two equally scored occurrences


# --------------------------------------------------------------------------
# Determinism and empty input
# --------------------------------------------------------------------------


def test_extraction_is_deterministic(idf: dict[str, float]) -> None:
    runs = [predict_intents(WORKED_QUESTION, idf, top_k=None) for _ in range(5)]
    assert all(run == runs[0] for run in runs)
    assert [intent.text for intent in runs[0]] == [
        "safer gene",
        "safer gene-editing medicine",
        "safer gene-editing",
        "gene-editing medicine",
        "editing medicine",
        "gene-editing",
    ]


def test_intents_are_hashable_value_objects(idf: dict[str, float]) -> None:
    one = SearchIntent(text="safer gene", tokens=("safer", "gene"), score=2.0, span=(9, 11))
    two = SearchIntent(text="safer gene", tokens=("safer", "gene"), score=2.0, span=(9, 11))
    assert one == two
    assert len({one, two}) == 1


@pytest.mark.parametrize("text", ["", "   ", "?!,.", "the and of"])
def test_empty_or_contentless_queries_yield_no_intents(text: str, idf: dict[str, float]) -> None:
    assert predict_intents(text, idf) == []
    assert tokenize_query(text) == [] or all(
        ClosedClassFunctionWords()(token) for token in tokenize_query(text)
    )


def test_extract_intents_on_an_empty_token_list_returns_an_empty_list() -> None:
    assert extract_intents([], {}) == []
    assert extract_intents([], {"gene": 1.0}, top_k=None) == []


def test_a_single_content_token_cannot_form_a_bigram(idf: dict[str, float]) -> None:
    assert predict_intents("What is a gene?", idf) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_n": 0}, "min_n"),
        ({"max_n": 1}, "max_n"),
        ({"max_gap": -1}, "max_gap"),
        # A negative top_k would slice off the *worst* |top_k| intents and look
        # like a working truncation, so it is rejected like its siblings.
        ({"top_k": -1}, "top_k"),
    ],
)
def test_invalid_bounds_are_rejected(kwargs: dict[str, Any], message: str) -> None:
    tokens = tokenize_query(WORKED_QUESTION)
    with pytest.raises(ValueError, match=message):
        extract_intents(tokens, {}, **kwargs)
