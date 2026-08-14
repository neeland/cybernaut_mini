"""Tests for blog stage 2, multilingual tokenization.

The headline test is :func:`test_english_worked_example_matches_the_post`, which is the
post's own published output treated as an acceptance criterion. Everything else guards
the failure modes that make a multilingual tokenizer look healthy while returning
nothing: silently dropping non-Latin script, mojibake from a bad normalisation, and
non-deterministic segmentation.

Every test runs offline. No API key, no model download, no network.
"""

# These tests exist to prove non-Latin script survives, so fullwidth, Cyrillic and CJK
# characters appear on purpose in nearly every assertion; RUF001 has nothing to catch here.
# RUF002 is silenced for the same reason: the docstrings quote the worked-example
# sentences verbatim, and "correcting" the fullwidth question mark would make a
# docstring disagree with the constant it is documenting.
# ruff: noqa: RUF001, RUF002

from __future__ import annotations

import pickle

import pytest

from cybernaut_mini.query.s2_tokenize import (
    MultilingualTokenizer,
    Tokenized,
    default_tokenizer,
    guess_script_language,
    normalize_language,
    tokenize,
)
from cybernaut_mini.query.s2_tokenize.segmentation import (
    PYSBD_LANGUAGES,
    char_bigrams,
    regex_sentences,
    segment_japanese,
    segment_words,
)
from cybernaut_mini.query.s2_tokenize.stemming import SNOWBALL_LANGUAGES, snowball_algorithm
from cybernaut_mini.query.s2_tokenize.stopwords import stopwords_for

# The post's worked example, verbatim.
QUESTION_EN = (
    "What lessons from bacteria and yeast actually translate into "
    "safer gene-editing medicines?"
)
EXPECTED_STEMS_EN = frozenset(
    {"yeast", "bacteria", "safer", "lesson", "translat", "gene", "medicin", "edit", "actual"}
)

QUESTION_JA = (
    "細菌と酵母から得られる教訓は、より安全な遺伝子編集医薬品に"
    "どのように応用されるのでしょうか？"
)
QUESTION_ZH = "从细菌和酵母中学到的经验如何应用于更安全的基因编辑药物？"


@pytest.fixture()
def tokenizer() -> MultilingualTokenizer:
    """The tokenizer as production actually configures it: MeCab auto-detected.

    This fixture used to pin ``use_mecab=False`` so Japanese assertions held with or
    without the then-optional ``cjk`` extra. That made every Japanese test exercise the
    character-bigram fallback and nothing else, which hid the fact that the fallback
    shreds the blog's own headline token 遺伝子. MeCab is now a core dependency
    precisely so this path is the tested one; tests that mean to cover the fallback
    construct their own tokenizer with ``use_mecab=False`` explicitly.
    """
    return MultilingualTokenizer()


# --------------------------------------------------------------------------- #
# The acceptance test: the post's published English output                     #
# --------------------------------------------------------------------------- #


def test_english_worked_example_matches_the_post(tokenizer: MultilingualTokenizer) -> None:
    result = tokenizer.tokenize(QUESTION_EN, "en")
    assert set(result.stems) == EXPECTED_STEMS_EN
    assert len(result.stems) == len(EXPECTED_STEMS_EN)  # no duplicates in this query


def test_worked_example_splits_the_hyphenated_compound(
    tokenizer: MultilingualTokenizer,
) -> None:
    """'gene-editing' must become two tokens or the post's token list is unreachable."""
    tokens = tokenizer.words(QUESTION_EN, "en")
    assert "gene" in tokens
    assert "editing" in tokens
    assert not any("-" in token for token in tokens)


def test_worked_example_drops_function_words(tokenizer: MultilingualTokenizer) -> None:
    tokens = tokenizer.tokenize(QUESTION_EN, "en").tokens
    for stopword in ("what", "from", "and", "into"):
        assert stopword not in tokens
    assert "actually" in tokens, "'actually' is a content word here — the post keeps it"


def test_tokens_and_stems_are_parallel(tokenizer: MultilingualTokenizer) -> None:
    result = tokenizer.tokenize(QUESTION_EN, "en")
    assert len(result.tokens) == len(result.stems) == len(result)
    pairs = dict(zip(result.tokens, result.stems, strict=True))
    assert pairs["medicines"] == "medicin"
    assert pairs["translate"] == "translat"


def test_module_level_helper_uses_the_shared_tokenizer() -> None:
    assert set(tokenize(QUESTION_EN).stems) == EXPECTED_STEMS_EN
    assert default_tokenizer() is default_tokenizer()


# --------------------------------------------------------------------------- #
# Acronyms must survive stop-word removal (US / IT / WHO are entities, not      #
# pronouns). The corpus this replica targets is news, where these are among     #
# the most frequent entities in the collection.                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("query", "must_keep"),
    [
        ("US sanctions on Iran", "us"),
        ("US-China trade war", "us"),
        ("IT department budget", "it"),
        ("WHO guidance on flu", "who"),
        ("The EU and the UN disagree", "eu"),
    ],
)
def test_acronyms_survive_stopword_removal(
    tokenizer: MultilingualTokenizer, query: str, must_keep: str
) -> None:
    assert must_keep in tokenizer.tokenize(query, "en").tokens


@pytest.mark.parametrize(
    ("query", "must_drop"),
    [
        ("It rains in Spain", "it"),
        ("The dog barks loudly", "the"),
        ("Who won the election?", "who"),
        ("They gave us the report", "us"),
    ],
)
def test_sentence_case_function_words_are_still_removed(
    tokenizer: MultilingualTokenizer, query: str, must_drop: str
) -> None:
    """The acronym exemption keys on ALL-CAPS, so ordinary capitalisation is unaffected."""
    assert must_drop not in tokenizer.tokenize(query, "en").tokens


def test_hyphenated_acronym_keeps_both_sides(tokenizer: MultilingualTokenizer) -> None:
    tokens = tokenizer.tokenize("US-China trade war", "en").tokens
    assert "us" in tokens
    assert "china" in tokens


def test_all_caps_headline_does_not_become_pure_noise(
    tokenizer: MultilingualTokenizer,
) -> None:
    """When everything is upper-case, capitalisation carries no signal, so ignore it.

    Otherwise an ALL-CAPS headline exempts every word from stop-word removal and the
    tokenizer returns the function words it exists to strip.
    """
    tokens = tokenizer.tokenize("US SANCTIONS HIT IT SECTOR AS WHO WARNS", "en").tokens
    assert "as" not in tokens
    assert "sanctions" in tokens
    assert "sector" in tokens


def test_short_all_caps_query_keeps_the_exemption(
    tokenizer: MultilingualTokenizer,
) -> None:
    """Below the ratio's noise floor, "US IT" must not be read as a shouted headline."""
    assert tokenizer.tokenize("US IT", "en").tokens == ("us", "it")


def test_acronym_exemption_does_not_disturb_non_latin_scripts(
    tokenizer: MultilingualTokenizer,
) -> None:
    """``isupper()`` is False for CJK, so the caps rule must be inert there."""
    japanese = tokenizer.tokenize(QUESTION_JA, "ja").tokens
    assert "遺伝子" in japanese
    chinese = tokenizer.tokenize(QUESTION_ZH, "zh").tokens
    assert "的" not in chinese  # Chinese function words still removed


def test_digit_only_tokens_are_not_mistaken_for_acronyms(
    tokenizer: MultilingualTokenizer,
) -> None:
    """``"2024".upper() == "2024"`` would fool a naive all-caps test."""
    assert MultilingualTokenizer._is_acronym("2024") is False
    assert MultilingualTokenizer._is_acronym("US") is True
    assert MultilingualTokenizer._is_acronym("Who") is False
    assert MultilingualTokenizer._is_acronym("A") is False
    assert MultilingualTokenizer._is_acronym("細菌") is False


# --------------------------------------------------------------------------- #
# Non-Latin script must survive (the [a-z0-9]+ bug this stage exists to fix)   #
# --------------------------------------------------------------------------- #


def test_japanese_is_not_silently_emptied(tokenizer: MultilingualTokenizer) -> None:
    result = tokenizer.tokenize(QUESTION_JA, "ja")
    assert result.language == "ja"
    assert result.tokens, "Japanese input produced zero tokens"
    joined = "".join(result.tokens)
    # No mojibake: every character emitted came from the input, and the characters that
    # matter came through intact.
    assert set(joined) <= set(QUESTION_JA)


def test_japanese_reproduces_the_blog_worked_example(
    tokenizer: MultilingualTokenizer,
) -> None:
    """The post's Japanese token list, minus the one token that is not in its sentence.

    The post prints nine tokens for the Japanese question. Eight of them are derivable
    from the sentence it also prints; 治療 ("treatment") is NOT — that word does not
    occur anywhere in

        細菌と酵母から得られる教訓は、より安全な遺伝子編集医薬品にどのように応用されるのでしょうか？

    so the post's own list is internally inconsistent and 8/9 is the ceiling for any
    tokenizer reading that sentence. We assert the achievable eight and pin 治療 as
    absent, so that if a future change ever conjures it the test says why that is wrong.

    This test is the reason mecab-python3 is a core dependency rather than an extra.
    Under the character-bigram fallback 遺伝子 shreds into 遺伝/伝子/子編 and this
    assertion fails — which is the correct outcome, not something to narrow the test for.
    """
    tokens = tokenizer.tokenize(QUESTION_JA, "ja").tokens

    derivable = ("細菌", "教訓", "遺伝子", "酵母", "応用", "編集", "どの", "安全")
    missing = [word for word in derivable if word not in tokens]
    assert not missing, f"blog tokens absent from output: {missing} (got {tokens})"

    assert "治療" not in QUESTION_JA, "the post's sentence changed; revisit this test"
    assert "治療" not in tokens


def test_japanese_stems_are_identity(tokenizer: MultilingualTokenizer) -> None:
    """Snowball has no Japanese algorithm, so stemming must be a no-op, not a mangle."""
    result = tokenizer.tokenize(QUESTION_JA, "ja")
    assert result.stems == result.tokens
    assert "ja" not in SNOWBALL_LANGUAGES


def test_japanese_katakana_loanwords_stay_whole(tokenizer: MultilingualTokenizer) -> None:
    assert "コンピュータ" in tokenizer.words("コンピュータで遺伝子を編集する", "ja")


def test_chinese_is_segmented_by_jieba(tokenizer: MultilingualTokenizer) -> None:
    tokens = tokenizer.tokenize(QUESTION_ZH, "zh").tokens
    for word in ("细菌", "酵母", "基因", "编辑"):
        assert word in tokens
    # jieba found real multi-character words rather than falling back to bigrams.
    assert any(len(token) >= 2 for token in tokens)
    # Chinese function words are removed.
    assert "的" not in tokens
    assert "和" not in tokens


def test_russian_survives_and_is_stemmed(tokenizer: MultilingualTokenizer) -> None:
    result = tokenizer.tokenize("Какие уроки бактерий действительно применимы?", "ru")
    assert result.tokens
    assert "уроки" in result.tokens
    assert "урок" in result.stems  # Snowball Russian trimmed the plural
    assert "какие" not in result.tokens  # Russian stopword


def test_korean_eojeol_are_kept(tokenizer: MultilingualTokenizer) -> None:
    """Hangul is space-delimited; the tokenizer must not n-gram it into dust."""
    tokens = tokenizer.tokenize("한국어 문장 입니다.", "ko").tokens
    assert tokens == ("한국어", "문장", "입니다")


def test_arabic_and_greek_survive(tokenizer: MultilingualTokenizer) -> None:
    assert tokenizer.words("الخميرة والبكتيريا", "ar")
    assert tokenizer.words("ζύμη και βακτήρια", "el")


def test_emoji_and_punctuation_are_dropped_but_do_not_break_anything(
    tokenizer: MultilingualTokenizer,
) -> None:
    result = tokenizer.tokenize("Gene editing 🧬 works!!! (mostly)", "en")
    assert "gene" in result.tokens
    assert "🧬" not in result.tokens


# --------------------------------------------------------------------------- #
# Unicode normalisation                                                        #
# --------------------------------------------------------------------------- #


def test_nfkc_folds_fullwidth_forms(tokenizer: MultilingualTokenizer) -> None:
    fullwidth = "ＧＥＮＥ－ＥＤＩＴＩＮＧ　Ｍｅｄｉｃｉｎｅｓ"
    assert tokenizer.tokenize(fullwidth, "en").stems == ("gene", "edit", "medicin")


def test_nfkc_folds_compatibility_characters(tokenizer: MultilingualTokenizer) -> None:
    # U+FB01 LATIN SMALL LIGATURE FI, the kind of thing PDF extraction produces.
    assert "modification" in tokenizer.words("modiﬁcation", "en")
    # Halfwidth katakana normalises to fullwidth, so it segments like normal Japanese.
    assert "カタカナ" in tokenizer.words("ｶﾀｶﾅ", "ja")


def test_casefolding_is_applied(tokenizer: MultilingualTokenizer) -> None:
    assert tokenizer.words("CRISPR Gene EDITING", "en") == ["crispr", "gene", "editing"]


def test_normalize_is_pure_and_strips(tokenizer: MultilingualTokenizer) -> None:
    assert tokenizer.normalize("  spaced　out  ") == "spaced out"
    assert tokenizer.normalize("") == ""


# --------------------------------------------------------------------------- #
# Sentences                                                                    #
# --------------------------------------------------------------------------- #


def test_sentences_are_split_and_keep_original_case(
    tokenizer: MultilingualTokenizer,
) -> None:
    text = "Yeast is a fungus. Bacteria are not! Is that clear?"
    sentences = tokenizer.sentences(text, "en")
    assert len(sentences) == 3
    assert sentences[0] == "Yeast is a fungus."
    assert tokenizer.tokenize(text, "en").sentences == tuple(sentences)


def test_sentences_survive_a_language_pysbd_does_not_support() -> None:
    assert "sw" not in PYSBD_LANGUAGES
    tok = MultilingualTokenizer()
    sentences = tok.sentences("Chachu ni kuvu. Bakteria si kuvu.", "sw")
    assert sentences == ["Chachu ni kuvu.", "Bakteria si kuvu."]


def test_regex_sentence_fallback_handles_cjk_terminators() -> None:
    assert regex_sentences("細菌です。酵母です。") == ["細菌です。", "酵母です。"]


def test_tokens_are_collected_across_every_sentence(
    tokenizer: MultilingualTokenizer,
) -> None:
    result = tokenizer.tokenize("Yeast ferments. Bacteria divide.", "en")
    assert len(result.sentences) == 2
    assert set(result.stems) == {"yeast", "ferment", "bacteria", "divid"}


# --------------------------------------------------------------------------- #
# Options                                                                      #
# --------------------------------------------------------------------------- #


def test_stopword_removal_can_be_disabled() -> None:
    tok = MultilingualTokenizer(remove_stopwords=False)
    assert "what" in tok.tokenize(QUESTION_EN, "en").tokens


def test_stemming_can_be_disabled() -> None:
    tok = MultilingualTokenizer(stem=False)
    result = tok.tokenize(QUESTION_EN, "en")
    assert result.stems == result.tokens
    assert "medicines" in result.stems


def test_min_token_length_filters_short_tokens() -> None:
    tok = MultilingualTokenizer(min_token_length=4)
    assert all(len(t) >= 4 for t in tok.tokenize("a bb ccc dddd eeeee", "en").tokens)


def test_extra_stopwords_extend_a_shipped_list() -> None:
    tok = MultilingualTokenizer(extra_stopwords={"en": ["yeast"]})
    stems = tok.tokenize(QUESTION_EN, "en").stems
    assert "yeast" not in stems
    assert "bacteria" in stems


def test_extra_stopwords_can_cover_an_unshipped_language() -> None:
    assert stopwords_for("sw") == frozenset()
    assert "na" in stopwords_for("sw", ["NA"])


# --------------------------------------------------------------------------- #
# Language handling                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("en", "en"),
        ("EN", "en"),
        ("en-US", "en"),
        ("en_GB", "en"),
        ("eng", "en"),
        ("English", "en"),
        ("zh-Hant", "zh"),
        ("jpn", "ja"),
        ("", "und"),
        (None, "und"),
        ("klingon", "und"),
    ],
)
def test_language_tags_normalize(given: str | None, expected: str) -> None:
    assert normalize_language(given) == expected


def test_every_tag_spelling_produces_the_same_tokens(
    tokenizer: MultilingualTokenizer,
) -> None:
    baseline = tokenizer.tokenize(QUESTION_EN, "en")
    for tag in ("EN", "en-US", "eng", "English"):
        assert tokenizer.tokenize(QUESTION_EN, tag) == baseline


def test_unknown_language_still_tokenizes_without_stemming(
    tokenizer: MultilingualTokenizer,
) -> None:
    result = tokenizer.tokenize("Chachu na bakteria", "klingon")
    assert result.language == "und"
    assert result.tokens == ("chachu", "na", "bakteria")
    assert result.stems == result.tokens


def test_missing_language_falls_back_to_the_script(
    tokenizer: MultilingualTokenizer,
) -> None:
    assert tokenizer.tokenize(QUESTION_JA, None).language == "ja"
    assert tokenizer.tokenize(QUESTION_ZH, None).language == "zh"
    assert tokenizer.tokenize("hello", None).language == "und"
    assert guess_script_language("한국어") == "ko"


def test_snowball_coverage_matches_the_posts_claim() -> None:
    """The post cites pyStemmer's 24 languages; this build must reach at least that."""
    assert len(SNOWBALL_LANGUAGES) >= 24
    assert snowball_algorithm("en") == "english"
    assert snowball_algorithm("ja") is None


# --------------------------------------------------------------------------- #
# Determinism, purity, pickling                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "language"),
    [(QUESTION_EN, "en"), (QUESTION_JA, "ja"), (QUESTION_ZH, "zh")],
)
def test_repeated_calls_are_identical(text: str, language: str) -> None:
    tok = MultilingualTokenizer(use_mecab=False)
    first = tok.tokenize(text, language)
    second = tok.tokenize(text, language)
    assert first == second
    # ... and a cold instance agrees with a warmed-up one, so the caches are pure.
    assert MultilingualTokenizer(use_mecab=False).tokenize(text, language) == first


def test_tokenizer_round_trips_through_pickle() -> None:
    tok = MultilingualTokenizer(use_mecab=False, min_token_length=2, stem=True)
    warm = tok.tokenize(QUESTION_EN, "en")  # populate the C-object caches first
    revived: MultilingualTokenizer = pickle.loads(pickle.dumps(tok))
    assert revived.tokenize(QUESTION_EN, "en") == warm
    assert revived.min_token_length == 2
    assert revived.use_mecab is False


def test_result_is_immutable_and_hashable(tokenizer: MultilingualTokenizer) -> None:
    result = tokenizer.tokenize(QUESTION_EN, "en")
    assert isinstance(result, Tokenized)
    assert hash(result)
    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass raises FrozenInstanceError
        result.tokens = ()  # type: ignore[misc]


def test_unique_stems_deduplicates_in_order(tokenizer: MultilingualTokenizer) -> None:
    result = tokenizer.tokenize("Yeast and yeast and bacteria.", "en")
    assert result.stems == ("yeast", "yeast", "bacteria")
    assert result.unique_stems() == ("yeast", "bacteria")


# --------------------------------------------------------------------------- #
# Degenerate input                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", ["", "   ", "\n\t ", "!!! ??? ---"])
def test_empty_and_punctuation_only_input(
    tokenizer: MultilingualTokenizer, text: str
) -> None:
    result = tokenizer.tokenize(text, "en")
    assert result.tokens == ()
    assert result.stems == ()
    assert result.language == "en"
    assert tokenizer.words(text, "en") == []


def test_empty_input_never_leaves_junk_sentences(
    tokenizer: MultilingualTokenizer,
) -> None:
    """Empty and whitespace-only input must produce no sentences at all.

    Punctuation-only input is deliberately *not* asserted here: the terminator regex
    still emits ``('!!! ???', '---')`` for it, which is a known wart, not a claim.
    """
    for text in ("", "   ", "\n\t "):
        assert tokenizer.tokenize(text, "en").sentences == ()


def test_falsy_text_never_raises_whatever_the_language_tag_is(
    tokenizer: MultilingualTokenizer,
) -> None:
    """Regression: the script guess used to run before the empty-text guard.

    ``tokenize(None, "en")`` returned an empty result while ``tokenize(None, None)``
    raised ``TypeError`` from ``re.search`` — two different answers to the same
    degenerate input. Both must now degrade the same way.
    """
    for text in ("", None):
        for language in ("en", None):
            result = tokenizer.tokenize(text, language)  # type: ignore[arg-type]
            assert result.tokens == ()
            assert result.stems == ()
            assert tokenizer.words(text, language) == []  # type: ignore[arg-type]
            assert tokenizer.sentences(text, language) == []  # type: ignore[arg-type]


def test_stopwords_only_input_yields_nothing(tokenizer: MultilingualTokenizer) -> None:
    assert tokenizer.tokenize("what is the and of it", "en").tokens == ()


def test_very_long_input_does_not_explode(tokenizer: MultilingualTokenizer) -> None:
    result = tokenizer.tokenize(" ".join([QUESTION_EN] * 200), "en")
    assert len(result.sentences) == 200
    assert len(result.tokens) == 9 * 200


# --------------------------------------------------------------------------- #
# Segmentation primitives                                                      #
# --------------------------------------------------------------------------- #


def test_mecab_is_optional_and_never_fatal() -> None:
    """The `cjk` extra may be absent; auto-detect and an explicit request must degrade."""
    auto = MultilingualTokenizer(use_mecab=None)
    demanded = MultilingualTokenizer(use_mecab=True)
    for tok in (auto, demanded):
        assert isinstance(tok.mecab_available, bool)
        assert tok.tokenize(QUESTION_JA, "ja").tokens, "Japanese lost when MeCab is absent"
    # Whatever this machine has, asking for the fallback explicitly must give bigrams.
    assert MultilingualTokenizer(use_mecab=False).mecab_available is False


def test_mecab_path_is_used_when_a_tagger_exists() -> None:
    """Exercise the MeCab branch with a stub, so the code runs with or without the extra."""

    class WakatiStub:
        def parse(self, text: str) -> str:
            return "遺伝子 編集 医薬品 \n"

    assert segment_japanese("遺伝子編集医薬品", WakatiStub()) == ["遺伝子", "編集", "医薬品"]
    # A tagger that blows up or returns nothing must fall back rather than propagate.

    class BrokenStub:
        def parse(self, text: str) -> str:
            raise RuntimeError("dictionary not found")

    assert segment_japanese("遺伝子編集", BrokenStub()) == char_bigrams("遺伝子編集")
    assert segment_japanese("遺伝子編集", None) == char_bigrams("遺伝子編集")


def test_char_bigrams_are_overlapping() -> None:
    assert char_bigrams("遺伝子編集") == ["遺伝", "伝子", "子編", "編集"]
    assert char_bigrams("細") == ["細"]
    assert char_bigrams("") == []


def test_mixed_script_input_is_segmented_run_by_run() -> None:
    tokens = segment_words("crispr技術で遺伝子編集", "ja")
    assert "crispr" in tokens
    assert "編集" in tokens


def test_numbers_and_alphanumerics_are_kept() -> None:
    assert segment_words("covid-19 cas9 type 2", "en") == ["covid", "19", "cas9", "type", "2"]


def test_underscore_separates_tokens() -> None:
    assert segment_words("gene_editing", "en") == ["gene", "editing"]
