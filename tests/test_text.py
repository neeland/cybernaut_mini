from __future__ import annotations

from cybernaut_mini.text import STOPWORDS, TextProcessor, lexical_form, normalize


def test_normalize_nfkc_and_whitespace() -> None:
    # \ufb01 is the "fi" ligature and \u00a0 a no-break space; NFKC folds both.
    assert normalize("\ufb01ne \u00a0spacing\n\ttext") == "fine spacing text"


def test_lexical_form_lowercases() -> None:
    assert lexical_form("Gene Therapy") == "gene therapy"


def test_tokenize_strips_punctuation(text_processor: TextProcessor) -> None:
    assert text_processor.tokenize("CRISPR-Cas9, edits DNA!") == ["crispr", "cas9", "edits", "dna"]


def test_content_tokens_remove_stopwords(text_processor: TextProcessor) -> None:
    tokens = text_processor.content_tokens("The gene is in the cell")
    assert tokens == ["gene", "cell"]
    assert all(token not in STOPWORDS for token in tokens)


def test_regex_backend_returns_no_entities(text_processor: TextProcessor) -> None:
    assert text_processor.backend == "regex"
    assert text_processor.entities("NOSIBLE built Cybernaut in Cape Town") == []


def test_tokenize_is_deterministic(text_processor: TextProcessor) -> None:
    text = "Solar panels reached 25% efficiency in 2025"
    assert text_processor.tokenize(text) == text_processor.tokenize(text)
