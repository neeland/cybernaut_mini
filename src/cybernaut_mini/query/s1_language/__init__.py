"""Hybrid-3 stage 1 — language detection and translation.

The first stage of the query path. It decides what language the question is in and
what text the remaining seven stages should actually work with, and it is the only
stage that may need to talk to an LLM before retrieval has begun.

Public surface::

    from cybernaut_mini.query.s1_language import prepare_question
    prepared = prepare_question(question, target_lang="ja", translator=translator)
    prepared.text_for_retrieval  # -> what stage 2 tokenizes

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — "1 // Language Detection
    and Translation", the first of the eight Hybrid-3 stages: fasttext-langdetect for
    detection (170+ languages), gemma-3n-e4b-it via OpenRouter for translation (140
    languages). Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions: the stage is split into detection (offline, deterministic, never raises)
and translation (network, optional, behind a protocol) because only the second needs
a credential — an offline install must still get a language code.

Alternatives rejected: one module exporting everything (simpler import, but it would
make the network client an unavoidable import of the query path); a class holding
both halves (state that a single function call does not need).
"""

from __future__ import annotations

from cybernaut_mini.query.s1_language.detect import (
    UNKNOWN_LANGUAGE,
    LanguageResult,
    detect_language,
    normalise_for_detection,
)
from cybernaut_mini.query.s1_language.prepare import (
    PreparedQuestion,
    normalise_lang_code,
    prepare_question,
)
from cybernaut_mini.query.s1_language.translate import (
    API_KEY_ENV_VARS,
    DEFAULT_BASE_URL,
    DEFAULT_TRANSLATION_MODEL,
    MissingAPIKeyError,
    NullTranslator,
    OpenAICompatibleTranslator,
    TranslationError,
    Translator,
    language_name,
)

__all__ = [
    "API_KEY_ENV_VARS",
    "DEFAULT_BASE_URL",
    "DEFAULT_TRANSLATION_MODEL",
    "UNKNOWN_LANGUAGE",
    "LanguageResult",
    "MissingAPIKeyError",
    "NullTranslator",
    "OpenAICompatibleTranslator",
    "PreparedQuestion",
    "TranslationError",
    "Translator",
    "detect_language",
    "language_name",
    "normalise_for_detection",
    "normalise_lang_code",
    "prepare_question",
]
