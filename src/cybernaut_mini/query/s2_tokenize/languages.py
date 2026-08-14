"""Language-code normalisation and script detection for the tokenizer.

Stage 1 of the pipeline (language detection) hands stage 2 a language tag, but the
tag's *shape* depends on who produced it: ``fasttext-langdetect`` returns ISO 639-1
(``"en"``), an HTTP ``Accept-Language`` header returns ``"en-US"``, a dataset column
might hold ``"eng"`` or ``"English"``. Every one of those must select the same English
pipeline, so all callers funnel through :func:`normalize_language` first. Anything
unrecognised collapses to ``"und"`` (the ISO 639-2 "undetermined" code), which the
tokenizer treats as "segment it, but do not pretend to know its morphology".

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 2, "Multilingual
    Tokenization": a standardized interface that wraps a dozen NLP packages and
    "abstracts away all language-specific complexities". Abstracting them away starts
    with agreeing on what a language *is*. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - The post never states the tag vocabulary, so ISO 639-1 (two letters) is chosen as
      the internal canonical form because it is what ``fasttext-langdetect`` — the
      library the post names for stage 1 — actually emits.
    - Macrolanguage and script subtags are dropped (``zh-Hant`` -> ``zh``). Traditional
      and Simplified Chinese are segmented by the same jieba dictionary here, so keeping
      the subtag would imply a distinction the pipeline cannot honour.
    - An unknown tag is not an error. A production search engine sees garbage language
      tags every day; degrading to ``"und"`` keeps the query searchable, whereas raising
      would drop it.

Alternatives rejected:
    - ``langcodes``/``babel`` for full BCP-47 parsing: correct and complete, but a new
      dependency for a lookup table of ~40 entries the tokenizer can actually act on.
      Nothing downstream can use a subtag we do not branch on.
    - Passing raw tags straight through and branching on them at every call site: this
      is exactly the "language-specific complexity" the post says the interface exists
      to hide, and it makes ``"en"`` and ``"en-GB"`` behave differently by accident.
"""

from __future__ import annotations

import re

UNDETERMINED = "und"
"""Canonical tag for "no usable language information"."""

# ISO 639-2/639-3 three-letter codes for the languages this stage has any special
# handling for (a stemmer, a stopword list, a segmenter or a sentence splitter).
_THREE_LETTER: dict[str, str] = {
    "ara": "ar",
    "bul": "bg",
    "cat": "ca",
    "ces": "cs",
    "cze": "cs",
    "dan": "da",
    "deu": "de",
    "ger": "de",
    "ell": "el",
    "gre": "el",
    "eng": "en",
    "epo": "eo",
    "est": "et",
    "eus": "eu",
    "baq": "eu",
    "fas": "fa",
    "per": "fa",
    "fin": "fi",
    "fra": "fr",
    "fre": "fr",
    "gle": "ga",
    "heb": "he",
    "hin": "hi",
    "hun": "hu",
    "hye": "hy",
    "arm": "hy",
    "ind": "id",
    "ita": "it",
    "jpn": "ja",
    "kaz": "kk",
    "kor": "ko",
    "lit": "lt",
    "mar": "mr",
    "mya": "my",
    "bur": "my",
    "nep": "ne",
    "nld": "nl",
    "dut": "nl",
    "nor": "no",
    "pol": "pl",
    "por": "pt",
    "ron": "ro",
    "rum": "ro",
    "rus": "ru",
    "slk": "sk",
    "slo": "sk",
    "spa": "es",
    "srp": "sr",
    "sot": "st",
    "swe": "sv",
    "tam": "ta",
    "tur": "tr",
    "ukr": "uk",
    "urd": "ur",
    "yid": "yi",
    "zho": "zh",
    "chi": "zh",
}

# English names, because corpora label rows "English" more often than anyone admits.
_ENGLISH_NAMES: dict[str, str] = {
    "arabic": "ar",
    "armenian": "hy",
    "basque": "eu",
    "bulgarian": "bg",
    "burmese": "my",
    "catalan": "ca",
    "chinese": "zh",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "esperanto": "eo",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "indonesian": "id",
    "irish": "ga",
    "italian": "it",
    "japanese": "ja",
    "kazakh": "kk",
    "korean": "ko",
    "lithuanian": "lt",
    "marathi": "mr",
    "nepali": "ne",
    "norwegian": "no",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "slovak": "sk",
    "spanish": "es",
    "swedish": "sv",
    "tamil": "ta",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "yiddish": "yi",
}

_TAG_RE = re.compile(r"^[a-z]{2,3}$")

# Script ranges. Hangul is deliberately absent: Korean is written with spaces between
# eojeol, so the generic word regex already segments it, and routing it through a
# character n-gram path would shred words that arrive correctly delimited.
HAN_RANGES = "㐀-䶿一-鿿豈-﫿"
# Hiragana, then Katakana (which opens at U+30A0, a double hyphen that looks like `=`
# but is a block boundary here, not punctuation), then the phonetic-extension small kana.
KANA_RANGES = "぀-ゟ゠-ヿㇰ-ㇿ"  # noqa: RUF001

HAN_RE = re.compile(f"[{HAN_RANGES}]")
KANA_RE = re.compile(f"[{KANA_RANGES}]")
HANGUL_RE = re.compile(r"[가-힯ᄀ-ᇿ]")
CJK_RUN_RE = re.compile(f"[{HAN_RANGES}{KANA_RANGES}]+")
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


def normalize_language(language: str | None) -> str:
    """Map any reasonable spelling of a language onto a canonical ISO 639-1 tag.

    Returns :data:`UNDETERMINED` when the input is empty or unrecognised, never raises.

    >>> normalize_language("en-US"), normalize_language("ENG"), normalize_language(None)
    ('en', 'en', 'und')
    """
    if not language:
        return UNDETERMINED
    tag = language.strip().lower().replace("_", "-")
    if not tag:
        return UNDETERMINED
    primary = tag.split("-", 1)[0]
    if primary in _ENGLISH_NAMES:
        return _ENGLISH_NAMES[primary]
    if len(primary) == 3 and primary in _THREE_LETTER:
        return _THREE_LETTER[primary]
    if _TAG_RE.match(primary) and len(primary) == 2:
        return primary
    # Full names that only appear unsplit, e.g. "simplified chinese".
    for name, code in _ENGLISH_NAMES.items():
        if name in tag:
            return code
    return UNDETERMINED


def guess_script_language(text: str) -> str:
    """Best-effort language guess from script alone, for when no tag was supplied.

    This is *not* language identification (that is stage 1's job with fastText); it only
    answers the question the segmenter actually needs: which word-splitting strategy
    does this string require? Latin text therefore returns :data:`UNDETERMINED` rather
    than guessing English.
    """
    if KANA_RE.search(text):
        return "ja"
    if HAN_RE.search(text):
        return "zh"
    if HANGUL_RE.search(text):
        return "ko"
    return UNDETERMINED
