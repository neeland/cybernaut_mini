"""Sentence boundary detection and word segmentation across scripts.

Two problems live here, and the post names a different library for each.

*Sentences.* pySBD ("support for 22 languages" in the post) implements the Golden Rules
sentence boundary set, so it is used wherever it has a language profile; everything else
falls back to a terminator regex that also understands the CJK full stops ``。！？``.

*Words.* Latin, Cyrillic, Greek and Hangul text is delimited by spaces and punctuation,
so a Unicode word regex is enough. Han and kana are not delimited at all, so those runs
are cut out of the string and handed to a real segmenter: jieba for Chinese (the post's
choice) and MeCab for Japanese when the optional ``cjk`` extra is installed.

The regex is the part that matters most. The repo's original ``text.py`` tokenizer uses
``[a-z0-9]+``, which does not merely segment non-Latin text badly — it *deletes* it, so
a Russian or Japanese query silently produces zero tokens and the lexical index returns
nothing while looking perfectly healthy. The pattern here is ``[^\\W_]+``: every Unicode
letter and digit, minus the underscore, splitting on hyphens and apostrophes. Splitting
on the hyphen is also what makes the post's worked example come out right — its expected
tokens contain ``gene`` and ``edit`` separately, so "gene-editing" must be two words.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 2, "Multilingual
    Tokenization": "sentence boundary detection, text segmentation ... Getting this to
    work equally well across many languages is difficult." Tech stack quoted there:
    pySBD (22 languages), jieba (Chinese), mecab-python3 / ipadic (Japanese), "regex —
    sigh. Way too much regex." Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - MeCab is optional (it needs a C extension plus a dictionary), so Japanese must
      still work without it. The fallback walks a Japanese run one script class at a
      time: kanji and hiragana runs become overlapping character bigrams — the classic
      CJK bigram index, and the reason 編集 falls out of 遺伝子編集医薬品 and どの out of
      にどのように — while katakana runs are kept whole, because a katakana run is almost
      always one loanword or name that bigrams would shred.
    - Korean is left to the word regex. Hangul is space-delimited at the eojeol level,
      so an n-gram path would destroy words that arrived already segmented. Without
      mecab-ko there is no morpheme analysis, so eojeol are the tokens.
    - jieba's HMM path is used for unknown words. It is a Viterbi decode over fixed
      tables, so it is deterministic; only the dictionary version can change results.
    - pySBD and MeCab objects are cached per language by the caller and are not
      thread-safe (pySBD stores the text being segmented on the instance).

Alternatives rejected:
    - wtpsplit for sentence splitting: the post lists it and also calls it "incredibly
      slow" — it is a neural model, which would add a torch dependency and a download to
      a stage that must run offline in tests.
    - spaCy as the universal segmenter: it is in the post's stack, but each language
      needs its own downloaded model, so an import that succeeds still fails at runtime.
      It stays optional and is not required by anything here.
    - Non-overlapping bigrams in the Japanese fallback: half the size, but they only
      match when the query happens to share the same parity offset as the document.
      Overlapping bigrams trade index size for recall, which is the trade the post makes
      everywhere else ("People need precision; AIs need recall").
"""  # noqa: RUF002 — the fullwidth CJK terminators are the characters being described.

from __future__ import annotations

import logging
import re
from typing import Any

from .languages import CJK_RUN_RE, HAN_RANGES, KANA_RE, normalize_language

# Unicode letters and digits, minus the underscore. Hyphens, apostrophes and every kind
# of punctuation act as separators, so "gene-editing" -> ["gene", "editing"].
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

# A sentence is a run of non-terminators followed by any terminators that trail it.
# ASCII and CJK terminators both count; the CJK ones are not followed by a space.
_SENTENCE_RE = re.compile(r"[^.!?。！？…؟\n\r]+[.!?。！？…؟]*")  # noqa: RUF001 — CJK terminators

_HIRAGANA_RANGES = "぀-ゟ"
# Range bounds, not punctuation: U+30A0 KATAKANA-HIRAGANA DOUBLE HYPHEN opens the block.
_KATAKANA_RANGES = "゠-ヿㇰ-ㇿ"  # noqa: RUF001

# One alternation per script class, so a Japanese run can be walked class by class.
_JA_CLASS_RE = re.compile(
    f"([{HAN_RANGES}]+)|([{_KATAKANA_RANGES}]+)|([{_HIRAGANA_RANGES}]+)"
)

# pySBD's own language registry, read at import time so this list cannot drift from the
# installed version. The literal set is only the fallback if that attribute moves.
_PYSBD_FALLBACK = frozenset(
    {
        "am", "ar", "bg", "da", "de", "el", "en", "es", "fa", "fr", "hi", "hy",
        "it", "ja", "kk", "mr", "my", "nl", "pl", "ru", "sk", "ur", "zh",
    }
)


def _pysbd_languages() -> frozenset[str]:
    try:
        from pysbd.languages import LANGUAGE_CODES  # type: ignore[import-untyped]
    except Exception:  # pragma: no cover - pysbd is a hard dependency
        return _PYSBD_FALLBACK
    try:
        return frozenset(str(code) for code in LANGUAGE_CODES)
    except Exception:  # pragma: no cover - defensive against an API change
        return _PYSBD_FALLBACK


PYSBD_LANGUAGES: frozenset[str] = _pysbd_languages()


def build_sentence_segmenter(language: str) -> Any | None:
    """Return a pySBD segmenter for ``language``, or ``None`` if it has no profile.

    The returned object is stateful and must not be shared across threads.
    """
    if language not in PYSBD_LANGUAGES:
        return None
    try:
        import pysbd  # type: ignore[import-untyped]

        return pysbd.Segmenter(language=language, clean=False)
    except Exception:  # pragma: no cover - unusable profile degrades to the regex
        return None


def regex_sentences(text: str) -> list[str]:
    """Terminator-based sentence split, used wherever pySBD has no language profile."""
    out: list[str] = []
    for line in text.splitlines():
        for match in _SENTENCE_RE.finditer(line):
            piece = match.group().strip()
            if piece:
                out.append(piece)
    return out


def split_sentences(text: str, segmenter: Any | None) -> list[str]:
    """Split ``text`` into sentences with ``segmenter`` if given, else the regex."""
    if not text.strip():
        return []
    if segmenter is not None:
        try:
            pieces = [str(s).strip() for s in segmenter.segment(text)]
            found = [p for p in pieces if p]
            if found:
                return found
        except Exception:  # pragma: no cover - pySBD raising is not fatal
            pass
    return regex_sentences(text)


def _jieba_cut(run: str) -> list[str]:
    import jieba  # type: ignore[import-untyped]

    jieba.setLogLevel(logging.ERROR)
    return [tok for tok in (t.strip() for t in jieba.lcut(run)) if tok]


def char_bigrams(run: str) -> list[str]:
    """Overlapping character bigrams; a one-character run yields itself."""
    if len(run) < 2:
        return [run] if run else []
    return [run[i : i + 2] for i in range(len(run) - 1)]


def _japanese_fallback(run: str) -> list[str]:
    """Segment a Japanese run without MeCab, one script class at a time.

    Kanji runs and hiragana runs become overlapping character bigrams; katakana runs are
    kept whole. The asymmetry is the point: a katakana run is nearly always a single
    loanword or name (``コンピュータ``) that bigrams would destroy, whereas a hiragana run
    is a string of particles and inflection whose only recoverable units are short
    n-grams — which is how ``どの`` falls out of ``にどのように``.
    """
    out: list[str] = []
    for kanji, katakana, hiragana in _JA_CLASS_RE.findall(run):
        if kanji:
            out.extend(char_bigrams(kanji))
        elif katakana:
            out.append(katakana)
        elif hiragana:
            out.extend(char_bigrams(hiragana))
    return out


def segment_japanese(run: str, tagger: Any | None) -> list[str]:
    """MeCab segmentation when the optional ``cjk`` extra is installed, else bigrams."""
    if tagger is not None:
        try:
            parsed = tagger.parse(run)
        except Exception:  # pragma: no cover - a broken dictionary must not be fatal
            parsed = None
        if parsed:
            tokens = [t for t in str(parsed).split() if t]
            if tokens:
                return tokens
    return _japanese_fallback(run)


def segment_cjk_run(run: str, language: str, tagger: Any | None) -> list[str]:
    """Segment one uninterrupted run of Han/kana characters."""
    if language == "ja" or (language != "zh" and KANA_RE.search(run)):
        return segment_japanese(run, tagger)
    try:
        return _jieba_cut(run)
    except Exception:  # pragma: no cover - jieba is a hard dependency
        return char_bigrams(run)


def segment_words(text: str, language: str, tagger: Any | None = None) -> list[str]:
    """Split ``text`` into surface words, routing Han/kana runs to a real segmenter.

    Mixed-script input is handled run by run, so "CRISPRで遺伝子編集" keeps ``crispr``
    as a Latin token and still segments the Japanese around it.
    """
    lang = normalize_language(language)
    out: list[str] = []
    pos = 0
    for match in CJK_RUN_RE.finditer(text):
        if match.start() > pos:
            out.extend(WORD_RE.findall(text[pos : match.start()]))
        out.extend(segment_cjk_run(match.group(), lang, tagger))
        pos = match.end()
    if pos < len(text):
        out.extend(WORD_RE.findall(text[pos:]))
    return [tok for tok in out if tok]
