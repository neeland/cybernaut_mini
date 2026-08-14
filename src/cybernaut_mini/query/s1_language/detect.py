"""Stage 1a: detect the language of a question with fastText ``lid.176``.

The pipeline needs a language code before anything else can happen: stage 2 picks a
stemmer and a sentence splitter per language, stage 4 interpolates the language name
into the E5 instruction, and stage 1b decides whether a translation is required at
all. Detection is therefore the first thing a question meets, and it must never be
the thing that crashes the query path — an unrecognisable question still deserves a
search, so every failure mode here degrades to :data:`UNKNOWN_LANGUAGE` instead of
raising.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 1 "Language
    Detection and Translation": "it is still necessary to detect what language the
    question and expansions were written in", with `fasttext-langdetect` (170+
    languages) named as the detector. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - fastText's ``predict`` treats a newline as a record separator and rejects any
      string containing one, so a pasted multi-line question is a real crash in the
      naive call. All whitespace is collapsed to single spaces before the call. This
      is lossless for language ID, which is a bag-of-character-ngrams model, and it
      is done here rather than trusting the installed ``ftlangdetect`` build, because
      only newer builds normalise internally.
    - Empty or whitespace-only input is not an error. It returns
      ``LanguageResult(UNKNOWN_LANGUAGE, 0.0)`` so the caller can route a blank
      question through the same code path as any other.
    - The ``str`` annotation is the contract, but it is not enforced by the runtime,
      and a ``None`` arriving from a caller that forgot to default an optional field
      is the likeliest way this function is ever called wrongly. Non-string input is
      therefore treated exactly like blank input — :data:`UNKNOWN_LANGUAGE` — rather
      than being allowed to surface as a ``TypeError`` from inside the regex layer.
      The annotation stays ``str`` because passing anything else is still a bug in
      the caller; this stage just refuses to be the place the query dies.
    - Very short input ("hi", "3.14", a bare URL) is detected anyway rather than
      rejected, and the raw fastText score is passed through undamped. Callers that
      care apply their own threshold; the post does not disclose one, and inventing a
      damping curve here would hide the model's own — already low — confidence.
    - Language codes are ISO-639-1/3 lowercase, exactly as ``lid.176`` emits them,
      with no mapping to BCP-47 regions. The label prefix ``__label__`` is stripped
      by ``ftlangdetect`` itself.

Alternatives rejected:
    - ``langdetect``/``langid`` (pure Python, no 131 MB model download): rejected
      because the post names fasttext-langdetect specifically, and its 170-language
      coverage is the reason the rest of the pipeline can claim to be multilingual.
    - Script-based detection (Unicode block heuristics) as a pre-filter: cheap and
      offline, but it cannot separate the languages that share a script (en/de/es,
      ja/zh at the Han level), which is precisely the discrimination stage 2 needs.
    - Raising on undetectable input: correct for a batch validator, wrong for a query
      path where the cost of a bad language guess is a slightly worse stemmer and the
      cost of an exception is no results at all.
    - Returning fastText's top-k candidates for mixed-script text: genuinely useful,
      but the downstream stages consume exactly one language code, so a second
      candidate would have no consumer and would only widen the API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ftlangdetect import detect as _ft_detect

__all__ = [
    "UNKNOWN_LANGUAGE",
    "LanguageResult",
    "detect_language",
    "normalise_for_detection",
]

#: Returned when the language cannot be determined (blank input, or a detector
#: failure). "und" is the ISO 639-2 code for "undetermined".
UNKNOWN_LANGUAGE = "und"

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class LanguageResult:
    """The detected language of a piece of text.

    Attributes
    ----------
    lang:
        Lowercase ISO language code as emitted by ``lid.176`` (``"en"``, ``"ja"``,
        ``"zh"``), or :data:`UNKNOWN_LANGUAGE` when detection was not possible.
    confidence:
        fastText's probability for that label, clamped to ``[0.0, 1.0]``. Exactly
        ``0.0`` when the language is unknown.
    """

    lang: str
    confidence: float

    @property
    def is_known(self) -> bool:
        """True when a real language code was determined."""
        return self.lang != UNKNOWN_LANGUAGE


def normalise_for_detection(text: str) -> str:
    """Collapse every run of whitespace to a single space and strip the ends.

    fastText rejects strings containing a newline, so this is not cosmetic: it is
    the difference between detecting a pasted multi-line question and raising.

    Anything that is not a ``str`` normalises to ``""``. That is the boundary guard
    for the whole module: it is the only place a caller's bad value can reach the
    regex, and returning empty here is what turns a ``TypeError`` into
    :data:`UNKNOWN_LANGUAGE` two frames up.
    """
    if not isinstance(text, str):
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _coerce_result(raw: Any) -> LanguageResult:
    """Turn whatever ``ftlangdetect`` returned into a :class:`LanguageResult`.

    ``detect(k=1)`` returns a single ``{"lang": ..., "score": ...}`` mapping, but
    older and newer builds have disagreed on whether a list is returned, so the
    first element of a sequence is accepted too.
    """
    if isinstance(raw, list | tuple):
        if not raw:
            return LanguageResult(UNKNOWN_LANGUAGE, 0.0)
        raw = raw[0]
    if not isinstance(raw, dict):
        return LanguageResult(UNKNOWN_LANGUAGE, 0.0)

    lang = str(raw.get("lang", "") or "").strip().lower()
    if not lang:
        return LanguageResult(UNKNOWN_LANGUAGE, 0.0)
    lang = lang.removeprefix("__label__")

    try:
        score = float(raw.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    if score != score:  # NaN — never let it reach a comparison downstream.
        score = 0.0
    return LanguageResult(lang, min(1.0, max(0.0, score)))


def detect_language(text: str) -> LanguageResult:
    """Detect the language of *text*.

    Deterministic: ``lid.176`` is a frozen model and the call takes no sampling
    parameters, so the same string always yields the same code and score.

    Blank input, non-string input, and any detector failure (a missing model file, a
    string fastText still refuses) all degrade to
    ``LanguageResult(UNKNOWN_LANGUAGE, 0.0)`` rather than propagating — losing the
    language of one question must not fail the search. This function does not raise.
    """
    normalised = normalise_for_detection(text)
    if not normalised:
        return LanguageResult(UNKNOWN_LANGUAGE, 0.0)
    try:
        raw = _ft_detect(normalised, low_memory=False, k=1)
    except (ValueError, RuntimeError, OSError):
        return LanguageResult(UNKNOWN_LANGUAGE, 0.0)
    return _coerce_result(raw)
