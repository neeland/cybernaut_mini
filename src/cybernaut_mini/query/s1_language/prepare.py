"""Stage 1 entry point: detect, optionally translate, hand the next stage one string.

:func:`prepare_question` is the whole of stage 1 as the pipeline sees it. It answers
the two questions every later stage asks — *what language is this?* and *what text do
I actually search with?* — and it answers them without ever failing the query.

The post's worked example runs through here unchanged:

>>> prepare_question(
...     "What lessons from bacteria and yeast actually translate into safer "
...     "gene-editing medicines?"
... ).translated
False

and with ``target_lang="ja"`` plus a translator it becomes
"細菌と酵母から得られる教訓は、より安全な遺伝子編集医薬品にどのように応用されるのでしょうか？".

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 1 "Language
    Detection and Translation": detect the language of the question and expansions
    and, "if results are requested in a different language, translate that question
    into the requested language". The post's own example passes the English question
    through untranslated and renders the same question in Japanese. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - "Results requested in a different language" is expressed as an explicit
      ``target_lang``. The post does not say how the request is made; a caller-supplied
      code is the only interpretation that does not require guessing user intent.
    - ``target_lang`` is normalised to its primary subtag, lowercased, so ``"ja-JP"``,
      ``"JA"`` and ``"ja"`` all mean Japanese. Regional variants are not a distinction
      the detector can make, so keeping one would only produce spurious mismatches.
    - Non-string input is coerced to ``""`` at the top of the function, which makes it
      indistinguishable from a blank question: language ``und``, no translator call,
      an empty ``text_for_retrieval``. The alternative is a ``TypeError`` out of the
      detector's regex, and stage 1's whole promise is that it does not kill the query
      path. Every field of :class:`PreparedQuestion` stays a real ``str``, so no
      downstream stage inherits the ``None``.
    - With no translator supplied, stage 1 is the identity: the original text is
      searched and ``translated`` is False. That is the offline default, and it is a
      decision the caller made by not passing a translator — not a silent recovery.
    - A translator that DOES get passed is trusted to be usable, so its exceptions
      propagate. A missing ``OPENROUTER_API_KEY`` is a configuration bug, and
      swallowing it would return results in the wrong language with no signal that
      anything went wrong, which is harder to diagnose than the exception.
    - ``translated`` reports what actually happened, not what was attempted: it is
      True only when a translator returned text different from the original. A
      translator that returns its input (e.g. :class:`NullTranslator`) leaves it False.
    - The language reported is always the language of the ORIGINAL question, not of
      ``text_for_retrieval``; ``retrieval_language`` gives the latter. Stage 4's E5
      instruction wants the language of the text it is about to embed, so it reads
      ``retrieval_language``.

Alternatives rejected:
    - Detecting the language of the translation as well (a second fastText call to
      confirm the model complied): rejected as a doubling of cost for a check whose
      only sane failure response — search the original anyway — is already what an
      empty/failed translation produces.
    - Translating first and detecting afterwards: cheaper by one call when a target
      is set, but then a same-language request pays for a pointless LLM round trip,
      which is exactly the case the post's English example is showing off.
    - Making stage 1 raise when a target language is requested and no translator is
      configured: honest, but it turns a configuration gap into a dead query path,
      and the pipeline has no fallback to offer above this layer.
"""  # noqa: RUF002 — the quoted Japanese is the post's own text, fullwidth mark and all.

from __future__ import annotations

from dataclasses import dataclass

from cybernaut_mini.query.s1_language.detect import LanguageResult, detect_language
from cybernaut_mini.query.s1_language.translate import Translator

__all__ = ["PreparedQuestion", "normalise_lang_code", "prepare_question"]


@dataclass(frozen=True, slots=True)
class PreparedQuestion:
    """The output of stage 1 — everything later stages need to know about language.

    Attributes
    ----------
    original:
        The question exactly as the user typed it, never rewritten.
    language:
        Detected language of ``original`` (:data:`UNKNOWN_LANGUAGE` if undetectable).
    confidence:
        Detector confidence in ``language``, in ``[0.0, 1.0]``.
    text_for_retrieval:
        The string every downstream stage should tokenize, embed and route with.
        Equal to ``original`` unless a translation actually happened.
    translated:
        True only if ``text_for_retrieval`` differs from ``original`` because a
        translator produced it.
    target_lang:
        Normalised language the results were requested in, or ``None`` if the caller
        did not ask for one.
    """

    original: str
    language: str
    confidence: float
    text_for_retrieval: str
    translated: bool
    target_lang: str | None = None

    @property
    def retrieval_language(self) -> str:
        """Language of ``text_for_retrieval`` — what stage 4's instruction needs."""
        if self.translated and self.target_lang:
            return self.target_lang
        return self.language


def normalise_lang_code(code: str | None) -> str | None:
    """Lowercase a language tag and reduce it to its primary subtag.

    ``"ja-JP"`` → ``"ja"``, ``"EN_US"`` → ``"en"``, ``""`` and ``None`` → ``None``.
    """
    if code is None:
        return None
    primary = code.strip().lower().replace("_", "-").split("-", 1)[0]
    return primary or None


def prepare_question(
    text: str,
    target_lang: str | None = None,
    translator: Translator | None = None,
) -> PreparedQuestion:
    """Run stage 1 over *text*.

    Detects the language of *text*; if *target_lang* names a different language and a
    *translator* was supplied, translates the question and marks it translated. With
    no *translator*, or when the target already matches the detected language, the
    original text is returned untouched and no translator call is made at all.

    Detection never raises. Translation does: a translator that was explicitly passed
    in and then fails is a configuration error the caller needs to see.

    A *text* that is not a string is treated as a blank question rather than raising,
    so the only exception this function can produce comes from a translator the caller
    chose to pass in.
    """
    if not isinstance(text, str):
        text = ""
    detected: LanguageResult = detect_language(text)
    target = normalise_lang_code(target_lang)

    text_for_retrieval = text
    translated = False
    if translator is not None and target is not None and text.strip() and detected.lang != target:
        source = detected.lang if detected.is_known else None
        candidate = translator.translate(text, target, source)
        if candidate.strip() and candidate != text:
            text_for_retrieval = candidate
            translated = True

    return PreparedQuestion(
        original=text,
        language=detected.lang,
        confidence=detected.confidence,
        text_for_retrieval=text_for_retrieval,
        translated=translated,
        target_lang=target,
    )
