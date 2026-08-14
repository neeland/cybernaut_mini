"""Closed-class (function word) detection for search-intent extraction.

Stage 3 of the retrieval pipeline rejects any candidate intent that "contains
certain parts-of-speech like, for example, pronouns, determiners, conjunctions,
adverbs, etc." Two detectors implement that rule and share one call signature so
they are interchangeable:

``ClosedClassFunctionWords``
    A curated closed-class word list. No model download, no optional dependency,
    deterministic on every machine — therefore the *default* and the detector the
    test-suite pins.
``PosFunctionWords``
    Consults the ``pos`` tag a real tagger (spaCy, the post's stack) attached to
    the token, and falls back to the word list when the tag is missing, so an
    environment without spaCy degrades instead of crashing.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 3, "Search
    Intent Prediction": intents "do not contain certain parts-of-speech like, for
    example, pronouns, determiners, conjunctions, adverbs, etc." Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - The post's "etc." is read as *the closed classes plus adverbs*: pronouns,
      determiners, conjunctions (coordinating and subordinating), adpositions,
      auxiliaries/modals, particles, interjections, and adverbs. Nouns,
      adjectives, verbs, numerals and proper nouns are kept, because the worked
      example's intents contain the verb-derived "editing" and the comparative
      adjective "safer".
    - Adverbs are only *mostly* closed-class in English. Rather than a
      ``-ly`` suffix rule — which would eat "family", "supply", "assembly" — the
      list enumerates the frequent adverbs, including the worked example's
      "actually". A rare adverb therefore survives; a false positive on a noun
      would silently delete a whole intent, and that is the worse error.
    - Detection is case-insensitive on the already-lowercased surface form. No
      language dispatch: this list is English. The post itself notes intents were
      not yet available in every language.

Alternatives rejected:
    - NLTK's stopword list: shorter to write, but tuned for *document* indexing,
      so it drops "not"/"no"/"own" while keeping many content words. Intent
      extraction needs a grammatical class filter, not a frequency filter.
    - Reusing ``cybernaut_mini.text.STOPWORDS``: same objection, and that list is
      owned by the tokenizer stage — coupling the intent filter to it would mean
      a stopword tweak silently changes which intents exist.
    - Requiring spaCy: it is the post's stack but an optional extra here, so
      making it mandatory would make the tested default path unrunnable on a
      clean checkout.
"""

from __future__ import annotations

from typing import Protocol

__all__ = [
    "DEFAULT_FUNCTION_WORDS",
    "FUNCTION_POS",
    "ClosedClassFunctionWords",
    "FunctionWordDetector",
    "PosFunctionWords",
    "default_function_word_detector",
]

_PRONOUNS = """
    i me my mine myself you your yours yourself yourselves he him his himself she
    her hers herself it its itself we us our ours ourselves they them their theirs
    themselves who whom whose which what whatever whoever whichever this that these
    those one ones oneself somebody someone something anybody anyone anything
    everybody everyone everything nobody nothing none each other another
    """

_DETERMINERS = """
    a an the this that these those my your his her its our their some any no every
    each either neither both all half such what which whose many much more most few
    fewer less least several enough certain various
    """

_CONJUNCTIONS = """
    and or but nor for yet so because although though while whereas whether if
    unless until since as than that when where whenever wherever before after once
    plus versus vs
    """

_ADPOSITIONS = """
    about above across after against along amid among around at before behind below
    beneath beside besides between beyond by concerning despite down during except
    for from in inside into like near of off on onto opposite out outside over past
    per regarding round since through throughout to toward towards under underneath
    until unto up upon versus via with within without
    """

_AUXILIARIES = """
    am is are was were be been being have has had having do does did doing will
    would shall should can could may might must ought need dare let s re ve ll d m t
    ain isn aren wasn weren hasn haven hadn doesn don didn won wouldn shan shouldn
    cannot couldn mustn mightn needn
    """

_PARTICLES_AND_INTERJECTIONS = """
    not to s oh ah eh hey hi hello please thanks thank yes no ok okay well um uh
    """

# Frequent English adverbs. Enumerated rather than suffix-matched: see the module
# docstring for why a "-ly" rule is unsafe.
_ADVERBS = """
    actually really basically essentially generally specifically particularly
    typically usually normally often always never sometimes seldom rarely frequently
    occasionally quickly slowly simply merely just only very quite rather somewhat
    still already yet again also too however therefore moreover furthermore thus
    hence instead otherwise nevertheless nonetheless perhaps maybe probably possibly
    definitely certainly surely clearly obviously apparently arguably mostly largely
    mainly primarily chiefly currently presently recently previously formerly lately
    finally eventually ultimately initially originally subsequently approximately
    roughly nearly almost exactly precisely directly indirectly effectively
    efficiently successfully significantly substantially considerably relatively
    comparatively increasingly decreasingly potentially theoretically practically
    virtually literally honestly frankly seriously truly indeed here there now then
    soon later afterwards ago away back backwards forwards forward together apart
    ahead abroad everywhere anywhere somewhere nowhere anyhow anyway somehow
    likewise meanwhile hardly barely scarcely entirely completely totally fully
    partly partially wholly equally similarly accordingly consequently respectively
    """

DEFAULT_FUNCTION_WORDS: frozenset[str] = frozenset(
    word
    for block in (
        _PRONOUNS,
        _DETERMINERS,
        _CONJUNCTIONS,
        _ADPOSITIONS,
        _AUXILIARIES,
        _PARTICLES_AND_INTERJECTIONS,
        _ADVERBS,
    )
    for word in block.split()
)

# Universal Dependencies coarse tags that a real tagger assigns to the classes the
# post excludes. NOUN/PROPN/ADJ/VERB/NUM are deliberately absent.
FUNCTION_POS: frozenset[str] = frozenset(
    {
        "PRON",
        "DET",
        "CCONJ",
        "SCONJ",
        "ADP",
        "AUX",
        "PART",
        "ADV",
        "INTJ",
        "PUNCT",
        "SPACE",
        "SYM",
        "X",
    }
)


class _HasSurface(Protocol):
    """Structural view of the token a detector needs: a surface and maybe a tag."""

    @property
    def text(self) -> str: ...

    @property
    def pos(self) -> str | None: ...


class FunctionWordDetector(Protocol):
    """Anything that can answer "is this token a function word?"."""

    def __call__(self, token: _HasSurface) -> bool: ...


class ClosedClassFunctionWords:
    """Word-list detector. Deterministic, dependency-free, the tested default.

    ``extra`` and ``allow`` let a caller tune the list without editing it:
    ``allow`` wins over ``extra`` and over the built-in list, so a domain where
    "back" or "will" is a noun can rescue it.
    """

    def __init__(
        self,
        words: frozenset[str] = DEFAULT_FUNCTION_WORDS,
        *,
        extra: frozenset[str] | None = None,
        allow: frozenset[str] | None = None,
    ) -> None:
        self._words = words | (extra or frozenset())
        self._allow = allow or frozenset()

    def __call__(self, token: _HasSurface) -> bool:
        surface = token.text.casefold()
        if surface in self._allow:
            return False
        return surface in self._words

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ClosedClassFunctionWords(words={len(self._words)})"


class PosFunctionWords:
    """Tag-driven detector, with the word list as a safety net.

    A token whose ``pos`` is unset (no tagger installed, or the tagger skipped the
    span) is judged by ``fallback`` instead of being waved through — a missing
    optional dependency must degrade the filter, never disable it.
    """

    def __init__(
        self,
        *,
        function_pos: frozenset[str] = FUNCTION_POS,
        fallback: FunctionWordDetector | None = None,
    ) -> None:
        self._function_pos = function_pos
        self._fallback: FunctionWordDetector = fallback or ClosedClassFunctionWords()

    def __call__(self, token: _HasSurface) -> bool:
        pos = token.pos
        if pos is None:
            return self._fallback(token)
        return pos.upper() in self._function_pos

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PosFunctionWords(pos={sorted(self._function_pos)})"


def default_function_word_detector() -> FunctionWordDetector:
    """The detector used when a caller passes none: the word list, never spaCy."""
    return ClosedClassFunctionWords()
