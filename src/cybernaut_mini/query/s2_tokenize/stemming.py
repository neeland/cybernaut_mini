"""Snowball stemming, cached per language, identity everywhere else.

The post's worked example is the specification for this module. Given "What lessons from
bacteria and yeast actually translate into safer gene-editing medicines?" it lists nine
tokens, and three of them — ``translat``, ``medicin``, ``edit`` — are not words. They are
Snowball English stems, which is how we know the post's ``pyStemmer`` entry sits *after*
segmentation and stopword removal rather than beside them.

PyStemmer's ``Stemmer`` objects are C wrappers holding a mutable stemming context. They
are **not thread-safe**: two threads calling ``stemWords`` on one instance can interleave
inside the C struct. They are also not picklable. Both facts are handled by keeping the
cache private to the owning :class:`~.tokenizer.MultilingualTokenizer` instance, which
drops it on pickling and gives each thread its own tokenizer.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 2, "Multilingual
    Tokenization", naming ``pyStemmer`` — "support for 24 languages" — alongside
    "text inflection" and "word stemming". Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - Stemming, not lemmatisation. The post's expected tokens are Snowball output
      (``translat`` is not a lemma), so an accurate lemmatiser would *fail* the worked
      example. The lexical index only needs query and document to collide on the same
      string, and a stem does that without a model.
    - Languages Snowball does not cover get the identity function rather than a
      transliteration or an approximation. A wrong stem merges unrelated words and is
      worse than no stem at all.
    - The installed PyStemmer exposes more algorithms (36) than the 24 the post cites.
      The map below is keyed on what ``Stemmer.algorithms()`` actually reports at import
      time, so a newer or older wheel neither crashes nor silently loses a language.

Alternatives rejected:
    - NLTK's ``SnowballStemmer``: same algorithms, pure Python, roughly an order of
      magnitude slower on the token volumes a search index produces, and NLTK is a much
      heavier dependency for the same output.
    - spaCy lemmatisation: better linguistics, but it needs a per-language model
      download, cannot run offline by default, and would not reproduce the post's tokens.
    - A module-level ``lru_cache`` of stemmers: convenient, but it would hand the same
      non-thread-safe C object to every caller in the process. Per-instance caching keeps
      the unsafe object's lifetime tied to something the caller owns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# ISO 639-1 -> Snowball algorithm name. Only languages Snowball actually implements.
_SNOWBALL_BY_CODE: dict[str, str] = {
    "ar": "arabic",
    "hy": "armenian",
    "eu": "basque",
    "ca": "catalan",
    "cs": "czech",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "eo": "esperanto",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "el": "greek",
    "hi": "hindi",
    "hu": "hungarian",
    "id": "indonesian",
    "ga": "irish",
    "it": "italian",
    "lt": "lithuanian",
    "ne": "nepali",
    "no": "norwegian",
    "nb": "norwegian",
    "nn": "norwegian",
    "fa": "persian",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sr": "serbian",
    "st": "sesotho",
    "es": "spanish",
    "sv": "swedish",
    "ta": "tamil",
    "tr": "turkish",
    "yi": "yiddish",
}


def _import_stemmer() -> Any:
    """Import PyStemmer, or return ``None`` if this build does not have it.

    One import site, so the module has one place that knows PyStemmer is untyped and
    one place that decides what "PyStemmer is missing" means.
    """
    try:
        import Stemmer  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - PyStemmer is a hard dependency
        return None
    return Stemmer


def _available_algorithms() -> frozenset[str]:
    module = _import_stemmer()
    if module is None:  # pragma: no cover - PyStemmer is a hard dependency
        return frozenset()
    try:
        return frozenset(str(name) for name in module.algorithms())
    except Exception:  # pragma: no cover - defensive against an API change
        return frozenset()


_AVAILABLE: frozenset[str] = _available_algorithms()

SNOWBALL_LANGUAGES: frozenset[str] = frozenset(
    code for code, algo in _SNOWBALL_BY_CODE.items() if not _AVAILABLE or algo in _AVAILABLE
)
"""ISO 639-1 codes this build can actually stem."""


def snowball_algorithm(language: str) -> str | None:
    """Return the Snowball algorithm name for ``language``, or ``None`` if unsupported."""
    algo = _SNOWBALL_BY_CODE.get(language)
    if algo is None:
        return None
    if _AVAILABLE and algo not in _AVAILABLE:
        return None
    return algo


def build_stemmer(language: str) -> Any | None:
    """Construct a PyStemmer instance for ``language``, or ``None`` if unsupported.

    The returned object is stateful C code and must not be shared across threads.
    """
    algo = snowball_algorithm(language)
    if algo is None:
        return None
    module = _import_stemmer()
    if module is None:  # pragma: no cover - PyStemmer is a hard dependency
        return None
    try:
        return module.Stemmer(algo)
    except Exception:  # pragma: no cover - a missing algorithm degrades to identity
        return None


def stem_words(tokens: Sequence[str], stemmer: Any | None) -> list[str]:
    """Stem ``tokens`` with ``stemmer``; identity when there is no stemmer.

    Never changes the length of the sequence, so ``tokens[i]`` and its stem stay aligned.
    """
    if stemmer is None or not tokens:
        return list(tokens)
    try:
        stemmed = stemmer.stemWords(list(tokens))
    except Exception:  # pragma: no cover - a stemmer failure must not lose the query
        return list(tokens)
    if len(stemmed) != len(tokens):  # pragma: no cover - contract violation upstream
        return list(tokens)
    return [str(s) for s in stemmed]
