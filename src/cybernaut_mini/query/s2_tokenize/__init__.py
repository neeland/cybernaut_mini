"""Stage 2 of the Hybrid-3 query pipeline: multilingual tokenization.

Public surface::

    from cybernaut_mini.query.s2_tokenize import MultilingualTokenizer, tokenize

    tokenize("What lessons from bacteria and yeast actually translate into "
             "safer gene-editing medicines?").stems
    # ('lesson', 'bacteria', 'yeast', 'actual', 'translat', 'safer', 'gene',
    #  'edit', 'medicin')

:func:`tokenize` uses one shared default tokenizer, which is convenient and therefore
worth a warning: the objects it caches are not thread-safe, so concurrent code should
construct its own :class:`MultilingualTokenizer` per thread.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 2 of the eight-stage
    pipeline, "Multilingual Tokenization", whose worked example this package reproduces
    exactly for English. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - The package is importable without the optional ``cjk``/``nlp`` extras; every
      optional path degrades instead of raising.
    - Stage 1 supplies the language tag. Nothing here calls fastText.

Alternatives rejected:
    - Re-exporting the whole internal surface (segmenters, stemmer builders): the post's
      point is that callers should see one interface, not twelve libraries. The helper
      modules stay importable for tests but are not part of the advertised API.
    - Making :func:`tokenize` build a fresh tokenizer per call: safe under threads, but
      it would rebuild a pySBD profile and a Snowball stemmer on every query.
"""

from __future__ import annotations

from .languages import UNDETERMINED, guess_script_language, normalize_language
from .stemming import SNOWBALL_LANGUAGES
from .stopwords import STOPWORDS
from .tokenizer import MultilingualTokenizer, Tokenized

__all__ = [
    "SNOWBALL_LANGUAGES",
    "STOPWORDS",
    "UNDETERMINED",
    "MultilingualTokenizer",
    "Tokenized",
    "default_tokenizer",
    "guess_script_language",
    "normalize_language",
    "tokenize",
]

_DEFAULT: MultilingualTokenizer | None = None


def default_tokenizer() -> MultilingualTokenizer:
    """The process-wide tokenizer used by :func:`tokenize`. Not thread-safe."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MultilingualTokenizer()
    return _DEFAULT


def tokenize(text: str, language: str | None = "en") -> Tokenized:
    """Tokenize ``text`` with the shared default tokenizer."""
    return default_tokenizer().tokenize(text, language)
