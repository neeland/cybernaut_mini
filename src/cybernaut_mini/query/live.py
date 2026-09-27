"""The live query path: stages 1, 2 and 4 wired in front of ``retrieve()``/``route()``.

Three stage packages in this repo were built, tested, and then never called by the
serving path: stage 1 (language detect/translate), stage 2 (the multilingual
tokenizer) and stage 4 (instruction-prefixed embeddings). This module is the glue
that puts them on it. It owns three decisions:

* every query entry point runs :func:`prepare` first, so the text the pipeline
  tokenizes, embeds and routes is stage 1's ``text_for_retrieval``, not the raw
  input string;
* query tokenization goes through :func:`live_query_tokens`, which keeps the
  build-time :class:`~cybernaut_mini.text.TextProcessor` for ASCII text (its tokens
  are what the BM25 index was built from) and hands everything the latin-only
  ``[a-z0-9]+`` regex would delete to stage 2's
  :class:`~cybernaut_mini.query.s2_tokenize.MultilingualTokenizer`;
* when the embedding provider is an E5 *instruct* checkpoint, the query-side string
  is composed by stage 4 (:meth:`InstructionSelector.embedding_input`), producing
  the ``Instruct: …\\nQuery: …`` wire format instead of the bare question.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stages 1, 2 and 4 of
    the eight-stage query pipeline: language detection/translation, "a standardized
    interface" for multilingual tokenization, and "we generate an appropriate
    instruction for the question and submit it, along with the expansions, to
    multilingual-e5-large-instruct". Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - [inferred] the BM25 side of the index is tokenized with ``TextProcessor``, so
      for text that processor can actually see (ASCII), its output remains the query
      tokenization — swapping in stage 2's stems there would stop query tokens from
      matching index tokens and regress every English eval. Stage 2 therefore takes
      over exactly where the regex fails: any text containing non-ASCII characters
      contributes stage-2 surface tokens as well, and text the regex deletes
      entirely (CJK, Cyrillic, Arabic…) is tokenized by stage 2 alone.
    - [inferred] the stage-2 tokenizer instance is a lazily built module-level
      singleton. It is documented as not thread-safe; this repo's query path is
      single-threaded per process (the agent's 18 calls are sequential), which is
      the same assumption the cached ``ShardSelector`` in ``routing`` already makes.
    - The instruct wire format is only composed for checkpoints whose identifier
      names both ``e5`` and ``instruct``; the non-instruct mE5 checkpoints get their
      ``query:``/``passage:`` prefixes inside the embedding provider instead (see
      ``providers/embeddings.py`` and ``s4_instruct/e5.py`` for why the two schemes
      must never mix).
    - A retrieval language stage 4 has no English name for (``"und"``, or a code
      outside ``LANGUAGE_NAMES``) falls back to the selector's default language
      rather than raising: stage 1 promises never to kill the query path, and this
      module keeps that promise on its behalf.

Alternatives rejected:
    - Replacing ``TextProcessor`` outright with the stage-2 tokenizer on the query
      path: honest to the post's "one standardized interface", but Snowball stems
      ("medicin") do not match the unstemmed tokens the index stores, so English
      lexical retrieval would silently degrade. Integration must not make the
      common case worse to fix the broken one.
    - Passing a ``MultilingualTokenizer`` through every ``retrieve()``/``route()``
      call site: the explicit-dependency ideal, but it would change two public
      signatures and every caller for an object with no per-call configuration.
    - Detecting the instruct flavour inside ``retrieve()`` with a string test at
      the call site: one character of drift between that test and the provider's
      own prefix logic would send double-prefixed queries to the encoder. The test
      lives here, once, and the provider imports the same predicate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cybernaut_mini.query.s1_language import PreparedQuestion, prepare_question
from cybernaut_mini.query.s2_tokenize import MultilingualTokenizer
from cybernaut_mini.query.s4_instruct import InstructionSelector
from cybernaut_mini.query.s4_instruct.templates import LANGUAGE_NAMES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cybernaut_mini.query.s1_language.translate import Translator
    from cybernaut_mini.text import TextProcessor

__all__ = [
    "embedding_query_text",
    "is_e5_instruct",
    "live_query_tokens",
    "prepare",
]

#: Lazily built stage-2 tokenizer. Not thread-safe; see the module docstring.
_TOKENIZER: MultilingualTokenizer | None = None

#: Lazily built stage-4 selector for the instruct wire format. Stateless and frozen.
_SELECTOR: InstructionSelector | None = None


def _tokenizer() -> MultilingualTokenizer:
    global _TOKENIZER  # module-level singleton, built once
    if _TOKENIZER is None:
        _TOKENIZER = MultilingualTokenizer()
    return _TOKENIZER


def _selector() -> InstructionSelector:
    global _SELECTOR  # module-level singleton, built once
    if _SELECTOR is None:
        _SELECTOR = InstructionSelector()
    return _SELECTOR


def prepare(
    question: str,
    *,
    target_lang: str | None = None,
    translator: Translator | None = None,
) -> PreparedQuestion:
    """Run stage 1 for a live query. Thin, named alias for :func:`prepare_question`.

    With no translator this is the offline identity path: the language is detected
    (or ``und`` when fastText is absent) and the original text is searched.
    """
    return prepare_question(question, target_lang, translator)


def live_query_tokens(
    text: str,
    *,
    language: str | None,
    processor: TextProcessor,
) -> list[str]:
    """Tokenize a query for the lexical side of retrieval.

    ASCII text keeps the index-compatible ``TextProcessor`` tokens. Text carrying
    any non-ASCII character also runs through stage 2, whose tokens are appended
    (first-appearance order, deduplicated) — and when the regex deletes the text
    entirely, stage 2's tokens are the only ones returned. ``language`` is stage
    1's detected retrieval language; ``None``/``"und"`` lets stage 2 guess from
    the script.
    """
    tokens = processor.content_tokens(text)
    if not tokens:
        tokens = processor.tokenize(text)
    if tokens and text.isascii():
        return tokens

    lang = None if language in (None, "", "und") else language
    stage2 = _tokenizer().tokenize(text, lang)
    merged = list(dict.fromkeys([*tokens, *stage2.tokens]))
    return merged


def is_e5_instruct(identifier: str) -> bool:
    """True when *identifier* names an E5 *instruct* checkpoint.

    Those models take the ``Instruct: …\\nQuery: …`` wire format on the query side
    and **no** prefix on the document side; the non-instruct mE5 checkpoints use
    ``query:``/``passage:`` prefixes instead. ``providers/embeddings.py`` uses this
    same predicate, so the two halves of the scheme cannot drift apart.
    """
    lowered = identifier.lower()
    return "e5" in lowered and "instruct" in lowered


def embedding_query_text(
    provider_identifier: str,
    question: str,
    *,
    language: str | None = None,
    expansions: Sequence[str] = (),
) -> str:
    """The exact string to embed for this query under this provider.

    For an E5 instruct checkpoint, stage 4 composes the full query-side wire
    format — instruction, question and expansions. For every other provider the
    question is returned unchanged (non-instruct E5 prefixes are the provider's
    own job, applied inside ``embed_queries``).
    """
    if not is_e5_instruct(provider_identifier):
        return question
    known = language if language is not None and language.split("-")[0] in LANGUAGE_NAMES else None
    return _selector().embedding_input(question, language=known, expansions=tuple(expansions))
