"""Stage 3 of the query pipeline: search intent prediction.

Input: the query's tokens *with positions* plus an IDF table. Output: a ranked
list of :class:`SearchIntent` — contiguous n-grams of content words with a high
harmonic mean of TF-IDF, which stage 6 tests against each shard's phrase bloom
filter and stage 8 matches with Aho-Corasick inside the retrieved documents.

    >>> from cybernaut_mini.query.s3_intents import compute_idf, predict_intents
    >>> idf = compute_idf([["gene", "editing"], ["safer", "medicine"], ["yeast"]])
    >>> q = ("What lessons from bacteria and yeast actually translate into "
    ...      "safer gene-editing medicines?")
    >>> sorted(i.text for i in predict_intents(q, idf))
    ['editing medicine', 'gene-editing', 'gene-editing medicine', 'safer gene', \
'safer gene-editing', 'safer gene-editing medicine']

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 3 of the
    eight-stage retrieval pipeline: "we run the tokens and their TF-IDF scores
    through a system that identifies search intents. We define a search intent as
    a sequence of proximal tokens that have a high 'harmonic TF-IDF score' and do
    not contain certain parts-of-speech like, for example, pronouns, determiners,
    conjunctions, adverbs, etc." Stack: NumPy, spaCy. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - The post prints the intents but not the machinery, so three parameters had
      to be chosen: n ranges over 2..4, "proximal" means strictly contiguous, and
      "high" means top-k rather than an absolute threshold. Each is a keyword
      argument, and each default is the one that reproduces the worked example.
    - spaCy is the post's tagger but an optional extra here, so the curated
      closed-class word list is the default and the tested path. The tag-driven
      detector is available and is used automatically by
      :func:`predict_intents(..., use_spacy=True)`.
    - The IDF table is a parameter, never an import. This stage must work for a
      global table, a per-shard table, or a fixture, and must not depend on the
      index build's internals.

Alternatives rejected:
    - Exposing one god-function: the four concerns (scoring, function words,
      tokens, assembly) are separately testable and separately swappable, so they
      are separate modules re-exported here.
    - Returning bare strings: stage 6 and stage 8 need the score to weight a
      bloom-filter miss and the span to build snippets, so an intent is a record.
"""

from __future__ import annotations

from .function_words import (
    DEFAULT_FUNCTION_WORDS,
    FUNCTION_POS,
    ClosedClassFunctionWords,
    FunctionWordDetector,
    PosFunctionWords,
    default_function_word_detector,
)
from .intents import (
    DEFAULT_MAX_N,
    DEFAULT_MIN_N,
    DEFAULT_TOP_K,
    SearchIntent,
    extract_intents,
    predict_intents,
)
from .scoring import IdfLookup, compute_idf, harmonic_tfidf, query_tfidf, resolve_idf
from .tokens import QueryToken, singularize, spacy_available, tokenize_query

__all__ = [
    "DEFAULT_FUNCTION_WORDS",
    "DEFAULT_MAX_N",
    "DEFAULT_MIN_N",
    "DEFAULT_TOP_K",
    "FUNCTION_POS",
    "ClosedClassFunctionWords",
    "FunctionWordDetector",
    "IdfLookup",
    "PosFunctionWords",
    "QueryToken",
    "SearchIntent",
    "compute_idf",
    "default_function_word_detector",
    "extract_intents",
    "harmonic_tfidf",
    "predict_intents",
    "query_tfidf",
    "resolve_idf",
    "singularize",
    "spacy_available",
    "tokenize_query",
]
