"""Stage 3: turn scored tokens into ranked search intents.

The algorithm is a direct reading of the post's one-sentence definition. An intent
is (a) *a sequence of proximal tokens* — a contiguous run in the query, so
"bacteria and yeast" is not one, the conjunction sits between them; (b) with *a
high harmonic TF-IDF score* — the harmonic mean punishes any n-gram containing a
common token; (c) that *does not contain* pronouns, determiners, conjunctions,
adverbs and friends — so function words both disqualify candidates and cut the
runs that candidates are drawn from.

Worked example (the post's, reproduced exactly by
:func:`predict_intents` — see ``tests/test_s3_intents.py``)::

    "What lessons from bacteria and yeast actually translate into
     safer gene-editing medicines?"

    what/PRON lessons from/ADP bacteria and/CCONJ yeast actually/ADV translate
    into/ADP [safer gene editing medicine]

    -> only one run of proximal content tokens survives, length 4, and its
       n-grams for n=2..4 are precisely the post's six intents:
       "safer gene", "gene-editing", "editing medicine",
       "safer gene-editing", "gene-editing medicine", "safer gene-editing medicine"

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 3, "Search
    Intent Prediction": "we run the tokens and their TF-IDF scores through a
    system that identifies search intents. We define a search intent as a sequence
    of proximal tokens that have a high 'harmonic TF-IDF score' and do not contain
    certain parts-of-speech like, for example, pronouns, determiners,
    conjunctions, adverbs, etc." Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - "Proximal" means *strictly contiguous in the raw token stream*, not "within a
      window of k". A window would let "bacteria yeast" through, and the post's own
      example contains no such gapped intent. ``max_gap`` exposes the loosening for
      callers who want it; the default of 0 is the tested behaviour.
    - n ranges over 2..4. The example's shortest intent is a bigram and its longest
      is the full four-token tail; a unigram intent would duplicate the stage-2
      token list, which is transmitted separately anyway.
    - "High" is a *relative* bar: candidates are ranked and truncated at ``top_k``
      rather than cut at an absolute TF-IDF threshold, because the scale of IDF
      depends on the corpus that produced it. ``min_score`` exists and defaults to
      0.0, which drops only the genuinely worthless (any n-gram containing a
      zero-IDF token).
    - Ranking ties are broken by the intent string, then by position, so equal
      scores — the common case when every token is unseen — still give a stable,
      reproducible order.
    - Longer intents naturally outrank their own sub-phrases only when their extra
      tokens are rarer. No length prior is applied; the post mentions none, and the
      downstream consumers (bloom-filter reranking in stage 6, Aho-Corasick
      matching in stage 8) want both the specific and the general form.

Alternatives rejected:
    - Noun-chunk extraction with spaCy's parser: linguistically tidier and would
      also yield "safer gene-editing medicines", but it emits *one* phrase per
      chunk, not the nested family of six the post prints, and it makes an optional
      dependency load-bearing.
    - RAKE / YAKE keyphrase extraction: they build phrases from stopword-delimited
      runs, which is the same shape, but they impose their own scoring and would
      discard the harmonic TF-IDF rule that the post specifically names.
    - Filtering candidates by an absolute harmonic-score threshold: needs a
      corpus-specific constant nobody can tune from the outside.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .function_words import (
    ClosedClassFunctionWords,
    FunctionWordDetector,
    PosFunctionWords,
    default_function_word_detector,
)
from .scoring import IdfLookup, harmonic_tfidf, query_tfidf
from .tokens import QueryToken, tokenize_query

__all__ = [
    "DEFAULT_MAX_N",
    "DEFAULT_MIN_N",
    "DEFAULT_TOP_K",
    "SearchIntent",
    "extract_intents",
    "predict_intents",
]

DEFAULT_MIN_N = 2
DEFAULT_MAX_N = 4
DEFAULT_TOP_K = 10


@dataclass(frozen=True, slots=True)
class SearchIntent:
    """One predicted search intent.

    Attributes:
        text: the rendered phrase, e.g. ``"safer gene-editing medicine"``.
        tokens: its member token surfaces, in order.
        score: the harmonic mean of the members' TF-IDF scores.
        span: half-open ``(start, end)`` span of raw-token indices it covers, so a
            caller can highlight it in the original query.
    """

    text: str
    tokens: tuple[str, ...]
    score: float
    span: tuple[int, int]

    def __len__(self) -> int:
        return len(self.tokens)


def _content_runs(
    tokens: Sequence[QueryToken],
    is_function_word: FunctionWordDetector,
    max_gap: int,
) -> list[list[QueryToken]]:
    """Split the stream into maximal runs of proximal, non-function tokens.

    Function words are dropped rather than treated as separators in their own
    right: they still occupy an index, so at the default ``max_gap=0`` a skipped
    "and" already makes its neighbours non-contiguous and the run breaks. A run
    also breaks on punctuation that is not intent glue (a comma, a full stop) —
    including punctuation that sits in front of a *dropped* function word, which
    is why ``pending_break`` is carried across the skip: at ``max_gap >= 1``
    "bacteria, and yeast" must not become one run just because the comma happened
    to attach to the conjunction.
    """
    runs: list[list[QueryToken]] = []
    current: list[QueryToken] = []
    pending_break = False
    for token in tokens:
        if is_function_word(token):
            pending_break = pending_break or not token.joins_previous
            continue
        if current:
            gap = token.index - current[-1].index - 1
            if gap > max_gap or pending_break or not token.joins_previous:
                runs.append(current)
                current = []
        current.append(token)
        pending_break = False
    if current:
        runs.append(current)
    return runs


def _render(window: Sequence[QueryToken]) -> str:
    """Join a window's surfaces, preserving the query's own hyphens."""
    parts = [window[0].text]
    for token in window[1:]:
        parts.append(token.glue)
        parts.append(token.text)
    return "".join(parts)


def extract_intents(
    tokens_with_positions: Sequence[QueryToken],
    idf_lookup: IdfLookup,
    *,
    min_n: int = DEFAULT_MIN_N,
    max_n: int = DEFAULT_MAX_N,
    top_k: int | None = DEFAULT_TOP_K,
    min_score: float = 0.0,
    max_gap: int = 0,
    function_word_detector: FunctionWordDetector | None = None,
    default_idf: float | None = None,
    sublinear_tf: bool = True,
) -> list[SearchIntent]:
    """Predict search intents from positioned query tokens and an IDF table.

    Args:
        tokens_with_positions: the query's tokens *with their positions* — see
            :func:`~cybernaut_mini.query.s3_intents.tokens.tokenize_query`.
            Positions are what make "proximal" computable.
        idf_lookup: ``{term: idf}`` mapping, or a callable term -> idf. Supplied by
            the caller (a shard's table, a global table) so this stage never
            imports the index build.
        min_n: shortest intent, in tokens.
        max_n: longest intent, in tokens.
        top_k: keep at most this many; ``None`` keeps all.
        min_score: drop candidates scoring at or below this. The default 0.0 drops
            exactly the n-grams containing a zero-TF-IDF token.
        max_gap: tokens allowed between two members. 0 = strictly contiguous.
        function_word_detector: predicate marking pronouns/determiners/etc.
            Defaults to the closed-class word list (never spaCy, which is optional
            here).
        default_idf: IDF for terms missing from ``idf_lookup``; defaults to the
            table's maximum, i.e. "unseen means rarest".
        sublinear_tf: ``1 + ln(count)`` term frequency over the query.

    Returns:
        Intents sorted by descending harmonic TF-IDF, then by text, then by
        position. Deduplicated on ``text``, keeping the best-scoring occurrence.
        Deterministic for a given input.

    Raises:
        ValueError: if ``min_n`` < 1, ``max_n`` < ``min_n``, ``max_gap`` < 0, or
            ``top_k`` < 0. A negative ``top_k`` is rejected rather than passed to
            the slice, where it would silently drop the *worst* ``|top_k|``
            intents and look like a working truncation.
    """
    if min_n < 1:
        msg = f"min_n must be >= 1, got {min_n}"
        raise ValueError(msg)
    if max_n < min_n:
        msg = f"max_n ({max_n}) must be >= min_n ({min_n})"
        raise ValueError(msg)
    if max_gap < 0:
        msg = f"max_gap must be >= 0, got {max_gap}"
        raise ValueError(msg)
    if top_k is not None and top_k < 0:
        msg = f"top_k must be >= 0 or None, got {top_k}"
        raise ValueError(msg)

    detector = function_word_detector or default_function_word_detector()
    runs = _content_runs(tokens_with_positions, detector, max_gap)
    if not runs:
        return []

    scores = query_tfidf(
        [token.text for run in runs for token in run],
        idf_lookup,
        default_idf=default_idf,
        sublinear_tf=sublinear_tf,
    )

    best: dict[str, SearchIntent] = {}
    for run in runs:
        for size in range(min_n, max_n + 1):
            for start in range(0, len(run) - size + 1):
                window = run[start : start + size]
                score = harmonic_tfidf([scores[token.text] for token in window])
                if score <= min_score:
                    continue
                intent = SearchIntent(
                    text=_render(window),
                    tokens=tuple(token.text for token in window),
                    score=score,
                    span=(window[0].index, window[-1].index + 1),
                )
                incumbent = best.get(intent.text)
                if incumbent is None or (intent.score, -intent.span[0]) > (
                    incumbent.score,
                    -incumbent.span[0],
                ):
                    best[intent.text] = intent

    ranked = sorted(best.values(), key=lambda i: (-i.score, i.text, i.span))
    return ranked if top_k is None else ranked[:top_k]


def predict_intents(
    text: str,
    idf_lookup: IdfLookup,
    *,
    use_spacy: bool = False,
    min_n: int = DEFAULT_MIN_N,
    max_n: int = DEFAULT_MAX_N,
    top_k: int | None = DEFAULT_TOP_K,
    min_score: float = 0.0,
    max_gap: int = 0,
    function_word_detector: FunctionWordDetector | None = None,
    default_idf: float | None = None,
    sublinear_tf: bool = True,
) -> list[SearchIntent]:
    """Tokenise a raw query and extract its intents in one call.

    ``use_spacy=True`` asks for real POS tags (the post's stack) and switches the
    default detector to the tag-driven one; when spaCy or ``en_core_web_sm`` is
    absent the tags are ``None`` and that detector falls back to the word list, so
    the call still returns intents rather than raising.
    """
    tokens = tokenize_query(text, use_spacy=use_spacy)
    detector = function_word_detector
    if detector is None and use_spacy:
        detector = PosFunctionWords(fallback=ClosedClassFunctionWords())
    return extract_intents(
        tokens,
        idf_lookup,
        min_n=min_n,
        max_n=max_n,
        top_k=top_k,
        min_score=min_score,
        max_gap=max_gap,
        function_word_detector=detector,
        default_idf=default_idf,
        sublinear_tf=sublinear_tf,
    )
