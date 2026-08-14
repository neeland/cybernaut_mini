"""Harmonic TF-IDF: the aggregator the post names, and the IDF that feeds it.

An intent is scored by the **harmonic mean** of its member tokens' TF-IDF scores.
That choice is the whole design: the harmonic mean is dominated by its *smallest*
input, so one boilerplate token drags an n-gram down no matter how distinctive its
neighbours are, and a token with a TF-IDF of zero — a term present in every
document — makes the n-gram worth exactly zero. An arithmetic mean would let
"safer" (rare) carry "the" (worthless) into the result set; a geometric mean also
zeroes on a zero but is far gentler on merely-weak tokens. Harmonic is the strict
one, which is why "sequence of proximal tokens that have a high harmonic TF-IDF
score" is a phrase filter and not just a ranking.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 3 runs "the
    tokens and their TF-IDF scores through a system that identifies search
    intents ... a sequence of proximal tokens that have a high 'harmonic TF-IDF
    score'". Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - "TF-IDF score" of a *query* token means TF over the query and IDF over the
      indexed corpus. TF is sublinear (``1 + ln(count)``), so a token repeated in
      a long question is boosted but does not dominate; with the usual count of 1
      it is exactly 1.0, making the score of a short query pure IDF.
    - IDF is the smoothed, always-positive ``ln((1+N)/(1+df)) + 1`` (scikit-learn's
      formula). Unsmoothed ``ln(N/df)`` is offered because it is the variant that
      can reach zero, which is precisely the "worthless token" case the harmonic
      mean is chosen to punish, and worth being able to demonstrate.
    - An unseen term is *rare*, not meaningless, so it defaults to the highest IDF
      in the supplied mapping rather than to zero. Defaulting to zero would delete
      every intent containing a novel entity — the opposite of what a news index
      wants.
    - A non-positive input yields 0.0 rather than an exception or an infinity:
      scoring runs inside a ranking loop where a raised error costs a query.

Alternatives rejected:
    - Arithmetic mean: the post explicitly says harmonic, and arithmetic would
      admit function-word-adjacent noise. Named here because it is what a reader
      assumes "average TF-IDF" means.
    - Summed TF-IDF: monotonically prefers longer n-grams, so the maximum-length
      window always wins and ranking carries no information.
    - Reaching into ``cybernaut_mini.indexing`` for a ready-made IDF table: that
      module belongs to the index build. This stage takes IDF as a parameter so a
      caller can pass a shard's table, a global table, or a test fixture.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "IdfLookup",
    "compute_idf",
    "harmonic_tfidf",
    "query_tfidf",
    "resolve_idf",
]

IdfLookup = Mapping[str, float] | Callable[[str], float]


def harmonic_tfidf(scores: Sequence[float]) -> float:
    """Harmonic mean of an n-gram's per-token TF-IDF scores.

    ``n / Σ(1/xᵢ)``. Returns 0.0 for an empty sequence and for any sequence
    containing a non-positive or non-finite value — a single zero-TF-IDF token
    (a term that occurs in every document) makes the whole n-gram worthless,
    which is exactly why the post picks the harmonic mean over the arithmetic
    mean: the arithmetic mean would let one distinctive neighbour hide it.

    Args:
        scores: per-token TF-IDF scores of one candidate intent.

    Returns:
        The harmonic mean, or 0.0 when the candidate is disqualified.
    """
    values = np.asarray(scores, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    if not bool(np.all(np.isfinite(values))) or bool(np.any(values <= 0.0)):
        return 0.0
    return float(values.size / np.sum(1.0 / values))


def compute_idf(
    documents: Iterable[Iterable[str]],
    *,
    smooth: bool = True,
) -> dict[str, float]:
    """Derive an IDF table from tokenised documents.

    A small helper so an index build can feed this stage without this stage
    importing the index. Pass the result straight to :func:`extract_intents`.

    Args:
        documents: one iterable of tokens per document. Repeats within a document
            are ignored — IDF is a document-frequency statistic.
        smooth: ``ln((1+N)/(1+df)) + 1`` when True (always positive, scikit-learn's
            convention); ``ln(N/df)`` when False, which is zero for a term that
            appears in every document.

    Returns:
        ``{term: idf}``, key-sorted so iteration order is reproducible.
    """
    document_frequency: dict[str, int] = {}
    n_documents = 0
    for tokens in documents:
        n_documents += 1
        for term in set(tokens):
            document_frequency[term] = document_frequency.get(term, 0) + 1
    if n_documents == 0:
        return {}
    idf: dict[str, float] = {}
    for term in sorted(document_frequency):
        df = document_frequency[term]
        if smooth:
            idf[term] = math.log((1.0 + n_documents) / (1.0 + df)) + 1.0
        else:
            idf[term] = math.log(n_documents / df)
    return idf


def resolve_idf(idf_lookup: IdfLookup, default_idf: float | None = None) -> Callable[[str], float]:
    """Normalise a mapping-or-callable IDF source into a total function.

    For a mapping, an absent term falls back to ``default_idf`` when given, else to
    the largest IDF present (an unseen term is the rarest thing there is), else to
    1.0 for an empty mapping. A callable is assumed total and is returned as-is.
    """
    if callable(idf_lookup):
        return idf_lookup
    mapping = idf_lookup
    if default_idf is not None:
        fallback = float(default_idf)
    elif mapping:
        fallback = float(max(mapping.values()))
    else:
        fallback = 1.0

    def lookup(term: str) -> float:
        return float(mapping.get(term, fallback))

    return lookup


def query_tfidf(
    tokens: Sequence[str],
    idf_lookup: IdfLookup,
    *,
    default_idf: float | None = None,
    sublinear_tf: bool = True,
) -> dict[str, float]:
    """TF-IDF of each distinct query token: ``(1 + ln(count)) * idf(term)``.

    Args:
        tokens: the query's tokens, repeats included (they are the TF).
        idf_lookup: mapping or callable from term to IDF.
        default_idf: IDF for terms missing from a mapping; see :func:`resolve_idf`.
        sublinear_tf: use ``1 + ln(count)``; ``False`` uses the raw count.

    Returns:
        ``{term: tf_idf}`` for every distinct token, key-sorted.
    """
    lookup = resolve_idf(idf_lookup, default_idf)
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    scores: dict[str, float] = {}
    for term in sorted(counts):
        count = counts[term]
        tf = 1.0 + math.log(count) if sublinear_tf else float(count)
        scores[term] = tf * lookup(term)
    return scores
