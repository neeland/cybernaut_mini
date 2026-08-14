"""The stage-2 -> stage-7 bridge: a question in, an expanded search-word set out.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 7's worked example
    starts from a question ("What lessons from bacteria and yeast actually translate into
    safer gene-editing medicines?") and lists its nine *stems* as the "Original Search
    Words", so the input to expansion is stage 2's output, not the raw string. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions: the originals are ``Tokenized.unique_stems()`` [inferred from the example]:
    the post's list is de-duplicated and stemmed, and the index's term graphs are keyed by
    the same stems, so anything else would simply miss every node. The language tag is an
    input because stage 1 owns detection.

    The post notes that "question expansions are not available in Japanese", i.e. this
    stage is language-gated in production. Nothing here gates it: an unstemmed language
    still produces stems (identical to its tokens), and a shard graph built from that
    language's documents is keyed the same way, so expansion either works or finds no
    nodes and degrades to a no-op. Encoding NOSIBLE's rollout state as a hard-coded
    language deny-list would be copying a schedule, not a technique.

Alternatives rejected:
    - Putting this function in ``expand.py``. It is the only thing in the package that
      imports stage 2, and keeping it separate is what lets ``expand_terms`` be tested
      with hand-written stems and no tokenizer at all.
    - Accepting a raw string in ``expand_query``. Two ways to pass the same argument, one
      of which silently retokenizes text the caller already tokenized.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from cybernaut_mini.query.s2_tokenize import MultilingualTokenizer, tokenize
from cybernaut_mini.query.s7_expand.expand import Expansion, ExpansionConfig, expand_query

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.query.s7_expand.filters import UbiquityFilter

__all__ = ["expand_question", "question_terms"]


def question_terms(
    question: str,
    *,
    language: str | None = "en",
    tokenizer: MultilingualTokenizer | None = None,
) -> tuple[str, ...]:
    """The question's original search words: stage 2's unique stems, in question order.

    >>> question_terms("What lessons from bacteria and yeast actually translate into "
    ...                "safer gene-editing medicines?")
    ('lesson', 'bacteria', 'yeast', 'actual', 'translat', 'safer', 'gene', 'edit', 'medicin')
    """
    tokenized = (
        tokenizer.tokenize(question, language)
        if tokenizer is not None
        else tokenize(question, language)
    )
    return tokenized.unique_stems()


def expand_question(
    index: LoadedIndex,
    question: str,
    *,
    shard_ids: Sequence[int],
    language: str | None = "en",
    shard_weights: Mapping[int, float] | None = None,
    config: ExpansionConfig | None = None,
    ubiquity: UbiquityFilter | None = None,
    tokenizer: MultilingualTokenizer | None = None,
) -> Expansion:
    """Tokenize ``question`` with stage 2, then expand it against the routed shards.

    A convenience over :func:`~cybernaut_mini.query.s7_expand.expand.expand_query` for
    callers holding a question rather than a term list; every keyword argument is passed
    straight through. Pass ``tokenizer`` to reuse one instance across queries — the
    default builds nothing per call but the shared process-wide tokenizer is not
    thread-safe, so a threaded caller should own one per thread.
    """
    terms = question_terms(question, language=language, tokenizer=tokenizer)
    return expand_query(
        index,
        terms,
        shard_ids=shard_ids,
        shard_weights=shard_weights,
        config=config,
        ubiquity=ubiquity,
    )
