"""Blog stage 8's final step: "a highly relevant snippet ... decided by the user".

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 8: "Once the
    search completes a highly relevant snippet is constructed for each search
    result. The maximum length of the snippet is decided by the user." [disclosed]
    The worked example's snippet is a contiguous window of the article body, with
    the passage that answers the question inside it, not a concatenation of
    fragments. Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - "Highly relevant" is anchored on the strongest *intent* match, falling back to
      a query token and then to the lead paragraph [inferred]. The post does not say
      what centres the window; anchoring on the intent is the reading that makes
      stage 8's own step 5 pay for itself twice — it reranks, and it explains.
    - Ties between equally weighted anchors break on the earliest offset, so the
      snippet is deterministic for a given (text, intents, tokens) triple.
    - The window is trimmed to whitespace boundaries so it never begins or ends
      mid-word, and the trimming only ever shortens it, so ``max_chars`` is a hard
      ceiling rather than a target.

Alternatives rejected:
    - Reusing :func:`cybernaut_mini.retrieval.make_snippet`: it hard-codes 500
      characters and knows nothing about intents, and the post explicitly makes the
      maximum a user parameter. This module keeps the same word-boundary discipline
      and adds the two missing degrees of freedom.
    - Sentence-segmented snippets (pysbd is in the dependency set): better prose,
      but the segmentation cost is per result and the post's example snippet clearly
      starts mid-sentence.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from cybernaut_mini.query.s8_retrieve.intent_scan import IntentMatch, fold

__all__ = [
    "DEFAULT_SNIPPET_MAX_CHARS",
    "SNIPPET_HARD_MAX_CHARS",
    "make_snippet",
]

#: Default ceiling. Matches :func:`cybernaut_mini.retrieval.make_snippet` so the two
#: snippet builders agree when the caller expresses no preference.
DEFAULT_SNIPPET_MAX_CHARS = 500

#: :class:`~cybernaut_mini.models.SearchHit` validates ``snippet`` at 500 characters,
#: so a request above this cannot be stored in a hit.
SNIPPET_HARD_MAX_CHARS = 500

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """NFKC + whitespace collapse, preserving case *and* length parity with ``fold``."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def _trim_to_word_boundary(text: str, start: int, end: int) -> str:
    """Trim ``text[start:end]`` so it neither starts nor ends mid-word."""
    snippet = text[start:end]
    if not snippet:
        return snippet
    if start > 0 and not snippet[0].isspace():
        space = snippet.find(" ")
        if space == -1:
            return ""
        snippet = snippet[space:].lstrip()
    if end < len(text) and snippet and not snippet[-1].isspace():
        space = snippet.rfind(" ")
        if space == -1:
            return ""
        snippet = snippet[:space].rstrip()
    return snippet.strip()


def _token_anchor(haystack: str, tokens: Sequence[str]) -> tuple[int, int] | None:
    """Earliest whole-word occurrence of any token in the folded ``haystack``."""
    best: tuple[int, int] | None = None
    for token in tokens:
        folded = fold(token)
        if not folded:
            continue
        pattern = re.compile(r"(?<!\w)" + re.escape(folded) + r"(?!\w)")
        match = pattern.search(haystack)
        if match is not None and (best is None or match.start() < best[0]):
            best = (match.start(), match.end())
    return best


def make_snippet(
    text: str,
    *,
    intent_matches: Sequence[IntentMatch] = (),
    query_tokens: Sequence[str] = (),
    max_chars: int = DEFAULT_SNIPPET_MAX_CHARS,
) -> str:
    """Return a snippet of at most ``max_chars`` characters, centred on the best anchor.

    Anchor precedence: the highest-weighted intent match (earliest offset wins ties),
    then the earliest whole-word query token, then the start of the text. The window
    is centred on the anchor, clamped to the text, and trimmed to word boundaries.

    Raises :class:`ValueError` for ``max_chars < 1``.
    """
    if max_chars < 1:
        msg = f"max_chars must be >= 1, got {max_chars}"
        raise ValueError(msg)

    normalized = _normalize(text)
    if not normalized:
        return ""
    haystack = fold(text)

    anchor: tuple[int, int] | None = None
    if intent_matches:
        best = min(intent_matches, key=lambda m: (-m.weight, m.start, m.end, m.intent))
        anchor = (best.start, best.end)
    if anchor is None:
        anchor = _token_anchor(haystack, query_tokens)

    if anchor is None or len(normalized) <= max_chars:
        return _trim_to_word_boundary(normalized, 0, min(max_chars, len(normalized)))

    # ``fold`` preserves length, so folded offsets index the normalised text too —
    # unless the caller handed us text whose NFKC form differs in length from the
    # folded form, which the guard below turns into a lead-of-text snippet rather
    # than a mis-centred window.
    if len(haystack) != len(normalized):
        return _trim_to_word_boundary(normalized, 0, max_chars)

    centre = (anchor[0] + anchor[1]) // 2
    half = max_chars // 2
    start = max(0, centre - half)
    end = min(len(normalized), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)
    return _trim_to_word_boundary(normalized, start, end)
