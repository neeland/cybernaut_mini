"""Query tokens carrying the two things intent extraction needs: position and glue.

Stage 3 consumes "the tokens and their TF-IDF scores", and its definition of an
intent — "a sequence of *proximal* tokens" — is positional. A bare list of strings
cannot express proximity, so this module defines :class:`QueryToken`, which keeps

``index``
    the token's ordinal in the *raw* stream, function words included. Two content
    tokens are proximal only when their indices differ by one, which is why
    "bacteria and yeast" never becomes an intent: the conjunction sits between
    them and breaks adjacency for free.
``separator_before``
    the literal characters that stood between the previous token and this one.
    That is what lets the renderer emit "gene-editing" rather than "gene editing",
    reproducing the hyphen the post's worked example prints, and what lets a comma
    or a full stop end a run of proximal tokens.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 3 ("a sequence
    of proximal tokens") reading the output of stage 2, "Multilingual
    Tokenization". Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - The post's stage-2 example emits *stems* ("medicin", "translat", "edit") but
      its stage-3 example emits *surface* forms ("medicine", "gene-editing"), so
      stage 3 cannot be reading the stemmer's output. This tokenizer therefore
      keeps surfaces and applies only English plural stripping —
      "medicines" -> "medicine", "lessons" -> "lesson" — which is exactly the one
      transformation the worked example shows.
    - Full lemmatisation is deliberately *not* applied even when spaCy is present:
      spaCy lemmatises "safer" -> "safe" and "editing" -> "edit", which would
      produce "safe gene-edit" and contradict the post's printed intents. spaCy is
      used for part-of-speech tags only.
    - Hyphens and apostrophes glue tokens together; every other punctuation run
      breaks a sequence, on the reading that "proximal" cannot span a clause
      boundary.
    - ``\\w+`` is the word pattern, so digits and non-Latin scripts survive. It is
      not a CJK segmenter — the post says intents were not yet available in
      Japanese either.

Alternatives rejected:
    - Passing ``(token, char_offset)`` pairs and re-slicing the query at render
      time: fewer fields, but the rendered text would then contain the raw
      "medicines" instead of the singularised surface, losing worked-example
      parity.
    - Treating every whitespace gap as proximal and ignoring punctuation: simpler,
      but "yeast, bacteria" would then yield an intent spanning a list boundary.
    - Reusing ``cybernaut_mini.text.TextProcessor``: it lowercases, stems and drops
      stopwords, discarding both the positions and the hyphens this stage needs.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

__all__ = [
    "QueryToken",
    "singularize",
    "spacy_available",
    "tokenize_query",
]

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Separators that keep two tokens inside one intent. Anything else (comma, colon,
# full stop, bracket, quote, dash-as-punctuation) terminates the sequence.
# The lookalikes are deliberate: a query pasted from a word processor carries
# U+2010/U+2011 hyphens and a U+2019 apostrophe, and those must glue exactly as
# ASCII "-" and "'" do, or a smart-quoted "gene-editing" loses its intent.
_GLUE_CHARS = frozenset({"-", "‐", "‑", "'", "’", "/"})  # noqa: RUF001

# Endings where a trailing "s" is not a plural marker.
_NON_PLURAL_ENDINGS = ("ss", "us", "is")
_ES_ENDINGS = ("ches", "shes", "sses", "xes")

# Singular nouns ending in "s" that the ending rules above would still mangle.
_INVARIANT_NOUNS = frozenset(
    [
        "alias",
        "atlas",
        "bias",
        "canvas",
        "chaos",
        "gas",
        "iris",
        "lens",
        "means",
        "news",
        "series",
        "species",
    ]
)


@dataclass(frozen=True, slots=True)
class QueryToken:
    """One token of the query, positioned in the raw stream.

    Attributes:
        text: the surface form used in intent strings (lowercased, singularised).
        index: ordinal in the raw token stream, function words included.
        start: character offset of the token in the NFKC-normalised query.
        end: character offset one past the token.
        separator_before: raw characters between the previous token and this one.
        pos: coarse part-of-speech tag when a tagger supplied one, else ``None``.
        raw: the untouched surface, before lowercasing and plural stripping.
    """

    text: str
    index: int
    start: int = -1
    end: int = -1
    separator_before: str = ""
    pos: str | None = None
    raw: str = ""

    @property
    def joins_previous(self) -> bool:
        """True when the gap before this token keeps it inside a sequence."""
        gap = self.separator_before
        if gap == "":
            return True
        stripped = gap.strip()
        if stripped == "":
            return True
        return all(char in _GLUE_CHARS for char in stripped)

    @property
    def glue(self) -> str:
        """The string to place before this token when rendering an intent."""
        stripped = self.separator_before.strip()
        if stripped and all(char in _GLUE_CHARS for char in stripped):
            return stripped
        return " "


def singularize(word: str) -> str:
    """Strip a regular English plural marker; leave everything else alone.

    Intentionally minimal. "medicines" -> "medicine" and "lessons" -> "lesson" are
    the transformations the post's worked example demands; a full lemmatiser would
    also rewrite "safer" and "editing" and break that example.

    Case-insensitive in its *decisions* and case-preserving in its *output*: the
    guard lists are lowercase, so testing them against the raw surface would let
    "News" through to the plural rule and return "New". :func:`tokenize_query`
    casefolds first, but this is public API and is called on raw words too.
    """
    lowered = word.lower()
    if len(lowered) != len(word):  # a case fold that changes width: leave it alone
        return word
    if len(word) <= 3 or not lowered.endswith("s"):
        return word
    if lowered in _INVARIANT_NOUNS or lowered.endswith(_NON_PLURAL_ENDINGS):
        return word
    if lowered.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if lowered.endswith(_ES_ENDINGS):
        return word[:-2]
    candidate = word[:-1]
    return candidate if len(candidate) >= 3 else word


@lru_cache(maxsize=1)
def _load_spacy() -> Any | None:
    """Load ``en_core_web_sm`` once, or return ``None`` if it is unavailable."""
    try:  # spaCy is an optional extra: absence must degrade, never crash.
        import spacy

        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except Exception:
        return None


def spacy_available() -> bool:
    """Whether the spaCy POS path can be used in this environment."""
    return _load_spacy() is not None


def _pos_by_character(text: str) -> list[str | None]:
    """Map every character offset of ``text`` to the POS tag covering it.

    Going through character offsets rather than token alignment means spaCy may
    tokenize differently from this module (it splits "gene-editing" into three
    tokens, including the hyphen) without the tags sliding out of position.
    """
    nlp = _load_spacy()
    tags: list[str | None] = [None] * len(text)
    if nlp is None:
        return tags
    try:
        doc = nlp(text)
    except Exception:  # a broken model must not take the query down
        return tags
    for token in doc:
        start = token.idx
        for offset in range(start, min(start + len(token.text), len(text))):
            tags[offset] = str(token.pos_)
    return tags


def tokenize_query(
    text: str,
    *,
    use_spacy: bool = False,
    singularise: bool = True,
) -> list[QueryToken]:
    """Split a query into positioned :class:`QueryToken` values.

    Args:
        text: the raw query.
        use_spacy: attach real POS tags when spaCy and ``en_core_web_sm`` are
            importable. Defaults to ``False`` so the deterministic, dependency-free
            path is what runs — and what the tests pin. When ``True`` but spaCy is
            missing, tags stay ``None`` and :class:`PosFunctionWords` falls back to
            its word list.
        singularise: apply the regular-plural rule to each surface.

    Returns:
        Tokens in query order. Deterministic: no clock, no RNG, no model sampling.
    """
    normalised = unicodedata.normalize("NFKC", text)
    tags = _pos_by_character(normalised) if use_spacy else []
    tokens: list[QueryToken] = []
    previous_end = 0
    for index, match in enumerate(_WORD_RE.finditer(normalised)):
        raw = match.group()
        surface = raw.casefold()
        tokens.append(
            QueryToken(
                text=singularize(surface) if singularise else surface,
                index=index,
                start=match.start(),
                end=match.end(),
                separator_before=normalised[previous_end : match.start()],
                pos=tags[match.start()] if tags else None,
                raw=raw,
            )
        )
        previous_end = match.end()
    return tokens
