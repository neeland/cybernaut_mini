"""Step 5 of blog stage 8: the Aho-Corasick full-text pass over the top results.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 8, step 5:
    "Then do a full-text search over top results for intents." The post's stage-8
    tech stack names ahocorasick-rs, which is the multi-pattern exact-phrase
    matcher used here [disclosed]. The intents themselves come from stage 3
    (:mod:`cybernaut_mini.query.s3_intents`) [disclosed]. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - Word boundaries [inferred]. Aho-Corasick matches raw substrings, so
      ``"gene editing"`` would otherwise fire inside ``"polygene editings"``. An
      intent is a phrase of whole words, so a match is rejected when it is glued to
      a word character on either side. The check is suppressed when either side is
      scriptio-continua (CJK ideographs, kana, hangul, Thai): Japanese is one of the
      post's worked examples and writes ``遺伝子編集`` with no spaces at all, so
      demanding a non-word neighbour there would reject every legitimate match.
    - Case/format folding [inferred]. Text and patterns are NFKC-normalised and
      lowercased with a length-preserving fold, because match offsets are reused to
      build snippets and ``"İ".lower()`` is two characters long — a plain
      ``str.lower()`` would silently shift every offset after it.
    - Overlapping matches are all reported [inferred]. Stage 3 emits nested n-grams
      ("gene-editing" and "safer gene-editing medicine"), and counting only the
      leftmost-longest of a nest would throw away the corroboration that makes the
      shorter intent's evidence real.
    - Evidence weight = the intent's stage-3 harmonic TF-IDF score, with repeats
      inside one document damped by a factor that saturates below 1.5 [inferred]:
      the post gives no formula, and only the *order* of the evidence ranking
      reaches the fused result, so the scale is free — but the damping has to be
      strong enough that a boilerplate phrase repeated twenty times never outranks
      a document that satisfies a second distinct intent.

Alternatives rejected:
    - A regex alternation over the intents: correct, but it is the exact linear-scan
      cost the post's choice of ahocorasick-rs exists to avoid — one automaton pass
      finds all phrases at once regardless of how many intents there are.
    - Stemming the haystack before matching: search intents are high-precision exact
      phrases in the post's telling, and stemming both sides trades that precision
      for recall the lexical BM25 leg already provides.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from ahocorasick_rs import AhoCorasick, MatchKind

from cybernaut_mini.query.s3_intents import SearchIntent

__all__ = [
    "IntentMatch",
    "IntentMatcher",
    "IntentScan",
    "fold",
    "intent_patterns",
]


# ------------------------------------------------------------------ #
# Folding                                                             #
# ------------------------------------------------------------------ #

# Scriptio-continua blocks: no inter-word spacing, so a word-boundary test on their
# neighbours is meaningless. Ranges are inclusive.
_BOUNDARYLESS_RANGES: tuple[tuple[int, int], ...] = (
    (0x0E00, 0x0E7F),  # Thai
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
)


def fold(text: str) -> str:
    """NFKC-normalise, collapse whitespace and lowercase *without changing length*.

    Length preservation matters because match offsets from the folded haystack are
    used to slice the same-length normalised text when building a snippet. Only
    characters whose lowercase form is a single character are lowered; the handful
    that expand (``İ`` -> ``i̇``) are left as-is rather than shifting every later
    offset by one.
    """
    normalized = unicodedata.normalize("NFKC", text)
    out: list[str] = []
    previous_space = False
    for char in normalized:
        if char.isspace():
            # Collapse runs of whitespace to a single space, mirroring text.normalize.
            if not previous_space:
                out.append(" ")
            previous_space = True
            continue
        previous_space = False
        lowered = char.lower()
        out.append(lowered if len(lowered) == 1 else char)
    return "".join(out).strip()


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def _is_boundaryless(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _BOUNDARYLESS_RANGES)


def _repeat_factor(count: int) -> float:
    """Saturating multiplier for repeats of one intent: 1.0 at one, sup 1.5."""
    return 1.0 + 0.5 * (1.0 - 1.0 / count)


def _boundary_ok(haystack: str, start: int, end: int) -> bool:
    """True when ``haystack[start:end]`` is not glued to a word character."""
    if start > 0:
        before, first = haystack[start - 1], haystack[start]
        if (
            _is_word_char(before)
            and _is_word_char(first)
            and not _is_boundaryless(before)
            and not _is_boundaryless(first)
        ):
            return False
    if end < len(haystack):
        after, last = haystack[end], haystack[end - 1]
        if (
            _is_word_char(after)
            and _is_word_char(last)
            and not _is_boundaryless(after)
            and not _is_boundaryless(last)
        ):
            return False
    return True


# ------------------------------------------------------------------ #
# Records                                                             #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class IntentMatch:
    """One occurrence of one intent inside one document's folded text."""

    intent: str
    start: int
    end: int
    weight: float

    def __len__(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class IntentScan:
    """The result of scanning one document for every intent at once.

    ``evidence`` is the reranking signal; ``coverage`` is the fraction of the
    query's intents this document contains, kept for tracing and explanation.
    """

    doc_id: str
    matches: tuple[IntentMatch, ...] = ()
    counts: Mapping[str, int] = field(default_factory=dict)
    evidence: float = 0.0
    coverage: float = 0.0

    @property
    def matched_intents(self) -> tuple[str, ...]:
        """Distinct intents found, in first-occurrence order."""
        seen: dict[str, None] = {}
        for match in self.matches:
            seen.setdefault(match.intent, None)
        return tuple(seen)

    def __bool__(self) -> bool:
        return bool(self.matches)


# ------------------------------------------------------------------ #
# Patterns                                                            #
# ------------------------------------------------------------------ #


def intent_patterns(
    intents: Iterable[SearchIntent | str],
) -> tuple[tuple[str, float], ...]:
    """Fold, de-duplicate and weight intents, best-weighted first.

    Two stage-3 intents can fold to the same pattern (``"Gene-Editing"`` and
    ``"gene-editing"``); the surviving copy keeps the higher score so a duplicate
    never silently downgrades the evidence. Bare strings weigh 1.0. Order is
    ``(-weight, pattern)`` so the automaton is built from the same pattern list
    whatever order the caller supplied.
    """
    weights: dict[str, float] = {}
    for intent in intents:
        if isinstance(intent, SearchIntent):
            text, weight = intent.text, float(intent.score)
        else:
            text, weight = intent, 1.0
        pattern = fold(text)
        if not pattern:
            continue
        if weight <= 0.0:
            weight = 1.0
        weights[pattern] = max(weights.get(pattern, 0.0), weight)
    return tuple(sorted(weights.items(), key=lambda kv: (-kv[1], kv[0])))


# ------------------------------------------------------------------ #
# Matcher                                                             #
# ------------------------------------------------------------------ #


class IntentMatcher:
    """A compiled Aho-Corasick automaton over the query's search intents.

    Build once per query and scan many documents with it — that reuse is the whole
    point of a multi-pattern automaton, and it is why stage 8 broadcasts the intents
    with the request instead of re-deriving them per shard.

        >>> matcher = IntentMatcher(["gene editing", "safer medicine"])
        >>> scan = matcher.scan("doc-1", "Safer medicine through gene editing.")
        >>> scan.matched_intents
        ('safer medicine', 'gene editing')
        >>> matcher.scan("doc-2", "polygene editingness").matched_intents
        ()
    """

    def __init__(self, intents: Iterable[SearchIntent | str]) -> None:
        self._patterns = intent_patterns(intents)
        self._weights = dict(self._patterns)
        self._texts = [pattern for pattern, _ in self._patterns]
        self._automaton = (
            AhoCorasick(self._texts, matchkind=MatchKind.Standard) if self._texts else None
        )

    @property
    def patterns(self) -> tuple[str, ...]:
        """The folded intent phrases this matcher looks for."""
        return tuple(self._texts)

    @property
    def weights(self) -> Mapping[str, float]:
        return dict(self._weights)

    def is_empty(self) -> bool:
        """True when there is nothing to match — the caller should skip the pass."""
        return self._automaton is None

    def find(self, text: str) -> tuple[IntentMatch, ...]:
        """All boundary-respecting occurrences in ``text``, earliest first.

        Overlapping matches are kept. Ordering is ``(start, end, intent)`` so the
        result is stable regardless of the order the automaton reports matches in.
        """
        if self._automaton is None:
            return ()
        haystack = fold(text)
        found: list[IntentMatch] = []
        for pattern_index, start, end in self._automaton.find_matches_as_indexes(
            haystack, overlapping=True
        ):
            if not _boundary_ok(haystack, start, end):
                continue
            pattern = self._texts[pattern_index]
            found.append(
                IntentMatch(intent=pattern, start=start, end=end, weight=self._weights[pattern])
            )
        found.sort(key=lambda m: (m.start, m.end, m.intent))
        return tuple(found)

    def scan(self, doc_id: str, text: str) -> IntentScan:
        """Scan one document and score its intent evidence."""
        matches = self.find(text)
        counts: dict[str, int] = {}
        for match in matches:
            counts[match.intent] = counts.get(match.intent, 0) + 1
        evidence = 0.0
        for pattern, count in counts.items():
            # Repeats corroborate, with saturating returns: 1 -> x1.00, 2 -> x1.25,
            # 10 -> x1.45, and never x1.50. Staying strictly below 2 is the point:
            # at equal weight, a document matching a second distinct intent always
            # outscores a document repeating the first one however many times.
            evidence += self._weights[pattern] * _repeat_factor(count)
        coverage = len(counts) / len(self._texts) if self._texts else 0.0
        return IntentScan(
            doc_id=doc_id,
            matches=matches,
            counts=counts,
            evidence=evidence,
            coverage=coverage,
        )

    def scan_many(self, documents: Sequence[tuple[str, str]]) -> list[IntentScan]:
        """Scan ``(doc_id, text)`` pairs in order, one automaton pass each."""
        return [self.scan(doc_id, text) for doc_id, text in documents]
