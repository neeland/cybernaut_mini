"""The dual tagger: an exact Aho-Corasick production tagger plus a sampled discoverer.

Each collection gets two taggers. The production tagger is a per-collection
Aho-Corasick automaton over the lowercased patterns the Resolver's distillations
produced; it is rebuilt at flush time and tags 100% of the chunks entering the
collection. The discovery tagger is a slow, universal NER model that sees only a
sampled X% of chunks and only ever *suggests* — its output goes to the suggestion
accumulator, never to the tags table.

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "Each collection contains two taggers: an ultra-fast and precise tagger created
    for that collection … not neural, it uses Aho-Corasick. It can process data going
    into a collection in real-time. A slow, universal tagger trained to identify
    named entities in general … it sees X% of the data added to each collection.
    X can be scaled to match compute capacity." Local copy:
    ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions:
    - Patterns are matched case-insensitively via the same length-preserving
      :func:`~cybernaut_mini.query.s8_retrieve.intent_scan.fold` the stage-8 intent
      scan uses — the post's distilled patterns are lowercased strings, so folding
      the haystack is the matching discipline they imply.
    - Word boundaries are enforced for ASCII word characters [inferred]: the post's
      own example pattern ``"chase"`` must not fire inside ``"purchase"``. The check
      only triggers when both the neighbour and the pattern edge are ASCII word
      characters, so patterns in scriptio-continua scripts (the MIRACL shards are
      multilingual) are never rejected for lacking spaces that their script does not
      use.
    - One surface may belong to several entities; the tagger reports every mapping
      and leaves disambiguation to collection scoping — the post's argument is that
      coherent collections make ambiguous surfaces rare *within* a collection, not
      that the tagger resolves them.
    - The sampling decision is one ``rng.random() < sample_rate`` per chunk with an
      injectable ``random.Random`` [inferred]; the post says only "sees X% of the
      data". The default rate is 7.5%, the midpoint of the 5-10% band this repo's
      gap analysis assigns to X at laptop scale.
    - GLiNER is the discovery model (``urchade/gliner_small-v2.1``) and is an
      optional, download-at-construction dependency behind a lazy import. The
      offline default is :class:`CapitalizedSpanDiscovery`, a deterministic
      capitalised-span heuristic over real text — a stand-in that keeps the loop
      testable with zero network, not a claim that regexes rival a neural tagger.

Alternatives rejected:
    - Reusing :class:`~cybernaut_mini.query.s8_retrieve.intent_scan.IntentMatcher`
      directly: it weights and scores matches for reranking evidence, and its
      pattern payload is a score, not an entity id. The automaton discipline is
      shared by importing ``fold``; the payload semantics are different enough that
      forcing one class to serve both would tangle the query path with the entity
      loop.
    - spaCy as the bundled discovery model: it is a heavier default than a loop
      whose whole point is that the *production* path never runs a neural model;
      GLiNER is what the post's gap analysis names, so it is the one worth an
      optional seam.
    - Sampling by hash of the chunk id instead of an RNG: deterministic across runs,
      but it fixes forever *which* chunks are never discovered on; the post's X% is
      a rate, and a seeded RNG reproduces any run exactly while still covering
      everything in expectation.
"""

from __future__ import annotations

import importlib
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ahocorasick_rs import AhoCorasick, MatchKind

from cybernaut_mini.query.s8_retrieve.intent_scan import fold

__all__ = [
    "DEFAULT_GLINER_LABELS",
    "DEFAULT_GLINER_MODEL",
    "DEFAULT_SAMPLE_RATE",
    "CapitalizedSpanDiscovery",
    "CollectionTagger",
    "DiscoveryTagger",
    "DualTagResult",
    "DualTagger",
    "GlinerDiscoveryTagger",
    "TagMatch",
    "normalise_surface",
]

#: Midpoint of the 5-10% band the gap analysis assigns to the post's "X%".
DEFAULT_SAMPLE_RATE = 0.075

#: The discovery model named by the gap analysis, and its entity labels.
DEFAULT_GLINER_MODEL = "urchade/gliner_small-v2.1"
DEFAULT_GLINER_LABELS: tuple[str, ...] = ("company", "person", "organization")


def normalise_surface(surface: str) -> str:
    """Fold a surface exactly as haystacks and patterns are folded before matching."""
    return fold(surface)


def _is_ascii_word_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char == "_")


def _boundary_ok(haystack: str, start: int, end: int) -> bool:
    """Reject matches glued to an ASCII word character on either side."""
    if start > 0 and _is_ascii_word_char(haystack[start - 1]) and _is_ascii_word_char(
        haystack[start]
    ):
        return False
    return not (
        end < len(haystack)
        and _is_ascii_word_char(haystack[end])
        and _is_ascii_word_char(haystack[end - 1])
    )


@dataclass(frozen=True)
class TagMatch:
    """One pattern of one entity found in one chunk's folded text."""

    pattern: str
    entity_id: str
    start: int
    end: int


class CollectionTagger:
    """The fast, precise, non-neural production tagger of one collection.

    Built from ``(pattern, entity_id)`` pairs — normally
    ``EntityStore.active_patterns`` — and rebuilt from scratch at every flush, which
    is cheap because the automaton construction is linear in total pattern length.

        >>> tagger = CollectionTagger([("chase", "Q192314")])
        >>> [m.pattern for m in tagger.tag("They purchased it near Chase Field.")]
        ['chase']
    """

    def __init__(self, patterns: Iterable[tuple[str, str]]) -> None:
        mapping: dict[str, set[str]] = {}
        for pattern, entity_id in patterns:
            folded = fold(pattern)
            if not folded:
                continue
            mapping.setdefault(folded, set()).add(entity_id)
        self._entities: dict[str, tuple[str, ...]] = {
            pattern: tuple(sorted(ids)) for pattern, ids in sorted(mapping.items())
        }
        self._texts: list[str] = list(self._entities)
        self._automaton = (
            AhoCorasick(self._texts, matchkind=MatchKind.Standard) if self._texts else None
        )

    @property
    def patterns(self) -> tuple[str, ...]:
        return tuple(self._texts)

    def is_empty(self) -> bool:
        return self._automaton is None

    def tag(self, text: str) -> tuple[TagMatch, ...]:
        """Every boundary-respecting occurrence of every pattern, earliest first."""
        if self._automaton is None:
            return ()
        haystack = fold(text)
        found: list[TagMatch] = []
        for pattern_index, start, end in self._automaton.find_matches_as_indexes(
            haystack, overlapping=True
        ):
            if not _boundary_ok(haystack, start, end):
                continue
            pattern = self._texts[pattern_index]
            found.extend(
                TagMatch(pattern=pattern, entity_id=entity_id, start=start, end=end)
                for entity_id in self._entities[pattern]
            )
        found.sort(key=lambda m: (m.start, m.end, m.pattern, m.entity_id))
        return tuple(found)

    def covers(self, surface: str) -> bool:
        """True when the automaton already matches somewhere inside *surface*.

        This is the suggestion filter: a discovered surface the production tagger
        already fires on needs no resolution.
        """
        return bool(self.tag(surface))


@runtime_checkable
class DiscoveryTagger(Protocol):
    """The slow, universal tagger: proposes entity surfaces, never writes tags."""

    def discover(self, text: str) -> Sequence[str]:
        """Candidate entity surfaces found in *text*, in occurrence order."""
        ...


#: A run of 2+ capitalised/acronym words ("Chase Field", "New York Stock Exchange"),
#: or one word with an interior capital ("JPMorgan"). ASCII-only on purpose: this is
#: the offline stand-in, not the universal model. A dot stays inside a word only when
#: glued to the next letter ("U.S"), so a sentence-final period ends the span instead
#: of welding "Chase Field. This" into one surface.
_CAP_WORD = r"[A-Z](?:[A-Za-z0-9&\-]|\.(?=[A-Za-z0-9]))*"
_CAP_SPAN_RE = re.compile(
    rf"\b(?:{_CAP_WORD})(?:\s+(?:of|the|for|and|&))?"
    rf"(?:\s+{_CAP_WORD})+\b|\b[A-Z][a-z]+[A-Z][A-Za-z]*\b"
)


class CapitalizedSpanDiscovery:
    """Deterministic offline discovery: capitalised multi-word spans in real text.

    A testing stand-in for the neural universal tagger. It only proposes; precision
    is the Resolver's problem, which is exactly the division of labour the post
    describes.
    """

    def __init__(self, min_words: int = 1) -> None:
        self.min_words = min_words

    def discover(self, text: str) -> list[str]:
        seen: dict[str, None] = {}
        for match in _CAP_SPAN_RE.finditer(text):
            surface = " ".join(match.group(0).split())
            if len(surface.split()) >= self.min_words:
                seen.setdefault(surface, None)
        return list(seen)


class GlinerDiscoveryTagger:
    """GLiNER-backed discovery, behind an optional import and an explicit download.

    Constructing this class loads (and on a cold cache, downloads) the model — it is
    never a default and never constructed on the offline path. ``gliner`` is not an
    installed dependency of this repo; the import error names the extra to install.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GLINER_MODEL,
        labels: Sequence[str] = DEFAULT_GLINER_LABELS,
        threshold: float = 0.5,
    ) -> None:
        try:
            gliner_module = importlib.import_module("gliner")
        except ImportError as exc:  # pragma: no cover — exercised only without gliner
            msg = (
                "GlinerDiscoveryTagger needs the optional 'gliner' package "
                "(pip install gliner); the offline default is CapitalizedSpanDiscovery"
            )
            raise ImportError(msg) from exc
        self.labels = list(labels)
        self.threshold = threshold
        self._model: Any = gliner_module.GLiNER.from_pretrained(model_name)

    def discover(self, text: str) -> list[str]:
        predictions = self._model.predict_entities(text, self.labels, threshold=self.threshold)
        seen: dict[str, None] = {}
        for prediction in predictions:
            surface = " ".join(str(prediction.get("text", "")).split())
            if surface:
                seen.setdefault(surface, None)
        return list(seen)


@dataclass(frozen=True)
class DualTagResult:
    """What happened to one chunk: production tags always, discoveries when sampled."""

    chunk_id: str
    tags: tuple[TagMatch, ...]
    discovered: tuple[str, ...]
    sampled: bool


class DualTagger:
    """The post's two-tagger arrangement for one collection.

    The production tagger runs on every chunk; the discovery tagger runs on a
    sampled ``sample_rate`` fraction of them. With no discovery tagger the class
    degrades to the production automaton alone — the cold-start configuration.
    """

    def __init__(
        self,
        production: CollectionTagger,
        discovery: DiscoveryTagger | None = None,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        rng: random.Random | None = None,
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            msg = f"sample_rate must be within [0, 1], got {sample_rate}"
            raise ValueError(msg)
        self.production = production
        self.discovery = discovery
        self.sample_rate = sample_rate
        self._rng = rng if rng is not None else random.Random(0)

    def process(self, chunk_id: str, text: str) -> DualTagResult:
        """Tag one chunk (always) and maybe pass it to discovery (sampled)."""
        tags = self.production.tag(text)
        sampled = self.discovery is not None and self._rng.random() < self.sample_rate
        discovered: tuple[str, ...] = ()
        if sampled and self.discovery is not None:
            discovered = tuple(self.discovery.discover(text))
        return DualTagResult(chunk_id=chunk_id, tags=tags, discovered=discovered, sampled=sampled)
