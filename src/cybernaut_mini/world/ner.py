"""Per-event named entities with mention counts, plus the two canonicalization rules.

Events carry their entities as ``{TYPE: {surface: count}}``. The offline default
extractor is a capitalized-span heuristic classified against gazetteers (countries
via :mod:`cybernaut_mini.world.countries`, listings via
:mod:`cybernaut_mini.world.tickers`, a small city list, corporate-suffix cues);
spaCy's OntoNotes NER is the opt-in quality path. Canonicalization is two pure
functions transcribing the post's stated rules: corporate-suffix folding and
surname folding within a bucket.

Blog ref: https://nosible.com/blog/point-in-time-knowledge-graphs-over-named-entities
    — "NOSIBLE World runs NER over every event and records many entity types; this
    post focuses on five: ORG, PERSON, GPE and LOC for places, and PRODUCT", the
    ``entities{TYPE:{name:count}}`` record shape, and "surface forms are
    canonicalized, so Apple, Apple Inc. and apple collapse to one node and a lone
    surname folds into the fuller name kept that year". Local copy:
    ``docs/blog-archive/point-in-time-knowledge-graphs-over-named-entities.md``.

Assumptions:
    - The regex extractor reads runs of capitalized tokens. Classification is
      gazetteer-first (GPE via the strict country resolver plus a small city list,
      ORG via the ticker alias table, suffix tokens, or ``&``), then a narrow
      PERSON heuristic (2-3 alphabetic title-case tokens containing no
      organization word), and ORG as the fallback. This is knowingly cruder than
      OntoNotes NER — it is the deterministic offline floor; ``load_spacy_ner``
      is the quality path when the model is installed.
    - Single-token spans are noise-prone (every sentence-initial word is
      capitalized), so they only count when the gazetteer knows them or the same
      surface repeats in the text — mirroring how mention *counts*, not one-off
      hits, carry the KG post's signal.
    - Suffix folding merges on the casefolded, suffix-stripped key and keeps the
      highest-count surface as the display form (ties: shorter, then
      lexicographic), so "Apple" wins over "Apple Inc." exactly as the post's
      example collapses.
    - Surname folding is scoped to whatever bucket the caller passes (one event
      here; one year in the KG builder): a single-token PERSON folds into a fuller
      name only when exactly ONE fuller name in the bucket ends in that token —
      an ambiguous surname stays its own node rather than guessing.
    - PRODUCT is not emitted by the offline extractor (no reliable cue without a
      model); the type appears when spaCy or GLiNER (optional, not installed)
      supplies it, and every consumer treats missing types as empty.

Alternatives rejected:
    - Shipping spaCy as the default: the repo's offline-first rule; the model is
      a download and the fixture tests must pass with zero network.
    - Classifying leftover spans as MISC: downstream layers (tickers, countries,
      the KG) only read typed layers, so an honest ORG guess is more useful than
      an untyped bucket nothing consumes.
    - Frequency-weighted fuzzy merging (edit distance): the post states exact
      canonicalization rules; fuzzy merges would quietly join "Meta" and "Mesa".
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache

from cybernaut_mini.config import ConfigError
from cybernaut_mini.models import Document
from cybernaut_mini.world import countries, tickers
from cybernaut_mini.world.events import WorldEvent

__all__ = [
    "aggregate_mentions",
    "extract_mentions",
    "fold_surnames",
    "load_spacy_ner",
    "merge_corporate_suffixes",
    "tag_events",
]

Mentions = dict[str, dict[str, int]]

_SPAN_RE = re.compile(r"(?<![\w'])[A-Z][\w&.\-']*(?:[ ][A-Z][\w&.\-']*)*")

_ORG_WORDS = frozenset(
    {"awards", "centre", "center", "bank", "group", "university", "institute", "company",
     "corporation", "association", "league", "club", "ministry", "department", "agency",
     "party", "hotel", "airlines", "airways", "motors", "records", "times", "post", "journal",
     "media", "network", "systems", "technologies", "capital", "partners", "board", "council",
     "committee", "commission", "union", "federation", "school", "college", "hospital",
     "church", "studios", "entertainment", "holdings", "industries", "foundation", "fund",
     "exchange", "reserve", "senate", "congress", "parliament", "court", "office", "press"}
)

_CITY_GAZETTEER = frozenset(
    {"new york", "san francisco", "san jose", "los angeles", "washington", "london", "paris",
     "beijing", "shanghai", "moscow", "atlanta", "chicago", "boston", "seattle", "tokyo",
     "hong kong", "brussels", "geneva", "glasgow", "edinburgh", "dublin", "berlin", "frankfurt",
     "madrid", "rome", "milan", "amsterdam", "vienna", "zurich", "singapore", "sydney",
     "toronto", "dubai", "mumbai", "delhi", "seoul", "taipei", "jerusalem", "tehran", "kyiv",
     "gaza", "houston", "detroit", "miami", "philadelphia", "denver", "dallas"}
)

_COMMON_SINGLE = frozenset(
    {"the", "a", "an", "i", "it", "he", "she", "we", "they", "this", "that", "these", "those",
     "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january",
     "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december", "mr", "mrs", "ms", "dr", "but", "and", "for", "with", "after",
     "before", "when", "while", "however", "meanwhile", "here", "there", "one", "two", "three",
     "four", "five", "six", "seven", "eight", "nine", "ten", "new", "last", "next", "first"}
)


def _classify(span: str) -> str | None:
    key = span.casefold()
    if countries.resolve(span) is not None or key in _CITY_GAZETTEER:
        return "GPE"
    tokens = span.split()
    if (
        tickers.resolve_org(span) is not None
        or "&" in span
        or tokens[-1].rstrip(".").casefold() in {"inc", "corp", "ltd", "plc", "llc", "co"}
    ):
        return "ORG"
    if (
        2 <= len(tokens) <= 3
        and all(token.isalpha() and token[0].isupper() for token in tokens)
        and not any(token.casefold() in _ORG_WORDS for token in tokens)
    ):
        return "PERSON"
    return "ORG"


def extract_mentions(text: str) -> Mentions:
    """Typed mention counts for one text via the offline heuristic extractor."""
    raw = Counter(match.group().rstrip(".,'") for match in _SPAN_RE.finditer(text))
    typed: dict[str, Counter[str]] = {}
    for surface, count in raw.items():
        if not surface:
            continue
        if " " not in surface:
            key = surface.casefold()
            known = (
                countries.resolve(surface) is not None
                or tickers.resolve_org(surface) is not None
                or key in _CITY_GAZETTEER
            )
            if key in _COMMON_SINGLE or len(surface) < 2 or (count < 2 and not known):
                continue
        label = _classify(surface)
        if label is not None:
            typed.setdefault(label, Counter())[surface] += count
    return {label: dict(counter) for label, counter in typed.items()}


def aggregate_mentions(texts: Sequence[str]) -> Mentions:
    """Mention counts summed over member texts, then canonicalized per type."""
    merged: dict[str, Counter[str]] = {}
    for text in texts:
        for label, counts in extract_mentions(text).items():
            merged.setdefault(label, Counter()).update(counts)
    result: Mentions = {}
    for label, counter in merged.items():
        counts = merge_corporate_suffixes(counter) if label == "ORG" else dict(counter)
        if label == "PERSON":
            counts = fold_surnames(counts)
        result[label] = dict(sorted(counts.items()))
    return result


def merge_corporate_suffixes(counts: Mapping[str, int]) -> dict[str, int]:
    """Fold "Apple", "Apple Inc." and "apple" into one surface, summing counts.

    The merge key is the casefolded, corporate-suffix-stripped surface; the kept
    display form is the highest-count original (ties: shorter, then lexicographic).
    """
    groups: dict[str, list[tuple[str, int]]] = {}
    for surface, count in counts.items():
        groups.setdefault(tickers.normalize_org(surface), []).append((surface, int(count)))
    merged: dict[str, int] = {}
    for members in groups.values():
        display = min(members, key=lambda item: (-item[1], len(item[0]), item[0]))[0]
        merged[display] = sum(count for _, count in members)
    return merged


def fold_surnames(counts: Mapping[str, int]) -> dict[str, int]:
    """Fold a lone surname into the bucket's unique fuller name ending in it."""
    fuller: dict[str, list[str]] = {}
    for surface in counts:
        tokens = surface.split()
        if len(tokens) > 1:
            fuller.setdefault(tokens[-1].casefold(), []).append(surface)
    folded: dict[str, int] = {}
    for surface, count in counts.items():
        targets = fuller.get(surface.casefold(), []) if " " not in surface else []
        key = targets[0] if len(targets) == 1 else surface
        folded[key] = folded.get(key, 0) + int(count)
    return folded


def tag_events(
    events: Iterable[WorldEvent],
    documents: Sequence[Document],
    *,
    extractor: Callable[[str], Mentions] | None = None,
) -> list[WorldEvent]:
    """Fill ``entities`` and ``ent_gpe`` from each event's member documents.

    ``extractor`` defaults to the offline heuristic; pass the callable returned by
    :func:`load_spacy_ner` for the quality path.
    """
    by_id = {doc.id: doc for doc in documents}
    tagged: list[WorldEvent] = []
    for event in events:
        texts = [
            f"{by_id[doc_id].title}\n{by_id[doc_id].text}"
            for doc_id in event.member_doc_ids
            if doc_id in by_id
        ]
        if extractor is None:
            entities = aggregate_mentions(texts)
        else:
            merged: dict[str, Counter[str]] = {}
            for text in texts:
                for label, counts in extractor(text).items():
                    merged.setdefault(label, Counter()).update(counts)
            entities = {label: dict(sorted(c.items())) for label, c in merged.items()}
            if "ORG" in entities:
                entities["ORG"] = merge_corporate_suffixes(entities["ORG"])
            if "PERSON" in entities:
                entities["PERSON"] = fold_surnames(entities["PERSON"])
        tagged.append(
            event.model_copy(
                update={
                    "entities": entities,
                    "ent_gpe": sorted(entities.get("GPE", {})),
                }
            )
        )
    return tagged


@lru_cache(maxsize=2)
def load_spacy_ner(model: str = "en_core_web_sm") -> Callable[[str], Mentions]:
    """Opt-in spaCy extractor (OntoNotes labels incl. ORG/PERSON/GPE/LOC/PRODUCT).

    Raises :class:`ConfigError` when spaCy or the model is not installed, so the
    offline default stays the heuristic extractor.
    """
    try:
        import spacy
    except ImportError as error:  # pragma: no cover - environment-specific
        msg = "spaCy is not installed; the offline heuristic extractor is the default"
        raise ConfigError(msg) from error
    try:
        nlp = spacy.load(model, disable=["parser", "lemmatizer"])
    except OSError as error:  # pragma: no cover - environment-specific
        msg = f"spaCy model {model!r} is not installed (python -m spacy download {model})"
        raise ConfigError(msg) from error

    keep = {"ORG", "PERSON", "GPE", "LOC", "PRODUCT"}

    def extract(text: str) -> Mentions:
        typed: dict[str, Counter[str]] = {}
        for entity in nlp(text).ents:
            if entity.label_ in keep:
                typed.setdefault(entity.label_, Counter())[entity.text.strip()] += 1
        return {label: dict(counter) for label, counter in typed.items()}

    return extract
