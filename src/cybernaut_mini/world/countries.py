"""Country attribution from GPE entities with strict whole-string alias matching.

An event's country attribution is its main country plus every country its named
geopolitical entities resolve to. Resolution is deliberately strict: whole-string,
case-insensitive lookups against a table of country names, official aliases,
demonyms of four letters or more, and native-script names — a partial match never
counts, and a small drop-set removes strings that are genuinely ambiguous.

Blog ref: https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world
    — "We match each entity whole-string and case-insensitive against a table of
    country names, official aliases, nationalities of four letters or more, and
    native-language names. A partial match never counts, so ``Indiana`` never
    resolves to India, and genuinely ambiguous strings such as a bare ``Georgia``
    (the US state) are dropped." and ``attribution(e) = { the event's main
    country } + { countries resolved from its named entities }``. Local copy:
    ``docs/blog-archive/rebuilding-the-geopolitical-risk-index-from-nosible-world.md``.

Assumptions:
    - The alias table is built from :mod:`pycountry` (name, official name, common
      name) plus committed demonym and endonym tables for the countries that
      dominate geopolitical news. pycountry's formal names ("Iran, Islamic
      Republic of") never appear verbatim in news text, so the common-name and
      demonym layers do the real work.
    - Alpha-3 codes match only when the surface is fully uppercase ("USA"): a
      casefolded whole-string match would resolve the English words
      "in"/"It"/"Us" to India/Italy/the United States the moment a
      sentence-initial token reaches the resolver. Alpha-2 codes are *not*
      matched except for the allow-list {"US", "UK"}: two-letter codes collide
      wholesale with US-state abbreviations and everyday abbreviations ("TV" is
      Tuvalu, "IN" is India, "ID" is Indonesia), which is the "Indiana" problem
      in code form — the same ambiguity rule that drops "Georgia" drops them.
    - The demonym floor of four letters is enforced at table-build time, and each
      demonym also matches its plural ("Americans"), since NER surfaces keep the
      plural form.
    - The drop-set is casefolded and checked first, so a dropped alias can never
      resolve through another layer. "Georgia" is the post's own example; "Jordan",
      "Chad" and "Guinea" are dropped for the same person/place ambiguity.
    - ``attribution`` is capped (default 15) by descending GPE mention count so a
      wire-roundup event naming forty countries cannot blow up the pair index
      quadratically. The cap keeps the most-covered countries, not the first seen.

Alternatives rejected:
    - Fuzzy or substring matching: strictness *is* the published method; every
      false attribution pollutes a per-country index forever.
    - Resolving cities/regions to their country (San Jose -> US): needs a
      world-scale gazetteer; the posts resolve country-level strings only and the
      main-country field covers the rest.
    - Storing alpha-3 or full names as the attribution key: alpha-2 is the
      shortest stable key, and :func:`country_name` recovers a display name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache

import pycountry

from cybernaut_mini.world.events import WorldEvent

__all__ = [
    "DROPPED_ALIASES",
    "attribution",
    "attributions",
    "country_name",
    "main_country",
    "resolve",
    "tag_events",
]

#: Casefolded strings that must never resolve: shared with common person names or
#: US states ("Georgia" is the post's own example of a genuinely ambiguous string).
DROPPED_ALIASES = frozenset({"georgia", "jordan", "chad", "guinea"})

_MIN_DEMONYM_LENGTH = 4

#: Demonym -> alpha-2 for the countries that dominate geopolitical coverage.
_DEMONYMS: dict[str, str] = {
    "american": "US",
    "argentine": "AR",
    "australian": "AU",
    "belgian": "BE",
    "brazilian": "BR",
    "british": "GB",
    "canadian": "CA",
    "chinese": "CN",
    "colombian": "CO",
    "cuban": "CU",
    "dutch": "NL",
    "egyptian": "EG",
    "french": "FR",
    "german": "DE",
    "greek": "GR",
    "indian": "IN",
    "indonesian": "ID",
    "iranian": "IR",
    "iraqi": "IQ",
    "israeli": "IL",
    "italian": "IT",
    "japanese": "JP",
    "lebanese": "LB",
    "mexican": "MX",
    "nigerian": "NG",
    "norwegian": "NO",
    "pakistani": "PK",
    "palestinian": "PS",
    "polish": "PL",
    "russian": "RU",
    "saudi": "SA",
    "scottish": "GB",
    "spanish": "ES",
    "swedish": "SE",
    "swiss": "CH",
    "syrian": "SY",
    "taiwanese": "TW",
    "turkish": "TR",
    "ukrainian": "UA",
    "venezuelan": "VE",
    "vietnamese": "VN",
}

#: Native-script and shorthand names -> alpha-2 (whole-string, casefolded).
_EXTRA_NAMES: dict[str, str] = {
    "america": "US",
    "britain": "GB",
    "great britain": "GB",
    "u.k.": "GB",
    "u.s.": "US",
    "u.s.a.": "US",
    "united states of america": "US",
    "deutschland": "DE",
    "españa": "ES",
    "italia": "IT",
    "nederland": "NL",
    "norge": "NO",
    "polska": "PL",
    "россия": "RU",
    "российская федерация": "RU",
    "украина": "UA",
    "україна": "UA",
    "ελλάδα": "GR",
    "ישראל": "IL",
    "مصر": "EG",
    "إيران": "IR",
    "ایران": "IR",
    "پاکستان": "PK",
    "भारत": "IN",
    "中国": "CN",
    "中國": "CN",
    "中华人民共和国": "CN",
    "日本": "JP",
    "대한민국": "KR",
    "한국": "KR",
    "việt nam": "VN",
    "south korea": "KR",
    "north korea": "KP",
}


@lru_cache(maxsize=1)
def _name_table() -> dict[str, str]:
    """Casefolded whole-string name/demonym/endonym -> alpha-2."""
    table: dict[str, str] = {}
    for country in pycountry.countries:
        for attribute in ("name", "official_name", "common_name"):
            value = getattr(country, attribute, None)
            if value:
                table[str(value).casefold()] = str(country.alpha_2)
    for demonym, code in _DEMONYMS.items():
        if len(demonym) < _MIN_DEMONYM_LENGTH:  # pragma: no cover - table invariant
            continue
        table[demonym] = code
        table[demonym + "s"] = code
    table.update(_EXTRA_NAMES)
    for alias in DROPPED_ALIASES:
        table.pop(alias, None)
    return table


#: The only two-letter code surfaces allowed to resolve ("UK" is not ISO but is
#: how news writes Britain; every other alpha-2 is ambiguous with state
#: abbreviations or everyday words — "TV", "IN", "ID" — and never matches).
_ALLOWED_ALPHA_2 = {"US": "US", "UK": "GB"}


@lru_cache(maxsize=1)
def _code_table() -> dict[str, str]:
    """Uppercase code surface -> alpha-2 (alpha-3 plus the alpha-2 allow-list)."""
    table: dict[str, str] = {}
    for country in pycountry.countries:
        table[str(country.alpha_3)] = str(country.alpha_2)
    table.update(_ALLOWED_ALPHA_2)
    return table


def resolve(surface: str) -> str | None:
    """Resolve one entity surface to an alpha-2 code, or ``None``.

    Whole-string only: names/demonyms/endonyms match casefolded, alpha codes match
    only fully-uppercase surfaces, and the drop-set wins over everything.
    """
    text = surface.strip().rstrip(",")
    if not text:
        return None
    key = text.casefold()
    if key in DROPPED_ALIASES:
        return None
    code = _name_table().get(key)
    if code is not None:
        return code
    if text.isupper() and len(text) in (2, 3):
        return _code_table().get(text)
    return None


def country_name(alpha_2: str) -> str:
    """Display name for an alpha-2 code (common name when pycountry has one)."""
    country = pycountry.countries.get(alpha_2=alpha_2)
    if country is None:
        msg = f"unknown alpha-2 country code {alpha_2!r}"
        raise ValueError(msg)
    common = getattr(country, "common_name", None)
    return str(common) if common else str(country.name)


def main_country(gpe_mentions: Mapping[str, int]) -> str | None:
    """The country the event is mainly about: the top-mentioned resolvable GPE."""
    best: tuple[int, str] | None = None
    for surface, count in gpe_mentions.items():
        code = resolve(surface)
        if code is None:
            continue
        candidate = (-int(count), code)
        if best is None or candidate < best:
            best = candidate
    return None if best is None else best[1]


def attribution(event: WorldEvent, cap: int = 15) -> tuple[str, ...]:
    """``{main country} | {countries resolved from ent_gpe}``, capped by mentions.

    Returns sorted alpha-2 codes. The cap keeps the main country plus the
    most-mentioned resolved countries so pair explosion stays bounded.
    """
    gpe_counts = event.entities.get("GPE", {})
    weights: dict[str, int] = {}
    for surface in event.ent_gpe:
        code = resolve(surface)
        if code is not None:
            weights[code] = weights.get(code, 0) + int(gpe_counts.get(surface, 1))
    main = resolve(event.event.country) if event.event.country else None
    if main is not None:
        weights[main] = max(weights.get(main, 0), 1) + 10**9  # the main country always survives
    ranked = sorted(weights, key=lambda code: (-weights[code], code))[:cap]
    return tuple(sorted(ranked))


def attributions(events: Iterable[WorldEvent], cap: int = 15) -> list[tuple[str, ...]]:
    """Attribution sets for a sequence of events (row-aligned convenience)."""
    return [attribution(event, cap=cap) for event in events]


def tag_events(events: Iterable[WorldEvent]) -> list[WorldEvent]:
    """Fill ``event.country`` with the display name of the main resolvable GPE."""
    tagged: list[WorldEvent] = []
    for event in events:
        code = main_country(event.entities.get("GPE", {}))
        if code is None:
            tagged.append(event)
            continue
        header = event.event.model_copy(update={"country": country_name(code)})
        tagged.append(event.model_copy(update={"event": header}))
    return tagged
