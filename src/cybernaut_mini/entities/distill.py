"""Distilling a Resolver record into the n-gram patterns an automaton can match.

The Resolver's intelligence lives in the strings it collects; this module is the
"distil that intelligence into a simple list of strings" step. Every rule here is
deterministic and reverse-engineered from the post's worked JPMorgan example: the
published record and the published pattern list are the specification, and the unit
tests hold this module to the derivable subset of that list verbatim.

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "it should therefore be possible to distil a great deal of that intelligence into
    a simple list of strings and regex patterns … it certainly works in specific
    cases like, for example, JP Morgan", followed by the JPMorgan record and the
    distilled unigram/bigram/trigram list. Local copy:
    ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions: each rule below is read off pairs in the post's example.
    - Cleaning: lowercase, collapse whitespace, drop commas, strip trailing periods,
      keep interior periods and ampersands — "JPMorgan Securities, LLC" →
      "jpmorgan securities llc" but "J.P. Morgan Cazenove" keeps its dots.
    - Initials variants: "J.P. Morgan" also yields "jp morgan" (dots removed),
      "j p morgan" (dots as spaces) and "jpmorgan" (initials collapsed onto the next
      word) — all four appear in the post's list.
    - Camel-case names split: "JPMorgan Chase" also yields "jp morgan chase" and
      "j p morgan chase", both in the post's list.
    - ``&`` ↔ ``and``: "jpmorgan chase & co" and "jpmorgan chase and co" both
      appear, so each form generates the other.
    - Corporate suffixes strip: "jpmorgan chase & co" → "jpmorgan chase".
    - Tickers become unigrams ("jpm") and an exchange-ticker bigram ("nyse jpm"),
      with the exchange short name taken from a small MIC/name table.
    - The website contributes its registrable domain ("jpmorganchase.com"). The
      post's list also contains "jpmorgan.com" and "chase.com", which are NOT
      derivable from the record's single ``website`` field — brand-level domains came
      from evidence the record truncates, so this module emits only what it can
      justify.
    - Three-token names contribute their leading bigram ("jpmorgan securities llc" →
      "jpmorgan securities"; "chase student loans" → "chase student"); longer names
      do not ("guaranty trust company of new york" stays whole in the post's list).
    - Two-token brand names also collapse ("JPM Coin" → "jpmcoin", as in the list).
    - Primary-name tokens of 4+ letters become unigrams ("jpmorgan", "chase") — for
      the ``name_*`` fields only, never for subsidiaries, where single tokens like
      "trust" would be noise.
    - Key people (not a record field — the post's "dimon" and "jamie dimon ceo" come
      from truncated evidence) are accepted as an explicit argument: full name,
      4+-letter surname, and the "<name> ceo" trigram.
    - The cap is 100 patterns [inferred from the post's ~90-entry list]. Buckets are
      emitted in identification-value order (names, tickers, people, domains,
      brands, subsidiaries, historical names) and truncated at the cap, so the
      patterns that make the entity *recognisable* survive truncation first.

Alternatives rejected:
    - Asking an LLM to write the patterns: that reintroduces the agent on a path the
      post built specifically to run without one; distillation is the deterministic
      half of the design.
    - Exact reproduction of the post's 90-entry list as the test oracle: the list
      provably contains evidence the published record truncates ("chase sapphire",
      "house of morgan", "jamie dimon ceo"), so field-for-field derivation cannot
      reach it; the tests instead pin the ~45-pattern derivable subset verbatim.
    - Regex patterns alongside literal strings: the post mentions them, but every
      published example is a literal n-gram, and ahocorasick-rs matches literals
      only — regexes would force a second, slower matcher into the 100% path.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

__all__ = ["DEFAULT_PATTERN_CAP", "distill", "name_variants"]

#: The post's list has ~90 entries; the loop caps a collection's per-entity
#: contribution at 100.
DEFAULT_PATTERN_CAP = 100

#: Trailing corporate designators, longest first so "& co" wins over "co".
_CORPORATE_SUFFIXES: tuple[str, ...] = (
    "& company",
    "and company",
    "& co",
    "and co",
    "corporation",
    "incorporated",
    "company",
    "limited",
    "corp",
    "inc",
    "ltd",
    "llc",
    "plc",
    "co",
)

#: MIC codes and exchange names → the short name people actually write in text.
_EXCHANGE_SHORT_NAMES: Mapping[str, str] = {
    "xnys": "nyse",
    "new york stock exchange": "nyse",
    "xnas": "nasdaq",
    "nasdaq stock market": "nasdaq",
    "nasdaq": "nasdaq",
    "xlon": "lse",
    "london stock exchange": "lse",
}

_NAME_FIELDS: tuple[str, ...] = (
    "name_orig",
    "name_kgid",
    "name_wiki",
    "name_figi",
    "name_eodhd",
    "name_lei",
)

_TICKER_FIELDS: tuple[str, ...] = ("ticker_wiki", "ticker_figi", "ticker_serp")

#: Leading run of single capitals (dotted or not) followed by a capitalised word:
#: "J.P. Morgan", "J P Morgan", "JPMorgan".
_CAMEL_RE = re.compile(r"^([A-Z]{2,4})([A-Z][a-z].*)$")


def _clean(name: str) -> str:
    """Lowercase, collapse whitespace, drop commas, strip trailing periods."""
    lowered = " ".join(name.lower().replace(",", " ").split())
    return lowered.rstrip(".").strip()


def _strip_suffix(name: str) -> str:
    for suffix in _CORPORATE_SUFFIXES:
        for form in (f" {suffix}", f" {suffix}."):
            if name.endswith(form) and len(name) > len(form):
                return name[: -len(form)].strip()
    return name


def _swap_ampersand(name: str) -> list[str]:
    variants: list[str] = []
    if "&" in name:
        variants.append(" ".join(name.replace("&", " and ").split()))
    if " and " in name:
        variants.append(" ".join(name.replace(" and ", " & ").split()))
    return variants


def _initials_variants(name: str) -> list[str]:
    """Variants of a leading dotted-initials or camel-case group.

    "j.p. morgan cazenove" → ["jp morgan cazenove", "j p morgan cazenove",
    "jpmorgan cazenove"]; camel-case input is split by :func:`name_variants` before
    reaching here.
    """
    tokens = name.split()
    if not tokens or "." not in tokens[0]:
        return []
    letters = [c for c in tokens[0] if c.isalpha()]
    if not (1 <= len(letters) <= 4) or len(tokens) < 2:
        return []
    rest = tokens[1:]
    joined = "".join(letters)
    return [
        " ".join([joined, *rest]),  # "jp morgan …"
        " ".join([*letters, *rest]),  # "j p morgan …"
        " ".join([joined + rest[0], *rest[1:]]),  # "jpmorgan …"
    ]


def _camel_split(raw: str) -> str | None:
    """"JPMorgan Chase" → "JP Morgan Chase"; None when no leading camel group."""
    tokens = raw.split()
    if not tokens:
        return None
    match = _CAMEL_RE.match(tokens[0])
    if match is None:
        return None
    acronym, rest = match.groups()
    return " ".join([acronym, rest, *tokens[1:]])


def name_variants(raw: str) -> list[str]:
    """Every deterministic surface variant of one name, cleaned, first-seen order."""
    seen: dict[str, None] = {}

    def add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate:
            seen.setdefault(candidate, None)

    base = _clean(raw)
    if not base:
        return []
    add(base)
    camel = _camel_split(raw)
    forms = [base] + ([_clean(camel)] if camel else [])
    for form in forms:
        add(form)
        for variant in _initials_variants(form):
            add(variant)
    if camel is not None:
        # "JPMorgan Chase" → "j p morgan chase": the acronym letters spaced out.
        tokens = _clean(camel).split()
        add(" ".join([*tokens[0], *tokens[1:]]))
    for form in list(seen):
        for variant in _swap_ampersand(form):
            add(variant)
    for form in list(seen):
        stripped = _strip_suffix(form)
        if stripped != form:
            add(stripped)
            for variant in _initials_variants(stripped):
                add(variant)
    for form in list(seen):
        tokens = form.split()
        if len(tokens) == 3:
            prefix = tokens[:2]
            # A prefix ending in a connective ("hambrecht &") or built from bare
            # initials ("j p") is not a pattern anyone types.
            if prefix[-1] not in {"&", "and", "of", "the"} and all(
                len(token) >= 2 for token in prefix
            ):
                add(" ".join(prefix))
    return list(seen)


def _domain(url: str) -> str | None:
    netloc = urlparse(url if "//" in url else f"http://{url}").netloc.lower()
    netloc = netloc.removeprefix("www.")
    return netloc or None


def _exchange_bigrams(record: Mapping[str, Any], tickers: Sequence[str]) -> list[str]:
    short = None
    for key in ("mic_code", "exchange"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            short = _EXCHANGE_SHORT_NAMES.get(value.strip().lower())
            if short:
                break
    if not short:
        return []
    return [f"{short} {ticker}" for ticker in tickers]


def distill(
    record: Mapping[str, Any],
    key_people: Iterable[str] = (),
    cap: int = DEFAULT_PATTERN_CAP,
) -> tuple[str, ...]:
    """Distil a Resolver record into at most *cap* lowercased automaton patterns.

    *record* is a mapping shaped like
    :class:`cybernaut_mini.entities.resolver.ResolverRecord` (the post's JSON schema);
    missing or null fields simply contribute nothing.
    """
    ordered: dict[str, None] = {}

    def add(pattern: str | None) -> None:
        if pattern and len(pattern) >= 2:
            ordered.setdefault(pattern, None)

    def names_from(value: Any) -> list[str]:
        if isinstance(value, str) and value.strip():
            return [value]
        if isinstance(value, list | tuple):
            return [item for item in value if isinstance(item, str) and item.strip()]
        return []

    # 1. Primary names: full variant set plus distinctive token unigrams.
    primary: list[str] = []
    for field in _NAME_FIELDS:
        primary.extend(names_from(record.get(field)))
    for name in primary:
        for variant in name_variants(name):
            add(variant)
    for name in primary:
        for token in _clean(name).replace("&", " ").replace(".", "").split():
            if len(token) >= 4 and token.isalpha() and token not in _CORPORATE_SUFFIXES:
                add(token)

    # 2. Tickers and the exchange-ticker bigram.
    tickers: list[str] = []
    for field in _TICKER_FIELDS:
        for ticker in names_from(record.get(field)):
            cleaned = ticker.strip().lower()
            if cleaned and cleaned not in tickers:
                tickers.append(cleaned)
    for ticker in tickers:
        add(ticker)
    for bigram in _exchange_bigrams(record, tickers):
        add(bigram)

    # 3. Key people: full name, surname, "<name> ceo".
    for person in key_people:
        cleaned = _clean(person)
        if not cleaned:
            continue
        add(cleaned)
        surname = cleaned.split()[-1]
        if len(surname) >= 4:
            add(surname)
        add(f"{cleaned} ceo")

    # 4. Domains.
    website = record.get("website")
    if isinstance(website, str) and website.strip():
        add(_domain(website))

    # 5. Brands, then subsidiaries, then historical names.
    for field in ("brand_names", "subsidiaries", "historical_names"):
        for name in names_from(record.get(field)):
            for variant in name_variants(name):
                add(variant)
            if field == "brand_names" and len(name.split()) == 2:
                add(_clean(name).replace(" ", ""))  # "JPM Coin" → "jpmcoin"

    return tuple(list(ordered)[:cap])
