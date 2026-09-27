"""Ticker resolution: the exact layer over the fuzzy entity layer.

Entities are fuzzy (surface strings with counts); tickers are exact (a resolved
listing or nothing). This module keeps that asymmetry from the posts: a hand-built
alias table for a few dozen large caps, resolved by normalized *exact* match of ORG
surfaces, emitting the ``tickers[{name, ticker_eodhd}]`` block verbatim.

Blog ref: https://nosible.com/blog/point-in-time-knowledge-graphs-over-named-entities
    — the record's ``"tickers": [{"name": "Microsoft Corporation", "ticker_eodhd":
    "MSFT.US"}, ...]`` block and "each record carries ... resolved company
    tickers"; https://nosible.com/blog/nosible-compare — World "covers 38 thousand
    tickers", resolved at ingest. Local copies under ``docs/blog-archive/``.

Assumptions:
    - The alias table is a committed constant of ~35 large caps in EODHD style
      (``NVDA.US``, ``6758.TSE``): the laptop stand-in for World's 38k-ticker
      resolver. Every alias resolves through casefolded, corporate-suffix-stripped
      exact match — no fuzzy matching, because a wrong ticker silently corrupts
      every per-ticker series downstream, while a missed one only shrinks it.
    - Resolution reads the event's already-canonicalized ORG mention layer, so
      "Nvidia Corp" and "NVIDIA" have usually merged before they get here; the
      table still lists both forms so resolution does not depend on that merge.
    - One canonical display name per ticker (the exchange-style long name the post
      prints), regardless of which alias matched.

Alternatives rejected:
    - Resolving from Wikidata P414/ticker claims at build time: that is the
      entities workstream's Resolver agent; this layer stays offline-deterministic.
    - Substring/prefix matching ("Apple" inside "Apple Daily"): the false-positive
      direction the exact layer exists to prevent.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache

from cybernaut_mini.world.events import TickerRef, WorldEvent

__all__ = [
    "TICKER_TABLE",
    "resolve_org",
    "resolve_tickers",
    "tag_events",
]

#: ticker_eodhd -> (canonical display name, alias surfaces).
TICKER_TABLE: dict[str, tuple[str, tuple[str, ...]]] = {
    "AAPL.US": ("Apple Inc", ("Apple", "Apple Inc", "Apple Computer")),
    "AMD.US": ("Advanced Micro Devices Inc", ("AMD", "Advanced Micro Devices")),
    "AMZN.US": ("Amazon.com Inc", ("Amazon", "Amazon.com")),
    "ATVI.US": ("Activision Blizzard Inc", ("Activision", "Activision Blizzard")),
    "BA.US": ("Boeing Company", ("Boeing", "The Boeing Company")),
    "BAC.US": ("Bank of America Corp", ("Bank of America", "BofA")),
    "CSCO.US": ("Cisco Systems Inc", ("Cisco", "Cisco Systems")),
    "CVX.US": ("Chevron Corp", ("Chevron",)),
    "DIS.US": ("Walt Disney Company", ("Disney", "Walt Disney", "The Walt Disney Company")),
    "F.US": ("Ford Motor Company", ("Ford", "Ford Motor")),
    "GE.US": ("General Electric Company", ("General Electric",)),
    "GM.US": ("General Motors Company", ("General Motors",)),
    "GOOGL.US": ("Alphabet Inc", ("Google", "Alphabet")),
    "GS.US": ("Goldman Sachs Group Inc", ("Goldman Sachs", "Goldman")),
    "IBM.US": ("International Business Machines Corp", ("IBM", "International Business Machines")),
    "INTC.US": ("Intel Corp", ("Intel",)),
    "JNJ.US": ("Johnson & Johnson", ("Johnson & Johnson", "Johnson and Johnson")),
    "JPM.US": ("JPMorgan Chase & Co", ("JPMorgan", "JPMorgan Chase", "JP Morgan", "J.P. Morgan")),
    "KO.US": ("Coca-Cola Company", ("Coca-Cola", "Coca Cola", "The Coca-Cola Company")),
    "MA.US": ("Mastercard Inc", ("Mastercard",)),
    "META.US": ("Meta Platforms Inc", ("Meta", "Meta Platforms", "Facebook")),
    "MSFT.US": ("Microsoft Corporation", ("Microsoft",)),
    "NFLX.US": ("Netflix Inc", ("Netflix",)),
    "NVDA.US": ("NVIDIA Corporation", ("NVIDIA", "Nvidia", "Nvidia Corp")),
    "ORCL.US": ("Oracle Corp", ("Oracle",)),
    "PEP.US": ("PepsiCo Inc", ("PepsiCo", "Pepsi")),
    "PFE.US": ("Pfizer Inc", ("Pfizer",)),
    "T.US": ("AT&T Inc", ("AT&T",)),
    "TSLA.US": ("Tesla Inc", ("Tesla", "Tesla Motors")),
    "UBER.US": ("Uber Technologies Inc", ("Uber", "Uber Technologies")),
    "V.US": ("Visa Inc", ("Visa",)),
    "VZ.US": ("Verizon Communications Inc", ("Verizon", "Verizon Communications")),
    "WFC.US": ("Wells Fargo & Company", ("Wells Fargo",)),
    "WMT.US": ("Walmart Inc", ("Walmart", "Wal-Mart")),
    "XOM.US": ("Exxon Mobil Corp", ("Exxon", "Exxon Mobil", "ExxonMobil")),
    "6758.TSE": ("Sony Corp", ("Sony", "Sony Corporation")),
}

_SUFFIX_TOKENS = frozenset(
    {"inc", "inc.", "corp", "corp.", "corporation", "co", "co.", "company", "ltd", "ltd.", "plc",
     "llc", "group", "holdings", "ag", "sa", "nv", "se"}
)


def normalize_org(surface: str) -> str:
    """Casefold and strip trailing corporate-suffix tokens for exact matching."""
    tokens = surface.strip().casefold().split()
    while len(tokens) > 1 and tokens[-1] in _SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for ticker, (name, aliases) in TICKER_TABLE.items():
        for alias in (name, *aliases):
            index[normalize_org(alias)] = ticker
    return index


def resolve_org(surface: str) -> TickerRef | None:
    """Resolve one ORG surface by normalized exact match, or ``None``."""
    ticker = _alias_index().get(normalize_org(surface))
    if ticker is None:
        return None
    return TickerRef(name=TICKER_TABLE[ticker][0], ticker_eodhd=ticker)


def resolve_tickers(org_mentions: Mapping[str, int]) -> list[TickerRef]:
    """Distinct resolved listings for an event's ORG layer, sorted by ticker."""
    refs: dict[str, TickerRef] = {}
    for surface in org_mentions:
        ref = resolve_org(surface)
        if ref is not None:
            refs[ref.ticker_eodhd] = ref
    return [refs[ticker] for ticker in sorted(refs)]


def tag_events(events: Iterable[WorldEvent]) -> list[WorldEvent]:
    """Fill each event's ``tickers`` block from its ORG entity layer."""
    return [
        event.model_copy(update={"tickers": resolve_tickers(event.entities.get("ORG", {}))})
        for event in events
    ]
