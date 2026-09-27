"""Distillation held to the post's own JPMorgan example, verbatim.

The oracle is the blog post itself: ``configs/entities/jpmorgan.json`` is the
published Resolver record and ``configs/entities/jpmorgan_patterns.json`` the
published distilled list, both copied verbatim from the archived post. The tests pin
(a) every pattern that is *derivable field-for-field from the record* — 55 of them,
listed explicitly — and (b) a coverage floor over the full published list.

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "here are the unigrams, bigrams, and trigrams distilled from the information the
    Resolution Agent extracted". Local copy:
    ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions:
    - Exact reproduction of the published list is impossible from the published
      record alone: entries like "chase sapphire", "house of morgan" and
      "jamie dimon ceo" come from evidence the record truncates
      (``short_wiki: "TRUNCATED"``). The coverage floor (≥65% of the unique list)
      plus the explicit derivable subset is therefore the honest fidelity contract.
    - "dimon" and "jamie dimon ceo" ARE tested — via the explicit ``key_people``
      argument, fed in the loop from Wikidata's P169 (chief executive) claim.

Alternatives rejected:
    - Asserting set equality with the published list: see above; it would force the
      code to hard-code JPMorgan-specific strings, which is the opposite of a rule.
"""

from __future__ import annotations

import json
from pathlib import Path

from cybernaut_mini.entities.distill import DEFAULT_PATTERN_CAP, distill, name_variants

RECORD_PATH = Path("configs/entities/jpmorgan.json")
PATTERNS_PATH = Path("configs/entities/jpmorgan_patterns.json")

#: Every pattern of the post's published list that the published record derives.
DERIVABLE_FROM_POST = [
    # unigrams
    "jpm",
    "jpmorgan",
    "jpmorganchase",
    "chase",
    "dimon",
    "jpmcoin",
    "indexgpt",
    # name variants
    "jpmorgan chase",
    "jp morgan chase",
    "j p morgan chase",
    "j.p. morgan",
    "jp morgan",
    "j p morgan",
    "jpmorgan chase & co",
    "jpmorgan chase and co",
    "nyse jpm",
    "chase uk",
    "chase student",
    "jpmorgan workplace",
    "jpmorgan workplace solutions",
    "jpmorgan securities",
    "jpmorgan cazenove",
    "jpmorgan europe",
    "jamie dimon ceo",
    # subsidiaries, verbatim from the record
    "chase bank",
    "jpmorgan securities llc",
    "jpmorgan europe ltd",
    "hambrecht & quist",
    "robert fleming & co",
    "texas commerce bank",
    "first chicago nbd",
    "first chicago bank",
    "banc one",
    "city national bank of columbus",
    "purdue national corporation",
    "bear stearns",
    "washington mutual",
    "first republic bank",
    "first republic",
    "collegiate funding services",
    "climatecare",
    "j.p. morgan cazenove",
    "global shares",
    "renovite technologies",
    "viva wallet",
    "chase manhattan bank",
    "j.p. morgan & co",
    "bank one",
    "chemical bank",
    "manufacturers hanover",
    "national bank of detroit",
    "providian financial",
    "great western bank",
    "chase national bank",
    "corn exchange bank",
    "guaranty trust company of new york",
    "jpmorgan ventures energy corporation",
]


def test_distill_derives_the_posts_patterns() -> None:
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    patterns = distill(record, key_people=("Jamie Dimon",))
    missing = [p for p in DERIVABLE_FROM_POST if p not in patterns]
    assert missing == []


def test_distill_covers_most_of_the_published_list() -> None:
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    published = set(json.loads(PATTERNS_PATH.read_text(encoding="utf-8")))
    patterns = set(distill(record, key_people=("Jamie Dimon",)))
    coverage = len(patterns & published) / len(published)
    assert coverage >= 0.65


def test_distill_output_shape() -> None:
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    patterns = distill(record, key_people=("Jamie Dimon",))
    assert len(patterns) <= DEFAULT_PATTERN_CAP
    assert len(set(patterns)) == len(patterns)  # deduplicated
    assert all(p == p.lower() for p in patterns)  # lowercased for the automaton
    assert all(len(p) >= 2 for p in patterns)
    # The cap parameter really truncates, keeping the identification-first order.
    capped = distill(record, key_people=("Jamie Dimon",), cap=10)
    assert len(capped) == 10
    assert capped == patterns[:10]
    assert "jpmorgan chase" in capped


def test_name_variants_reproduce_the_initials_rules() -> None:
    variants = name_variants("J.P. Morgan & Co.")
    for expected in (
        "j.p. morgan & co",
        "jp morgan & co",
        "j p morgan & co",
        "jpmorgan & co",
        "j.p. morgan and co",
        "j.p. morgan",
        "jp morgan",
        "j p morgan",
        "jpmorgan",
    ):
        assert expected in variants
    # Camel-case splitting: the post's "jp morgan chase" / "j p morgan chase".
    camel = name_variants("JPMorgan Chase")
    assert "jp morgan chase" in camel
    assert "j p morgan chase" in camel
    # Cleaning: commas drop, trailing periods drop, interior periods stay.
    assert name_variants("JPMorgan Securities, LLC")[0] == "jpmorgan securities llc"
    # No junk bigram prefixes from connectives or bare initials.
    assert "hambrecht &" not in name_variants("Hambrecht & Quist")
    assert "j p" not in name_variants("J.P. Morgan")


def test_distill_handles_sparse_records() -> None:
    assert distill({}) == ()
    only_ticker = distill({"ticker_wiki": "GE", "mic_code": "XNYS"})
    assert "ge" in only_ticker
    assert "nyse ge" in only_ticker
    website_only = distill({"website": "http://www.example.org/"})
    assert website_only == ("example.org",)
