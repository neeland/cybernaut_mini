"""The dual tagger over real CC-News/MIRACL fixture text — boundaries, sampling, GLiNER.

The corpus rows come from ``data/01_raw/fixtures/documents.jsonl`` (committed real
CC-News and MIRACL documents). They happen to contain the exact ambiguity the post's
JPMorgan example turns on: the unigram "chase" appears as a verb ("to chase
Arrieta"), inside "purchase(d)" (where a substring matcher without boundaries would
fire), and as the venue "Chase Field".

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "an ultra-fast and precise tagger … it uses Aho-Corasick", "a slow, universal
    tagger … sees X% of the data added to each collection". Local copy:
    ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions:
    - The word-boundary rule is asserted against an independent regex oracle over
      the same folded text, so the automaton and the test cannot share a bug.
    - Sampling is a rate, not a quota: over 460 chunks at 7.5% the binomial 99.9%
      band is roughly [1%, 16%], and the seeded RNG makes the exact count stable
      anyway.
    - GLiNER is not an installed dependency; its test is ``importorskip``-gated and
      additionally requires the entities network env var, since constructing the
      tagger downloads the checkpoint.

Alternatives rejected:
    - Synthetic sentences for the boundary cases: the committed corpus already
      contains "purchase"/"chase" collisions for real, which is strictly better
      evidence.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from cybernaut_mini.entities.tagger import (
    CapitalizedSpanDiscovery,
    CollectionTagger,
    DualTagger,
    normalise_surface,
)
from cybernaut_mini.query.s8_retrieve.intent_scan import fold

FIXTURE_DOCS = Path("data/01_raw/fixtures/documents.jsonl")

#: Independent oracle for a standalone "chase": not glued to ASCII word characters.
_CHASE_ORACLE = re.compile(r"(?<![0-9a-z_])chase(?![0-9a-z_])")


def _fixture_docs() -> list[dict[str, str]]:
    rows = [json.loads(line) for line in FIXTURE_DOCS.read_text(encoding="utf-8").splitlines()]
    return [{"id": row["id"], "text": row["text"]} for row in rows]


def test_boundary_rule_matches_regex_oracle_on_real_corpus() -> None:
    tagger = CollectionTagger([("chase", "Q192314")])
    docs = _fixture_docs()
    tagged = {doc["id"] for doc in docs if tagger.tag(doc["text"])}
    oracle = {doc["id"] for doc in docs if _CHASE_ORACLE.search(fold(doc["text"]))}
    assert tagged == oracle
    # The corpus really exercises both sides of the boundary rule.
    assert "ccn-4fff8c81dee64495" in tagged  # "… to chase Arrieta …"
    assert "ccn-ecd900247927d44d" in tagged  # "… and Chase Field …"
    assert "ccn-c3af993ec0753111" in tagged  # "The chase ended …" (noun, standalone)
    assert "ccn-0a7e4496a9cac29e" not in tagged  # "purchased"/"chased" must not fire


def test_overlapping_patterns_and_multi_entity_surfaces() -> None:
    tagger = CollectionTagger(
        [("jpmorgan chase", "Q192314"), ("chase", "Q192314"), ("chase", "Q524629")]
    )
    matches = tagger.tag("JPMorgan Chase reported earnings.")
    assert [(m.pattern, m.entity_id) for m in matches] == [
        ("jpmorgan chase", "Q192314"),
        ("chase", "Q192314"),
        ("chase", "Q524629"),
    ]
    # Offsets index the folded haystack.
    folded = fold("JPMorgan Chase reported earnings.")
    for match in matches:
        assert folded[match.start : match.end] == match.pattern


def test_covers_is_the_suggestion_filter() -> None:
    tagger = CollectionTagger([("jpmorgan chase", "Q192314")])
    assert tagger.covers("JPMorgan Chase & Co.")
    assert not tagger.covers("Acme Robotics")
    assert CollectionTagger([]).is_empty()
    assert CollectionTagger([]).tag("anything") == ()


def test_normalise_surface_folds_like_the_haystack() -> None:
    assert normalise_surface("  JPMorgan   Chase  ") == "jpmorgan chase"
    assert normalise_surface("ACME") == "acme"


def test_dual_tagger_tags_everything_and_samples_discovery() -> None:
    docs = _fixture_docs()
    production = CollectionTagger([("chase", "Q192314")])
    dual = DualTagger(
        production,
        CapitalizedSpanDiscovery(),
        sample_rate=0.075,
        rng=random.Random(7),
    )
    results = [dual.process(doc["id"], doc["text"]) for doc in docs]
    # 100% of chunks pass through the production tagger …
    assert {r.chunk_id for r in results if r.tags} == {
        doc["id"] for doc in docs if production.tag(doc["text"])
    }
    # … while discovery sees roughly X% (binomial 99.9% band at n=460, p=0.075).
    sampled = sum(r.sampled for r in results)
    assert 5 <= sampled <= 75
    assert all(r.discovered == () for r in results if not r.sampled)
    assert any(r.discovered for r in results if r.sampled)


def test_capitalized_span_discovery_finds_real_surfaces() -> None:
    doc = next(d for d in _fixture_docs() if d["id"] == "ccn-ecd900247927d44d")
    discovered = CapitalizedSpanDiscovery(min_words=2).discover(doc["text"])
    assert "Chase Field" in discovered


def test_dual_tagger_rejects_bad_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        DualTagger(CollectionTagger([]), sample_rate=1.5)


def test_gliner_discovery_optional() -> None:
    pytest.importorskip("gliner")
    import os

    if not os.environ.get("CYBERNAUT_MINI_ENTITIES_NETWORK"):
        pytest.skip("GLiNER checkpoint download is opt-in (CYBERNAUT_MINI_ENTITIES_NETWORK=1)")
    from cybernaut_mini.entities.tagger import GlinerDiscoveryTagger

    tagger = GlinerDiscoveryTagger()
    doc = next(d for d in _fixture_docs() if d["id"] == "ccn-ecd900247927d44d")
    assert isinstance(tagger.discover(doc["text"]), list)
