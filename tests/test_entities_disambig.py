"""Two-stage company-name disambiguation over real colliding fixture stories.

The committed CC-News fixture genuinely collides on the surface "chase": a Cubs
pitcher gets chased in an NLCS report (denverpost.com), a driver flees a police
chase (wigantoday.net), and a Phoenix hiking guide overlooks Chase Field
(phoenixnewtimes.com). Stage one anchors each sense with keywords that really occur
in the bodies; stage two's ensemble must then route a held-out phoenixnewtimes
story by its context alone. The Wikidata anchor path answers from the committed
warm cache — zero network.

Blog ref: https://nosible.com/blog/using-vector-search-to-see-signals-in-company-news
    — "In the first stage we scan the body of each news story for known keywords
    associated with the overlapping entities. For example, the names of their CEO
    and board members. … In the second stage we fit probability models. … The best
    solution is simply to ensemble." Local copy:
    ``docs/blog-archive/using-vector-search-to-see-signals-in-company-news.md``.

Assumptions:
    - The fixture's "chase" senses stand in for the post's Discovery cluster: the
      collision is real (four unrelated stories, one surface), only the candidate
      entities are senses instead of listed companies, because the committed corpus
      does not carry two tickered companies sharing a name.
    - The held-out prediction is pinned to the domain-prior mechanism the post
      describes ("if we see a story about 'Discovery' on latimes.com …"): the
      held-out story shares only its domain (and city vocabulary) with one seed.
    - The ambiguity rule is tested with anchors from *within one real story* (Cubs
      and Dodgers both occur in the NLCS report), because ambiguity means matching
      anchors of several candidates, not matching several anchors.

Alternatives rejected:
    - Synthesising colliding company stories: forbidden (real data only), and the
      fixture already collides naturally.
    - Asserting exact ensemble probabilities: NB/LR internals may shift across
      sklearn versions; the tests pin the argmax, the probability-simplex property,
      and the seed labels, which are the stated behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cybernaut_mini.entities.disambig import (
    ContextEnsemble,
    NewsStory,
    anchor_seed_labels,
    anchors_from_wikidata,
    country_mentions,
    story_from_document,
    two_stage_disambiguator,
)
from cybernaut_mini.ingest import load_documents
from cybernaut_mini.models import Document

FIXTURE_DOCS = Path("data/01_raw/fixtures/documents.jsonl")

#: Real fixture stories that collide on the surface "chase".
BASEBALL = "ccn-4fff8c81dee64495"  # denverpost.com — "… to chase Arrieta …"
POLICE = "ccn-c3af993ec0753111"  # wigantoday.net — "Driver on the run after police chase"
HIKES = "ccn-ecd900247927d44d"  # phoenixnewtimes.com — "… and Chase Field."
ROOSEVELT = "ccn-0a7e4496a9cac29e"  # phoenixnewtimes.com — held out, no anchor occurs
WASPS = "ccn-9ce2c5fcc61deab3"  # coventrytelegraph.net — mentions South Africa

#: Anchor keywords per candidate; each occurs verbatim in exactly one story body.
ANCHORS = {
    "nlcs-race": ["arrieta"],
    "police-pursuit": ["officers"],
    "chase-field": ["superstition"],
}

JPM = "Q192314"


@pytest.fixture(scope="module")
def fixture_documents() -> list[Document]:
    return load_documents(FIXTURE_DOCS)


@pytest.fixture(scope="module")
def stories(fixture_documents: list[Document]) -> list[NewsStory]:
    by_id = {document.id: document for document in fixture_documents}
    return [story_from_document(by_id[doc_id]) for doc_id in (BASEBALL, POLICE, HIKES, ROOSEVELT)]


def test_story_from_document_prefers_publisher_then_netloc(
    fixture_documents: list[Document],
) -> None:
    by_id = {document.id: document for document in fixture_documents}
    assert story_from_document(by_id[BASEBALL]).domain == "denverpost.com"
    # Same real document with the publisher key withheld: netloc fallback.
    stripped = by_id[BASEBALL].model_copy(
        update={"metadata": {k: v for k, v in by_id[BASEBALL].metadata.items() if k != "publisher"}}
    )
    assert story_from_document(stripped).domain == "www.denverpost.com"
    orphan = by_id[BASEBALL].model_copy(update={"metadata": {}, "url": None})
    assert story_from_document(orphan).domain == ""


def test_anchor_seed_labels_single_candidate_hits(stories: list[NewsStory]) -> None:
    seeds = anchor_seed_labels(stories, ANCHORS)
    assert seeds.labels == {
        BASEBALL: "nlcs-race",
        POLICE: "police-pursuit",
        HIKES: "chase-field",
    }
    assert seeds.ambiguous == ()
    assert seeds.unmatched == (ROOSEVELT,)


def test_multi_candidate_match_is_ambiguity_not_evidence(stories: list[NewsStory]) -> None:
    # Cubs and Dodgers both occur in the real NLCS report: anchors of two
    # candidates fire in one story, so it must be reported, not labelled.
    seeds = anchor_seed_labels(stories, {"cubs-side": ["cubs"], "dodgers-side": ["dodgers"]})
    assert seeds.labels == {}
    assert seeds.ambiguous == (BASEBALL,)
    assert set(seeds.unmatched) == {POLICE, HIKES, ROOSEVELT}


def test_ensemble_routes_held_out_story_by_domain_prior(stories: list[NewsStory]) -> None:
    ensemble, seeds = two_stage_disambiguator(stories, ANCHORS)
    assert ROOSEVELT in seeds.unmatched
    held_out = next(story for story in stories if story.doc_id == ROOSEVELT)
    probabilities = ensemble.predict_proba(held_out)
    assert set(probabilities) == set(ANCHORS)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    # phoenixnewtimes.com is only ever seen with the Chase Field sense: the
    # post's latimes.com/Warner-Bros-Discovery mechanism, on real fixture data.
    entity_id, probability = ensemble.predict(held_out)
    assert entity_id == "chase-field"
    assert probability == pytest.approx(probabilities["chase-field"])
    assert probability > max(p for cls, p in probabilities.items() if cls != "chase-field")


def test_single_class_ensemble_degenerates_to_certainty(stories: list[NewsStory]) -> None:
    ensemble = ContextEnsemble().fit(stories, {BASEBALL: "nlcs-race"})
    assert ensemble.predict_proba(stories[1]) == {"nlcs-race": 1.0}
    assert ensemble.predict(stories[1]) == ("nlcs-race", 1.0)


def test_ensemble_raises_without_seeds_or_fit(stories: list[NewsStory]) -> None:
    with pytest.raises(ValueError, match="at least one labelled story"):
        ContextEnsemble().fit(stories, {})
    with pytest.raises(ValueError, match="before fit"):
        ContextEnsemble().predict_proba(stories[0])


def test_country_mentions_on_real_text(fixture_documents: list[Document]) -> None:
    by_id = {document.id: document for document in fixture_documents}
    # The Wasps rugby story really says "South Africa"; pycountry maps it to ZA.
    assert "ZA" in country_mentions(by_id[WASPS].text)
    assert country_mentions("no geography in this sentence") == ()


def test_anchors_from_wikidata_answers_from_warm_cache_offline() -> None:
    # data/01_raw/entities/wikidata/entity_Q192314.json is committed; no network.
    assert anchors_from_wikidata(JPM) == ("jamie dimon",)
