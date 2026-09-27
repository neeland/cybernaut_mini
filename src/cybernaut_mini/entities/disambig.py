"""Two-stage company-name disambiguation: anchor keywords, then a context ensemble.

Many entities share one surface ("Discovery", "Chase", "Jordan"). Stage one scans
story bodies with Aho-Corasick for anchor keywords tied to exactly one candidate —
the names of executives and board members pulled from Wikidata — producing a seed
set of stories whose identity is high-conviction. Stage two fits probability models
over context features (source domain, country mentions, body TF-IDF) on those seeds
and ensembles them by probability averaging, so held-out stories that carry no
anchor still get a calibrated identity guess.

Blog ref: https://nosible.com/blog/using-vector-search-to-see-signals-in-company-news
    — "We solved this using a two-stage model. In the first stage we scan the body of
    each news story for known keywords associated with the overlapping entities. For
    example, the names of their CEO and board members. … For stage one we used the
    Aho-Corasick algorithm implementation from G-Research. In the second stage we fit
    probability models. For example, if we see a story about 'Discovery' on
    latimes.com we know there is a high probability of it being about Warner Bros
    Discovery. Similarly, if we see an article that mentions South Africa, we know it
    has a nearly 100% probability of being about Discovery (DSY). … Yes,
    probabilities can be spurious. The best solution is simply to ensemble." Local
    copy: ``docs/blog-archive/using-vector-search-to-see-signals-in-company-news.md``.

Assumptions:
    - The stage-two ensemble is Naive Bayes + logistic regression + a domain-prior
      lookup, probability-averaged [inferred]: the post says only "probability
      models" and "ensemble"; NB and LR are the two standard probabilistic text
      classifiers over TF-IDF, and the latimes.com example *is* a domain prior, so
      it participates as its own equally weighted member rather than only as a
      feature the other two may ignore.
    - Features are exactly the post's two worked examples plus the body: a one-hot
      source domain, per-country mention flags (the "mentions South Africa" signal,
      matched by Aho-Corasick over pycountry names with word boundaries), and body
      TF-IDF. The ambiguous surface itself is *not* excluded from country matching
      ("Jordan" fires the country flag in every story about any Jordan) — a flag
      that fires for every class is uninformative to the fitted models, so
      special-casing it would add code without changing behaviour.
    - Anchor keywords come from the Wikidata record the Resolver already caches:
      :meth:`~cybernaut_mini.entities.resolver.WikidataTool.entity` (opt-in network
      behind ``CYBERNAUT_MINI_ENTITIES_NETWORK``, warm cache fully offline). The
      normalised record carries the chief executive (P169); board members (P3320)
      are not in that shared schema, so callers append further anchor names — the
      lexicon is a plain mapping.
    - A story anchors a seed only when its anchors match exactly one candidate;
      stories matching anchors of several candidates are reported as ambiguous and
      kept out of training rather than resolved by fiat.
    - Sparse TF-IDF is densified before stacking with the one-hot and flag columns:
      at this repo's laptop scale (thousands of stories, small vocabularies) the
      dense matrix is megabytes, and it spares a scipy dependency in typed code.

Alternatives rejected:
    - A single logistic regression with the domain as a feature (no ensemble): the
      post's Hoya Corp story is the argument — any one model's probabilities "can be
      spurious", and its stated fix is "simply to ensemble".
    - Class-weighted voting instead of probability averaging: the gap analysis and
      the post's language ("probability models") both point at averaging the
      calibrated outputs, which also keeps the ensemble's output a probability.
    - Demonym matching ("South African") in the country flags: pycountry's demonym
      coverage is patchy and the post's example is the country name itself; names
      keep the flag precise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import numpy as np

from cybernaut_mini.entities.resolver import WikidataTool
from cybernaut_mini.entities.tagger import CollectionTagger, normalise_surface

if TYPE_CHECKING:
    import numpy.typing as npt

    from cybernaut_mini.models import Document

    FloatArray = npt.NDArray[np.float32]

__all__ = [
    "ContextEnsemble",
    "NewsStory",
    "SeedSet",
    "anchor_seed_labels",
    "anchors_from_wikidata",
    "country_mentions",
    "story_from_document",
    "two_stage_disambiguator",
]


@dataclass(frozen=True)
class NewsStory:
    """One story as the disambiguator sees it: id, source domain, body text."""

    doc_id: str
    domain: str
    text: str


def story_from_document(document: Document) -> NewsStory:
    """Adapt a corpus :class:`~cybernaut_mini.models.Document` to a story.

    The source domain is the ``publisher`` metadata the CC-News fixtures carry,
    falling back to the URL's netloc, falling back to ``""`` (an unknown domain —
    the prior treats it as uniform).
    """
    domain = str(document.metadata.get("publisher", "") or "")
    if not domain and document.url:
        domain = urlparse(document.url).netloc
    return NewsStory(doc_id=document.id, domain=domain.lower(), text=document.text)


def anchors_from_wikidata(qid: str, tool: WikidataTool | None = None) -> tuple[str, ...]:
    """Executive anchor names for *qid* from the Resolver's cached Wikidata record.

    A warm cache (``data/01_raw/entities/wikidata``) answers offline; a cold cache
    requires the opt-in ``CYBERNAUT_MINI_ENTITIES_NETWORK=1``. The shared record
    schema carries the chief executive; board members are a caller-side extension.
    """
    wikidata = tool if tool is not None else WikidataTool()
    record = wikidata.entity(qid)
    anchors = [
        normalise_surface(str(value))
        for value in (record.get("chief_executive"),)
        if value
    ]
    return tuple(dict.fromkeys(anchors))


@dataclass(frozen=True)
class SeedSet:
    """Stage one's output: the high-conviction seeds, plus what it refused to label."""

    labels: dict[str, str]  # doc_id -> entity_id
    ambiguous: tuple[str, ...]  # doc ids matching anchors of >1 candidate
    unmatched: tuple[str, ...]  # doc ids matching no anchors at all


def anchor_seed_labels(
    stories: Sequence[NewsStory], anchors: Mapping[str, Sequence[str]]
) -> SeedSet:
    """Scan bodies for per-candidate anchor keywords; label single-candidate hits.

    *anchors* maps each candidate entity id to its anchor keyword list. A story is
    a seed for the one candidate whose anchors it matches; matching several
    candidates is ambiguity, not evidence, and matching none leaves the story for
    stage two.
    """
    tagger = CollectionTagger(
        (keyword, entity_id)
        for entity_id, keywords in anchors.items()
        for keyword in keywords
    )
    labels: dict[str, str] = {}
    ambiguous: list[str] = []
    unmatched: list[str] = []
    for story in stories:
        matched = {match.entity_id for match in tagger.tag(story.text)}
        if len(matched) == 1:
            labels[story.doc_id] = matched.pop()
        elif matched:
            ambiguous.append(story.doc_id)
        else:
            unmatched.append(story.doc_id)
    return SeedSet(labels=labels, ambiguous=tuple(ambiguous), unmatched=tuple(unmatched))


@lru_cache(maxsize=1)
def _country_tagger() -> CollectionTagger:
    """Aho-Corasick over pycountry names (and common names), payload = alpha-2."""
    import pycountry

    patterns: list[tuple[str, str]] = []
    for country in pycountry.countries:
        names = {country.name, getattr(country, "common_name", None)}
        patterns.extend(
            (name, country.alpha_2) for name in names if isinstance(name, str) and name
        )
    return CollectionTagger(patterns)


def country_mentions(text: str) -> tuple[str, ...]:
    """Sorted alpha-2 codes of every country whose name occurs in *text*."""
    return tuple(sorted({match.entity_id for match in _country_tagger().tag(text)}))


class ContextEnsemble:
    """NB + LR + domain-prior over (domain one-hot, country flags, body TF-IDF).

    Fit on the anchored seed set; ``predict_proba`` averages the three members'
    probabilities, aligned by class label. With a single training class every
    member is a constant, so the ensemble degenerates honestly to certainty.
    """

    def __init__(self, prior_smoothing: float = 1.0) -> None:
        self.prior_smoothing = prior_smoothing
        self.classes_: tuple[str, ...] = ()
        self._domains: tuple[str, ...] = ()
        self._countries: tuple[str, ...] = ()
        self._vectorizer: Any = None
        self._nb: Any = None
        self._lr: Any = None
        self._domain_counts: dict[str, dict[str, int]] = {}
        self._class_totals: dict[str, int] = {}

    def _features(self, stories: Sequence[NewsStory]) -> FloatArray:
        tfidf = np.asarray(
            self._vectorizer.transform([story.text for story in stories]).toarray(),
            dtype=np.float32,
        )
        domain_onehot = np.zeros((len(stories), len(self._domains)), dtype=np.float32)
        flags = np.zeros((len(stories), len(self._countries)), dtype=np.float32)
        domain_col = {domain: j for j, domain in enumerate(self._domains)}
        country_col = {code: j for j, code in enumerate(self._countries)}
        for i, story in enumerate(stories):
            j = domain_col.get(story.domain)
            if j is not None:
                domain_onehot[i, j] = 1.0
            for code in country_mentions(story.text):
                col = country_col.get(code)
                if col is not None:
                    flags[i, col] = 1.0
        return np.hstack([tfidf, domain_onehot, flags])

    def fit(self, stories: Sequence[NewsStory], labels: Mapping[str, str]) -> ContextEnsemble:
        """Fit all three members on the stories that carry a label."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB

        train = [story for story in stories if story.doc_id in labels]
        if not train:
            msg = "ContextEnsemble.fit needs at least one labelled story"
            raise ValueError(msg)
        y = [labels[story.doc_id] for story in train]
        self.classes_ = tuple(sorted(set(y)))
        self._domains = tuple(sorted({story.domain for story in train if story.domain}))
        self._countries = tuple(
            sorted({code for story in train for code in country_mentions(story.text)})
        )
        self._vectorizer = TfidfVectorizer(lowercase=True)
        self._vectorizer.fit([story.text for story in train])
        features = self._features(train)

        self._domain_counts = {cls: {} for cls in self.classes_}
        self._class_totals = dict.fromkeys(self.classes_, 0)
        for story, cls in zip(train, y, strict=True):
            self._class_totals[cls] += 1
            if story.domain:
                counts = self._domain_counts[cls]
                counts[story.domain] = counts.get(story.domain, 0) + 1

        if len(self.classes_) > 1:
            self._nb = MultinomialNB().fit(features, y)
            self._lr = LogisticRegression(max_iter=1000).fit(features, y)
        return self

    def _domain_prior(self, domain: str) -> dict[str, float]:
        """Laplace-smoothed ``P(entity | domain)`` from the seed counts."""
        smoothing = self.prior_smoothing
        weights = {
            cls: self._domain_counts[cls].get(domain, 0) + smoothing for cls in self.classes_
        }
        total = sum(weights.values())
        return {cls: weight / total for cls, weight in weights.items()}

    def predict_proba(self, story: NewsStory) -> dict[str, float]:
        """Probability per candidate: the mean of NB, LR and the domain prior."""
        if not self.classes_:
            msg = "ContextEnsemble.predict_proba called before fit"
            raise ValueError(msg)
        if len(self.classes_) == 1:
            return {self.classes_[0]: 1.0}
        features = self._features([story])
        members: list[dict[str, float]] = []
        for model in (self._nb, self._lr):
            row = model.predict_proba(features)[0]
            members.append(
                {str(cls): float(p) for cls, p in zip(model.classes_, row, strict=True)}
            )
        members.append(self._domain_prior(story.domain))
        return {
            cls: sum(member.get(cls, 0.0) for member in members) / len(members)
            for cls in self.classes_
        }

    def predict(self, story: NewsStory) -> tuple[str, float]:
        """The most probable candidate and its averaged probability."""
        probabilities = self.predict_proba(story)
        entity_id = min(probabilities, key=lambda cls: (-probabilities[cls], cls))
        return entity_id, probabilities[entity_id]


def two_stage_disambiguator(
    stories: Sequence[NewsStory], anchors: Mapping[str, Sequence[str]]
) -> tuple[ContextEnsemble, SeedSet]:
    """Run both stages: anchor the seeds, fit the ensemble on them, return both.

    The returned :class:`SeedSet` says which stories trained the model (and which
    were ambiguous or unmatched), so evaluation can hold out honestly.
    """
    seeds = anchor_seed_labels(stories, anchors)
    ensemble = ContextEnsemble().fit(stories, seeds.labels)
    return ensemble, seeds
