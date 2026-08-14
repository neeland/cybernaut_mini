"""Stage 5 — Shard Selection: three ranking factors fused with RRF.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 5, "Shard Selection":
    "arguably the single most important stage in our retrieval pipeline because, if we
    route the question to the wrong shards, we won't return the best document". The post
    lists four ranking factors — Vanilla Dense, Bayesian Dense, Vanilla Sparse and Entity
    Sparse — and combines them with reciprocal rank fusion, "a simple but powerful
    ensembling method ... This gives robust results even if one of the ranking factors is
    noisy." Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

The Bayesian Dense Similarity factor is deliberately absent
-----------------------------------------------------------
The post's second factor reads, in full: "This selector is derived from a vector search
index we developed that often beats the current state-of-the-art. I can't discuss it
because we are pursuing a patent on it 😉." Nothing about it is disclosed — not the prior,
not the likelihood, not what is being estimated. This replica therefore ships **three**
factors, not four, and says so rather than shipping some plausible-looking Bayesian
reranker under that name. Inventing one would be the worst of both worlds: it would not
be NOSIBLE's method, and a reader comparing this code to the post would conclude the
withheld factor had been reconstructed when it had only been guessed at. Its absence is
reported in :attr:`ShardSelectionSignals.omitted_factors` so a caller sees the gap at
runtime, not only in a docstring. Expect selection quality below the post's as a result:
the missing factor is one of two dense signals, and RRF with three lists is more exposed
to any one of them being noisy than RRF with four.

Techniques
----------
- Vanilla Dense Similarity over an HNSW graph [disclosed] — see :mod:`.dense`.
- Vanilla Sparse and Entity Sparse Similarity over sparse TF-IDF matrices [disclosed] —
  see :mod:`.sparse`.
- RRF over the surviving factors [disclosed] — the post names reciprocal rank fusion and
  this repo already has a weighted implementation in :mod:`cybernaut_mini.rrf`.
- Per-factor candidate depth [inferred] — the post's worked example routes a question to
  100 shards ("+97 others"), which is the default here, but it does not say whether each
  factor produces a full ranking or a truncated one. Truncating each factor to the same
  depth is what makes the stage sublinear; the alternative, a complete ranking per
  factor, is what ``routing.py`` does today and is O(n_shards) by construction.

Assumptions: the question's "keywords" are
    :meth:`cybernaut_mini.text.TextProcessor.content_tokens` — stopword-free, lemmatised
    when spaCy is present — because that is what shard keywords were computed from at
    build time, and comparing a question tokenised one way against shard keywords
    tokenised another way is the kind of mismatch that silently halves a sparse score.
    The question's entities come from :meth:`~cybernaut_mini.text.TextProcessor.entities`
    by default, which returns nothing without spaCy; ``entity_extractor`` exists so a
    caller with a real NER model, or a test, can supply its own.

Alternatives rejected: folding stage 6's rerankers in here, which is what
    ``cybernaut_mini.routing.route`` does. The post is explicit that reranking is a
    separate stage operating on the shards selection has already chosen, and merging them
    hides which stage is responsible when routing goes wrong. Also rejected: normalising
    the three factors' scores onto a common scale and summing them. Score fusion needs
    the factors to be calibrated against each other — a cosine in [0, 1], a sparse cosine
    whose maximum depends on vocabulary overlap, and an entity cosine that only exists
    for some questions — which is precisely the calibration RRF's rank-only view avoids
    needing, and the post chose RRF.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from cybernaut_mini.query.s5_select.dense import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,
    HnswCacheError,
    HnswShardIndex,
    RecallReport,
)
from cybernaut_mini.query.s5_select.sparse import SparseShardMatrix, term_frequencies
from cybernaut_mini.rrf import FusedItem, RankedList, rrf_fuse

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from cybernaut_mini.config import RRFConfig
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.providers.embeddings import EmbeddingProvider, FloatArray
    from cybernaut_mini.text import TextProcessor

#: The post's worked example routes each question to 100 shards.
DEFAULT_CANDIDATE_DEPTH = 100

#: The factor the post withholds. Named so callers can assert on the gap.
BAYESIAN_DENSE = "bayesian_dense"

DENSE = "dense"
SPARSE = "sparse"
ENTITY = "entity"


@dataclass(frozen=True)
class FactorRanking:
    """One ranking factor's contribution, best-first, with why it did or did not run."""

    name: str
    weight: float
    #: Shard ids best-first. Empty when ``applied`` is False.
    ranked: tuple[int, ...]
    #: The factor's own scores by shard id, for tracing. Not comparable across factors.
    scores: dict[int, float]
    applied: bool
    #: Human-readable reason, always set — "applied" or why not.
    reason: str

    def as_ranked_list(self) -> RankedList:
        return RankedList(
            name=self.name,
            weight=self.weight,
            ids=tuple(str(shard_id) for shard_id in self.ranked),
            scores=tuple(self.scores[shard_id] for shard_id in self.ranked),
        )


@dataclass(frozen=True)
class ShardSelectionSignals:
    """Everything stage 5 looked at, for tracing and for tests."""

    question: str
    query_terms: dict[str, float]
    query_entities: dict[str, float]
    factors: dict[str, FactorRanking]
    fused: list[FusedItem]
    candidate_depth: int
    exact_dense: bool
    #: Factors the post describes that this replica does not implement, and why.
    omitted_factors: dict[str, str] = field(default_factory=dict)

    def factor(self, name: str) -> FactorRanking:
        return self.factors[name]

    def applied_factors(self) -> tuple[str, ...]:
        return tuple(name for name, factor in self.factors.items() if factor.applied)


@dataclass(frozen=True)
class ShardSelection:
    """The stage's output: ranked shard ids plus the signals that produced them."""

    shard_ids: list[int]
    signals: ShardSelectionSignals

    def __iter__(self) -> Iterator[int]:
        return iter(self.shard_ids)

    def __len__(self) -> int:
        return len(self.shard_ids)


_OMITTED = {
    BAYESIAN_DENSE: (
        "withheld by the source post: 'I can't discuss it because we are pursuing a "
        "patent on it'. Not implemented rather than guessed at."
    )
}


class ShardSelector:
    """Select the shards a question should be routed to, from a :class:`LoadedIndex`.

    Construction is cheap: the HNSW graph and the two sparse matrices are built on first
    use and then cached on the instance, so a long-lived selector pays for them once. Only
    :attr:`LoadedIndex.manifests` is read — never ``documents``, ``doc_tokens`` or
    ``vectors`` — so a 1,000,000-document index costs the same here as a 24-document one.
    """

    def __init__(
        self,
        index: LoadedIndex,
        *,
        provider: EmbeddingProvider,
        processor: TextProcessor,
        rrf_config: RRFConfig | None = None,
        candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
        entity_extractor: Callable[[str], Sequence[str]] | None = None,
        exact_dense: bool = False,
        connectivity: int = DEFAULT_CONNECTIVITY,
        expansion_add: int = DEFAULT_EXPANSION_ADD,
        expansion_search: int = DEFAULT_EXPANSION_SEARCH,
        hnsw_cache_dir: Path | None = None,
    ) -> None:
        if candidate_depth < 1:
            msg = f"candidate_depth must be >= 1, got {candidate_depth}"
            raise ValueError(msg)
        self.index = index
        self.provider = provider
        self.processor = processor
        if rrf_config is None:
            from cybernaut_mini.config import RRFConfig as _RRFConfig

            rrf_config = _RRFConfig()
        self.rrf_config = rrf_config
        self.candidate_depth = candidate_depth
        self.entity_extractor = entity_extractor or processor.entities
        #: When True, the dense factor scans every shard vector instead of using HNSW.
        #: Exact and slower; the switch exists so recall can be compared like for like.
        self.exact_dense = exact_dense
        self.connectivity = connectivity
        self.expansion_add = expansion_add
        self.expansion_search = expansion_search
        self.hnsw_cache_dir = Path(hnsw_cache_dir) if hnsw_cache_dir is not None else None

        self._hnsw: HnswShardIndex | None = None
        self._keyword_matrix: SparseShardMatrix | None = None
        self._entity_matrix: SparseShardMatrix | None = None

    # ------------------------------------------------------------------ #
    # Lazily built, then cached, structures                               #
    # ------------------------------------------------------------------ #

    @property
    def hnsw(self) -> HnswShardIndex:
        """The HNSW graph over shard summary embeddings, built or loaded on first use."""
        if self._hnsw is None:
            self._hnsw = self._load_or_build_hnsw()
        return self._hnsw

    def _load_or_build_hnsw(self) -> HnswShardIndex:
        model = self.index.meta.embedding_model
        if self.hnsw_cache_dir is not None:
            try:
                cached = HnswShardIndex.load(self.hnsw_cache_dir, embedding_model=model)
            except HnswCacheError:
                cached = None
            if cached is not None and cached.shard_ids == tuple(sorted(self.index.manifests)):
                return cached
        built = HnswShardIndex.from_manifests(
            self.index.manifests,
            self.provider,
            connectivity=self.connectivity,
            expansion_add=self.expansion_add,
            expansion_search=self.expansion_search,
        )
        if self.hnsw_cache_dir is not None:
            built.save(self.hnsw_cache_dir, embedding_model=model)
        return built

    @property
    def keyword_matrix(self) -> SparseShardMatrix:
        if self._keyword_matrix is None:
            self._keyword_matrix = SparseShardMatrix.from_keywords(self.index.manifests)
        return self._keyword_matrix

    @property
    def entity_matrix(self) -> SparseShardMatrix:
        if self._entity_matrix is None:
            self._entity_matrix = SparseShardMatrix.from_entities(self.index.manifests)
        return self._entity_matrix

    def build(self) -> None:
        """Materialise all three structures now instead of on the first query."""
        _ = self.hnsw
        _ = self.keyword_matrix
        _ = self.entity_matrix

    # ------------------------------------------------------------------ #
    # Question analysis                                                   #
    # ------------------------------------------------------------------ #

    def question_terms(self, question: str) -> dict[str, float]:
        """The question's keywords with their term frequencies."""
        return term_frequencies(self.processor.content_tokens(question))

    def question_entities(self, question: str) -> dict[str, float]:
        """The question's entities with their counts. Empty means the entity factor is off."""
        return term_frequencies(list(self.entity_extractor(question)))

    def embed_question(self, question: str) -> FloatArray:
        vector: FloatArray = self.provider.embed_queries([question])[0]
        return vector

    # ------------------------------------------------------------------ #
    # The three factors, each usable on its own                           #
    # ------------------------------------------------------------------ #

    def dense_factor(
        self,
        question_vector: FloatArray,
        *,
        depth: int | None = None,
        exact: bool | None = None,
    ) -> FactorRanking:
        """(a) Vanilla Dense Similarity: cosine to each shard's summary embedding."""
        depth = self.candidate_depth if depth is None else depth
        use_exact = self.exact_dense if exact is None else exact
        hnsw = self.hnsw
        hits = (
            hnsw.exact_neighbours(question_vector, depth)
            if use_exact
            else hnsw.neighbours(question_vector, depth)
        )
        return FactorRanking(
            name=DENSE,
            weight=self.rrf_config.dense_weight,
            ranked=hits.shard_ids,
            scores=hits.as_dict(),
            applied=True,
            reason="exact cosine scan" if use_exact else "HNSW approximate cosine",
        )

    def sparse_factor(
        self, query_terms: Mapping[str, float], *, depth: int | None = None
    ) -> FactorRanking:
        """(c) Vanilla Sparse Similarity: TF-IDF cosine over question and shard keywords."""
        depth = self.candidate_depth if depth is None else depth
        hits = self.keyword_matrix.score(query_terms, top_n=depth)
        applied = bool(query_terms)
        return FactorRanking(
            name=SPARSE,
            weight=self.rrf_config.lexical_weight,
            ranked=hits.shard_ids,
            scores=hits.as_dict(),
            applied=applied,
            reason="applied" if applied else "question has no content keywords",
        )

    def entity_factor(
        self, query_entities: Mapping[str, float], *, depth: int | None = None
    ) -> FactorRanking:
        """(d) Entity Sparse Similarity — computed only when the question has entities.

        With no entities the factor is *not* run and contributes no ranked list to the
        fusion, which is what the post's "If the question contains one or more entities"
        requires. A zero-filled list would not be equivalent: RRF ranks, so a list of
        equal zeros still injects an arbitrary ordering into the fused score.
        """
        depth = self.candidate_depth if depth is None else depth
        if not query_entities:
            return FactorRanking(
                name=ENTITY,
                weight=self.rrf_config.entity_weight,
                ranked=(),
                scores={},
                applied=False,
                reason="question contains no entities",
            )
        hits = self.entity_matrix.score(query_entities, top_n=depth)
        return FactorRanking(
            name=ENTITY,
            weight=self.rrf_config.entity_weight,
            ranked=hits.shard_ids,
            scores=hits.as_dict(),
            applied=True,
            reason="applied",
        )

    # ------------------------------------------------------------------ #
    # The stage                                                           #
    # ------------------------------------------------------------------ #

    def select(
        self,
        question: str,
        *,
        top_n: int | None = None,
        candidate_depth: int | None = None,
        question_vector: FloatArray | None = None,
        query_terms: Mapping[str, float] | None = None,
        query_entities: Mapping[str, float] | None = None,
        exact_dense: bool | None = None,
    ) -> ShardSelection:
        """Rank the shards for ``question`` and return the ids best-first.

        Parameters
        ----------
        question:
            The user question, already language-detected and translated by stage 1.
        top_n:
            Cap on the returned shard ids. ``None`` returns every fused candidate.
        candidate_depth:
            How many shards each factor may contribute. Defaults to the selector's.
        question_vector:
            A pre-computed query embedding — stage 4 has already produced one, and
            passing it avoids a second encoder call.
        query_terms, query_entities:
            Overrides for the question's keywords and entities, for callers that
            tokenise or recognise entities themselves.
        exact_dense:
            Force the dense factor onto (or off) the exact scan for this call.
        """
        depth = self.candidate_depth if candidate_depth is None else candidate_depth
        if depth < 1:
            msg = f"candidate_depth must be >= 1, got {depth}"
            raise ValueError(msg)
        if top_n is not None and top_n < 1:
            msg = f"top_n must be >= 1, got {top_n}"
            raise ValueError(msg)

        terms = dict(query_terms) if query_terms is not None else self.question_terms(question)
        entities = (
            dict(query_entities)
            if query_entities is not None
            else self.question_entities(question)
        )
        vector = self.embed_question(question) if question_vector is None else question_vector
        use_exact = self.exact_dense if exact_dense is None else exact_dense

        dense = self.dense_factor(vector, depth=depth, exact=use_exact)
        sparse = self.sparse_factor(terms, depth=depth)
        entity = self.entity_factor(entities, depth=depth)

        ranked_lists = [
            factor.as_ranked_list()
            for factor in (dense, sparse, entity)
            if factor.applied and factor.ranked
        ]
        fused = rrf_fuse(ranked_lists, k=self.rrf_config.k)

        shard_ids = [int(item.id) for item in fused]
        if top_n is not None:
            shard_ids = shard_ids[:top_n]

        signals = ShardSelectionSignals(
            question=question,
            query_terms=terms,
            query_entities=entities,
            factors={DENSE: dense, SPARSE: sparse, ENTITY: entity},
            fused=fused,
            candidate_depth=depth,
            exact_dense=use_exact,
            omitted_factors=dict(_OMITTED),
        )
        return ShardSelection(shard_ids=shard_ids, signals=signals)

    # ------------------------------------------------------------------ #
    # Measurement                                                         #
    # ------------------------------------------------------------------ #

    def dense_recall(self, questions: Sequence[str], k: int) -> RecallReport:
        """Recall@k of the HNSW dense factor against an exact cosine scan.

        Approximate search buys sublinear query time with recall, and this reports how
        much: the fraction of the exact top-k shards the graph actually returned.
        """
        vectors = self.provider.embed_queries(list(questions))
        return self.hnsw.recall_report(vectors, k)
