"""Tests for stage 5 — Shard Selection (``cybernaut_mini.query.s5_select``).

The synthetic index below is built by hand rather than through ``write_index`` because
stage 5 reads *only* shard manifests. Building it in memory lets one test hand the
selector a :class:`LoadedIndex` whose documents and tokens explode on any access, which
is the only way to prove the stage never materialises the corpus.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from cybernaut_mini.config import RRFConfig
from cybernaut_mini.indexing import LoadedIndex
from cybernaut_mini.models import Document, IndexMeta, ShardEntity, ShardKeyword, ShardManifest
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.query.s5_select import (
    BAYESIAN_DENSE,
    DENSE,
    ENTITY,
    SPARSE,
    HnswCacheError,
    HnswShardIndex,
    ShardSelector,
    SparseShardMatrix,
    summary_vectors,
    term_frequencies,
)
from cybernaut_mini.text import TextProcessor

if TYPE_CHECKING:
    from cybernaut_mini.providers.embeddings import FloatArray

# --------------------------------------------------------------------------- #
# A synthetic multi-shard index                                               #
# --------------------------------------------------------------------------- #

_CONSONANTS = "bdfgklmnprstvz"
_VOWELS = "aeiou"

#: A term every shard carries, so the sparse matrix is not block-diagonal and IDF has
#: something to discount. Without it the keyword test would pass on a degenerate matrix.
_SHARED_TERM = "research"
_SHARED_ENTITY = "global consortium"


def _pseudo_word(n: int) -> str:
    """A distinct six-letter pseudo-word per ``n``, injective for n < 686,000.

    Real English nouns would collide on character trigrams, which the hash embedder
    consumes, and a topic test that fails because "batteries" looks like "bacteria" is
    testing the fixture rather than the selector. The alphabet is large enough that a
    shard-selection fixture can be scaled past the post's 250,000 shards without two
    shards ever ending up with the same topic words — which would make "did the right
    shard win?" unanswerable rather than merely hard.
    """
    letters: list[str] = []
    remainder = n
    for alphabet in (_CONSONANTS, _VOWELS, _CONSONANTS, _VOWELS, _CONSONANTS, _VOWELS):
        letters.append(alphabet[remainder % len(alphabet)])
        remainder //= len(alphabet)
    return "".join(letters)


def shard_words(shard_id: int) -> tuple[str, str, str]:
    """The three topic words that belong to ``shard_id`` and to no other shard."""
    return (
        _pseudo_word(shard_id * 3),
        _pseudo_word(shard_id * 3 + 1),
        _pseudo_word(shard_id * 3 + 2),
    )


def shard_entity(shard_id: int) -> str:
    return f"entity {_pseudo_word(shard_id * 3)}"


def build_synthetic_index(
    n_shards: int,
    embedder: HashEmbedder,
    *,
    with_entities: bool = True,
    documents_explode: bool = False,
) -> LoadedIndex:
    """A LoadedIndex of ``n_shards`` topically disjoint shards, one document each."""
    documents: list[Document] = []
    summaries: list[str] = []
    manifests: dict[int, ShardManifest] = {}

    for shard_id in range(n_shards):
        first, second, third = shard_words(shard_id)
        summary = f"{first} {second} {third} {_SHARED_TERM} overview"
        doc_id = f"doc-{shard_id:05d}"
        documents.append(
            Document(
                id=doc_id,
                title=f"{first} {second}",
                text=f"{summary}. A study of {first} and {second} and {third}.",
                language="en",
            )
        )
        summaries.append(summary)
        entities = (
            [
                ShardEntity(text=shard_entity(shard_id), count=5),
                ShardEntity(text=_SHARED_ENTITY, count=1),
            ]
            if with_entities
            else []
        )
        manifests[shard_id] = ShardManifest(
            shard_id=shard_id,
            document_ids=[doc_id],
            centroid=[0.0] * embedder.dim,
            title=f"{first}, {second}, {third}",
            summary=summary,
            keywords=[
                ShardKeyword(term=first, weight=1.0),
                ShardKeyword(term=second, weight=0.7),
                ShardKeyword(term=third, weight=0.4),
                ShardKeyword(term=_SHARED_TERM, weight=0.05),
            ],
            entities=entities,
            term_graph={},
            document_count=1,
            embedding_model=embedder.identifier,
        )

    vectors = embedder.embed_documents([f"{doc.title}\n{doc.text}" for doc in documents])
    meta = IndexMeta(
        embedding_model=embedder.identifier,
        embedding_dim=embedder.dim,
        n_shards=n_shards,
        n_documents=len(documents),
        seed=7,
    )
    if documents_explode:
        return LoadedIndex(
            meta=meta,
            documents=_ExplodingSequence(),
            vectors=np.zeros((0, embedder.dim), dtype=np.float32),
            row_map=_ExplodingMapping(),
            manifests=manifests,
            doc_tokens=_ExplodingMapping(),
            by_id=_ExplodingMapping(),
        )
    return LoadedIndex(
        meta=meta,
        documents=documents,
        vectors=vectors,
        row_map={doc.id: row for row, doc in enumerate(documents)},
        manifests=manifests,
        doc_tokens={doc.id: doc.text.lower().split() for doc in documents},
    )


class _Exploded(AssertionError):
    """Raised when stage 5 touches something it promised not to touch."""


class _ExplodingSequence(Sequence[Document]):
    def __getitem__(self, index: object) -> Document:  # type: ignore[override]
        msg = "stage 5 read index.documents"
        raise _Exploded(msg)

    def __len__(self) -> int:
        msg = "stage 5 took len(index.documents)"
        raise _Exploded(msg)

    def __iter__(self) -> Iterator[Document]:
        msg = "stage 5 iterated index.documents"
        raise _Exploded(msg)


class _ExplodingMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        msg = f"stage 5 read a per-document mapping for {key!r}"
        raise _Exploded(msg)

    def __iter__(self) -> Iterator[str]:
        msg = "stage 5 iterated a per-document mapping"
        raise _Exploded(msg)

    def __len__(self) -> int:
        msg = "stage 5 took the length of a per-document mapping"
        raise _Exploded(msg)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

SMALL_SHARDS = 12
LARGE_SHARDS = 220


@pytest.fixture(scope="module")
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=64)


@pytest.fixture(scope="module")
def processor() -> TextProcessor:
    return TextProcessor(use_spacy=False)


@pytest.fixture(scope="module")
def small_index(embedder: HashEmbedder) -> LoadedIndex:
    return build_synthetic_index(SMALL_SHARDS, embedder)


@pytest.fixture(scope="module")
def large_index(embedder: HashEmbedder) -> LoadedIndex:
    return build_synthetic_index(LARGE_SHARDS, embedder)


@pytest.fixture(scope="module")
def small_selector(
    small_index: LoadedIndex, embedder: HashEmbedder, processor: TextProcessor
) -> ShardSelector:
    return ShardSelector(small_index, provider=embedder, processor=processor)


@pytest.fixture(scope="module")
def large_selector(
    large_index: LoadedIndex, embedder: HashEmbedder, processor: TextProcessor
) -> ShardSelector:
    return ShardSelector(large_index, provider=embedder, processor=processor)


def question_for(shard_id: int) -> str:
    first, second, _third = shard_words(shard_id)
    return f"what is known about {first} and {second}?"


# --------------------------------------------------------------------------- #
# Factor (a): Vanilla Dense Similarity over HNSW                               #
# --------------------------------------------------------------------------- #


def test_summary_vectors_are_normalised_and_row_aligned(
    small_index: LoadedIndex, embedder: HashEmbedder
) -> None:
    ids, vectors = summary_vectors(small_index.manifests, embedder)
    assert ids == tuple(range(SMALL_SHARDS))
    assert vectors.shape == (SMALL_SHARDS, embedder.dim)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    # Row i really is shard i's summary, not some other shard's.
    direct = embedder.embed_documents([small_index.manifests[3].summary])[0]
    np.testing.assert_allclose(vectors[3], direct, atol=1e-6)


def test_dense_factor_finds_the_shard_whose_summary_matches(
    small_selector: ShardSelector, embedder: HashEmbedder
) -> None:
    for shard_id in range(SMALL_SHARDS):
        vector = embedder.embed_queries([question_for(shard_id)])[0]
        factor = small_selector.dense_factor(vector, depth=3)
        assert factor.applied
        assert factor.ranked[0] == shard_id, f"dense missed shard {shard_id}"
        assert factor.scores[shard_id] == max(factor.scores.values())


def test_dense_factor_scores_are_cosines_in_range(
    small_selector: ShardSelector, embedder: HashEmbedder
) -> None:
    vector = embedder.embed_queries([question_for(5)])[0]
    factor = small_selector.dense_factor(vector, depth=SMALL_SHARDS)
    assert len(factor.ranked) == SMALL_SHARDS
    for score in factor.scores.values():
        assert -1.0001 <= score <= 1.0001
    # Ranked strictly by descending score.
    ordered = [factor.scores[shard_id] for shard_id in factor.ranked]
    assert ordered == sorted(ordered, reverse=True)


def test_hnsw_construction_is_deterministic(
    small_index: LoadedIndex, embedder: HashEmbedder
) -> None:
    first = HnswShardIndex.from_manifests(small_index.manifests, embedder)
    second = HnswShardIndex.from_manifests(small_index.manifests, embedder)
    for shard_id in range(SMALL_SHARDS):
        query = embedder.embed_queries([question_for(shard_id)])[0]
        assert first.neighbours(query, 5) == second.neighbours(query, 5)


def test_hnsw_agrees_with_exact_cosine_on_a_small_index(
    small_selector: ShardSelector, embedder: HashEmbedder
) -> None:
    hnsw = small_selector.hnsw
    for shard_id in range(SMALL_SHARDS):
        query = embedder.embed_queries([question_for(shard_id)])[0]
        approximate = hnsw.neighbours(query, 5)
        exact = hnsw.exact_neighbours(query, 5)
        assert approximate.shard_ids == exact.shard_ids
        np.testing.assert_allclose(approximate.scores, exact.scores, atol=1e-6)


def test_exact_neighbours_matches_a_hand_rolled_scan(
    small_selector: ShardSelector, embedder: HashEmbedder
) -> None:
    hnsw = small_selector.hnsw
    query = embedder.embed_queries([question_for(9)])[0]
    query = query / np.linalg.norm(query)
    manual = sorted(
        ((float(hnsw.vectors[row] @ query), shard) for row, shard in enumerate(hnsw.shard_ids)),
        key=lambda pair: (-pair[0], pair[1]),
    )[:4]
    exact = hnsw.exact_neighbours(query, 4)
    assert exact.shard_ids == tuple(shard for _score, shard in manual)
    np.testing.assert_allclose(exact.scores, [score for score, _ in manual], atol=1e-6)


def test_hnsw_recall_at_10_over_220_shards(
    large_selector: ShardSelector,
) -> None:
    """The measured price of approximate search, pinned as a floor.

    Approximate search buys sublinear query time with recall, and the size of that trade
    is a fact worth writing down. Measured on this fixture with USearch's default
    connectivity=16 / expansion_add=128 / expansion_search=64:

    ======  =========  ==========  ==========
    shards  queries    recall@10   recall@100
    ======  =========  ==========  ==========
       220         44      1.0000      0.9914
     2,000         40      0.9850      0.9625
    20,000         40      0.9875      0.9203
    ======  =========  ==========  ==========

    So the dense factor loses roughly 1% of the exact top-10 and up to 8% of the exact
    top-100 at twenty thousand shards, while per-question selection stays at ~0.3-0.4 ms
    across all three scales — an exact scan at 20,000 shards is what that buys back.
    Only the 220-shard row is asserted here; the larger rows are measurements, not tests,
    because building a 20,000-shard graph costs ~3s and does not belong in the suite.

    The assertions are floors rather than equalities: a USearch upgrade may legitimately
    reshape the graph, but a drop below them means the dense factor started losing shards
    the exact scan would have found, which is a routing regression.
    """
    questions = [question_for(shard_id) for shard_id in range(0, LARGE_SHARDS, 5)]
    report = large_selector.dense_recall(questions, k=10)
    assert report.n_queries == len(questions)
    assert report.n_shards == LARGE_SHARDS
    assert report.k == 10
    assert report.recall >= 0.98, f"HNSW recall@10 regressed: {report}"
    assert min(report.overlaps) >= 9

    deep = large_selector.dense_recall(questions, k=100)
    assert deep.k == 100
    assert deep.recall >= 0.95, f"HNSW recall@100 regressed: {deep}"


def test_hnsw_cache_round_trips(
    small_index: LoadedIndex, embedder: HashEmbedder, tmp_path: Path
) -> None:
    built = HnswShardIndex.from_manifests(small_index.manifests, embedder)
    built.save(tmp_path, embedding_model=embedder.identifier)
    restored = HnswShardIndex.load(tmp_path, embedding_model=embedder.identifier)
    assert restored.shard_ids == built.shard_ids
    np.testing.assert_allclose(restored.vectors, built.vectors)
    query = embedder.embed_queries([question_for(2)])[0]
    assert restored.neighbours(query, 5) == built.neighbours(query, 5)


def test_hnsw_cache_rejects_a_different_embedding_model(
    small_index: LoadedIndex, embedder: HashEmbedder, tmp_path: Path
) -> None:
    HnswShardIndex.from_manifests(small_index.manifests, embedder).save(
        tmp_path, embedding_model=embedder.identifier
    )
    with pytest.raises(HnswCacheError, match="embedding model"):
        HnswShardIndex.load(tmp_path, embedding_model="hash-999")


def test_hnsw_cache_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(HnswCacheError, match="no persisted"):
        HnswShardIndex.load(tmp_path / "absent", embedding_model="hash-64")


def test_selector_reuses_a_saved_hnsw_cache(
    small_index: LoadedIndex, embedder: HashEmbedder, processor: TextProcessor, tmp_path: Path
) -> None:
    cache = tmp_path / "hnsw"
    first = ShardSelector(
        small_index, provider=embedder, processor=processor, hnsw_cache_dir=cache
    )
    first.build()
    assert (cache / "hnsw_fingerprint.json").exists()

    class _RefusingProvider:
        """Any call to this would mean the cache was ignored."""

        @property
        def identifier(self) -> str:
            return embedder.identifier

        @property
        def dim(self) -> int:
            return embedder.dim

        def embed_documents(self, texts: list[str]) -> FloatArray:
            msg = "cache was ignored: summaries were re-embedded"
            raise AssertionError(msg)

        def embed_queries(self, texts: list[str]) -> FloatArray:
            return embedder.embed_queries(texts)

    second = ShardSelector(
        small_index,
        provider=_RefusingProvider(),
        processor=processor,
        hnsw_cache_dir=cache,
    )
    assert second.hnsw.shard_ids == first.hnsw.shard_ids
    query = embedder.embed_queries([question_for(4)])[0]
    assert second.hnsw.neighbours(query, 3) == first.hnsw.neighbours(query, 3)


def test_hnsw_rejects_a_query_of_the_wrong_dimension(small_selector: ShardSelector) -> None:
    with pytest.raises(ValueError, match="dim"):
        small_selector.hnsw.neighbours(np.zeros(7, dtype=np.float32), 3)


def test_hnsw_rejects_mismatched_inputs(embedder: HashEmbedder) -> None:
    vectors = np.zeros((3, embedder.dim), dtype=np.float32)
    with pytest.raises(ValueError, match="shard ids but"):
        HnswShardIndex([0, 1], vectors)
    with pytest.raises(ValueError, match="unique"):
        HnswShardIndex([0, 1, 1], vectors)


def test_summary_vectors_rejects_an_empty_index(embedder: HashEmbedder) -> None:
    with pytest.raises(ValueError, match="zero shards"):
        summary_vectors({}, embedder)


# --------------------------------------------------------------------------- #
# Factor (c): Vanilla Sparse Similarity                                        #
# --------------------------------------------------------------------------- #


def test_sparse_matrix_rows_are_unit_length(small_index: LoadedIndex) -> None:
    matrix = SparseShardMatrix.from_keywords(small_index.manifests)
    assert matrix.shape == (SMALL_SHARDS, SMALL_SHARDS * 3 + 1)
    for shard_id in small_index.manifests:
        row = matrix.row(shard_id)
        norm = sum(value * value for value in row.values()) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-9)


def test_sparse_matrix_is_sparse(large_index: LoadedIndex) -> None:
    matrix = SparseShardMatrix.from_keywords(large_index.manifests)
    # Four keywords per shard out of a vocabulary of 3*n + 1.
    assert matrix.nnz == LARGE_SHARDS * 4
    assert matrix.density() < 0.01


def test_sparse_factor_selects_the_shard_holding_the_query_keywords(
    small_selector: ShardSelector,
) -> None:
    for shard_id in range(SMALL_SHARDS):
        terms = small_selector.question_terms(question_for(shard_id))
        factor = small_selector.sparse_factor(terms, depth=3)
        assert factor.applied
        assert factor.ranked[0] == shard_id, f"sparse missed shard {shard_id}"


def test_sparse_factor_only_scores_shards_sharing_a_term(
    large_selector: ShardSelector,
) -> None:
    """The whole point of the inverted index: most shards are never looked at."""
    first, second, _third = shard_words(17)
    terms = term_frequencies([first, second])
    factor = large_selector.sparse_factor(terms, depth=LARGE_SHARDS)
    assert factor.ranked == (17,)
    assert len(factor.ranked) < LARGE_SHARDS


def test_sparse_factor_shared_term_reaches_every_shard_but_scores_lower(
    small_selector: ShardSelector,
) -> None:
    specific, _second, _third = shard_words(6)
    shared_only = small_selector.sparse_factor(term_frequencies([_SHARED_TERM]), depth=SMALL_SHARDS)
    assert len(shared_only.ranked) == SMALL_SHARDS
    specific_hit = small_selector.sparse_factor(term_frequencies([specific]), depth=SMALL_SHARDS)
    # IDF: a term in every shard is worth far less than a term in exactly one.
    assert specific_hit.scores[6] > max(shared_only.scores.values())


def test_sparse_factor_is_inactive_without_content_keywords(
    small_selector: ShardSelector,
) -> None:
    terms = small_selector.question_terms("what is it about the and of")
    assert terms == {}
    factor = small_selector.sparse_factor(terms)
    assert not factor.applied
    assert factor.ranked == ()
    assert "no content keywords" in factor.reason


def test_sparse_query_vector_drops_out_of_vocabulary_terms(
    small_selector: ShardSelector,
) -> None:
    matrix = small_selector.keyword_matrix
    known, _second, _third = shard_words(1)
    with_noise = matrix.query_vector(term_frequencies([known, "qqqqzz", "wwwwvv"]))
    without_noise = matrix.query_vector(term_frequencies([known]))
    assert with_noise == without_noise
    assert matrix.query_vector(term_frequencies(["qqqqzz"])) == {}


def test_sparse_scores_are_cosines_bounded_by_one(small_selector: ShardSelector) -> None:
    first, second, third = shard_words(2)
    factor = small_selector.sparse_factor(term_frequencies([first, second, third]))
    for score in factor.scores.values():
        assert 0.0 < score <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# Factor (d): Entity Sparse Similarity                                         #
# --------------------------------------------------------------------------- #


def test_entity_factor_is_omitted_when_the_question_has_no_entities(
    small_selector: ShardSelector,
) -> None:
    factor = small_selector.entity_factor({})
    assert not factor.applied
    assert factor.ranked == ()
    assert factor.scores == {}
    assert "no entities" in factor.reason


def test_entity_factor_never_reaches_rrf_without_entities(
    small_selector: ShardSelector,
) -> None:
    selection = small_selector.select(question_for(3))
    assert selection.signals.factor(ENTITY).applied is False
    # No fused item may carry an entity contribution.
    for item in selection.signals.fused:
        assert ENTITY not in item.contributions
    assert selection.signals.applied_factors() == (DENSE, SPARSE)


def test_entity_factor_selects_the_shard_holding_the_entity(
    small_selector: ShardSelector,
) -> None:
    for shard_id in range(SMALL_SHARDS):
        factor = small_selector.entity_factor(term_frequencies([shard_entity(shard_id)]), depth=3)
        assert factor.applied
        assert factor.ranked[0] == shard_id
        assert factor.reason == "applied"


def test_entity_factor_discounts_an_entity_every_shard_shares(
    small_selector: ShardSelector,
) -> None:
    shared = small_selector.entity_factor(term_frequencies([_SHARED_ENTITY]), depth=SMALL_SHARDS)
    assert len(shared.ranked) == SMALL_SHARDS
    specific = small_selector.entity_factor(term_frequencies([shard_entity(8)]), depth=SMALL_SHARDS)
    assert specific.scores[8] > max(shared.scores.values())


def test_entity_factor_uses_an_injected_extractor(
    small_index: LoadedIndex, embedder: HashEmbedder, processor: TextProcessor
) -> None:
    """The default extractor is spaCy-backed and yields nothing on the regex path."""
    assert processor.entities(question_for(4)) == []

    def extractor(question: str) -> list[str]:
        return [shard_entity(4)] if "extract me" in question else []

    selector = ShardSelector(
        small_index, provider=embedder, processor=processor, entity_extractor=extractor
    )
    without = selector.select(question_for(4))
    assert without.signals.factor(ENTITY).applied is False

    with_entity = selector.select(question_for(4) + " extract me")
    entity_factor = with_entity.signals.factor(ENTITY)
    assert entity_factor.applied
    assert entity_factor.ranked[0] == 4
    assert with_entity.signals.query_entities == {shard_entity(4): 1.0}
    assert with_entity.signals.applied_factors() == (DENSE, SPARSE, ENTITY)


def test_entity_matrix_is_empty_when_no_shard_has_entities(
    embedder: HashEmbedder, processor: TextProcessor
) -> None:
    index = build_synthetic_index(6, embedder, with_entities=False)
    selector = ShardSelector(index, provider=embedder, processor=processor)
    factor = selector.entity_factor(term_frequencies(["anything at all"]))
    assert factor.applied
    assert factor.ranked == ()


# --------------------------------------------------------------------------- #
# RRF fusion                                                                   #
# --------------------------------------------------------------------------- #


def test_selection_is_deterministic(small_selector: ShardSelector) -> None:
    first = small_selector.select(question_for(7), top_n=5)
    second = small_selector.select(question_for(7), top_n=5)
    assert first.shard_ids == second.shard_ids
    assert [item.score for item in first.signals.fused] == [
        item.score for item in second.signals.fused
    ]


def test_a_fresh_selector_reproduces_the_same_selection(
    small_index: LoadedIndex, embedder: HashEmbedder, processor: TextProcessor
) -> None:
    baseline = ShardSelector(small_index, provider=embedder, processor=processor).select(
        question_for(11)
    )
    rebuilt = ShardSelector(small_index, provider=embedder, processor=processor).select(
        question_for(11)
    )
    assert baseline.shard_ids == rebuilt.shard_ids


def test_fused_score_matches_the_weighted_rrf_formula(small_selector: ShardSelector) -> None:
    config = small_selector.rrf_config
    selection = small_selector.select(question_for(0))
    signals = selection.signals
    for item in selection.signals.fused:
        expected = 0.0
        for name in (DENSE, SPARSE, ENTITY):
            factor = signals.factor(name)
            if not factor.applied or not factor.ranked:
                continue
            if int(item.id) in factor.ranked:
                rank = factor.ranked.index(int(item.id)) + 1
                expected += factor.weight / (config.k + rank)
        assert item.score == pytest.approx(expected)


def test_rrf_weights_are_taken_from_the_config(
    small_index: LoadedIndex, embedder: HashEmbedder, processor: TextProcessor
) -> None:
    sparse_only = ShardSelector(
        small_index,
        provider=embedder,
        processor=processor,
        rrf_config=RRFConfig(dense_weight=0.0, lexical_weight=1.0, entity_weight=0.0),
    )
    selection = sparse_only.select(question_for(5))
    assert selection.signals.factor(DENSE).weight == 0.0
    # With no dense weight the top shard is whatever the sparse factor put first.
    assert selection.shard_ids[0] == selection.signals.factor(SPARSE).ranked[0]


def test_a_noisy_factor_does_not_capture_the_ranking(
    small_index: LoadedIndex, embedder: HashEmbedder, processor: TextProcessor
) -> None:
    """The post's stated reason for RRF: robustness when one factor is wrong."""
    selector = ShardSelector(small_index, provider=embedder, processor=processor)
    target = 6
    question = question_for(target)
    # A misleading entity pointing at a completely different shard.
    selection = selector.select(
        question, query_entities=term_frequencies([shard_entity(target + 3)])
    )
    assert selection.signals.factor(ENTITY).ranked[0] == target + 3
    assert selection.shard_ids[0] == target


# --------------------------------------------------------------------------- #
# The stage as a whole                                                         #
# --------------------------------------------------------------------------- #


def test_correct_shard_is_selected_for_every_topic(large_selector: ShardSelector) -> None:
    """Routing quality: the shard that actually holds the topic must come first."""
    misses: list[int] = []
    for shard_id in range(0, LARGE_SHARDS, 3):
        selection = large_selector.select(question_for(shard_id), top_n=5)
        if selection.shard_ids[0] != shard_id:
            misses.append(shard_id)
    assert misses == []


def test_correct_shard_survives_a_top_1_cap_at_scale(large_selector: ShardSelector) -> None:
    for shard_id in (0, 57, 118, 219):
        selection = large_selector.select(question_for(shard_id), top_n=1)
        assert selection.shard_ids == [shard_id]


def test_top_n_caps_the_result(large_selector: ShardSelector) -> None:
    uncapped = large_selector.select(question_for(40))
    capped = large_selector.select(question_for(40), top_n=7)
    assert len(capped.shard_ids) == 7
    assert capped.shard_ids == uncapped.shard_ids[:7]


def test_candidate_depth_bounds_every_factor(large_selector: ShardSelector) -> None:
    selection = large_selector.select(question_for(12), candidate_depth=4)
    assert selection.signals.candidate_depth == 4
    for name in (DENSE, SPARSE, ENTITY):
        assert len(selection.signals.factor(name).ranked) <= 4
    # Union of at most two applied factors of depth 4.
    assert len(selection.shard_ids) <= 8


def test_default_candidate_depth_limits_the_dense_factor(large_selector: ShardSelector) -> None:
    selection = large_selector.select(question_for(3))
    assert len(selection.signals.factor(DENSE).ranked) == 100
    assert len(selection.shard_ids) < LARGE_SHARDS


def test_selection_iterates_and_measures_like_its_shard_ids(
    small_selector: ShardSelector,
) -> None:
    selection = small_selector.select(question_for(2), top_n=3)
    assert list(selection) == selection.shard_ids
    assert len(selection) == 3


def test_precomputed_question_vector_is_used_verbatim(
    small_selector: ShardSelector, embedder: HashEmbedder
) -> None:
    """Stage 4 already embedded the question; stage 5 must accept that vector."""
    target = 9
    vector = embedder.embed_queries([question_for(target)])[0]
    # Ask about shard 1 lexically but hand over shard 9's embedding.
    selection = small_selector.select(question_for(1), question_vector=vector, top_n=SMALL_SHARDS)
    assert selection.signals.factor(DENSE).ranked[0] == target
    assert selection.signals.factor(SPARSE).ranked[0] == 1


def test_exact_dense_switch_is_recorded_and_agrees(small_selector: ShardSelector) -> None:
    approximate = small_selector.select(question_for(8), exact_dense=False)
    exact = small_selector.select(question_for(8), exact_dense=True)
    assert approximate.signals.exact_dense is False
    assert exact.signals.exact_dense is True
    assert "HNSW" in approximate.signals.factor(DENSE).reason
    assert "exact" in exact.signals.factor(DENSE).reason
    assert approximate.shard_ids == exact.shard_ids


def test_bayesian_dense_factor_is_declared_omitted(small_selector: ShardSelector) -> None:
    signals = small_selector.select(question_for(1)).signals
    assert BAYESIAN_DENSE not in signals.factors
    assert BAYESIAN_DENSE in signals.omitted_factors
    assert "patent" in signals.omitted_factors[BAYESIAN_DENSE]
    for item in signals.fused:
        assert BAYESIAN_DENSE not in item.contributions


def test_signals_expose_every_factor_even_when_inactive(small_selector: ShardSelector) -> None:
    signals = small_selector.select(question_for(4)).signals
    assert set(signals.factors) == {DENSE, SPARSE, ENTITY}
    for factor in signals.factors.values():
        assert factor.reason
        assert set(factor.scores) == set(factor.ranked)


def test_selection_never_reads_documents_or_tokens(
    embedder: HashEmbedder, processor: TextProcessor
) -> None:
    """A 1,000,000-document index must cost the same here as a 12-document one."""
    index = build_synthetic_index(SMALL_SHARDS, embedder, documents_explode=True)
    selector = ShardSelector(index, provider=embedder, processor=processor)
    selection = selector.select(question_for(5), top_n=3)
    assert selection.shard_ids[0] == 5


def test_invalid_arguments_are_rejected(
    small_selector: ShardSelector, small_index: LoadedIndex, embedder: HashEmbedder,
    processor: TextProcessor,
) -> None:
    with pytest.raises(ValueError, match="top_n"):
        small_selector.select(question_for(1), top_n=0)
    with pytest.raises(ValueError, match="candidate_depth"):
        small_selector.select(question_for(1), candidate_depth=0)
    with pytest.raises(ValueError, match="candidate_depth"):
        ShardSelector(small_index, provider=embedder, processor=processor, candidate_depth=0)


def test_build_materialises_all_three_structures(
    small_index: LoadedIndex, embedder: HashEmbedder, processor: TextProcessor
) -> None:
    selector = ShardSelector(small_index, provider=embedder, processor=processor)
    selector.build()
    assert len(selector.hnsw) == SMALL_SHARDS
    assert selector.keyword_matrix.shape[0] == SMALL_SHARDS
    assert selector.entity_matrix.shape[0] == SMALL_SHARDS


# --------------------------------------------------------------------------- #
# Against the real built index from conftest                                   #
# --------------------------------------------------------------------------- #


def test_selects_shards_from_a_real_built_index(
    built_index: LoadedIndex, processor: TextProcessor
) -> None:
    selector = ShardSelector(
        built_index, provider=HashEmbedder(dim=64), processor=processor, candidate_depth=4
    )
    selection = selector.select("solar panel efficiency record")
    valid = set(built_index.manifests)
    assert selection.shard_ids
    assert set(selection.shard_ids) <= valid
    assert selection.signals.factor(DENSE).applied
    assert selection.signals.factor(SPARSE).applied
