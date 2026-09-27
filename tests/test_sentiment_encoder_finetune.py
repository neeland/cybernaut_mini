"""Encoder fine-tune tests: the beat/missed demo, triplet mining, cluster adapters.

Everything offline is pure math over synthetic vectors plus real text — the
post's verbatim beat/missed sentence pair and real fixture headlines from
``data/01_raw/fixtures/documents.jsonl``. The actual
MultipleNegativesRankingLoss fine-tune downloads MiniLM weights and is opt-in
behind ``CYBERNAUT_MINI_SENTIMENT_DOWNLOADS=1``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from cybernaut_mini.config import ConfigError
from cybernaut_mini.dedup import EventCluster
from cybernaut_mini.sentiment.data import DOWNLOAD_ENV, load_fixture_stories
from cybernaut_mini.sentiment.encoder_finetune import (
    BEAT_SENTENCE,
    MISSED_SENTENCE,
    POLARITY_PAIRS,
    beat_missed_cosine,
    clusters_from_events,
    cosine,
    finetune_encoder,
    mine_triplets,
    opposite_terms,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DOCUMENTS = REPO_ROOT / "data" / "01_raw" / "fixtures" / "documents.jsonl"

_DOWNLOADS = bool(os.environ.get(DOWNLOAD_ENV, "").strip())


@pytest.fixture(scope="module")
def distractors() -> list[str]:
    """Real fixture headlines carrying no polarity-pair words."""
    titles = load_fixture_stories(FIXTURE_DOCUMENTS, max_rows=30)
    clean = [title for title in titles if not opposite_terms(title)]
    assert len(clean) >= 2
    return clean[:2]


# ── the domain-gap demo ──────────────────────────────────────────────────────


def test_demo_sentences_are_the_post_pair_verbatim() -> None:
    assert BEAT_SENTENCE == "The company beat analyst estimates."
    assert MISSED_SENTENCE == "The company missed analyst estimates."


def test_cosine_matches_hand_math_and_rejects_zero_vectors() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 2.0]) == pytest.approx(0.0)
    assert cosine([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1.0 / np.sqrt(2.0))
    with pytest.raises(ConfigError):
        cosine([0.0, 0.0], [1.0, 0.0])


def test_beat_missed_cosine_encodes_the_verbatim_pair() -> None:
    seen: list[list[str]] = []

    def encode(texts: list[str]) -> np.ndarray:
        seen.append(list(texts))
        return np.array([[1.0, 0.0], [1.0, 0.1]])

    value = beat_missed_cosine(encode)
    assert seen == [[BEAT_SENTENCE, MISSED_SENTENCE]]
    assert value == pytest.approx(cosine([1.0, 0.0], [1.0, 0.1]))


# ── polarity vocabulary ──────────────────────────────────────────────────────


def test_polarity_pairs_seed_with_the_post_terms() -> None:
    assert ("beat", "missed") in POLARITY_PAIRS
    assert ("grew", "fell") in POLARITY_PAIRS
    assert ("strong", "weak") in POLARITY_PAIRS
    assert ("increases", "decreases") in POLARITY_PAIRS


def test_opposite_terms_work_both_ways_and_strip_punctuation() -> None:
    assert opposite_terms(BEAT_SENTENCE) == {"missed"}
    assert opposite_terms(MISSED_SENTENCE) == {"beat"}
    assert opposite_terms("Sales grew, margins strong.") == {"fell", "weak"}
    assert opposite_terms("Nothing polar here.") == set()


# ── triplet mining ───────────────────────────────────────────────────────────


def _corpus(distractors: list[str]) -> tuple[list[str], np.ndarray]:
    """Anchor + near-dup cluster, an opposite-polarity row, real distractors."""
    texts = [BEAT_SENTENCE, BEAT_SENTENCE, MISSED_SENTENCE, *distractors]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # anchor (apex)
            [0.99, 0.1, 0.0],  # near-duplicate cluster member
            [0.9, 0.2, 0.0],  # the opposite-polarity hard negative
            [0.0, 1.0, 0.0],  # distractor
            [0.0, 0.0, 1.0],  # distractor
        ]
    )
    return texts, embeddings


def test_mine_triplets_builds_anchor_positive_hard_negative(distractors: list[str]) -> None:
    texts, embeddings = _corpus(distractors)
    (triplet,) = mine_triplets(texts, embeddings, clusters=[[0, 1]])
    assert triplet.anchor == BEAT_SENTENCE
    assert triplet.positive == BEAT_SENTENCE
    assert (triplet.anchor_index, triplet.positive_index) == (0, 1)
    # The negative is the highest-cosine OUT-of-cluster row containing "missed".
    assert triplet.negative == MISSED_SENTENCE
    assert triplet.negative_index == 2


def test_mine_triplets_skips_unusable_clusters(distractors: list[str]) -> None:
    texts, embeddings = _corpus(distractors)
    # Singleton cluster, polarity-free anchor, and a cluster whose opposite
    # word appears nowhere outside it: all yield nothing, never a made-up row.
    assert mine_triplets(texts, embeddings, clusters=[[0]]) == []
    assert mine_triplets(texts, embeddings, clusters=[[3, 4]]) == []
    everything = [[0, 1, 2]]  # the only "missed" row is inside the cluster
    assert mine_triplets(texts, embeddings, clusters=everything) == []


def test_mine_triplets_honours_max_triplets(distractors: list[str]) -> None:
    texts, embeddings = _corpus(distractors)
    clusters = [[0, 1], [2, 1]]  # the second would mine a beat-negative triplet
    assert len(mine_triplets(texts, embeddings, clusters, max_triplets=1)) == 1
    assert len(mine_triplets(texts, embeddings, clusters)) == 2


def test_mine_triplets_rejects_bad_shapes_and_indices(distractors: list[str]) -> None:
    texts, embeddings = _corpus(distractors)
    with pytest.raises(ConfigError):
        mine_triplets(texts, embeddings[:-1], clusters=[[0, 1]])
    with pytest.raises(ConfigError):
        mine_triplets(texts, embeddings, clusters=[[0, 99]])


# ── the dedup adapter ────────────────────────────────────────────────────────


def _event(apex: str, members: list[str]) -> EventCluster:
    return EventCluster(
        event_id="evt-0000000000000000",
        apex_doc_id=apex,
        member_doc_ids=sorted(members),
        total_netlocs=0,
    )


def test_clusters_from_events_puts_the_apex_first() -> None:
    doc_ids = ["doc-a", "doc-b", "doc-c"]
    events = [_event("doc-b", ["doc-a", "doc-b", "doc-c"])]
    assert clusters_from_events(events, doc_ids) == [[1, 0, 2]]


def test_clusters_from_events_drops_missing_and_small_clusters() -> None:
    doc_ids = ["doc-a", "doc-b"]
    events = [
        _event("doc-z", ["doc-a", "doc-z"]),  # apex not in the corpus
        _event("doc-a", ["doc-a", "doc-x"]),  # reduced below two members
        _event("doc-a", ["doc-a", "doc-b"]),
    ]
    assert clusters_from_events(events, doc_ids) == [[0, 1]]


# ── the fine-tune scaffold ───────────────────────────────────────────────────


def test_finetune_encoder_requires_triplets() -> None:
    with pytest.raises(ConfigError):
        finetune_encoder([])


@pytest.mark.skipif(not _DOWNLOADS, reason=f"{DOWNLOAD_ENV} not set; downloads MiniLM")
def test_finetune_closes_the_beat_missed_gap(distractors: list[str]) -> None:
    """The post's arc end to end: ~0.9 cosine before, measurably lower after."""
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    base = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    before = beat_missed_cosine(base.encode)
    assert before > 0.8  # the domain gap the post demonstrates

    texts, _ = _corpus(distractors)
    embeddings = np.asarray(base.encode(texts), dtype=np.float64)
    triplets = mine_triplets(texts, embeddings, clusters=[[0, 1]])
    model = finetune_encoder(triplets * 8, epochs=1, batch_size=8)
    after = beat_missed_cosine(model.encode)
    assert after < before
