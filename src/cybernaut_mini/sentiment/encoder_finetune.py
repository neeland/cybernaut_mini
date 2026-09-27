"""Domain-gap diagnosis + index-bootstrapped contrastive encoder fine-tune.

Three pieces: the beat/missed cosine demo showing general-corpus encoders
cannot separate opposite financial outcomes; triplet mining from the dedup
index (anchor + hard positive from the same apex cluster, hard negative = the
highest-cosine snippet containing the opposite polarity word); and a
MultipleNegativesRankingLoss fine-tune scaffold that trains MiniLM for a few
epochs and re-measures the beat/missed gap.

Blog ref: https://nosible.com/blog/using-vector-search-to-see-signals-in-company-news —
    "off-the-shelf embedding models are not discriminative enough ... 'The
    company beat analyst estimates.' / 'The company missed analyst
    estimates.' To a human investor these two sentences have very different
    meanings ... to off the shelf embedding models trained on general corpora
    these sentences look almost exactly the same", and the next step:
    "we can use our index to bootstrap a high-quality dataset that we can use
    to finetune nuanced encoders optimized for financial documents." Local
    copy: ``docs/blog-archive/using-vector-search-to-see-signals-in-company-news.md``.

Assumptions:
    - The demo sentences are the post's pair verbatim (with the post's
      sentence-final periods); under all-MiniLM-L6-v2 their cosine is ~0.9,
      which the gated test asserts drops after fine-tuning.
    - Clusters arrive as row-index lists with the APEX FIRST —
      :func:`clusters_from_events` adapts :class:`cybernaut_mini.dedup.EventCluster`
      records (WS2's output contract) into that shape, so this module never
      re-implements dedup and tests can hand-build clusters.
    - "The opposite polarity word" generalizes the post's beat/missed example
      through :data:`POLARITY_PAIRS`, seeded with the polarity vocabulary of
      the post's own triplet terms (beat/missed, grew/fell, strong/weak,
      increases/decreases); callers can pass their own pairs.
    - Negative mining is exact numpy cosine over L2-normalized embeddings —
      corpora at laptop scale need no ANN, matching how ``dedup`` shards.
    - The fine-tune uses sentence-transformers' ``InputExample`` +
      ``model.fit`` with ``MultipleNegativesRankingLoss`` (in-batch negatives
      plus our mined hard negative as the third text); it downloads weights
      and is opt-in behind the sentiment download gate, and runs only a few
      epochs — a scaffold proving the loop, not a production encoder.

Alternatives rejected:
    - TripletLoss: MNRL is the standard choice for (anchor, positive,
      hard-negative) triplets with in-batch negatives, needs no margin
      tuning, and is what the gap matrix prescribes.
    - Mining negatives by label (e.g. opposite sentiment class): polarity
      WORDS inside near-identical sentences are precisely the hard negatives
      the post's demo motivates; class-level negatives are mostly easy.
    - Fine-tuning the repo's production embedder: the demo's point is fixing
      a general-corpus encoder; the repo's retrieval embedder stays frozen
      (the WORLD pillar's foreknowledge-bias argument).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from cybernaut_mini.config import ConfigError
from cybernaut_mini.dedup import EventCluster

__all__ = [
    "BEAT_SENTENCE",
    "MISSED_SENTENCE",
    "POLARITY_PAIRS",
    "Triplet",
    "beat_missed_cosine",
    "clusters_from_events",
    "cosine",
    "finetune_encoder",
    "mine_triplets",
    "opposite_terms",
]

FloatArray = npt.NDArray[np.float64]

#: The post's demo pair, verbatim.
BEAT_SENTENCE = "The company beat analyst estimates."
MISSED_SENTENCE = "The company missed analyst estimates."

#: Opposite-polarity word pairs, from the post's own beat/missed example and
#: the polarity vocabulary of its triplet terms (Grew/Fell, Strong/Weak,
#: Increases/Decreases, Beats/Misses).
POLARITY_PAIRS: tuple[tuple[str, str], ...] = (
    ("beat", "missed"),
    ("beats", "misses"),
    ("grew", "fell"),
    ("strong", "weak"),
    ("increases", "decreases"),
)


def cosine(u: npt.ArrayLike, v: npt.ArrayLike) -> float:
    """Plain cosine similarity between two vectors."""
    a = np.asarray(u, dtype=np.float64)
    b = np.asarray(v, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        msg = "cosine undefined for a zero vector"
        raise ConfigError(msg)
    return float(np.dot(a, b) / denom)


def beat_missed_cosine(encode: Callable[[Sequence[str]], npt.ArrayLike]) -> float:
    """The domain-gap demo: how similar does *encode* think beat and missed are?"""
    vectors = np.asarray(encode([BEAT_SENTENCE, MISSED_SENTENCE]), dtype=np.float64)
    return cosine(vectors[0], vectors[1])


@dataclass(frozen=True)
class Triplet:
    """One mined training example: texts plus their corpus row indices."""

    anchor: str
    positive: str
    negative: str
    anchor_index: int
    positive_index: int
    negative_index: int


def opposite_terms(
    text: str, pairs: Sequence[tuple[str, str]] = POLARITY_PAIRS
) -> set[str]:
    """The opposite polarity words for every polarity word present in *text*."""
    tokens = set(text.casefold().split())
    tokens = {token.strip(".,;:!?'\"()") for token in tokens}
    opposites: set[str] = set()
    for left, right in pairs:
        if left in tokens:
            opposites.add(right)
        if right in tokens:
            opposites.add(left)
    return opposites


def _l2_normalize(matrix: FloatArray) -> FloatArray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.asarray(matrix / norms, dtype=np.float64)


def mine_triplets(
    texts: Sequence[str],
    embeddings: npt.ArrayLike,
    clusters: Sequence[Sequence[int]],
    *,
    polarity_pairs: Sequence[tuple[str, str]] = POLARITY_PAIRS,
    max_triplets: int | None = None,
) -> list[Triplet]:
    """Bootstrap (anchor, hard positive, hard negative) triplets from the index.

    Per cluster (apex first, >=2 members): anchor = apex text, positive = the
    member closest to the anchor by cosine, negative = the highest-cosine row
    OUTSIDE the cluster whose text contains an opposite polarity word for a
    polarity word in the anchor. Clusters whose anchor has no polarity word,
    or with no opposite-word candidates, yield nothing — never a made-up
    negative.
    """
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        msg = f"embeddings {matrix.shape} do not align with {len(texts)} texts"
        raise ConfigError(msg)
    normalized = _l2_normalize(matrix)

    lowered = [text.casefold() for text in texts]
    triplets: list[Triplet] = []
    for cluster in clusters:
        members = [int(index) for index in cluster]
        if len(members) < 2:
            continue
        if any(index < 0 or index >= len(texts) for index in members):
            msg = f"cluster {members} indexes outside the corpus of {len(texts)} rows"
            raise ConfigError(msg)
        anchor_index = members[0]
        opposites = opposite_terms(texts[anchor_index], polarity_pairs)
        if not opposites:
            continue

        anchor_vec = normalized[anchor_index]
        others = members[1:]
        similarity = normalized[others] @ anchor_vec
        positive_index = others[int(np.argmax(similarity))]

        member_set = set(members)
        candidates = [
            index
            for index in range(len(texts))
            if index not in member_set
            and any(opposite in lowered[index] for opposite in opposites)
        ]
        if not candidates:
            continue
        candidate_sims = normalized[candidates] @ anchor_vec
        negative_index = candidates[int(np.argmax(candidate_sims))]

        triplets.append(
            Triplet(
                anchor=texts[anchor_index],
                positive=texts[positive_index],
                negative=texts[negative_index],
                anchor_index=anchor_index,
                positive_index=positive_index,
                negative_index=negative_index,
            )
        )
        if max_triplets is not None and len(triplets) >= max_triplets:
            break
    return triplets


def clusters_from_events(
    events: Sequence[EventCluster], doc_ids: Sequence[str]
) -> list[list[int]]:
    """Adapt WS2's dedup clusters into apex-first row-index lists.

    *doc_ids* gives the row order of the text/embedding arrays. Members
    missing from *doc_ids* are skipped; clusters reduced below two members
    contribute nothing.
    """
    positions = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    clusters: list[list[int]] = []
    for event in events:
        if event.apex_doc_id not in positions:
            continue
        rows = [positions[event.apex_doc_id]]
        rows.extend(
            positions[doc_id]
            for doc_id in event.member_doc_ids
            if doc_id != event.apex_doc_id and doc_id in positions
        )
        if len(rows) >= 2:
            clusters.append(rows)
    return clusters


def finetune_encoder(
    triplets: Sequence[Triplet],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    revision: str | None = None,
    epochs: int = 1,
    batch_size: int = 16,
) -> Any:
    """MultipleNegativesRankingLoss fine-tune scaffold (opt-in download).

    Returns the fine-tuned model so the caller can re-run
    :func:`beat_missed_cosine` over ``model.encode`` and watch the gap open.
    A few epochs over mined triplets is the whole point — proving the
    index-bootstrapped loop, not shipping an encoder.
    """
    if not triplets:
        msg = "cannot fine-tune without triplets; mine some first"
        raise ConfigError(msg)
    try:
        from sentence_transformers import (  # type: ignore[attr-defined]
            InputExample,
            SentenceTransformer,
            losses,
        )
        from torch.utils.data import DataLoader
    except ImportError as exc:
        msg = (
            "the encoder fine-tune needs the optional 'st' extra "
            "(sentence-transformers + torch); install it with `uv sync --extra st`."
        )
        raise ConfigError(msg) from exc

    model = SentenceTransformer(
        model_name_or_path=model_name, revision=revision, trust_remote_code=True
    )
    examples = [
        InputExample(texts=[triplet.anchor, triplet.positive, triplet.negative])
        for triplet in triplets
    ]
    loader: Any = DataLoader(examples, shuffle=True, batch_size=batch_size)  # type: ignore[arg-type]
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs, show_progress_bar=False)
    return model
