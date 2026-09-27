"""IPTC-Media-Topics-ish single-label classification and the exact geopolitical filter.

Every event gets exactly one topic path (``iptc_level_1..3``). The offline default
classifier is embedding-nearest-topic: embed the ontology rows' one-sentence
descriptions with the same frozen model as the events and take the argmax cosine —
single label, preserving the constraint the GPR post documents as a real
limitation ("a Gulf war is tagged conflict, never energy"). The geopolitical
filter is a transcription of the post's exact boolean over the three ontology
fields, compared by code equality, never by words in the labels.

Blog ref: https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world
    — ``geopolitical(e) is TRUE when ANY of these hold: iptc_level_1 == "conflict,
    war and peace" / iptc_level_2 == "international relations" / iptc_level_3 in
    {"war crime", "genocide", "terrorism", "nuclear policy"}`` and "We match the
    ontology codes exactly, never the words in the labels, so 'weather warning' or
    'tug-of-war' never leak in." Local copy:
    ``docs/blog-archive/rebuilding-the-geopolitical-risk-index-from-nosible-world.md``.

Assumptions:
    - The ontology subset lives in ``configs/world/iptc_topics.yaml``: the 17
      top-level IPTC Media Topics plus the level-2/3 rows the geopolitical filter
      needs, each with a one-sentence description this repo wrote as the embedding
      target. The topic *labels* are the exact strings the filter compares against.
    - Nearest-topic runs over the event's stored embedding (no re-embedding of
      text), so classification is a single ``(n_events, n_topics)`` matmul and is
      deterministic for a frozen embedder.
    - A leaf row assigns its whole path (level 1, 2 and 3); a level-1 row leaves
      the deeper levels ``None``. Single label means single row — the argmax —
      exactly one path per event.
    - The opt-in quality path (``load_transformer_classifier``) wraps the
      ``classla/multilingual-IPTC-news-topic-classifier`` checkpoint for level 1;
      it raises :class:`~cybernaut_mini.config.ConfigError` when transformers or
      the download is unavailable so the offline default stays the embedding path.

Alternatives rejected:
    - Multi-label classification with a score floor: strictly more useful (the
      post says so) but it would silently fix the single-label limitation the GPR
      reproduction exists to demonstrate — the trade-coercion and oil patches in
      :mod:`cybernaut_mini.world.indices.gpr` are the posts' own fix, built on
      keeping this constraint.
    - Keyword matching on labels ("war" in title): the exact failure mode the post
      calls out; the tug-of-war unit test pins it shut.
    - Zero-shot mDeBERTa for levels 2-3 as a default: a model download on the
      default path violates the offline rule; the embedding path covers all three
      levels from one table.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml

from cybernaut_mini.config import ConfigError
from cybernaut_mini.world.events import WorldEvent, embedding_matrix
from cybernaut_mini.world.vectors import FrozenEmbedder, matryoshka_truncate

__all__ = [
    "DEFAULT_TOPICS_PATH",
    "GEOPOLITICAL_LEVEL_1",
    "GEOPOLITICAL_LEVEL_2",
    "GEOPOLITICAL_LEVEL_3",
    "TopicRow",
    "TopicTable",
    "geopolitical",
    "geopolitical_mask",
    "load_topics",
    "load_transformer_classifier",
    "nearest_topics",
    "tag_events",
]

FloatArray = npt.NDArray[np.float32]

DEFAULT_TOPICS_PATH = Path("configs/world/iptc_topics.yaml")

#: The exact ontology codes of the published geopolitical filter — compared by
#: equality, never by words in the labels.
GEOPOLITICAL_LEVEL_1 = "conflict, war and peace"
GEOPOLITICAL_LEVEL_2 = "international relations"
GEOPOLITICAL_LEVEL_3 = frozenset({"war crime", "genocide", "terrorism", "nuclear policy"})


@dataclass(frozen=True)
class TopicRow:
    """One ontology row: a topic path plus the description the classifier embeds."""

    level_1: str
    level_2: str | None
    level_3: str | None
    description: str


@dataclass(frozen=True)
class TopicTable:
    """The ontology subset, row-aligned with its embedded description matrix."""

    rows: tuple[TopicRow, ...]

    def descriptions(self) -> list[str]:
        return [row.description for row in self.rows]


@lru_cache(maxsize=4)
def load_topics(path: Path = DEFAULT_TOPICS_PATH) -> TopicTable:
    """Load the ontology subset from YAML; every row needs a label and a description."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = payload.get("topics") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        msg = f"{path} must contain a non-empty 'topics' list"
        raise ConfigError(msg)
    rows: list[TopicRow] = []
    for entry in entries:
        level_1 = entry.get("level_1")
        description = entry.get("description")
        if not level_1 or not description:
            msg = f"{path}: every topic row needs 'level_1' and 'description', got {entry!r}"
            raise ConfigError(msg)
        rows.append(
            TopicRow(
                level_1=str(level_1),
                level_2=str(entry["level_2"]) if entry.get("level_2") else None,
                level_3=str(entry["level_3"]) if entry.get("level_3") else None,
                description=" ".join(str(description).split()),
            )
        )
    return TopicTable(rows=tuple(rows))


def nearest_topics(
    event_matrix: FloatArray, topic_matrix: FloatArray, table: TopicTable
) -> list[TopicRow]:
    """Argmax-cosine topic row per event — the single-label constraint, preserved.

    Both matrices must live in the same frozen space (L2-normalized rows); the
    argmax over one matmul is the whole classifier.
    """
    if topic_matrix.shape[0] != len(table.rows):
        msg = f"topic matrix has {topic_matrix.shape[0]} rows for {len(table.rows)} topics"
        raise ValueError(msg)
    if event_matrix.shape[0] == 0:
        return []
    scores = matryoshka_truncate(event_matrix, None) @ matryoshka_truncate(topic_matrix, None).T
    return [table.rows[int(index)] for index in np.argmax(scores, axis=1)]


def tag_events(
    events: Sequence[WorldEvent],
    embedder: FrozenEmbedder,
    *,
    table: TopicTable | None = None,
) -> list[WorldEvent]:
    """Fill ``iptc_level_1..3`` via embedding-nearest-topic over stored embeddings.

    ``embedder`` must be the same frozen instance that produced the event vectors;
    the topic descriptions are embedded through it so both sides share one space.
    """
    table = table if table is not None else load_topics()
    if not events:
        return []
    topic_matrix = embedder.embed_documents(table.descriptions())
    rows = nearest_topics(embedding_matrix(events), topic_matrix, table)
    return [
        event.model_copy(
            update={
                "iptc_level_1": row.level_1,
                "iptc_level_2": row.level_2,
                "iptc_level_3": row.level_3,
            }
        )
        for event, row in zip(events, rows, strict=True)
    ]


def geopolitical(event: WorldEvent) -> bool:
    """The published filter, transcribed: exact code equality on three fields."""
    return (
        event.iptc_level_1 == GEOPOLITICAL_LEVEL_1
        or event.iptc_level_2 == GEOPOLITICAL_LEVEL_2
        or event.iptc_level_3 in GEOPOLITICAL_LEVEL_3
    )


def geopolitical_mask(events: Sequence[WorldEvent]) -> npt.NDArray[np.bool_]:
    """Row-aligned boolean mask of :func:`geopolitical` over an event table."""
    return np.asarray([geopolitical(event) for event in events], dtype=bool)


def load_transformer_classifier(
    model: str = "classla/multilingual-IPTC-news-topic-classifier",
) -> Callable[[Sequence[str]], list[str]]:
    """Opt-in level-1 quality path: the classla IPTC classifier over event titles.

    Returns a callable mapping texts to level-1 labels. Raises
    :class:`ConfigError` when transformers or the checkpoint is unavailable, so
    offline installs keep the embedding-nearest-topic default.
    """
    try:
        from transformers import pipeline
    except ImportError as error:  # pragma: no cover - environment-specific
        msg = "transformers is not installed; embedding-nearest-topic is the default"
        raise ConfigError(msg) from error
    try:
        classifier = pipeline("text-classification", model=model)
    except Exception as error:  # pragma: no cover - network/download-specific
        msg = f"could not load {model!r} (downloads are opt-in): {error}"
        raise ConfigError(msg) from error

    def classify(texts: Sequence[str]) -> list[str]:
        results = classifier(list(texts), truncation=True)
        return [str(result["label"]) for result in results]

    return classify
