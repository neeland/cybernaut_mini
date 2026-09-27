"""The anchor-signal engine every WORLD index shares: relevance, polarity, gates.

Each index is a handful of sentences compared against event embeddings. This module
is the one implementation of that recipe: load an anchor set from YAML (the posts'
sentences copied verbatim into ``configs/anchors/*.yaml``), embed the sentences with
the frozen model, and score events — max-cosine relevance with a floor, matched
uncertain/certain pairs routed through the best pair to a tanh polarity, the
``w_unc`` uncertainty weight, category masks, and the polarity-separation
diagnostics that are the retuning target for open models.

Blog ref: https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty
    — ``relevant(e) = max cosine(event e, the 3 topic phrases) >= 0.35``,
    ``polarity(e) = tanh((uncertain_sim - certain_sim) / 0.1)``,
    ``w_unc(e) = (1 + polarity(e)) / 2``, "A lever's score is the highest of its
    two framings. Relevance is the highest score over all levers. Each event's
    polarity comes from the pair whose better framing scores highest.", the 0.25
    category rule, and the separation stats ("events clearly not about trade
    average 0.44 ... events that pass the trade filter average 0.63 ... 60% land
    clearly on the uncertain side, above 0.6, and 21% clearly on the settled side,
    below 0.4"). https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world
    — the trade-coercion OR-gate and oil AND-gate these predicates compose into.
    https://nosible.com/blog/turning-news-into-a-risk-on-risk-off-equity-signal —
    the 17-anchor flat set with floor 0.30. Local copies under ``docs/blog-archive/``.

Assumptions:
    - An anchor set has up to three blocks: flat ``anchors`` (name -> sentence),
      ``levers`` (name -> list of {uncertain, certain} pairs) and ``categories``
      (same shape as levers). Relevance reads ``anchors`` when present (trade,
      stress, oil), otherwise the lever sentences themselves (EPU, whose appendix
      says "There is no separate topic block").
    - Published thresholds (0.35/0.40/0.30/0.25, temperature 0.1) were tuned
      against ``text-embedding-3-large`` cosines. Open-model cosine scales differ,
      so every threshold is data (YAML), :func:`sweep_relevance_floor` exists, and
      :func:`polarity_separation` emits the four stats the posts publish as the
      retuning target — the *shape* of the recipe is what is fixed.
    - Anchors are embedded through :class:`~cybernaut_mini.world.vectors.FrozenEmbedder`
      ``embed_documents`` — the posts embed anchor sentences with the same model
      and transform as the stored event vectors, with no query-side instruction.
    - Scoring is matrix-in, arrays-out (one matmul, row-wise max/argmax) so a
      leave-one-lever-out rerun over a cached event matrix costs seconds.

Alternatives rejected:
    - Mean-cosine over an anchor set instead of max: averaging dilutes a strong
      match on one concept with irrelevance to sixteen others; the posts say
      "highest" at every step.
    - Learning polarity from labels: the whole point of the matched-pair design is
      that it ships with zero training data and no foreknowledge of the corpus.
    - A per-pair polarity averaged over pairs: the post routes each event through
      its single best-matching pair; averaging would let nine off-topic pairs pull
      a monetary-policy event toward neutral.
    - Hard-coding the sentences in Python: the sentences are the published
      artifact; YAML keeps them verbatim, diffable, and swappable per experiment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import BaseModel, ConfigDict, Field

from cybernaut_mini.config import ConfigError
from cybernaut_mini.world.vectors import FrozenEmbedder, matryoshka_truncate

__all__ = [
    "ANCHOR_DIR",
    "AnchorPair",
    "AnchorScores",
    "AnchorSet",
    "EmbeddedAnchorSet",
    "category_mask",
    "load_anchor_set",
    "net_polarity_weights",
    "polarity_separation",
    "score_events",
    "sweep_relevance_floor",
    "thresholds_from",
]

FloatArray = npt.NDArray[np.float32]

ANCHOR_DIR = Path("configs/anchors")


class AnchorPair(BaseModel):
    """One matched uncertain/certain framing pair."""

    model_config = ConfigDict(extra="forbid")

    uncertain: str = Field(min_length=1)
    certain: str = Field(min_length=1)


class AnchorSet(BaseModel):
    """One YAML anchor config: sentences verbatim from a post, thresholds as data."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    thresholds: dict[str, float] = Field(default_factory=dict)
    anchors: dict[str, str] = Field(default_factory=dict)
    levers: dict[str, list[AnchorPair]] = Field(default_factory=dict)
    categories: dict[str, list[AnchorPair]] = Field(default_factory=dict)

    @property
    def relevance_floor(self) -> float:
        return float(self.thresholds.get("relevance_floor", 0.35))

    @property
    def polarity_temperature(self) -> float:
        return float(self.thresholds.get("polarity_temperature", 0.1))

    @property
    def category_floor(self) -> float:
        return float(self.thresholds.get("category_floor", 0.25))

    def sentence_count(self) -> int:
        """Total sentences carried (the posts count their recipes in sentences)."""
        pairs = sum(len(p) for p in self.levers.values())
        pairs += sum(len(p) for p in self.categories.values())
        return len(self.anchors) + 2 * pairs

    def lever_pairs(
        self, include: Sequence[str] | None = None
    ) -> list[tuple[str, AnchorPair]]:
        """(lever name, pair) rows, optionally restricted to ``include`` levers.

        The restriction is the leave-one-lever-out ablation hook: pass all lever
        names minus one and rescore — embeddings are cached by the caller, so a
        rerun is one matmul.
        """
        names = list(self.levers) if include is None else list(include)
        unknown = [name for name in names if name not in self.levers]
        if unknown:
            msg = f"unknown levers {unknown!r}; available: {sorted(self.levers)}"
            raise ConfigError(msg)
        return [(name, pair) for name in names for pair in self.levers[name]]


def load_anchor_set(name_or_path: str | Path, *, anchor_dir: Path = ANCHOR_DIR) -> AnchorSet:
    """Load one anchor set by name (``"trade"``) or explicit YAML path."""
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = anchor_dir / f"{path.name}.yaml"
    if not path.exists():
        msg = f"anchor set {str(name_or_path)!r} not found at {path}"
        raise ConfigError(msg)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for block in ("anchors",):
        if block in payload and isinstance(payload[block], dict):
            payload[block] = {
                key: " ".join(str(value).split()) for key, value in payload[block].items()
            }
    anchor_set = AnchorSet.model_validate(payload)
    if not anchor_set.anchors and not anchor_set.levers:
        msg = f"{path} defines neither 'anchors' nor 'levers'"
        raise ConfigError(msg)
    return anchor_set


# ---------------------------------------------------------------------- #
# Embedding and scoring                                                  #
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class EmbeddedAnchorSet:
    """An anchor set with every sentence embedded once in the frozen space.

    ``relevance_matrix`` rows align with ``relevance_labels``; ``pair_uncertain``
    and ``pair_certain`` rows align with ``pair_labels`` (lever name per pair).
    """

    config: AnchorSet
    relevance_labels: tuple[str, ...]
    relevance_matrix: FloatArray
    pair_labels: tuple[str, ...]
    pair_uncertain: FloatArray
    pair_certain: FloatArray

    @classmethod
    def embed(
        cls,
        anchor_set: AnchorSet,
        embedder: FrozenEmbedder,
        *,
        levers: Sequence[str] | None = None,
    ) -> EmbeddedAnchorSet:
        """Embed the set's sentences. ``levers`` restricts pairs (ablation hook).

        Relevance reads the flat ``anchors`` block when present; otherwise the
        lever sentences themselves (both framings), as the EPU appendix specifies.
        """
        pairs = anchor_set.lever_pairs(include=levers)
        pair_labels = tuple(name for name, _ in pairs)
        if pairs:
            uncertain = embedder.embed_documents([pair.uncertain for _, pair in pairs])
            certain = embedder.embed_documents([pair.certain for _, pair in pairs])
        else:
            uncertain = np.zeros((0, embedder.dim), dtype=np.float32)
            certain = np.zeros((0, embedder.dim), dtype=np.float32)
        if anchor_set.anchors:
            labels = tuple(anchor_set.anchors)
            matrix = embedder.embed_documents([anchor_set.anchors[label] for label in labels])
        else:
            labels = pair_labels
            matrix = np.zeros((0, embedder.dim), dtype=np.float32)
        return cls(
            config=anchor_set,
            relevance_labels=labels,
            relevance_matrix=matrix,
            pair_labels=pair_labels,
            pair_uncertain=uncertain,
            pair_certain=certain,
        )

    def category_pairs(self, category: str, embedder: FrozenEmbedder) -> tuple[FloatArray, ...]:
        """(uncertain, certain) matrices for one category block."""
        if category not in self.config.categories:
            msg = f"unknown category {category!r}; available: {sorted(self.config.categories)}"
            raise ConfigError(msg)
        pairs = self.config.categories[category]
        return (
            embedder.embed_documents([pair.uncertain for pair in pairs]),
            embedder.embed_documents([pair.certain for pair in pairs]),
        )


@dataclass(frozen=True)
class AnchorScores:
    """Row-aligned per-event scores from one anchor set."""

    relevance: npt.NDArray[np.float64]
    nearest_anchor: tuple[str, ...]
    polarity: npt.NDArray[np.float64]
    w_unc: npt.NDArray[np.float64]
    best_pair: tuple[str, ...]

    def relevant_mask(self, floor: float) -> npt.NDArray[np.bool_]:
        """``relevant(e) = relevance(e) >= floor`` — the OR/AND-gate building block."""
        return np.asarray(self.relevance >= floor, dtype=bool)


def score_events(event_matrix: FloatArray, embedded: EmbeddedAnchorSet) -> AnchorScores:
    """Score every event against one embedded anchor set.

    - relevance: max cosine over the relevance block (flat anchors, or the lever
      sentences' better framing when the set has no flat block), with the argmax
      anchor label carried as the nearest-anchor tag;
    - polarity: ``bp = argmax_p max(cos(e, U_p), cos(e, C_p))`` then
      ``tanh((U_bp - C_bp) / T)``; ``w_unc = (1 + polarity) / 2``.

    Events with no pairs in the set get polarity 0 and ``w_unc`` 0.5 (neutral).
    """
    matrix = matryoshka_truncate(event_matrix, None)
    n_events = matrix.shape[0]

    pair_best: npt.NDArray[np.float64] | None = None
    if embedded.pair_labels:
        u_sims = (matrix @ embedded.pair_uncertain.T).astype(np.float64)
        c_sims = (matrix @ embedded.pair_certain.T).astype(np.float64)
        pair_best = np.maximum(u_sims, c_sims)
        best_index = np.argmax(pair_best, axis=1)
        rows = np.arange(n_events)
        gap = u_sims[rows, best_index] - c_sims[rows, best_index]
        polarity = np.tanh(gap / embedded.config.polarity_temperature)
        best_pair = tuple(embedded.pair_labels[int(i)] for i in best_index)
    else:
        polarity = np.zeros(n_events, dtype=np.float64)
        best_pair = ("",) * n_events

    if embedded.relevance_matrix.shape[0] > 0:
        rel_sims = (matrix @ embedded.relevance_matrix.T).astype(np.float64)
        relevance = rel_sims.max(axis=1) if n_events else np.zeros(0, dtype=np.float64)
        nearest = tuple(
            embedded.relevance_labels[int(i)] for i in np.argmax(rel_sims, axis=1)
        )
    elif pair_best is not None:
        # EPU shape: relevance is the highest lever score (a lever's score being
        # the highest of its two framings), and the nearest anchor is that lever.
        relevance = pair_best.max(axis=1) if n_events else np.zeros(0, dtype=np.float64)
        nearest = best_pair
    else:  # pragma: no cover - load_anchor_set forbids empty sets
        msg = "anchor set has neither a relevance block nor lever pairs"
        raise ConfigError(msg)

    return AnchorScores(
        relevance=relevance,
        nearest_anchor=nearest,
        polarity=polarity,
        w_unc=(1.0 + polarity) / 2.0,
        best_pair=best_pair,
    )


def category_mask(
    event_matrix: FloatArray,
    category_matrices: tuple[FloatArray, ...],
    floor: float,
) -> npt.NDArray[np.bool_]:
    """``max cosine(e, the category's pair sentences) >= floor`` (the 0.25 rule).

    AND this with the headline relevant mask to get a category sub-index filter.
    """
    matrix = matryoshka_truncate(event_matrix, None)
    sentences = np.concatenate([np.asarray(m, dtype=np.float32) for m in category_matrices])
    scores = (matrix @ sentences.T).astype(np.float64)
    best = scores.max(axis=1) if matrix.shape[0] else np.zeros(0, dtype=np.float64)
    return np.asarray(best >= floor, dtype=bool)


# ---------------------------------------------------------------------- #
# Diagnostics                                                            #
# ---------------------------------------------------------------------- #


def polarity_separation(
    scores: AnchorScores, relevant: npt.NDArray[np.bool_]
) -> dict[str, float]:
    """The four published separation stats — the retuning target for open models.

    The TPU post's yardstick: off-topic events average ``w_unc`` ~0.44, on-topic
    ~0.63, with 60% of counted events above 0.6 and 21% below 0.4. An open
    embedder should be tuned (floors, temperature) until these four move toward
    that shape; ``nan`` marks an empty side.
    """
    on = scores.w_unc[relevant]
    off = scores.w_unc[~relevant]
    return {
        "off_topic_mean_w_unc": float(np.mean(off)) if off.size else float("nan"),
        "on_topic_mean_w_unc": float(np.mean(on)) if on.size else float("nan"),
        "share_above_0.6": float(np.mean(on > 0.6)) if on.size else float("nan"),
        "share_below_0.4": float(np.mean(on < 0.4)) if on.size else float("nan"),
    }


def sweep_relevance_floor(
    scores: AnchorScores, floors: Sequence[float] | None = None
) -> dict[float, float]:
    """Share of events retained at each candidate floor.

    The published floors were tuned on ``text-embedding-3-large``; sweep and pick
    the floor whose retained share (and downstream correlations) look sane for
    the embedder in use. "The 0.35 cutoff is not load-bearing."
    """
    candidates = tuple(floors) if floors is not None else (0.25, 0.30, 0.35, 0.40, 0.45)
    total = max(1, scores.relevance.shape[0])
    return {
        float(floor): float(np.count_nonzero(scores.relevance >= floor)) / total
        for floor in candidates
    }


def net_polarity_weights(
    scores: AnchorScores, relevant: npt.NDArray[np.bool_], breadths: Sequence[float]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """(breadth * polarity, breadth) rows over relevant events, zero elsewhere.

    The building block for ``net_polarity(t)`` in
    :mod:`cybernaut_mini.world.indices.policy`: sum both per day and divide.
    """
    breadth = np.asarray(list(breadths), dtype=np.float64)
    mask = np.asarray(relevant, dtype=bool)
    weighted = np.where(mask, breadth * scores.polarity, 0.0)
    return weighted, np.where(mask, breadth, 0.0)


def thresholds_from(anchor_set: AnchorSet | Mapping[str, float]) -> dict[str, float]:
    """Threshold mapping from a set or raw mapping (convenience for index configs)."""
    if isinstance(anchor_set, AnchorSet):
        return dict(anchor_set.thresholds)
    return dict(anchor_set)
