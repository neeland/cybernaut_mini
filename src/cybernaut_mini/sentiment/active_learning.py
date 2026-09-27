"""Active-learning label refinement: linear baselines, disagreement mining, oracle.

Per-embedding ``SGDClassifier(loss='hinge', penalty='l2', alpha=1e-5)``
baselines (embeddings memoized as ``.npy`` under ``data/06_models/``),
pure-numpy mining of rows where every baseline agrees but the majority vote
does not, verbatim nested oracle adjudication, relabel-on-oracle-agreement,
drop-the-worst-model per round, loop to convergence with a per-iteration
accuracy log.

Blog ref: https://nosible.com/blog/fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction —
    Step 3/4: the ``embedding_model`` script (memoized ``np.load``/slice,
    ``SGDClassifier(loss='hinge', penalty='l2', alpha=0.00001)``, 80/20
    subscript split "Samples already shuffled", nested confusion-matrix
    dicts) and the relabeling algorithm ("Identify disagreements where all
    linear models agree but the majority vote label does not ... Consult an
    oracle ... Drop the worst performing linear model on the validation set
    from the ensemble ... Repeat until no additional samples require
    relabeling"), with the nested oracle prompt shown verbatim and the +3.5%
    validation improvement table. Local copy:
    ``docs/blog-archive/fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction.md``.

Assumptions:
    - ``SGDClassifier`` gets ``random_state=42`` (the post sets none): SGD is
      stochastic and the repo's replay/testing discipline requires
      deterministic baselines; every other hyperparameter is the post's.
    - "If the oracle agrees with your baselines, relabel. If not, keep the
      original label" is enforced belt-and-braces: a row is relabeled only
      when the parsed verdict says ``correct == "true"`` AND its ``label``
      equals the consensus prediction; any unparseable verdict keeps the
      original label, because silently mutating gold on a garbled reply is
      worse than a wasted oracle call.
    - The oracle is an injected callable ``(text, prediction_label) ->
      OracleVerdict | None`` so the loop is offline-testable;
      :class:`OpenAICompatibleOracle` is the real client (the post used
      GPT-5.1 via OpenRouter; any strong local model works through the same
      seam) and is never constructed in tests.
    - The oracle system prompt is stored post-``textwrap.dedent`` (the post
      builds it inside an indented function and dedents); the canonical config
      copy lives in ``configs/sentiment/oracle_adjudication.txt`` and tests
      assert both match.
    - Embedding memoization follows the post's cache-then-slice pattern
      (``np.load(...)[: n_rows]``): a cache may hold MORE rows than the
      current run but never fewer — fewer raises instead of silently training
      on a truncated dataset.
    - The loop stops when no candidates remain, when ``max_iterations`` is
      hit, or when only ``min_models`` baselines are left; the post's stop
      rule is the first of these, the other two are laptop guardrails.

Alternatives rejected:
    - A Kedro pipeline: the relabel/drop/retrain loop is dynamic (data-driven
      iteration count), which the repo keeps as library code by convention.
    - Cross-validated baselines: the post uses a single subscript split; CV
      would change every accuracy the log reports.
    - Auto-accepting the oracle's label when it disagrees with both the vote
      and the consensus: the post's rule is binary (side with baselines or
      keep the vote), and inventing a third path would be replica drift.
"""

# ruff: noqa: E501 — the referenced post's URL slug alone exceeds the 100-char line limit.

from __future__ import annotations

import re
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from cybernaut_mini.config import ConfigError
from cybernaut_mini.models import canonical_dumps
from cybernaut_mini.providers.embeddings import EmbeddingProvider
from cybernaut_mini.sentiment.prompts import TaskSpec, _extract_json_object

__all__ = [
    "DEFAULT_CACHE_DIR",
    "ORACLE_SYSTEM_TEMPLATE",
    "ActiveLearningResult",
    "BaselineResult",
    "IterationLog",
    "OpenAICompatibleOracle",
    "OracleVerdict",
    "active_learning_loop",
    "build_oracle_messages",
    "embedding_cache_path",
    "label_options",
    "memoized_embeddings",
    "mine_disagreements",
    "parse_oracle_reply",
    "train_baseline",
    "write_history",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Where the per-model ``.npy`` embedding caches live, per the gap matrix.
DEFAULT_CACHE_DIR = Path("data/06_models")

#: Copied VERBATIM from the post's "Prompt used to consult the oracle" block,
#: after the ``textwrap.dedent`` the post itself applies. Slots:
#: ``{prediction}`` (the linear-consensus label) and ``{label_options}``.
#: Canonical config copy: ``configs/sentiment/oracle_adjudication.txt``.
ORACLE_SYSTEM_TEMPLATE = """
# TASK DESCRIPTION

Given the following LLM prompt, another model labelled this text as "{prediction}".
Is this model correct? Provide the correct label and a rationale for your answer.

# RESPONSE FORMAT

You must respond using this exact format. DO NOT RESPOND IN ANY OTHER WAY:

```json
{{
    "correct": "true" | "false",
    "label": {label_options},
    "reason": "A rationale for the correctness or incorrectness of your label."
}}
```

P.S. Think fast there is no time to waste.
"""


def label_options(task: TaskSpec) -> str:
    """The post's ``" | ".join([f'"{{label}}"' ...])`` options string."""
    return " | ".join(f'"{label}"' for label in task.labels)


def build_oracle_messages(
    task: TaskSpec, text: str, prediction_label: str
) -> list[dict[str, Any]]:
    """The nested oracle conversation, structured exactly as the post builds it.

    The original labeling prompt travels in the USER turn; the challenger
    label and reply contract live in the SYSTEM turn.
    """
    if prediction_label not in task.labels:
        msg = f"prediction {prediction_label!r} not in task vocabulary {task.labels}"
        raise ConfigError(msg)
    system_text = ORACLE_SYSTEM_TEMPLATE.format(
        prediction=prediction_label, label_options=label_options(task)
    )
    labelling_prompt = textwrap.dedent(task.build_prompt(text))
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "text", "text": labelling_prompt}]},
    ]


@dataclass(frozen=True)
class OracleVerdict:
    """One parsed oracle reply: the post's ``{correct, label, reason}`` JSON."""

    correct: bool
    label: str
    reason: str


def parse_oracle_reply(raw: str, task: TaskSpec) -> OracleVerdict | None:
    """Parse the oracle's fenced JSON; ``None`` keeps the original label."""
    obj = _extract_json_object(raw)
    if not isinstance(obj, dict):
        return None
    correct_raw = obj.get("correct")
    if isinstance(correct_raw, bool):
        correct = correct_raw
    elif isinstance(correct_raw, str) and correct_raw.strip().casefold() in {"true", "false"}:
        correct = correct_raw.strip().casefold() == "true"
    else:
        return None
    label = obj.get("label")
    if not isinstance(label, str) or label.strip().casefold() not in task.labels:
        return None
    reason = obj.get("reason")
    return OracleVerdict(
        correct=correct,
        label=label.strip().casefold(),
        reason=reason.strip() if isinstance(reason, str) else "",
    )


class OpenAICompatibleOracle:
    """The real oracle client: one strong model behind any OpenAI-compatible URL.

    The post used GPT-5.1 through OpenRouter; a strong local model (e.g.
    ``qwen3:32b`` on Ollama) works through the same seam. Never constructed in
    offline tests — the loop takes any ``(text, prediction) -> verdict``
    callable.
    """

    def __init__(
        self,
        *,
        model: str,
        task: TaskSpec,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        timeout: float = 300.0,
        max_retries: int = 1,
        client_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.model = model
        self.task = task
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._api_key = api_key
        self._client_factory = client_factory
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory(self._api_key, self.base_url)
            else:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self._api_key, base_url=self.base_url, timeout=self.timeout
                )
        return self._client

    def __call__(self, text: str, prediction_label: str) -> OracleVerdict | None:
        client = self._get_client()
        messages = build_oracle_messages(self.task, text, prediction_label)
        for _ in range(self.max_retries + 1):
            response = client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.0
            )
            try:
                content = response.choices[0].message.content or ""
            except (AttributeError, IndexError, KeyError, TypeError):
                content = ""
            verdict = parse_oracle_reply(content, self.task)
            if verdict is not None:
                return verdict
        return None


# ── memoized embeddings + linear baselines ───────────────────────────────────


def embedding_cache_path(
    model_identifier: str, task_name: str, directory: Path = DEFAULT_CACHE_DIR
) -> Path:
    """``data/06_models/{task}__{model-slug}.npy`` — the post's ``.npy`` sidecar."""
    slug = re.sub(r"[^a-z0-9]+", "-", model_identifier.casefold()).strip("-")
    if not slug:
        msg = f"cannot derive a cache slug from {model_identifier!r}"
        raise ConfigError(msg)
    return directory / f"{task_name}__{slug}.npy"


def memoized_embeddings(
    texts: Sequence[str], provider: EmbeddingProvider, *, cache_path: Path
) -> FloatArray:
    """Load the ``.npy`` cache or embed-and-save, then slice to the run size.

    The post's exact pattern: generate once, ``np.load`` thereafter, and take
    ``embeddings[: n_rows]`` so smaller runs reuse the big cache.
    """
    if not texts:
        msg = "cannot embed an empty text list"
        raise ConfigError(msg)
    if not cache_path.exists():
        vectors = np.asarray(provider.embed_documents(list(texts)), dtype=np.float64)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, vectors)
    embeddings = np.asarray(np.load(cache_path), dtype=np.float64)
    if embeddings.shape[0] < len(texts):
        msg = (
            f"embedding cache {cache_path} holds {embeddings.shape[0]} rows but the "
            f"run needs {len(texts)}; delete the cache to re-embed"
        )
        raise ConfigError(msg)
    return embeddings[: len(texts)]


@dataclass(frozen=True)
class BaselineResult:
    """One linear baseline: accuracies, nested confusion dicts, full-row predictions."""

    name: str
    train_accuracy: float
    val_accuracy: float
    confusion_train: dict[str, dict[str, int]]
    confusion_val: dict[str, dict[str, int]]
    predictions: IntArray

    def as_dict(self) -> dict[str, object]:
        return {
            "confusion_train": self.confusion_train,
            "confusion_val": self.confusion_val,
            "name": self.name,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
        }


def _nested_confusion(
    y_true: IntArray, y_pred: IntArray, task: TaskSpec
) -> dict[str, dict[str, int]]:
    """The post's ``{row_label: {col_label: count}}`` confusion dict."""
    targets = [task.label_to_int[label] for label in task.labels]
    grid = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=targets)
    return {
        row: {col: int(grid[i, j]) for j, col in enumerate(task.labels)}
        for i, row in enumerate(task.labels)
    }


def train_baseline(
    name: str,
    embeddings: FloatArray,
    targets: Sequence[int] | IntArray,
    task: TaskSpec,
    *,
    train_pct: float = 0.8,
    random_state: int = 42,
) -> BaselineResult:
    """The post's baseline: hinge-loss SGD on frozen embeddings, subscript split."""
    features = np.asarray(embeddings, dtype=np.float64)
    target_arr = np.asarray(targets, dtype=np.int64)
    if features.ndim != 2 or features.shape[0] != target_arr.shape[0]:
        msg = f"embeddings {features.shape} do not align with {target_arr.shape[0]} targets"
        raise ConfigError(msg)
    n_train = int(features.shape[0] * train_pct)
    if n_train < 1 or n_train >= features.shape[0]:
        msg = f"cannot split {features.shape[0]} rows at train_pct={train_pct}"
        raise ConfigError(msg)
    x_train, y_train = features[:n_train], target_arr[:n_train]
    x_val, y_val = features[n_train:], target_arr[n_train:]

    classifier = SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=0.00001,
        random_state=random_state,
    )
    classifier.fit(x_train, y_train)
    train_preds = np.asarray(classifier.predict(x_train), dtype=np.int64)
    val_preds = np.asarray(classifier.predict(x_val), dtype=np.int64)
    predictions = np.asarray(classifier.predict(features), dtype=np.int64)
    return BaselineResult(
        name=name,
        train_accuracy=float(accuracy_score(y_true=y_train, y_pred=train_preds)),
        val_accuracy=float(accuracy_score(y_true=y_val, y_pred=val_preds)),
        confusion_train=_nested_confusion(y_train, train_preds, task),
        confusion_val=_nested_confusion(y_val, val_preds, task),
        predictions=predictions,
    )


def mine_disagreements(
    predictions: Mapping[str, IntArray], votes: Sequence[int] | IntArray
) -> IntArray:
    """Rows where every baseline predicts the same class and it differs from the vote.

    Pure numpy: stack the prediction rows, require column-wise unanimity, then
    keep the columns whose consensus contradicts the majority-vote label.
    """
    if not predictions:
        msg = "disagreement mining needs at least one prediction column"
        raise ConfigError(msg)
    votes_arr = np.asarray(votes, dtype=np.int64)
    stacked = np.stack([np.asarray(column, dtype=np.int64) for column in predictions.values()])
    if stacked.shape[1] != votes_arr.shape[0]:
        msg = f"predictions cover {stacked.shape[1]} rows, votes cover {votes_arr.shape[0]}"
        raise ConfigError(msg)
    unanimous = np.all(stacked == stacked[0], axis=0)
    contested = unanimous & (stacked[0] != votes_arr)
    return np.flatnonzero(contested).astype(np.int64)


# ── the refinement loop ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class IterationLog:
    """One round of the loop, matching the post's before/after accuracy table."""

    iteration: int
    model_accuracies: dict[str, dict[str, float]]
    n_candidates: int
    n_relabelled: int
    dropped_model: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "dropped_model": self.dropped_model,
            "iteration": self.iteration,
            "model_accuracies": self.model_accuracies,
            "n_candidates": self.n_candidates,
            "n_relabelled": self.n_relabelled,
        }


@dataclass(frozen=True)
class ActiveLearningResult:
    """The refined label column plus the full per-iteration history."""

    labels: IntArray
    history: list[IterationLog] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return bool(self.history) and self.history[-1].n_candidates == 0


def active_learning_loop(
    embeddings: Mapping[str, FloatArray],
    texts: Sequence[str],
    votes: Sequence[int],
    task: TaskSpec,
    oracle: Callable[[str, str], OracleVerdict | None],
    *,
    train_pct: float = 0.8,
    max_iterations: int = 10,
    min_models: int = 1,
    random_state: int = 42,
) -> ActiveLearningResult:
    """Train → mine → adjudicate → relabel → drop worst → repeat to convergence.

    *embeddings* maps embedding-model name to its (n_rows, dim) matrix, all
    aligned with *texts* and *votes*. The returned labels start as the votes
    and mutate only where the oracle sides with the unanimous consensus.
    """
    if not embeddings:
        msg = "active learning needs at least one embedding model"
        raise ConfigError(msg)
    labels = np.asarray(votes, dtype=np.int64).copy()
    if labels.shape[0] != len(texts):
        msg = f"{labels.shape[0]} votes vs {len(texts)} texts"
        raise ConfigError(msg)
    for name, matrix in embeddings.items():
        if matrix.shape[0] != labels.shape[0]:
            msg = f"embedding {name!r} covers {matrix.shape[0]} rows, votes {labels.shape[0]}"
            raise ConfigError(msg)
    if min_models < 1:
        msg = f"min_models must be >= 1, got {min_models}"
        raise ConfigError(msg)

    active = dict(embeddings)
    history: list[IterationLog] = []
    for iteration in range(max_iterations):
        results = {
            name: train_baseline(
                name, matrix, labels, task, train_pct=train_pct, random_state=random_state
            )
            for name, matrix in active.items()
        }
        accuracies = {
            name: {"train": result.train_accuracy, "val": result.val_accuracy}
            for name, result in results.items()
        }
        candidates = mine_disagreements(
            {name: result.predictions for name, result in results.items()}, labels
        )
        if candidates.size == 0:
            history.append(
                IterationLog(
                    iteration=iteration,
                    model_accuracies=accuracies,
                    n_candidates=0,
                    n_relabelled=0,
                    dropped_model=None,
                )
            )
            break

        relabelled = 0
        first = next(iter(results.values()))
        for index in candidates.tolist():
            consensus = int(first.predictions[index])
            prediction_label = task.int_to_label[consensus]
            verdict = oracle(texts[index], prediction_label)
            if verdict is not None and verdict.correct and verdict.label == prediction_label:
                labels[index] = consensus
                relabelled += 1

        dropped: str | None = None
        if len(active) > min_models:
            dropped = min(results, key=lambda name: results[name].val_accuracy)
            active.pop(dropped)

        history.append(
            IterationLog(
                iteration=iteration,
                model_accuracies=accuracies,
                n_candidates=int(candidates.size),
                n_relabelled=relabelled,
                dropped_model=dropped,
            )
        )
    return ActiveLearningResult(labels=labels, history=history)


def write_history(path: Path, result: ActiveLearningResult) -> None:
    """Persist the per-iteration log as canonical JSON (the repo's artifact rule)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "converged": result.converged,
        "history": [entry.as_dict() for entry in result.history],
    }
    path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")
