"""LLM-ensemble labeling: decision-tree prompts, strict JSON, majority vote, wide table.

One :class:`JsonTaskLabeler` per model, all running the same verbatim
decision-tree prompt from :mod:`.prompts` against any OpenAI-compatible
endpoint (local Ollama by default), with format-constrained JSON decoding and
retry-on-invalid-JSON. Columns land in a wide polars table
(``text | qwen3_0_6b | ... | majority_vote``) persisted as parquet under
iteration-versioned filenames, plus the 200-row hand-label CSV protocol with a
tiny CLI labeling loop for validating the prompt before scaling.

Blog ref: https://nosible.com/blog/fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction —
    "We used eight models ... To ensemble their labels, use majority vote.
    Start with your hand-labeled 200 samples to validate the prompt works
    before scaling to thousands." The dataset-transformation diagram shows the
    wide table (``text | grok_4_fast | ... | qwen3_32b | majority_vote``,
    shape (10_000, 10)) with the hand-labeled 200 kept separate ("*10_200?
    Nope. We kept the samples for the human set separate"), and the training
    script reads the iteration-versioned artifact
    ``financial_sentiment_100.0k_iter_18.ipc``. Local copy:
    ``docs/blog-archive/fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction.md``.

Assumptions:
    - The post's eight hosted OpenRouter models are replaced by any
      OpenAI-compatible endpoint — local Ollama (``http://localhost:11434/v1``)
      by default — because the recipe is (verbatim prompt, temperature 0,
      strict JSON, majority vote), not the vendor list. Column names snake_case
      the model id's last path segment, reproducing the post's ``grok_4_fast``
      / ``qwen3_32b`` column style.
    - "Format-constrained" decoding maps to
      ``response_format={"type": "json_object"}`` (honoured by Ollama, vLLM,
      and OpenAI); a reply :meth:`.prompts.TaskSpec.parse_reply` rejects is
      retried up to ``max_retries`` and then falls back to the task's
      when-in-doubt default label — the prompt's own tie rule extended to
      transport/format failures, exactly as :mod:`.labelers` does with NEU.
    - Majority-vote ties resolve to the task's default label. The post says
      only "use majority vote"; with an even model count a tie needs a rule,
      and every prompt already names its when-in-doubt class.
    - The post persists Arrow ``.ipc``; the repo standardises on parquet for
      wide tables, keeping the post's exact stem pattern
      ``{task}_{n/1000:.1f}k_iter_{i}`` so ``financial_sentiment_100.0k_iter_18``
      round-trips.
    - The reasoning-enabled variant ("Grok 4 Fast (reasoning enabled)") is
      mirrored by ``think=True``, which forwards
      ``chat_template_kwargs={"enable_thinking": True}`` and suffixes the
      column ``_reasoning`` — the local-Qwen equivalent of the post's toggle.
    - Rationales ride along as ``{column}__rationale`` columns because the
      post's refinement loop is driven by them ("Use those disagreements and
      the LLMs rationale to re-label examples and refine your prompt").

Alternatives rejected:
    - A Kedro pipeline: the retry loop, the interactive hand-label CLI, and
      the prompt-refinement iteration are dynamic flows; per the repo's split
      (static DAGs in pipelines/, dynamic loops as library code) this is
      library code.
    - Confidence/logprob-weighted voting: the post uses plain majority vote;
      weighting would be replica drift.
    - Storing integer labels in the table: the post's tables hold label
      strings; integers appear only at classifier boundaries via
      :attr:`.prompts.TaskSpec.label_to_int`.
"""

# ruff: noqa: E501 — the referenced post's URL slug alone exceeds the 100-char line limit.

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from cybernaut_mini.config import ConfigError
from cybernaut_mini.sentiment.prompts import TaskSpec

__all__ = [
    "DEFAULT_BASE_URL",
    "HAND_LABEL_COLUMNS",
    "JsonTaskLabeler",
    "LabelColumn",
    "ValidationReport",
    "build_vote_table",
    "column_name_for_model",
    "hand_label_loop",
    "majority_vote",
    "read_hand_label_csv",
    "read_vote_table",
    "sample_hand_label_rows",
    "validate_against_hand_labels",
    "vote_table_path",
    "write_hand_label_csv",
    "write_vote_table",
]

#: Local Ollama's OpenAI-compatible endpoint — the default model host.
DEFAULT_BASE_URL = "http://localhost:11434/v1"

#: Header of the hand-label CSV (``data/01_raw/human_labels.csv``).
HAND_LABEL_COLUMNS: tuple[str, str] = ("text", "human_label")


def column_name_for_model(model: str) -> str:
    """``qwen/qwen3-32b`` -> ``qwen3_32b`` — the post's column naming."""
    stem = model.split("/")[-1]
    name = re.sub(r"[^a-z0-9]+", "_", stem.casefold()).strip("_")
    if not name:
        msg = f"cannot derive a column name from model id {model!r}"
        raise ConfigError(msg)
    return name


@dataclass(frozen=True)
class LabelColumn:
    """One model's labels over the story list, with rationales and retry stats."""

    name: str
    labels: list[str]
    rationales: list[str]
    parse_failures: int


class JsonTaskLabeler:
    """One model labeling one task under the strict-JSON decision-tree prompt.

    ``client_factory`` is the network seam, identical in spirit to
    :class:`cybernaut_mini.sentiment.labelers.FewShotLLMLabeler`; tests inject
    a scripted fake and never touch a socket. Local endpoints need no real
    credential, so ``api_key`` defaults to the string Ollama ignores.
    """

    def __init__(
        self,
        *,
        model: str,
        task: TaskSpec,
        name: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "ollama",
        timeout: float = 120.0,
        max_retries: int = 2,
        think: bool = False,
        client_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        if max_retries < 0:
            msg = f"max_retries must be >= 0, got {max_retries}"
            raise ConfigError(msg)
        self.model = model
        self.task = task
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.think = think
        suffix = "_reasoning" if think else ""
        self._name = name or column_name_for_model(model) + suffix
        self._api_key = api_key
        self._client_factory = client_factory
        self._client: Any | None = None

    @property
    def name(self) -> str:
        return self._name

    def _default_client_factory(self, api_key: str, base_url: str) -> Any:
        from openai import OpenAI

        return OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)

    def _get_client(self) -> Any:
        if self._client is None:
            factory = self._client_factory or self._default_client_factory
            self._client = factory(self._api_key, self.base_url)
        return self._client

    def _request_kwargs(self, text: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.task.build_prompt(text)}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        if self.think:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        return kwargs

    def label_one(self, text: str) -> tuple[str, str, int]:
        """Label one snippet: ``(label, rationale, attempts)``.

        Retries on any reply the task parser rejects; after ``max_retries``
        extra attempts the task's default label is recorded with an empty
        rationale — never a dropped row.
        """
        client = self._get_client()
        attempts = 0
        for _ in range(self.max_retries + 1):
            attempts += 1
            response = client.chat.completions.create(**self._request_kwargs(text))
            try:
                content = response.choices[0].message.content or ""
            except (AttributeError, IndexError, KeyError, TypeError):
                content = ""
            label, rationale = self.task.parse_reply(content)
            if label is not None:
                return label, rationale, attempts
        return self.task.default_label, "", attempts

    def label(self, texts: Sequence[str]) -> LabelColumn:
        labels: list[str] = []
        rationales: list[str] = []
        failures = 0
        for text in texts:
            label, rationale, attempts = self.label_one(text)
            if attempts > 1:
                failures += attempts - 1
            labels.append(label)
            rationales.append(rationale)
        return LabelColumn(
            name=self.name, labels=labels, rationales=rationales, parse_failures=failures
        )


def majority_vote(columns: Mapping[str, Sequence[str]], task: TaskSpec) -> list[str]:
    """Row-wise majority over the model columns; ties fall to the default label."""
    if not columns:
        msg = "majority vote needs at least one model column"
        raise ConfigError(msg)
    lengths = {len(labels) for labels in columns.values()}
    if len(lengths) != 1:
        msg = f"model columns have differing lengths: {sorted(lengths)}"
        raise ConfigError(msg)
    votes: list[str] = []
    for row in zip(*columns.values(), strict=True):
        counts = Counter(row)
        ranked = counts.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            votes.append(task.default_label)
        else:
            votes.append(ranked[0][0])
    return votes


def build_vote_table(
    texts: Sequence[str],
    labelers: Sequence[JsonTaskLabeler],
    *,
    with_rationales: bool = True,
) -> pl.DataFrame:
    """The post's wide table: ``text | <model columns...> | majority_vote``.

    Rationale columns (``{model}__rationale``) trail the vote column so the
    label block matches the post's diagram column-for-column.
    """
    if not texts:
        msg = "cannot vote over an empty text list"
        raise ConfigError(msg)
    if not labelers:
        msg = "cannot vote without labelers"
        raise ConfigError(msg)
    tasks = {labeler.task.name for labeler in labelers}
    if len(tasks) != 1:
        msg = f"labelers span multiple tasks: {sorted(tasks)}"
        raise ConfigError(msg)

    columns: dict[str, LabelColumn] = {}
    for labeler in labelers:
        if labeler.name in columns:
            msg = f"duplicate vote column {labeler.name!r}"
            raise ConfigError(msg)
        columns[labeler.name] = labeler.label(texts)

    task = labelers[0].task
    label_block = {name: column.labels for name, column in columns.items()}
    data: dict[str, Sequence[str]] = {"text": list(texts)}
    data.update(label_block)
    data["majority_vote"] = majority_vote(label_block, task)
    if with_rationales:
        for name, column in columns.items():
            data[f"{name}__rationale"] = column.rationales
    return pl.DataFrame(data)


def vote_table_path(directory: Path, task_name: str, n_rows: int, iteration: int) -> Path:
    """``financial_sentiment_100.0k_iter_18.parquet`` — the post's stem, verbatim."""
    if n_rows <= 0:
        msg = f"n_rows must be positive, got {n_rows}"
        raise ConfigError(msg)
    if iteration < 0:
        msg = f"iteration must be >= 0, got {iteration}"
        raise ConfigError(msg)
    return directory / f"{task_name}_{n_rows / 1000:.1f}k_iter_{iteration}.parquet"


def write_vote_table(frame: pl.DataFrame, directory: Path, task: TaskSpec, iteration: int) -> Path:
    """Persist the wide table under its iteration-versioned filename."""
    path = vote_table_path(directory, task.name, frame.height, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return path


def read_vote_table(path: Path) -> pl.DataFrame:
    if not path.exists():
        msg = f"vote table not found at {path}"
        raise ConfigError(msg)
    return pl.read_parquet(path)


# ── the 200-row hand-label protocol ──────────────────────────────────────────


def sample_hand_label_rows(
    texts: Sequence[str], *, n: int = 200, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Uniform sample of *n* rows for hand labeling, plus the disjoint remainder.

    The post keeps the human set separate from the scaled voting set; returning
    both halves makes the disjointness structural rather than a convention.
    """
    import numpy as np

    if n <= 0:
        msg = f"hand-label sample size must be positive, got {n}"
        raise ConfigError(msg)
    if len(texts) <= n:
        msg = f"need more than {n} texts to keep a disjoint remainder, got {len(texts)}"
        raise ConfigError(msg)
    order = np.random.default_rng(seed).permutation(len(texts))
    hand = [texts[int(i)] for i in order[:n]]
    rest = [texts[int(i)] for i in order[n:]]
    return hand, rest


def write_hand_label_csv(texts: Sequence[str], path: Path) -> None:
    """Start the protocol: a two-column CSV with empty ``human_label`` cells."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HAND_LABEL_COLUMNS)
        for text in texts:
            writer.writerow([text, ""])


def read_hand_label_csv(path: Path) -> list[tuple[str, str]]:
    """All rows as ``(text, human_label)``; unlabeled rows carry ``""``."""
    if not path.exists():
        msg = f"hand-label CSV not found at {path}; write it with write_hand_label_csv"
        raise ConfigError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or tuple(header) != HAND_LABEL_COLUMNS:
            msg = f"{path} is not a hand-label CSV (header {header!r})"
            raise ConfigError(msg)
        return [(row[0], row[1].strip()) for row in reader if row]


def hand_label_loop(
    path: Path,
    task: TaskSpec,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    """The tiny CLI labeling loop over the unlabeled rows of the CSV.

    Per row: show the text and the numbered label vocabulary; accept a number,
    a label string, ``s`` to skip, or ``q`` to stop. Progress is written back
    after the loop so a quit never loses completed labels. Returns the number
    of rows labeled this session.
    """
    rows = read_hand_label_csv(path)
    options = ", ".join(f"[{i + 1}] {label}" for i, label in enumerate(task.labels))
    labeled = 0
    for index, (text, label) in enumerate(rows):
        if label:
            continue
        output_fn(f"\n({index + 1}/{len(rows)}) {text}")
        answer = input_fn(f"{options}, [s]kip, [q]uit > ").strip().casefold()
        if answer == "q":
            break
        if answer == "s" or not answer:
            continue
        if answer.isdigit() and 1 <= int(answer) <= len(task.labels):
            choice = task.labels[int(answer) - 1]
        elif answer in task.labels:
            choice = answer
        else:
            output_fn(f"unrecognised reply {answer!r}; skipping")
            continue
        rows[index] = (text, choice)
        labeled += 1
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HAND_LABEL_COLUMNS)
        writer.writerows(rows)
    return labeled


@dataclass(frozen=True)
class ValidationReport:
    """Prompt validation against the human 200: accuracy plus the refinement fuel."""

    accuracy: float
    n_scored: int
    disagreements: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "disagreements": self.disagreements,
            "n_scored": self.n_scored,
        }


def validate_against_hand_labels(
    frame: pl.DataFrame, hand_rows: Sequence[tuple[str, str]], task: TaskSpec
) -> ValidationReport:
    """Score ``majority_vote`` against the hand labels, keeping the rationales.

    Each disagreement carries the text, both labels, and every model's
    rationale column present in the table — the post's refinement loop input
    ("Use those disagreements and the LLMs rationale to re-label examples and
    refine your prompt").
    """
    labeled = {text: label for text, label in hand_rows if label}
    if not labeled:
        msg = "no labeled rows in the hand-label sample"
        raise ConfigError(msg)
    for label in labeled.values():
        if label not in task.labels:
            msg = f"hand label {label!r} not in task vocabulary {task.labels}"
            raise ConfigError(msg)

    rationale_columns = [name for name in frame.columns if name.endswith("__rationale")]
    hits = 0
    scored = 0
    disagreements: list[dict[str, str]] = []
    for row in frame.iter_rows(named=True):
        text = str(row["text"])
        if text not in labeled:
            continue
        scored += 1
        vote = str(row["majority_vote"])
        if vote == labeled[text]:
            hits += 1
            continue
        record = {"text": text, "human_label": labeled[text], "majority_vote": vote}
        for name in rationale_columns:
            record[name] = str(row[name])
        disagreements.append(record)
    if scored == 0:
        msg = "no overlap between the vote table and the hand-label sample"
        raise ConfigError(msg)
    return ValidationReport(
        accuracy=hits / scored, n_scored=scored, disagreements=disagreements
    )
