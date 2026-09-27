"""Logprobs-as-classifier inference: one forward pass, softmax over three token ids.

The fine-tuned causal model never generates: run the chat prompt through a
single forward pass, take the logits at the FINAL position, and softmax over
ONLY the first token id of each label. The argmax is the classification, the
softmax values are the per-label confidences — the clean local equivalent of
the post's ``regex + logprobs + top_logprobs`` endpoint trick.

Blog ref: https://nosible.com/blog/fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction —
    the "Client code to interact with your HF Inference Endpoint" block:
    ``max_tokens=1, logprobs=True, top_logprobs=len(labels)`` with
    ``"regex": "(positive|neutral|negative)"`` and thinking disabled, plus
    the closing insight: "causal models aren't just for text generation.
    When you understand how to leverage logprobs ... they can be powerful,
    production ready classification models too." Local copy:
    ``docs/blog-archive/fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction.md``.

Assumptions:
    - Restricting the softmax to the three label first-token ids IS the
      post's regex constraint, done locally: the regex forbids any other
      first token, so renormalizing over exactly those ids reproduces the
      constrained distribution without a server.
    - One token identifies each label because :mod:`.finetune` asserts every
      answer is exactly (label token, EOS); :func:`first_token_ids` re-checks
      that the ids are distinct, since two labels sharing a first token would
      silently merge classes.
    - The prompt is built exactly like training (same system prompt, chat
      template, generation prompt on, thinking off) — train/inference skew in
      the template would shift every logit.
    - The softmax runs in float64 numpy on the extracted three logits; at
      three values the dtype dance is free and keeps the function pure and
      testable without torch.
    - The released NOSIBLE checkpoints (financial-sentiment-v1.1-base,
      forward-looking-v1.2-base, prediction-v1.1-base) are the default model
      per task; loading them downloads weights, so :class:`LocalClassifier`
      is opt-in like every other model download in the repo.

Alternatives rejected:
    - ``model.generate(max_new_tokens=1)``: generation adds sampling/config
      surface for zero benefit; the forward pass already contains the answer.
    - Softmax over the full vocabulary then re-normalizing three entries:
      identical result, but it hides the constraint that makes the trick a
      classifier rather than a language model.
    - Averaging logprobs over all label tokens for multi-token labels: the
      training contract (two-token answers) guarantees single-token labels,
      so the extra machinery would be dead code.
"""

# ruff: noqa: E501 — the referenced post's URL slug alone exceeds the 100-char line limit.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from cybernaut_mini.config import ConfigError
from cybernaut_mini.sentiment.finetune import configure_tokenizer
from cybernaut_mini.sentiment.prompts import TaskSpec

__all__ = [
    "DEFAULT_CHECKPOINTS",
    "Classification",
    "LocalClassifier",
    "classify",
    "classify_logits",
    "first_token_ids",
    "softmax_over_labels",
]

FloatArray = npt.NDArray[np.float64]

#: The released 0.6B checkpoints, one per task (opt-in downloads).
DEFAULT_CHECKPOINTS: dict[str, str] = {
    "financial_sentiment": "NOSIBLE/financial-sentiment-v1.1-base",
    "forward_looking": "NOSIBLE/forward-looking-v1.2-base",
    "prediction": "NOSIBLE/prediction-v1.1-base",
}


@dataclass(frozen=True)
class Classification:
    """One classified snippet: the winning label and every label's confidence."""

    label: str
    probabilities: dict[str, float]

    @property
    def confidence(self) -> float:
        return self.probabilities[self.label]


def first_token_ids(tokenizer: Any, labels: Sequence[str]) -> list[int]:
    """First token id per label, with a collision check.

    The training contract makes each label a single token, but nothing stops
    a caller pointing this at a different tokenizer — two labels sharing a
    first token would merge classes, so that fails loudly.
    """
    ids: list[int] = []
    for label in labels:
        token_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if not token_ids:
            msg = f"label {label!r} tokenizes to nothing"
            raise ConfigError(msg)
        ids.append(int(token_ids[0]))
    if len(set(ids)) != len(ids):
        msg = f"labels {list(labels)} collide at their first token ids {ids}"
        raise ConfigError(msg)
    return ids


def softmax_over_labels(final_logits: npt.ArrayLike, token_ids: Sequence[int]) -> FloatArray:
    """Softmax over ONLY the label token logits at the final position."""
    logits = np.asarray(final_logits, dtype=np.float64)
    if logits.ndim != 1:
        msg = f"expected a 1-D vocab-sized logit vector, got shape {logits.shape}"
        raise ConfigError(msg)
    selected = logits[np.asarray(token_ids, dtype=np.int64)]
    shifted = selected - float(np.max(selected))
    exps = np.exp(shifted)
    return np.asarray(exps / np.sum(exps), dtype=np.float64)


def classify_logits(
    final_logits: npt.ArrayLike, task: TaskSpec, token_ids: Sequence[int]
) -> Classification:
    """Turn one final-position logit vector into a labeled classification."""
    if len(token_ids) != len(task.labels):
        msg = f"{len(token_ids)} token ids for {len(task.labels)} labels"
        raise ConfigError(msg)
    probabilities = softmax_over_labels(final_logits, token_ids)
    ranked = dict(zip(task.labels, (float(p) for p in probabilities), strict=True))
    winner = max(ranked, key=lambda label: ranked[label])
    return Classification(label=winner, probabilities=ranked)


def classify(model: Any, tokenizer: Any, text: str, task: TaskSpec) -> Classification:
    """One forward pass -> softmax over the three label first-token ids.

    The prompt matches training byte-for-byte: same system prompt, chat
    template with the generation prompt appended, thinking disabled.
    """
    import torch

    msgs = [
        {"role": "system", "content": task.system_prompt},
        {"role": "user", "content": text},
    ]
    prompt_str = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    batch = torch.tensor([input_ids], dtype=torch.long)
    if hasattr(model, "device"):
        batch = batch.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=batch)
    final_logits = outputs.logits[0, -1, :].to(torch.float64).cpu().numpy()
    token_ids = first_token_ids(tokenizer, task.labels)
    return classify_logits(final_logits, task, token_ids)


class LocalClassifier:
    """A fine-tuned checkpoint as a batch classifier (opt-in download).

    Defaults to the released NOSIBLE checkpoint for the task; pass
    ``model_id`` to point at a local fine-tune output directory instead.
    ``revision`` pins the HF revision per the repo's reproducibility contract.
    """

    def __init__(
        self,
        task: TaskSpec,
        *,
        model_id: str | None = None,
        revision: str | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.task = task
        resolved = model_id or DEFAULT_CHECKPOINTS.get(task.name)
        if resolved is None:
            msg = f"no default checkpoint for task {task.name!r}; pass model_id="
            raise ConfigError(msg)
        self.model_id = resolved
        self._tokenizer = configure_tokenizer(
            AutoTokenizer.from_pretrained(resolved, revision=revision, trust_remote_code=True)
        )
        self._model: Any = AutoModelForCausalLM.from_pretrained(
            resolved, revision=revision, trust_remote_code=True, device_map="auto"
        )
        self._model.eval()  # type: ignore[no-untyped-call]

    def classify(self, text: str) -> Classification:
        return classify(self._model, self._tokenizer, text, self.task)

    def classify_many(self, texts: Sequence[str]) -> list[Classification]:
        return [self.classify(text) for text in texts]
