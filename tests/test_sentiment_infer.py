"""Logprobs-as-classifier tests: softmax over exactly the label token ids.

Everything is offline: the tokenizer is a tiny whitespace fake, the model is a
scripted callable returning a fixed logit tensor, and the softmax math is
checked against a hand computation. Loading the released NOSIBLE checkpoints
(:class:`LocalClassifier` with a real model id) downloads weights and is never
exercised here.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cybernaut_mini.config import ConfigError
from cybernaut_mini.sentiment.finetune import IM_END
from cybernaut_mini.sentiment.infer import (
    DEFAULT_CHECKPOINTS,
    Classification,
    classify,
    classify_logits,
    first_token_ids,
    softmax_over_labels,
)
from cybernaut_mini.sentiment.prompts import TASKS, TaskSpec, task_for

SENTIMENT = task_for("financial_sentiment")


class _FakeTokenizer:
    """Whitespace fake mirroring the finetune test tokenizer."""

    eos_token = IM_END

    def __init__(self) -> None:
        self.pad_token: str | None = None
        self.padding_side = "right"
        self._vocab: dict[str, int] = {}
        self.template_calls: list[dict[str, Any]] = []

    def _id(self, token: str) -> int:
        return self._vocab.setdefault(token, len(self._vocab))

    def apply_chat_template(
        self,
        msgs: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
    ) -> str:
        self.template_calls.append(
            {
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        body = " ".join(f"<{msg['role']}> {msg['content']} {IM_END}" for msg in msgs)
        return f"{body} <assistant>"

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        assert add_special_tokens is False
        tokens: list[str] = []
        for chunk in re.split(r"(<\|im_end\|>)", text):
            if chunk == IM_END:
                tokens.append(chunk)
            else:
                tokens.extend(chunk.split())
        ids = [self._id(token) for token in tokens]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


# ── first_token_ids ──────────────────────────────────────────────────────────


def test_first_token_ids_are_distinct_per_label() -> None:
    tokenizer = _FakeTokenizer()
    ids = first_token_ids(tokenizer, SENTIMENT.labels)
    assert len(ids) == 3
    assert len(set(ids)) == 3
    # Multi-token labels contribute their FIRST token id.
    assert first_token_ids(tokenizer, ["negative growth"])[0] == ids[0]


def test_first_token_ids_reject_collisions_and_empties() -> None:
    tokenizer = _FakeTokenizer()
    with pytest.raises(ConfigError):
        first_token_ids(tokenizer, ["positive", "positive outlook"])
    with pytest.raises(ConfigError):
        first_token_ids(tokenizer, [""])


# ── the constrained softmax ──────────────────────────────────────────────────


def test_softmax_over_labels_matches_hand_math() -> None:
    logits = np.zeros(10, dtype=np.float64)
    logits[[2, 5, 7]] = [1.0, 2.0, 3.0]
    logits[0] = 50.0  # a huge non-label logit must not matter
    probabilities = softmax_over_labels(logits, [2, 5, 7])
    expected = np.exp(np.array([1.0, 2.0, 3.0]) - 3.0)
    expected = expected / expected.sum()
    np.testing.assert_allclose(probabilities, expected)
    assert probabilities.sum() == pytest.approx(1.0)


def test_softmax_over_labels_rejects_matrices() -> None:
    with pytest.raises(ConfigError):
        softmax_over_labels(np.zeros((2, 4)), [0, 1])


def test_classify_logits_picks_the_argmax_label() -> None:
    logits = np.zeros(10, dtype=np.float64)
    logits[[1, 2, 3]] = [0.0, 4.0, 1.0]  # negative, neutral, positive slots
    result = classify_logits(logits, SENTIMENT, [1, 2, 3])
    assert isinstance(result, Classification)
    assert result.label == "neutral"
    assert set(result.probabilities) == set(SENTIMENT.labels)
    assert result.confidence == result.probabilities["neutral"]
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_classify_logits_rejects_id_label_mismatch() -> None:
    with pytest.raises(ConfigError):
        classify_logits(np.zeros(10), SENTIMENT, [1, 2])


# ── one forward pass end to end ──────────────────────────────────────────────


class _FakeModel:
    """Returns fixed final-position logits; records the input batch."""

    def __init__(self, vocab_size: int, hot: dict[int, float]) -> None:
        import torch

        self.calls: list[Any] = []
        logits = torch.zeros((1, 1, vocab_size), dtype=torch.float32)
        for token_id, value in hot.items():
            logits[0, 0, token_id] = value
        self._logits = logits

    def __call__(self, *, input_ids: Any) -> Any:
        import torch

        self.calls.append(input_ids)
        # Expand to the sequence length so [-1] is genuinely the final position.
        seq_len = input_ids.shape[1]
        logits = self._logits.expand(1, seq_len, -1).clone()
        logits[0, :-1, :] = 0.0  # earlier positions carry no signal
        return SimpleNamespace(logits=torch.as_tensor(logits))


def test_classify_runs_one_forward_pass_with_the_training_prompt() -> None:
    pytest.importorskip("torch")
    tokenizer = _FakeTokenizer()
    token_ids = first_token_ids(tokenizer, SENTIMENT.labels)
    hot = {token_ids[2]: 5.0, token_ids[0]: 1.0}  # "positive" wins
    model = _FakeModel(vocab_size=200, hot=hot)
    result = classify(model, tokenizer, "Profit rose 12% on record demand.", SENTIMENT)
    assert result.label == "positive"
    assert result.confidence > result.probabilities["neutral"]
    assert len(model.calls) == 1  # ONE forward pass, no generation loop
    # The prompt is built exactly like training: generation prompt on, thinking off.
    template_call = tokenizer.template_calls[0]
    assert template_call == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


# ── checkpoint registry ──────────────────────────────────────────────────────


def test_default_checkpoints_cover_every_task() -> None:
    assert set(DEFAULT_CHECKPOINTS) == set(TASKS)
    for model_id in DEFAULT_CHECKPOINTS.values():
        assert model_id.startswith("NOSIBLE/")


def test_local_classifier_requires_a_checkpoint_for_unknown_tasks() -> None:
    pytest.importorskip("transformers")
    from cybernaut_mini.sentiment.infer import LocalClassifier

    custom = TaskSpec(
        name="custom_task",
        template="{text}",
        response_field="label",
        labels=("a", "b"),
        label_ints=(0, 1),
        default_label="a",
        system_prompt="s",
    )
    with pytest.raises(ConfigError):
        LocalClassifier(custom)  # fails BEFORE any weight download
