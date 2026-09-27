"""Fine-tune mechanics tests: the three critical details, with tiny tensors.

The post's pedagogical core is unit-tested exactly: (1) the
``[-100] * len(prompt) + answer`` label mask with the two-token answer assert;
(2) LEFT padding with ``pad = eos``; (3) the shifted
``(labels != -100) & (labels != eos)`` metric mask — including a synthetic
batch PROVING that keeping EOS inflates accuracy (their bug, reproduced).

Everything is offline: the tokenizer is a tiny whitespace fake, metric inputs
are hand-built numpy arrays, logit preprocessing uses tiny torch tensors, and
``TrainingArguments`` construction touches no network. Loading real Qwen3
weights (``load_tokenizer``/``load_model``/``build_trainer``) is opt-in and
never happens here; NO training run happens anywhere in this repo's tests.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

import numpy as np
import pytest

from cybernaut_mini.config import ConfigError
from cybernaut_mini.sentiment.finetune import (
    IM_END,
    TRAINING_RECIPE,
    build_training_arguments,
    compute_metrics_with_eos,
    compute_metrics_without_eos,
    configure_tokenizer,
    encode_example,
    preprocess_logits_for_metrics,
    tokenize_batch,
)
from cybernaut_mini.sentiment.prompts import task_for

SENTIMENT = task_for("financial_sentiment")

class _FakeTokenizer:
    """Whitespace tokenizer with a stable vocabulary and a chat template.

    ``<|im_end|>`` is a single token, so ``f"{label}{IM_END}"`` tokenizes to
    exactly two tokens for single-word labels — the training contract.
    """

    eos_token = IM_END

    def __init__(self) -> None:
        self.pad_token: str | None = None
        self.padding_side = "right"
        self._vocab: dict[str, int] = {}
        self.template_calls: list[dict[str, Any]] = []

    def _id(self, token: str) -> int:
        return self._vocab.setdefault(token, len(self._vocab) + 10)

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

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._id(token)


# ── critical detail #2: left padding, pad = eos ──────────────────────────────


def test_configure_tokenizer_sets_left_padding_and_pad_eq_eos() -> None:
    tokenizer = _FakeTokenizer()
    configured = configure_tokenizer(tokenizer)
    assert configured is tokenizer
    assert tokenizer.pad_token == IM_END
    assert tokenizer.padding_side == "left"


def test_configure_tokenizer_keeps_an_existing_pad_token() -> None:
    tokenizer = _FakeTokenizer()
    tokenizer.pad_token = "<pad>"
    configure_tokenizer(tokenizer)
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.padding_side == "left"  # regression: LEFT is unconditional


# ── critical detail #1: the label mask ───────────────────────────────────────


def test_encode_example_masks_the_prompt_and_keeps_the_answer() -> None:
    tokenizer = _FakeTokenizer()
    encoded = encode_example(
        tokenizer,
        text="Shares fell 8% after the recall.",
        label="negative",
        system_prompt=SENTIMENT.system_prompt,
    )
    prompt_len = len(encoded["input_ids"]) - 2
    answer_ids = tokenizer(f"negative{IM_END}", add_special_tokens=False)["input_ids"]
    assert len(answer_ids) == 2
    assert encoded["labels"] == [-100] * prompt_len + answer_ids
    assert encoded["input_ids"][-2:] == answer_ids
    assert encoded["attention_mask"] == [1] * len(encoded["input_ids"])
    # The chat template ran exactly as in the post: generation prompt on,
    # thinking off, string output.
    (call,) = tokenizer.template_calls
    assert call == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}


def test_encode_example_asserts_two_answer_tokens() -> None:
    tokenizer = _FakeTokenizer()
    with pytest.raises(AssertionError):
        encode_example(
            tokenizer,
            text="Shares fell.",
            label="very negative",  # two words -> three answer tokens
            system_prompt=SENTIMENT.system_prompt,
        )


def test_encode_example_head_truncates_to_max_length() -> None:
    tokenizer = _FakeTokenizer()
    encoded = encode_example(
        tokenizer,
        text="word " * 50,
        label="neutral",
        system_prompt=SENTIMENT.system_prompt,
        max_length=8,
    )
    assert len(encoded["input_ids"]) == 8
    assert len(encoded["attention_mask"]) == 8
    assert len(encoded["labels"]) == 8


def test_tokenize_batch_encodes_column_wise() -> None:
    tokenizer = _FakeTokenizer()
    batch = {
        "text": ["Shares fell 8%.", "Profit rose 12%."],
        "labels": ["negative", "positive"],
    }
    columns = tokenize_batch(tokenizer, batch, system_prompt=SENTIMENT.system_prompt)
    assert sorted(columns) == ["attention_mask", "input_ids", "labels"]
    assert len(columns["input_ids"]) == 2
    for input_ids, labels in zip(columns["input_ids"], columns["labels"], strict=True):
        assert len(input_ids) == len(labels)
        assert labels.count(-100) == len(labels) - 2


def test_tokenize_batch_rejects_missing_columns() -> None:
    with pytest.raises(ConfigError):
        tokenize_batch(_FakeTokenizer(), {"text": ["x"]}, system_prompt="s")


# ── critical detail #3: EOS-free metrics ─────────────────────────────────────

EOS_ID = 99
POS_ID = 7
NEG_ID = 8


def _synthetic_batch() -> tuple[np.ndarray, np.ndarray]:
    """Two rows: label predicted right in row 0, wrong in row 1; EOS right in both.

    Layout per row (length 5): three prompt positions (-100), the label token,
    then EOS. ``predictions[i]`` holds the argmaxed NEXT-token prediction made
    AT position i, so the prediction aligned with ``labels[:, 1:][i, j]`` is
    ``predictions[:, :-1][i, j]``.
    """
    labels = np.array(
        [
            [-100, -100, -100, POS_ID, EOS_ID],
            [-100, -100, -100, POS_ID, EOS_ID],
        ],
        dtype=np.int64,
    )
    # predictions[:, 2] is the prediction FOR the label slot (labels[:, 3]);
    # predictions[:, 3] is the prediction FOR the EOS slot (labels[:, 4]).
    predictions = np.array(
        [
            [0, 0, POS_ID, EOS_ID, 0],
            [0, 0, NEG_ID, EOS_ID, 0],
        ],
        dtype=np.int64,
    )
    return predictions, labels


def test_metric_without_eos_measures_only_the_label_token() -> None:
    metrics = compute_metrics_without_eos(_synthetic_batch(), eos_token_id=EOS_ID)
    assert metrics["accuracy"] == pytest.approx(0.5)  # 1 of 2 labels right


def test_eos_inclusion_inflates_accuracy() -> None:
    """Reproduce the post's bug: 'suspiciously good results' with EOS kept.

    True classification accuracy is 0.5 (one of two labels right), but every
    correctly predicted ``<|im_end|>`` pads the numerator: the buggy metric
    reports 3/4. This test is the PROOF the EOS mask matters.
    """
    batch = _synthetic_batch()
    buggy = compute_metrics_with_eos(batch)
    fixed = compute_metrics_without_eos(batch, eos_token_id=EOS_ID)
    assert buggy["accuracy"] == pytest.approx(0.75)
    assert fixed["accuracy"] == pytest.approx(0.5)
    assert buggy["accuracy"] > fixed["accuracy"]


def test_metric_shift_aligns_predictions_with_next_tokens() -> None:
    """A row whose only correct prediction sits at the shifted position scores 1.0."""
    labels = np.array([[-100, -100, POS_ID, EOS_ID]], dtype=np.int64)
    predictions = np.array([[0, POS_ID, EOS_ID, 0]], dtype=np.int64)
    metrics = compute_metrics_without_eos((predictions, labels), eos_token_id=EOS_ID)
    assert metrics["accuracy"] == pytest.approx(1.0)
    # Without the shift the same arrays would score zero on the label slot.
    unshifted_mask = (labels != -100) & (labels != EOS_ID)
    assert (predictions[unshifted_mask] == labels[unshifted_mask]).sum() == 0


def test_perfect_batch_scores_one_on_both_metrics() -> None:
    labels = np.array([[-100, POS_ID, EOS_ID]], dtype=np.int64)
    predictions = np.array([[POS_ID, EOS_ID, 0]], dtype=np.int64)
    assert compute_metrics_without_eos((predictions, labels), eos_token_id=EOS_ID)[
        "accuracy"
    ] == pytest.approx(1.0)
    assert compute_metrics_with_eos((predictions, labels))["accuracy"] == pytest.approx(1.0)


def test_preprocess_logits_keeps_only_the_argmax() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [[[0.1, 2.0, 0.3], [4.0, 0.2, 0.1]]], dtype=torch.float32
    )  # (batch=1, seq=2, vocab=3)
    ids = preprocess_logits_for_metrics(logits, labels=None)
    assert ids.tolist() == [[1, 0]]
    # Tuple-shaped model outputs: logits always come first.
    ids_from_tuple = preprocess_logits_for_metrics((logits, object()), labels=None)
    assert ids_from_tuple.tolist() == [[1, 0]]


# ── the verbatim recipe ──────────────────────────────────────────────────────


def test_training_recipe_keeps_the_post_hyperparameters() -> None:
    assert TRAINING_RECIPE["learning_rate"] == 2e-5
    assert TRAINING_RECIPE["lr_scheduler_type"] == "cosine"
    assert TRAINING_RECIPE["warmup_ratio"] == 0.03
    assert TRAINING_RECIPE["weight_decay"] == 0.1
    assert TRAINING_RECIPE["neftune_noise_alpha"] == 5
    assert TRAINING_RECIPE["group_by_length"] is True
    assert TRAINING_RECIPE["max_grad_norm"] == 1.0
    assert TRAINING_RECIPE["load_best_model_at_end"] is True
    assert TRAINING_RECIPE["metric_for_best_model"] == "eval_val_loss"


def test_build_training_arguments_applies_recipe_and_overrides(tmp_path: Any) -> None:
    transformers = pytest.importorskip("transformers")
    args = build_training_arguments(
        tmp_path,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        bogus_future_flag=True,  # unsupported keys are dropped, not fatal
    )
    assert args.learning_rate == pytest.approx(2e-5)
    assert args.warmup_ratio == pytest.approx(0.03)
    assert args.weight_decay == pytest.approx(0.1)
    assert args.neftune_noise_alpha == 5
    assert args.metric_for_best_model == "eval_val_loss"
    # The post's full-scale overrides land verbatim.
    assert args.per_device_train_batch_size == 16
    assert args.gradient_accumulation_steps == 4
    assert args.num_train_epochs == 5
    supported = set(inspect.signature(transformers.TrainingArguments.__init__).parameters)
    if "group_by_length" in supported:
        assert args.group_by_length is True  # honoured where transformers still has it
