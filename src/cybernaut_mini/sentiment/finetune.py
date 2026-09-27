"""Qwen3-0.6B next-token classification fine-tune: the post's recipe, mechanics-first.

Classification as next-token prediction: the chat prompt carries the snippet,
the assistant turn is the label token plus ``<|im_end|>``, loss is masked to
the answer, padding is LEFT, and evaluation accuracy excludes the EOS token.
Everything unit-testable (tokenization, masking, metrics, recipe) is a pure
function over tiny arrays; assembling the real Trainer downloads weights and
is opt-in. NO training run happens anywhere in this repo's tests.

Blog ref: https://nosible.com/blog/fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction —
    "The implementation is straightforward, but three details are critical":
    (1) label masking ``labels = [-100] * len(prompt_tokens["input_ids"]) +
    answer_tokens["input_ids"]``; (2) left padding with ``pad = eos``;
    (3) EOS exclusion in metrics ("We initially got suspiciously good results
    because we included EOS tokens"). Plus the full training script: the
    ``assert len(answer_tokens["input_ids"]) == 2`` check, the
    ``compute_metrics_without_eos`` / ``preprocess_logits_for_metrics`` pair,
    the TrainingArguments recipe (lr 2e-5, cosine, warmup 0.03, weight decay
    0.1, NEFTune alpha 5, group_by_length), and the dict-of-eval-sets
    {train 5%, val, phrasebank} with best-on-``eval_val_loss``. Local copy:
    ``docs/blog-archive/fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction.md``.

Assumptions:
    - The three pedagogical-core details are kept EXACT and unit-tested with
      synthetic tokenizers/arrays: the mask layout, the two-token answer
      assert, the left-padding config, and the shifted
      ``(labels != -100) & (labels != eos)`` metric mask. The buggy pre-fix
      metric (:func:`compute_metrics_with_eos`) is kept alongside so a test
      can PROVE EOS inclusion inflates accuracy, reproducing their bug.
    - The verbatim recipe lives in :data:`TRAINING_RECIPE`;
      :func:`build_training_arguments` filters keys the installed
      ``transformers`` no longer accepts (5.x dropped ``group_by_length``) —
      a documented deviation forced by the pinned dependency, not a choice.
    - Laptop scaling per the gap matrix: batch 4 x grad-accum 8 (the post ran
      16 x 4), epochs 3 (post: 5), ``optim="adamw_torch"`` because the fused
      variant is CUDA-only and this repo is MPS-first. The post's values are
      recoverable by passing them explicitly.
    - The LoRA path (:func:`apply_lora`, peft r=16) is a FAST-PATH DEVIATION:
      the post fine-tunes all 0.6B parameters. It exists so a laptop can
      exercise the code path; full fine-tuning stays the default
      (``lora=False``).
    - ``eval_pred`` reaches the metric functions as numpy arrays of token ids
      (predictions already argmaxed by ``preprocess_logits_for_metrics``),
      matching how ``Trainer`` calls ``compute_metrics`` under that hook.

Alternatives rejected:
    - ``AutoModelForSequenceClassification`` with a 3-way head: the post's
      whole point is hijacking next-token prediction — no classification
      head, no special architecture.
    - ``DataCollatorForLanguageModeling``: it re-derives labels from input
      ids and would destroy the prompt mask; the post uses
      ``DataCollatorForSeq2Seq`` which pads ``labels`` with -100.
    - Dropping the two-token assert for multi-token labels: the single-label-
      token contract is what makes :mod:`.infer`'s one-forward softmax valid;
      a label that tokenizes to more than one token must fail loudly here.
"""

# ruff: noqa: E501 — the referenced post's URL slug alone exceeds the 100-char line limit.

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score, f1_score

from cybernaut_mini.config import ConfigError
from cybernaut_mini.sentiment.prompts import TaskSpec

__all__ = [
    "IM_END",
    "QWEN3_MODEL_ID",
    "TRAINING_RECIPE",
    "apply_lora",
    "build_trainer",
    "build_training_arguments",
    "compute_metrics_with_eos",
    "compute_metrics_without_eos",
    "configure_tokenizer",
    "encode_example",
    "load_model",
    "load_tokenizer",
    "preprocess_logits_for_metrics",
    "tokenize_batch",
]

IntArray = npt.NDArray[np.int64]

#: The post's production model — ~0.6B parameters, MPS-capable in bf16.
QWEN3_MODEL_ID = "Qwen/Qwen3-0.6B"

#: "Standard Qwen end token" appended to every answer string.
IM_END = "<|im_end|>"

#: The post's TrainingArguments, verbatim, EXCEPT the laptop scaling noted in
#: the module docstring (batch 4 x accum 8 vs 16 x 4; epochs 3 vs 5;
#: adamw_torch vs adamw_torch_fused; bf16 off by default on MPS).
TRAINING_RECIPE: Mapping[str, object] = {
    "gradient_accumulation_steps": 8,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "max_grad_norm": 1.0,
    "weight_decay": 0.1,
    "neftune_noise_alpha": 5,
    "bf16": False,
    "optim": "adamw_torch",
    "logging_strategy": "steps",
    "logging_steps": 10,
    "logging_first_step": True,
    "group_by_length": True,
    "torch_compile": False,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_val_loss",
    "report_to": "none",
    "save_strategy": "steps",
    "save_steps": 500,
}


def configure_tokenizer(tokenizer: Any) -> Any:
    """The post's critical detail #2: ``pad = eos`` and LEFT padding, in place."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_tokenizer(model_id: str = QWEN3_MODEL_ID, revision: str | None = None) -> Any:
    """Download-gated: the Qwen3 tokenizer, configured for causal fine-tuning."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True
    )
    return configure_tokenizer(tokenizer)


def encode_example(
    tokenizer: Any,
    *,
    text: str,
    label: str,
    system_prompt: str,
    max_length: int = 2048,
) -> dict[str, list[int]]:
    """One training example, steps 1-5 of the post's ``tokenize`` verbatim.

    Build the chat prompt (generation prompt on, thinking off), tokenize the
    prompt and ``f"{label}<|im_end|>"`` separately so the lengths are known,
    assert the answer is exactly two tokens (label + EOS), then combine with
    the ``[-100] * len(prompt) + answer`` label mask and head-truncate.
    """
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    prompt_str = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    answer_str = f"{label}{IM_END}"

    prompt_tokens = tokenizer(prompt_str, add_special_tokens=False)
    answer_tokens = tokenizer(answer_str, add_special_tokens=False)

    assert len(answer_tokens["input_ids"]) == 2

    input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
    attention_mask = prompt_tokens["attention_mask"] + answer_tokens["attention_mask"]
    labels = [-100] * len(prompt_tokens["input_ids"]) + answer_tokens["input_ids"]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tokenize_batch(
    tokenizer: Any,
    batch: Mapping[str, Sequence[str]],
    *,
    system_prompt: str,
    max_length: int = 2048,
) -> dict[str, list[list[int]]]:
    """The post's batched ``tokenize`` function over ``{text, labels}`` columns."""
    if "text" not in batch or "labels" not in batch:
        msg = f"batch needs 'text' and 'labels' columns, got {sorted(batch)}"
        raise ConfigError(msg)
    input_ids_list: list[list[int]] = []
    attention_mask_list: list[list[int]] = []
    labels_list: list[list[int]] = []
    for text, label in zip(batch["text"], batch["labels"], strict=True):
        encoded = encode_example(
            tokenizer,
            text=text,
            label=label,
            system_prompt=system_prompt,
            max_length=max_length,
        )
        input_ids_list.append(encoded["input_ids"])
        attention_mask_list.append(encoded["attention_mask"])
        labels_list.append(encoded["labels"])
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def compute_metrics_without_eos(
    eval_pred: tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]], eos_token_id: int
) -> dict[str, float]:
    """The post's fixed metric: shift, then mask BOTH the prompt and the EOS token.

    ``predictions[:, :-1]`` against ``labels[:, 1:]`` aligns next-token
    predictions with their targets; the mask
    ``(labels != -100) & (labels != eos_token_id)`` keeps only the label
    token, so accuracy measures classification, not conversational markup.
    """
    predictions, labels = eval_pred
    predictions = predictions[:, :-1]
    labels = labels[:, 1:]

    # Exclude Prompt (-100) AND EOS.
    valid_mask = (labels != -100) & (labels != eos_token_id)

    pred_flat = predictions[valid_mask]
    label_flat = labels[valid_mask]

    f1 = f1_score(label_flat, pred_flat, average="macro")
    accuracy = accuracy_score(label_flat, pred_flat)
    return {"f1": float(f1), "accuracy": float(accuracy)}


def compute_metrics_with_eos(
    eval_pred: tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]],
) -> dict[str, float]:
    """The pre-fix BUGGY metric: masks the prompt but keeps the EOS token.

    Kept so tests can reproduce the post's "suspiciously good results" —
    every correctly predicted ``<|im_end|>`` pads the numerator. Never use
    this for a real evaluation.
    """
    predictions, labels = eval_pred
    predictions = predictions[:, :-1]
    labels = labels[:, 1:]

    valid_mask = labels != -100

    pred_flat = predictions[valid_mask]
    label_flat = labels[valid_mask]

    f1 = f1_score(label_flat, pred_flat, average="macro")
    accuracy = accuracy_score(label_flat, pred_flat)
    return {"f1": float(f1), "accuracy": float(accuracy)}


def preprocess_logits_for_metrics(logits: Any, labels: Any) -> Any:
    """Original logits are (Batch, Seq, Vocab); keep only the argmax indices."""
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def build_training_arguments(output_dir: Path, **overrides: object) -> Any:
    """The verbatim recipe as ``TrainingArguments``, minus unsupported keys.

    *overrides* replace recipe entries (e.g. the post's full-scale
    ``per_device_train_batch_size=16, gradient_accumulation_steps=4,
    num_train_epochs=5``). Keys the installed transformers rejects
    (``group_by_length`` disappeared in 5.x) are dropped, not renamed.
    """
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = dict(TRAINING_RECIPE)
    kwargs.update(overrides)
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    return TrainingArguments(output_dir=str(output_dir), **filtered)


def load_model(
    model_id: str = QWEN3_MODEL_ID, revision: str | None = None, *, lora: bool = False
) -> Any:
    """Download-gated: the causal LM in bf16 with automatic device placement."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if lora:
        model = apply_lora(model)
    return model


def apply_lora(model: Any, *, r: int = 16, lora_alpha: int = 32, dropout: float = 0.05) -> Any:
    """DEVIATION: peft LoRA fast path (r=16); the post fine-tunes every weight."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - peft ships with the repo env
        msg = "the LoRA fast path needs the optional 'peft' package"
        raise ConfigError(msg) from exc

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(model, config)


def build_trainer(
    task: TaskSpec,
    train_rows: Mapping[str, Sequence[str]],
    val_rows: Mapping[str, Sequence[str]],
    phrasebank_rows: Mapping[str, Sequence[str]],
    output_dir: Path,
    *,
    model_id: str = QWEN3_MODEL_ID,
    revision: str | None = None,
    lora: bool = False,
    max_length: int = 2048,
    **argument_overrides: object,
) -> Any:
    """Assemble the post's Trainer: masked datasets, Seq2Seq collator, EOS-free metrics.

    Download-gated (model + tokenizer weights) and NEVER called from tests;
    ``trainer.train()`` is left to the caller, keeping training runs out of
    this repo's automated paths. The eval dict is the post's
    ``{train: first 5%, val, phrasebank}`` with best-on-``eval_val_loss``.
    """
    from functools import partial

    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer

    tokenizer = load_tokenizer(model_id, revision)
    model = load_model(model_id, revision, lora=lora)

    def encode(rows: Mapping[str, Sequence[str]]) -> Any:
        columns = tokenize_batch(
            tokenizer, rows, system_prompt=task.system_prompt, max_length=max_length
        )
        return Dataset.from_dict(columns)

    train_ds = encode(train_rows)
    n_eval_train = max(1, int(0.05 * len(train_ds)))
    eval_datasets = {
        "train": train_ds.select(list(range(n_eval_train))),
        "val": encode(val_rows),
        "phrasebank": encode(phrasebank_rows),
    }
    chat_end_token_id = tokenizer.convert_tokens_to_ids(IM_END)

    return Trainer(
        model=model,
        args=build_training_arguments(output_dir, **argument_overrides),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        train_dataset=train_ds,
        eval_dataset=eval_datasets,
        compute_metrics=partial(compute_metrics_without_eos, eos_token_id=chat_end_token_id),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
