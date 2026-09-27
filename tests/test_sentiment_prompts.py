"""Prompt registry tests: the three verbatim decision-tree prompts, the config
copies, the archived-post fidelity, and the strict-JSON reply parser.

Everything is offline: the prompts are constants, the reply parser is pure,
and the archived blog post is committed under ``docs/blog-archive/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cybernaut_mini.config import ConfigError
from cybernaut_mini.sentiment.prompts import (
    FINANCIAL_SENTIMENT_PROMPT,
    FORWARD_LOOKING_PROMPT,
    PREDICTION_PROMPT,
    TASKS,
    TaskSpec,
    task_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "sentiment"
ARCHIVE = (
    REPO_ROOT
    / "docs"
    / "blog-archive"
    / "fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction.md"
)

_PROMPTS = {
    "financial_sentiment": FINANCIAL_SENTIMENT_PROMPT,
    "forward_looking": FORWARD_LOOKING_PROMPT,
    "prediction": PREDICTION_PROMPT,
}


# ── verbatim fidelity ────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(_PROMPTS))
def test_config_copies_are_byte_identical(name: str) -> None:
    config = CONFIG_DIR / f"decision_tree_{name}.txt"
    assert config.read_text(encoding="utf-8") == _PROMPTS[name]


@pytest.mark.parametrize("name", sorted(_PROMPTS))
def test_prompts_match_the_archived_post(name: str) -> None:
    """The constants are substrings of the committed archive — nothing paraphrased."""
    archive_text = ARCHIVE.read_text(encoding="utf-8")
    assert _PROMPTS[name].strip() in archive_text


def test_prompts_carry_the_shared_scaffolding() -> None:
    for prompt in _PROMPTS.values():
        assert "START" in prompt
        assert "DO NOT WRITE ANY PREAMBLE JUST RETURN JSON." in prompt
        assert '"rationale"' in prompt
        assert "{text}" in prompt
    assert "Materiality note:" in FINANCIAL_SENTIMENT_PROMPT
    assert "Pay careful attention to the logic. Don't deviate." in FINANCIAL_SENTIMENT_PROMPT
    assert 'DEFER TO A "neutral" CLASSIFICATION' in FINANCIAL_SENTIMENT_PROMPT
    assert 'DEFER TO A "not-forward" CLASSIFICATION' in FORWARD_LOOKING_PROMPT
    assert 'DEFER TO A "not-predictive" CLASSIFICATION' in PREDICTION_PROMPT


def test_build_prompt_fills_only_the_text_slot() -> None:
    task = task_for("financial_sentiment")
    rendered = task.build_prompt("Shares fell 8% after the recall.")
    assert "Shares fell 8% after the recall." in rendered
    assert "{text}" not in rendered
    # The JSON example's escaped braces survive formatting.
    assert '"financial_sentiment": "either negative, neutral, or positive"' in rendered


# ── the task registry ────────────────────────────────────────────────────────


def test_registry_has_the_three_tasks_with_their_contracts() -> None:
    assert sorted(TASKS) == ["financial_sentiment", "forward_looking", "prediction"]
    sentiment = TASKS["financial_sentiment"]
    assert sentiment.response_field == "financial_sentiment"
    assert sentiment.labels == ("negative", "neutral", "positive")
    assert sentiment.label_to_int == {"negative": -1, "neutral": 0, "positive": 1}
    assert sentiment.default_label == "neutral"
    forward = TASKS["forward_looking"]
    assert forward.response_field == "tense"
    assert forward.default_label == "not-forward"
    prediction = TASKS["prediction"]
    assert prediction.response_field == "causal"
    assert prediction.default_label == "not-predictive"


def test_int_mappings_round_trip() -> None:
    for task in TASKS.values():
        for label in task.labels:
            assert task.int_to_label[task.label_to_int[label]] == label


def test_task_for_unknown_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="unknown sentiment task"):
        task_for("vibes")


# ── the strict-JSON reply parser ─────────────────────────────────────────────


def _sentiment() -> TaskSpec:
    return TASKS["financial_sentiment"]


def test_parse_reply_accepts_a_clean_json_object() -> None:
    label, rationale = _sentiment().parse_reply(
        '{"rationale": "Profit warning.", "financial_sentiment": "negative"}'
    )
    assert label == "negative"
    assert rationale == "Profit warning."


def test_parse_reply_tolerates_key_order_whitespace_and_case() -> None:
    label, _ = _sentiment().parse_reply(
        '\n  {"financial_sentiment":   "POSITIVE",\n "rationale": "r"}  \n'
    )
    assert label == "positive"


def test_parse_reply_unwraps_one_json_fence() -> None:
    raw = '```json\n{"rationale": "r", "financial_sentiment": "neutral"}\n```'
    label, _ = _sentiment().parse_reply(raw)
    assert label == "neutral"


def test_parse_reply_finds_a_bare_object_after_preamble() -> None:
    raw = 'Sure! Here is the JSON: {"rationale": "r", "financial_sentiment": "positive"}'
    label, _ = _sentiment().parse_reply(raw)
    assert label == "positive"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "negative",
        '{"rationale": "r"}',
        '{"rationale": "r", "financial_sentiment": "bullish"}',
        '{"rationale": "r", "sentiment": "negative"}',
        '{"rationale": "r", "financial_sentiment": 1}',
    ],
)
def test_parse_reply_rejects_noncompliant_replies(raw: str) -> None:
    label, _ = _sentiment().parse_reply(raw)
    assert label is None


def test_parse_reply_uses_each_tasks_own_field() -> None:
    label, _ = TASKS["forward_looking"].parse_reply(
        '{"tense": "forward", "rationale": "launch is future"}'
    )
    assert label == "forward"
    label, _ = TASKS["prediction"].parse_reply(
        '{"rationale": "analyst estimate", "causal": "predictive"}'
    )
    assert label == "predictive"
