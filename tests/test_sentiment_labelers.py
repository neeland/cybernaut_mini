"""Labeler pool tests: offline lexicon labelers on real fixture headlines,
the verbatim few-shot prompt, and the gated optional labelers.

Story text is real: the committed CC-News fixture corpus's titles
(``data/01_raw/fixtures/documents.jsonl``). Flair tests skip when the optional
package is absent; FinBERT tests are opt-in behind
``CYBERNAUT_MINI_SENTIMENT_DOWNLOADS=1`` (they download weights); the LLM
labeler is exercised through an injected fake client, never a network.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cybernaut_mini.config import ConfigError
from cybernaut_mini.sentiment.data import DOWNLOAD_ENV, load_fixture_stories
from cybernaut_mini.sentiment.labelers import (
    DEFAULT_TEXTBLOB_THRESHOLDS,
    DEFAULT_VADER_THRESHOLDS,
    FEW_SHOT_PROMPT,
    FewShotLLMLabeler,
    FlairLabeler,
    RandomLabeler,
    TextBlobLabeler,
    VaderLabeler,
    build_label_matrix,
    build_pool,
    parse_llm_reply,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DOCUMENTS = REPO_ROOT / "data" / "01_raw" / "fixtures" / "documents.jsonl"
PROMPT_CONFIG = REPO_ROOT / "configs" / "sentiment" / "few_shot_prompt.txt"

_FLAIR_INSTALLED = importlib.util.find_spec("flair") is not None
_DOWNLOADS = bool(os.environ.get(DOWNLOAD_ENV, "").strip())


@pytest.fixture(scope="module")
def stories() -> list[str]:
    return load_fixture_stories(FIXTURE_DOCUMENTS, max_rows=24)


# ── lexicon labelers ─────────────────────────────────────────────────────────


def test_textblob_labels_are_three_class_and_deterministic(stories: list[str]) -> None:
    labeler = TextBlobLabeler(0.10)
    labels = labeler.label(stories)
    assert len(labels) == len(stories)
    assert set(labels) <= {-1, 0, 1}
    assert labels == labeler.label(stories)


def test_vader_labels_are_three_class_and_deterministic(stories: list[str]) -> None:
    labeler = VaderLabeler(0.10)
    labels = labeler.label(stories)
    assert len(labels) == len(stories)
    assert set(labels) <= {-1, 0, 1}
    assert labels == labeler.label(stories)


@pytest.mark.parametrize("cls", [TextBlobLabeler, VaderLabeler])
def test_higher_thresholds_never_add_non_neutral_labels(
    cls: type[TextBlobLabeler] | type[VaderLabeler], stories: list[str]
) -> None:
    """Raising the threshold can only push scores into the NEU band."""
    counts = []
    for threshold in (0.10, 0.30, 0.45):
        labels = cls(threshold).label(stories)
        counts.append(sum(1 for label in labels if label != 0))
    assert counts[0] >= counts[1] >= counts[2]


def test_lexicon_column_names_follow_the_post() -> None:
    assert TextBlobLabeler(0.1).name == "TextBlob-0.10"
    assert VaderLabeler(0.3).name == "VADER-0.30"


def test_non_positive_thresholds_rejected() -> None:
    with pytest.raises(ConfigError):
        TextBlobLabeler(0.0)
    with pytest.raises(ConfigError):
        VaderLabeler(-0.1)


# ── random baseline ──────────────────────────────────────────────────────────


def test_random_labeler_is_seeded_and_uniformly_three_class(stories: list[str]) -> None:
    labeler = RandomLabeler(seed=42)
    labels = labeler.label(stories)
    assert labels == RandomLabeler(seed=42).label(stories)
    assert set(labels) <= {-1, 0, 1}
    assert labels != RandomLabeler(seed=7).label(stories)


# ── the verbatim few-shot prompt + LLM labeler ───────────────────────────────


def test_prompt_config_file_matches_module_constant() -> None:
    """configs/sentiment/few_shot_prompt.txt is the canonical verbatim copy."""
    assert PROMPT_CONFIG.read_text(encoding="utf-8") == FEW_SHOT_PROMPT + "\n"


def test_prompt_contains_the_nine_examples_and_the_rules() -> None:
    assert FEW_SHOT_PROMPT.count("Story:") == 10  # 9 examples + the slot
    assert FEW_SHOT_PROMPT.count("Reply: POS") == 3
    assert FEW_SHOT_PROMPT.count("Reply: NEU") == 3
    assert FEW_SHOT_PROMPT.count("Reply: NEG") == 3
    assert "you MUST reply with NEU" in FEW_SHOT_PROMPT
    assert FEW_SHOT_PROMPT.rstrip().endswith("Reply:")


@pytest.mark.parametrize(
    ("reply", "label"),
    [
        ("POS", 1),
        ("neg", -1),
        ("NEU", 0),
        (" POS \n", 1),
        ("Reply: NEG", -1),
        ("positive vibes only", 0),  # not a bare POS/NEU/NEG token → unsure → NEU
        ("", 0),
        ("garbage", 0),
    ],
)
def test_parse_llm_reply_maps_and_defaults_to_neu(reply: str, label: int) -> None:
    assert parse_llm_reply(reply) == label


class _FakeClient:
    """Records requests; replies from a scripted list."""

    def __init__(self, replies: list[str]) -> None:
        self.requests: list[dict[str, Any]] = []
        self._replies = iter(replies)
        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        message = SimpleNamespace(content=next(self._replies))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_llm_labeler_uses_verbatim_prompt_at_temperature_zero(
    stories: list[str],
) -> None:
    client = _FakeClient(["POS", "NEG", "not-a-label"])
    labeler = FewShotLLMLabeler(
        model="test/model",
        name="TestLLM",
        api_key="key",
        client_factory=lambda api_key, base_url: client,
    )
    labels = labeler.label(stories[:3])
    assert labels == [1, -1, 0]  # NEU on parse failure
    assert len(client.requests) == 3
    for request, story in zip(client.requests, stories[:3], strict=True):
        assert request["temperature"] == 0.0
        assert request["model"] == "test/model"
        (message,) = request["messages"]
        assert message["content"] == FEW_SHOT_PROMPT.format(story=story)
        # The nine examples travel with every request, unmodified.
        assert "Company releases profit warning after the sales of XYZ disappoint." in (
            message["content"]
        )


def test_llm_labeler_without_key_raises(stories: list[str]) -> None:
    labeler = FewShotLLMLabeler(model="test/model", environ={})
    with pytest.raises(ConfigError, match="No API key"):
        labeler.label(stories[:1])


# ── optional / gated labelers ────────────────────────────────────────────────


@pytest.mark.skipif(_FLAIR_INSTALLED, reason="flair installed; the guard cannot fire")
def test_flair_labeler_without_flair_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="flair"):
        FlairLabeler(0.70)


@pytest.mark.skipif(not _FLAIR_INSTALLED, reason="optional flair not installed")
def test_flair_labeler_confidence_gates_to_neu(stories: list[str]) -> None:
    pytest.importorskip("flair")
    strict = FlairLabeler(0.99).label(stories[:8])
    loose = FlairLabeler(0.70).label(stories[:8])
    assert set(strict) <= {-1, 0, 1}
    assert sum(1 for label in strict if label != 0) <= sum(
        1 for label in loose if label != 0
    )


@pytest.mark.skipif(not _DOWNLOADS, reason=f"{DOWNLOAD_ENV} not set; FinBERT downloads")
def test_finbert_labelers_download_and_label(stories: list[str]) -> None:
    from cybernaut_mini.sentiment.labelers import FinbertLabeler

    finbert = FinbertLabeler("ProsusAI/finbert")
    assert finbert.name == "FinBERT"
    labels = finbert.label(stories[:4])
    assert len(labels) == 4
    assert set(labels) <= {-1, 0, 1}


# ── pool construction + label matrix ─────────────────────────────────────────


def test_build_pool_default_is_the_offline_sweep() -> None:
    pool = build_pool({})
    names = [labeler.name for labeler in pool]
    assert names == (
        [f"TextBlob-{t:.2f}" for t in DEFAULT_TEXTBLOB_THRESHOLDS]
        + [f"VADER-{t:.2f}" for t in DEFAULT_VADER_THRESHOLDS]
        + ["Random"]
    )


def test_build_pool_respects_params() -> None:
    pool = build_pool(
        {"textblob_thresholds": [0.15], "vader_thresholds": [], "random_seed": None}
    )
    assert [labeler.name for labeler in pool] == ["TextBlob-0.15"]


def test_build_label_matrix_one_column_per_labeler(stories: list[str]) -> None:
    pool = build_pool({"textblob_thresholds": [0.10], "vader_thresholds": [0.10]})
    matrix = build_label_matrix(stories, pool)
    assert sorted(matrix) == ["Random", "TextBlob-0.10", "VADER-0.10"]
    assert all(len(column) == len(stories) for column in matrix.values())


def test_build_label_matrix_rejects_empty_and_duplicates(stories: list[str]) -> None:
    with pytest.raises(ConfigError):
        build_label_matrix([], [TextBlobLabeler(0.1)])
    with pytest.raises(ConfigError, match="duplicate"):
        build_label_matrix(stories[:2], [TextBlobLabeler(0.1), TextBlobLabeler(0.1)])
