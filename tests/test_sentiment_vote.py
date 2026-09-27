"""Vote-table tests: strict-JSON labelers through a fake client, majority vote,
iteration-versioned parquet round-trips, and the 200-row hand-label protocol.

Everything is offline: the OpenAI-compatible client is an injected scripted
fake (never a socket), snippets are real fixture headlines from
``data/01_raw/fixtures/documents.jsonl``, and the CLI labeling loop runs
against scripted ``input_fn``/``output_fn`` callables.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cybernaut_mini.config import ConfigError
from cybernaut_mini.sentiment.data import load_fixture_stories
from cybernaut_mini.sentiment.prompts import task_for
from cybernaut_mini.sentiment.vote import (
    HAND_LABEL_COLUMNS,
    JsonTaskLabeler,
    build_vote_table,
    column_name_for_model,
    hand_label_loop,
    majority_vote,
    read_hand_label_csv,
    read_vote_table,
    sample_hand_label_rows,
    validate_against_hand_labels,
    vote_table_path,
    write_hand_label_csv,
    write_vote_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DOCUMENTS = REPO_ROOT / "data" / "01_raw" / "fixtures" / "documents.jsonl"

SENTIMENT = task_for("financial_sentiment")


@pytest.fixture(scope="module")
def stories() -> list[str]:
    return load_fixture_stories(FIXTURE_DOCUMENTS, max_rows=12)


def _reply(label: str, rationale: str = "because") -> str:
    return json.dumps({"rationale": rationale, "financial_sentiment": label})


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


def _labeler(replies: list[str], **kwargs: Any) -> tuple[JsonTaskLabeler, _FakeClient]:
    client = _FakeClient(replies)
    labeler = JsonTaskLabeler(
        model=kwargs.pop("model", "qwen/qwen3-0.6b"),
        task=SENTIMENT,
        client_factory=lambda api_key, base_url: client,
        **kwargs,
    )
    return labeler, client


# ── column naming ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "column"),
    [
        ("qwen/qwen3-32b", "qwen3_32b"),
        ("x-ai/grok-4-fast", "grok_4_fast"),
        ("llama3.2:3b", "llama3_2_3b"),
        ("Gemma3:4B", "gemma3_4b"),
    ],
)
def test_column_name_for_model(model: str, column: str) -> None:
    assert column_name_for_model(model) == column


def test_column_name_for_model_rejects_unusable_ids() -> None:
    with pytest.raises(ConfigError):
        column_name_for_model("///")


# ── the strict-JSON labeler ──────────────────────────────────────────────────


def test_labeler_sends_verbatim_prompt_with_json_format(stories: list[str]) -> None:
    labeler, client = _labeler([_reply("negative")])
    label, rationale, attempts = labeler.label_one(stories[0])
    assert (label, rationale, attempts) == ("negative", "because", 1)
    (request,) = client.requests
    assert request["temperature"] == 0.0
    assert request["response_format"] == {"type": "json_object"}
    assert "extra_body" not in request
    (message,) = request["messages"]
    assert message["content"] == SENTIMENT.build_prompt(stories[0])


def test_labeler_retries_on_invalid_json_then_succeeds(stories: list[str]) -> None:
    labeler, client = _labeler(["not json", _reply("positive")], max_retries=2)
    label, rationale, attempts = labeler.label_one(stories[0])
    assert (label, attempts) == ("positive", 2)
    assert rationale == "because"
    assert len(client.requests) == 2


def test_labeler_falls_back_to_default_label_after_retries(stories: list[str]) -> None:
    labeler, client = _labeler(["nope"] * 3, max_retries=2)
    label, rationale, attempts = labeler.label_one(stories[0])
    assert (label, rationale, attempts) == (SENTIMENT.default_label, "", 3)
    assert len(client.requests) == 3


def test_labeler_column_counts_parse_failures(stories: list[str]) -> None:
    replies = [_reply("neutral"), "garbage", _reply("positive")]
    labeler, _ = _labeler(replies, max_retries=1)
    column = labeler.label(stories[:2])
    assert column.labels == ["neutral", "positive"]
    assert column.parse_failures == 1


def test_think_variant_gets_reasoning_column_and_extra_body(stories: list[str]) -> None:
    labeler, client = _labeler([_reply("neutral")], model="qwen3:4b", think=True)
    assert labeler.name == "qwen3_4b_reasoning"
    labeler.label_one(stories[0])
    (request,) = client.requests
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_labeler_rejects_negative_max_retries() -> None:
    with pytest.raises(ConfigError):
        JsonTaskLabeler(model="m", task=SENTIMENT, max_retries=-1)


# ── majority vote ────────────────────────────────────────────────────────────


def test_majority_vote_takes_the_modal_label() -> None:
    columns = {
        "a": ["positive", "negative"],
        "b": ["positive", "negative"],
        "c": ["neutral", "positive"],
    }
    assert majority_vote(columns, SENTIMENT) == ["positive", "negative"]


def test_majority_vote_ties_fall_to_the_default_label() -> None:
    columns = {"a": ["positive"], "b": ["negative"]}
    assert majority_vote(columns, SENTIMENT) == ["neutral"]


def test_majority_vote_rejects_empty_and_ragged_input() -> None:
    with pytest.raises(ConfigError):
        majority_vote({}, SENTIMENT)
    with pytest.raises(ConfigError):
        majority_vote({"a": ["positive"], "b": []}, SENTIMENT)


# ── the wide table ───────────────────────────────────────────────────────────


def test_build_vote_table_matches_the_post_layout(stories: list[str]) -> None:
    texts = stories[:2]
    first, _ = _labeler([_reply("positive"), _reply("negative")], model="qwen3:0.6b")
    second, _ = _labeler([_reply("positive"), _reply("neutral")], model="llama3.2:3b")
    frame = build_vote_table(texts, [first, second])
    assert frame.columns == [
        "text",
        "qwen3_0_6b",
        "llama3_2_3b",
        "majority_vote",
        "qwen3_0_6b__rationale",
        "llama3_2_3b__rationale",
    ]
    assert frame["text"].to_list() == list(texts)
    assert frame["majority_vote"].to_list() == ["positive", "neutral"]


def test_build_vote_table_can_drop_rationales(stories: list[str]) -> None:
    labeler, _ = _labeler([_reply("neutral")])
    frame = build_vote_table(stories[:1], [labeler], with_rationales=False)
    assert not any(name.endswith("__rationale") for name in frame.columns)


def test_build_vote_table_rejects_bad_inputs(stories: list[str]) -> None:
    labeler, _ = _labeler([])
    with pytest.raises(ConfigError):
        build_vote_table([], [labeler])
    with pytest.raises(ConfigError):
        build_vote_table(stories[:1], [])
    twin_a, _ = _labeler([_reply("neutral")], model="qwen3:0.6b")
    twin_b, _ = _labeler([_reply("neutral")], model="qwen3:0.6b")
    with pytest.raises(ConfigError):
        build_vote_table(stories[:1], [twin_a, twin_b])
    other = JsonTaskLabeler(model="qwen3:0.6b", task=task_for("prediction"))
    mixed, _ = _labeler([_reply("neutral")], model="llama3.2:3b")
    with pytest.raises(ConfigError):
        build_vote_table(stories[:1], [mixed, other])


def test_vote_table_path_reproduces_the_post_stem(tmp_path: Path) -> None:
    path = vote_table_path(tmp_path, "financial_sentiment", 100_000, 18)
    assert path.name == "financial_sentiment_100.0k_iter_18.parquet"
    small = vote_table_path(tmp_path, "prediction", 3_000, 0)
    assert small.name == "prediction_3.0k_iter_0.parquet"
    with pytest.raises(ConfigError):
        vote_table_path(tmp_path, "prediction", 0, 1)
    with pytest.raises(ConfigError):
        vote_table_path(tmp_path, "prediction", 10, -1)


def test_vote_table_parquet_round_trip(tmp_path: Path, stories: list[str]) -> None:
    labeler, _ = _labeler([_reply("positive"), _reply("positive")])
    frame = build_vote_table(stories[:2], [labeler])
    path = write_vote_table(frame, tmp_path / "tables", SENTIMENT, iteration=3)
    assert path.name == "financial_sentiment_0.0k_iter_3.parquet"
    assert read_vote_table(path).equals(frame)
    with pytest.raises(ConfigError):
        read_vote_table(tmp_path / "missing.parquet")


# ── the hand-label protocol ──────────────────────────────────────────────────


def test_sample_hand_label_rows_is_disjoint_and_deterministic(stories: list[str]) -> None:
    hand, rest = sample_hand_label_rows(stories, n=4, seed=42)
    assert len(hand) == 4
    assert len(rest) == len(stories) - 4
    assert set(hand).isdisjoint(rest)
    assert sorted(hand + rest) == sorted(stories)
    again, _ = sample_hand_label_rows(stories, n=4, seed=42)
    assert again == hand


def test_sample_hand_label_rows_rejects_bad_sizes(stories: list[str]) -> None:
    with pytest.raises(ConfigError):
        sample_hand_label_rows(stories, n=0)
    with pytest.raises(ConfigError):
        sample_hand_label_rows(stories, n=len(stories))


def test_hand_label_csv_round_trip(tmp_path: Path, stories: list[str]) -> None:
    path = tmp_path / "human_labels.csv"
    write_hand_label_csv(stories[:3], path)
    rows = read_hand_label_csv(path)
    assert rows == [(text, "") for text in stories[:3]]
    with pytest.raises(ConfigError):
        read_hand_label_csv(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("text,label\nfoo,bar\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        read_hand_label_csv(bad)


def test_hand_label_loop_labels_skips_and_quits(tmp_path: Path, stories: list[str]) -> None:
    path = tmp_path / "human_labels.csv"
    write_hand_label_csv(stories[:4], path)
    answers = iter(["1", "positive", "s", "q"])
    shown: list[str] = []
    labeled = hand_label_loop(
        path,
        SENTIMENT,
        input_fn=lambda prompt: next(answers),
        output_fn=shown.append,
    )
    assert labeled == 2
    rows = read_hand_label_csv(path)
    assert rows[0][1] == "negative"  # option [1] in prompt order
    assert rows[1][1] == "positive"
    assert rows[2][1] == ""
    assert rows[3][1] == ""
    assert any(stories[0] in line for line in shown)


def test_hand_label_loop_skips_unrecognised_and_already_labeled(
    tmp_path: Path, stories: list[str]
) -> None:
    path = tmp_path / "human_labels.csv"
    write_hand_label_csv(stories[:2], path)
    first_answers = iter(["3", "q"])
    hand_label_loop(
        path, SENTIMENT, input_fn=lambda _: next(first_answers), output_fn=lambda _: None
    )
    answers = iter(["bogus"])
    shown: list[str] = []
    labeled = hand_label_loop(
        path, SENTIMENT, input_fn=lambda _: next(answers), output_fn=shown.append
    )
    assert labeled == 0  # row 0 already labeled; row 1 got an unrecognised reply
    assert any("unrecognised" in line for line in shown)
    rows = read_hand_label_csv(path)
    assert rows[0][1] == "positive"
    assert rows[1][1] == ""


# ── prompt validation against the human 200 ──────────────────────────────────


def test_validate_against_hand_labels_scores_and_keeps_rationales(
    stories: list[str],
) -> None:
    texts = stories[:3]
    labeler, _ = _labeler(
        [_reply("positive", "went up"), _reply("negative", "went down"), _reply("neutral", "flat")]
    )
    frame = build_vote_table(texts, [labeler])
    hand = [(texts[0], "positive"), (texts[1], "neutral"), (texts[2], "")]
    report = validate_against_hand_labels(frame, hand, SENTIMENT)
    assert report.n_scored == 2  # the unlabeled row never scores
    assert report.accuracy == pytest.approx(0.5)
    (disagreement,) = report.disagreements
    assert disagreement["text"] == texts[1]
    assert disagreement["human_label"] == "neutral"
    assert disagreement["majority_vote"] == "negative"
    assert disagreement["qwen3_0_6b__rationale"] == "went down"
    assert report.as_dict()["n_scored"] == 2


def test_validate_against_hand_labels_rejects_bad_input(stories: list[str]) -> None:
    labeler, _ = _labeler([_reply("neutral")])
    frame = build_vote_table(stories[:1], [labeler])
    with pytest.raises(ConfigError):
        validate_against_hand_labels(frame, [(stories[0], "")], SENTIMENT)
    with pytest.raises(ConfigError):
        validate_against_hand_labels(frame, [(stories[0], "bullish")], SENTIMENT)
    with pytest.raises(ConfigError):
        validate_against_hand_labels(frame, [("unseen text", "neutral")], SENTIMENT)


def test_hand_label_columns_are_the_protocol_header() -> None:
    assert HAND_LABEL_COLUMNS == ("text", "human_label")
