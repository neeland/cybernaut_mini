"""Active-learning tests: the verbatim oracle prompt, ``.npy`` memoization,
hinge-SGD baselines, pure-numpy disagreement mining, and the refinement loop.

Everything is offline: baselines and the loop run over synthetic numeric
vectors (pure-math unit tests), texts are real fixture headlines, embedding
memoization uses the repo's deterministic :class:`HashEmbedder`, and the
oracle is always an injected scripted callable — never a network client.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cybernaut_mini.config import ConfigError
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.sentiment.active_learning import (
    ORACLE_SYSTEM_TEMPLATE,
    ActiveLearningResult,
    IterationLog,
    OpenAICompatibleOracle,
    OracleVerdict,
    active_learning_loop,
    build_oracle_messages,
    embedding_cache_path,
    label_options,
    memoized_embeddings,
    mine_disagreements,
    parse_oracle_reply,
    train_baseline,
    write_history,
)
from cybernaut_mini.sentiment.data import load_fixture_stories
from cybernaut_mini.sentiment.prompts import task_for

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DOCUMENTS = REPO_ROOT / "data" / "01_raw" / "fixtures" / "documents.jsonl"
ORACLE_CONFIG = REPO_ROOT / "configs" / "sentiment" / "oracle_adjudication.txt"

SENTIMENT = task_for("financial_sentiment")
FORWARD = task_for("forward_looking")


@pytest.fixture(scope="module")
def stories() -> list[str]:
    return load_fixture_stories(FIXTURE_DOCUMENTS, max_rows=40)


# ── the verbatim oracle prompt ───────────────────────────────────────────────


def test_oracle_template_matches_the_config_copy() -> None:
    assert ORACLE_CONFIG.read_text(encoding="utf-8") == ORACLE_SYSTEM_TEMPLATE


def test_oracle_template_carries_the_post_scaffolding() -> None:
    assert "Is this model correct?" in ORACLE_SYSTEM_TEMPLATE
    assert "DO NOT RESPOND IN ANY OTHER WAY" in ORACLE_SYSTEM_TEMPLATE
    assert "P.S. Think fast there is no time to waste." in ORACLE_SYSTEM_TEMPLATE
    assert '"correct": "true" | "false"' in ORACLE_SYSTEM_TEMPLATE


def test_label_options_is_the_post_join() -> None:
    assert label_options(SENTIMENT) == '"negative" | "neutral" | "positive"'
    assert label_options(FORWARD) == '"forward" | "not-forward"'


def test_build_oracle_messages_nests_the_labelling_prompt(stories: list[str]) -> None:
    messages = build_oracle_messages(SENTIMENT, stories[0], "positive")
    system, user = messages
    assert system["role"] == "system"
    assert user["role"] == "user"
    system_text = system["content"][0]["text"]
    assert 'labelled this text as "positive"' in system_text
    assert label_options(SENTIMENT) in system_text
    # The original labeling prompt travels in the USER turn, snippet included.
    user_text = user["content"][0]["text"]
    assert stories[0] in user_text
    assert "DO NOT WRITE ANY PREAMBLE JUST RETURN JSON." in user_text


def test_build_oracle_messages_rejects_foreign_labels(stories: list[str]) -> None:
    with pytest.raises(ConfigError):
        build_oracle_messages(SENTIMENT, stories[0], "bullish")


# ── verdict parsing ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            '```json\n{"correct": "true", "label": "positive", "reason": "yes"}\n```',
            OracleVerdict(correct=True, label="positive", reason="yes"),
        ),
        (
            '{"correct": "false", "label": "neutral", "reason": "no"}',
            OracleVerdict(correct=False, label="neutral", reason="no"),
        ),
        (
            '{"correct": true, "label": "Negative"}',
            OracleVerdict(correct=True, label="negative", reason=""),
        ),
    ],
)
def test_parse_oracle_reply_accepts_the_contract(raw: str, expected: OracleVerdict) -> None:
    assert parse_oracle_reply(raw, SENTIMENT) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"correct": "maybe", "label": "positive"}',
        '{"correct": "true", "label": "bullish"}',
        '{"correct": "true"}',
        "[1, 2, 3]",
    ],
)
def test_parse_oracle_reply_rejects_everything_else(raw: str) -> None:
    assert parse_oracle_reply(raw, SENTIMENT) is None


class _FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.requests: list[dict[str, Any]] = []
        self._replies = iter(replies)
        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        message = SimpleNamespace(content=next(self._replies))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_openai_oracle_retries_then_returns_verdict(stories: list[str]) -> None:
    client = _FakeClient(
        ["garbage", '{"correct": "true", "label": "positive", "reason": "clear"}']
    )
    oracle = OpenAICompatibleOracle(
        model="qwen3:32b",
        task=SENTIMENT,
        max_retries=1,
        client_factory=lambda api_key, base_url: client,
    )
    verdict = oracle(stories[0], "positive")
    assert verdict == OracleVerdict(correct=True, label="positive", reason="clear")
    assert len(client.requests) == 2
    assert client.requests[0]["temperature"] == 0.0


def test_openai_oracle_gives_up_to_none(stories: list[str]) -> None:
    client = _FakeClient(["nope", "still nope"])
    oracle = OpenAICompatibleOracle(
        model="qwen3:32b",
        task=SENTIMENT,
        max_retries=1,
        client_factory=lambda api_key, base_url: client,
    )
    assert oracle(stories[0], "neutral") is None


# ── memoized embeddings ──────────────────────────────────────────────────────


def test_embedding_cache_path_slugs_the_model_id(tmp_path: Path) -> None:
    path = embedding_cache_path("Qwen/Qwen3-Embedding-0.6B", "financial_sentiment", tmp_path)
    assert path == tmp_path / "financial_sentiment__qwen-qwen3-embedding-0-6b.npy"
    with pytest.raises(ConfigError):
        embedding_cache_path("///", "financial_sentiment", tmp_path)


class _CountingEmbedder:
    """HashEmbedder that counts embed calls, to prove memoization."""

    def __init__(self, dim: int = 16) -> None:
        self._inner = HashEmbedder(dim=dim)
        self.calls = 0

    @property
    def identifier(self) -> str:
        return self._inner.identifier

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed_documents(self, texts: list[str]) -> Any:
        self.calls += 1
        return self._inner.embed_documents(texts)


def test_memoized_embeddings_embed_once_then_load(tmp_path: Path, stories: list[str]) -> None:
    provider = _CountingEmbedder()
    cache = tmp_path / "sent__hash.npy"
    first = memoized_embeddings(stories[:8], provider, cache_path=cache)
    second = memoized_embeddings(stories[:8], provider, cache_path=cache)
    assert provider.calls == 1
    assert cache.exists()
    np.testing.assert_array_equal(first, second)
    assert first.shape == (8, 16)


def test_memoized_embeddings_slice_the_bigger_cache(tmp_path: Path, stories: list[str]) -> None:
    provider = _CountingEmbedder()
    cache = tmp_path / "sent__hash.npy"
    full = memoized_embeddings(stories[:8], provider, cache_path=cache)
    sliced = memoized_embeddings(stories[:5], provider, cache_path=cache)
    assert provider.calls == 1  # the post's np.load(...)[: n_rows] pattern
    np.testing.assert_array_equal(sliced, full[:5])


def test_memoized_embeddings_reject_a_smaller_cache(tmp_path: Path, stories: list[str]) -> None:
    provider = _CountingEmbedder()
    cache = tmp_path / "sent__hash.npy"
    memoized_embeddings(stories[:4], provider, cache_path=cache)
    with pytest.raises(ConfigError):
        memoized_embeddings(stories[:8], provider, cache_path=cache)
    with pytest.raises(ConfigError):
        memoized_embeddings([], provider, cache_path=cache)


# ── linear baselines ─────────────────────────────────────────────────────────


def _separable(n: int, dim: int = 4, *, flip: list[int] | None = None) -> tuple[Any, list[int]]:
    """Linearly separable two-class vectors with optional label corruption."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n, dim))
    truth = [1 if features[i, 0] > 0.0 else 0 for i in range(n)]
    features[:, 0] = np.where(np.asarray(truth) == 1, features[:, 0] + 3.0, features[:, 0] - 3.0)
    labels = list(truth)
    for index in flip or []:
        labels[index] = 1 - labels[index]
    return features, labels


def test_train_baseline_learns_separable_data() -> None:
    features, labels = _separable(50)
    result = train_baseline("hash", features, labels, FORWARD)
    assert result.train_accuracy == 1.0
    assert result.val_accuracy == 1.0
    assert result.predictions.shape == (50,)
    # Nested confusion dicts are keyed by label strings, post-style.
    assert set(result.confusion_val) == {"forward", "not-forward"}
    assert set(result.confusion_val["forward"]) == {"forward", "not-forward"}
    total = sum(sum(row.values()) for row in result.confusion_val.values())
    assert total == 10  # the 20% validation tail of the subscript split
    assert result.as_dict()["name"] == "hash"


def test_train_baseline_rejects_misaligned_input() -> None:
    features, labels = _separable(10)
    with pytest.raises(ConfigError):
        train_baseline("hash", features, labels[:-1], FORWARD)
    with pytest.raises(ConfigError):
        train_baseline("hash", features, labels, FORWARD, train_pct=1.0)


# ── disagreement mining (pure numpy) ─────────────────────────────────────────


def test_mine_disagreements_requires_unanimity_and_contradiction() -> None:
    predictions = {
        "a": np.array([1, 0, 1, 0], dtype=np.int64),
        "b": np.array([1, 0, 1, 1], dtype=np.int64),
    }
    votes = [1, 1, 1, 0]
    # Row 0: unanimous, agrees with vote -> not mined.
    # Row 1: unanimous, contradicts vote -> mined.
    # Row 2: unanimous, agrees -> not mined. Row 3: split -> not mined.
    assert mine_disagreements(predictions, votes).tolist() == [1]


def test_mine_disagreements_single_model_is_all_disagreements() -> None:
    predictions = {"a": np.array([1, 0], dtype=np.int64)}
    assert mine_disagreements(predictions, [0, 0]).tolist() == [0]


def test_mine_disagreements_rejects_bad_shapes() -> None:
    with pytest.raises(ConfigError):
        mine_disagreements({}, [1])
    with pytest.raises(ConfigError):
        mine_disagreements({"a": np.array([1, 0], dtype=np.int64)}, [1])


# ── the refinement loop ──────────────────────────────────────────────────────


def _loop_inputs(stories: list[str]) -> tuple[dict[str, Any], list[str], list[int], list[int]]:
    flips = [3, 17]
    features, labels = _separable(40, flip=flips)
    embeddings = {
        "model_a": np.asarray(features, dtype=np.float64),
        "model_b": np.asarray(features + 0.01, dtype=np.float64),
    }
    return embeddings, stories[:40], labels, flips


def test_loop_relabels_when_the_oracle_sides_with_consensus(stories: list[str]) -> None:
    embeddings, texts, votes, flips = _loop_inputs(stories)
    oracle_calls: list[tuple[str, str]] = []

    def oracle(text: str, prediction: str) -> OracleVerdict:
        oracle_calls.append((text, prediction))
        return OracleVerdict(correct=True, label=prediction, reason="separable")

    result = active_learning_loop(embeddings, texts, votes, FORWARD, oracle, min_models=1)
    assert result.converged
    # Every flipped vote was mined, adjudicated, and restored.
    truth = [1 - votes[i] if i in flips else votes[i] for i in range(len(votes))]
    assert result.labels.tolist() == truth
    mined_texts = {text for text, _ in oracle_calls}
    assert {texts[i] for i in flips} <= mined_texts
    first = result.history[0]
    assert first.n_candidates >= len(flips)
    assert first.n_relabelled >= len(flips)
    # Drop-worst-model fired while more than min_models remained.
    assert first.dropped_model in {"model_a", "model_b"}
    assert result.history[-1].n_candidates == 0
    assert result.history[-1].dropped_model is None
    for entry in result.history:
        assert set(entry.model_accuracies[next(iter(entry.model_accuracies))]) == {"train", "val"}


def test_loop_keeps_labels_when_the_oracle_disagrees(stories: list[str]) -> None:
    embeddings, texts, votes, _ = _loop_inputs(stories)

    def oracle(text: str, prediction: str) -> OracleVerdict:
        other = "forward" if prediction != "forward" else "not-forward"
        return OracleVerdict(correct=False, label=other, reason="keep the vote")

    result = active_learning_loop(
        embeddings, texts, votes, FORWARD, oracle, max_iterations=3, min_models=1
    )
    assert result.labels.tolist() == votes  # nothing mutated
    assert all(entry.n_relabelled == 0 for entry in result.history)


def test_loop_none_verdicts_never_mutate_labels(stories: list[str]) -> None:
    embeddings, texts, votes, _ = _loop_inputs(stories)
    result = active_learning_loop(
        embeddings, texts, votes, FORWARD, lambda text, label: None, max_iterations=2
    )
    assert result.labels.tolist() == votes


def test_loop_rejects_misaligned_inputs(stories: list[str]) -> None:
    embeddings, texts, votes, _ = _loop_inputs(stories)
    oracle = lambda text, label: None  # noqa: E731
    with pytest.raises(ConfigError):
        active_learning_loop({}, texts, votes, FORWARD, oracle)
    with pytest.raises(ConfigError):
        active_learning_loop(embeddings, texts[:-1], votes, FORWARD, oracle)
    short = {"model_a": embeddings["model_a"][:-1]}
    with pytest.raises(ConfigError):
        active_learning_loop(short, texts, votes, FORWARD, oracle)
    with pytest.raises(ConfigError):
        active_learning_loop(embeddings, texts, votes, FORWARD, oracle, min_models=0)


def test_write_history_is_canonical_json(tmp_path: Path) -> None:
    result = ActiveLearningResult(
        labels=np.array([1, 0], dtype=np.int64),
        history=[
            IterationLog(
                iteration=0,
                model_accuracies={"hash": {"train": 1.0, "val": 0.9}},
                n_candidates=0,
                n_relabelled=0,
                dropped_model=None,
            )
        ],
    )
    path = tmp_path / "history.json"
    write_history(path, result)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["converged"] is True
    assert payload["history"][0]["model_accuracies"]["hash"]["val"] == 0.9
    # Canonical form: sorted keys, trailing newline.
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert list(payload["history"][0]) == sorted(payload["history"][0])
