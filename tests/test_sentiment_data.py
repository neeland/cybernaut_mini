"""Sentiment data loading: offline fixture stories, the opt-in download gate,
and the canonical-JSONL cache round trip (exercised without any network)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cybernaut_mini.config import ConfigError
from cybernaut_mini.models import canonical_dumps
from cybernaut_mini.sentiment.data import (
    DOWNLOAD_ENV,
    LABEL_TO_INT,
    SentimentStory,
    downloads_enabled,
    load_financial_sentiment,
    load_fixture_stories,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DOCUMENTS = REPO_ROOT / "data" / "01_raw" / "fixtures" / "documents.jsonl"

_DOWNLOADS = bool(os.environ.get(DOWNLOAD_ENV, "").strip())


def test_fixture_stories_are_real_titles() -> None:
    stories = load_fixture_stories(FIXTURE_DOCUMENTS, max_rows=8)
    assert len(stories) == 8
    assert all(isinstance(story, str) and story for story in stories)
    # Deterministic: same file, same slice.
    assert stories == load_fixture_stories(FIXTURE_DOCUMENTS, max_rows=8)


def test_fixture_stories_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="fixture corpus not found"):
        load_fixture_stories(tmp_path / "nope.jsonl")


def test_label_mapping_is_the_three_class_convention() -> None:
    assert LABEL_TO_INT == {"positive": 1, "neutral": 0, "negative": -1}


def test_downloads_enabled_reads_the_env_gate() -> None:
    assert not downloads_enabled({})
    assert not downloads_enabled({DOWNLOAD_ENV: "  "})
    assert downloads_enabled({DOWNLOAD_ENV: "1"})


def test_cold_cache_without_opt_in_raises_with_instructions(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=DOWNLOAD_ENV):
        load_financial_sentiment(
            cache_path=tmp_path / "financial-sentiment.jsonl", download=False
        )


def test_warm_cache_loads_offline(tmp_path: Path) -> None:
    """A cache written in the canonical-JSONL shape loads with no network."""
    cache = tmp_path / "financial-sentiment.jsonl"
    rows = [
        SentimentStory(text="cached row one", label=1, netloc="example.com"),
        SentimentStory(text="cached row two", label=-1),
    ]
    cache.write_text(
        "".join(canonical_dumps(row.as_dict()) + "\n" for row in rows), encoding="utf-8"
    )
    loaded = load_financial_sentiment(cache_path=cache, download=False)
    assert loaded == rows
    assert load_financial_sentiment(cache_path=cache, download=False, max_rows=1) == rows[:1]


@pytest.mark.skipif(not _DOWNLOADS, reason=f"{DOWNLOAD_ENV} not set; dataset downloads")
def test_financial_sentiment_download_and_cache(tmp_path: Path) -> None:
    cache = tmp_path / "financial-sentiment.jsonl"
    stories = load_financial_sentiment(cache_path=cache, download=True, max_rows=64)
    assert len(stories) == 64
    assert all(story.label in {-1, 0, 1} for story in stories)
    assert cache.exists()
    # Second load is pure cache.
    again = load_financial_sentiment(cache_path=cache, download=False, max_rows=64)
    assert again == stories
