"""Story loading for the sentiment lab: canonical HF dataset + offline fixtures.

Two sources, one output shape (``SentimentStory``): the maintained
NOSIBLE/financial-sentiment dataset (opt-in download, cached as JSONL under
``data/01_raw/financial_sentiment/``) for real runs, and headline-style slices
of the committed CC-News/MIRACL fixture corpus for offline tests.

Blog ref: https://nosible.com/blog/news-sentiment-showdown-who-checks-vibes-best —
    "The original 2024 export has been superseded by the maintained NOSIBLE
    Financial Sentiment dataset, which contains 100,000 cleaned, deduplicated,
    and labelled news samples." That successor dataset (columns: text, label,
    netloc, url) is the canonical input here. Local copy:
    ``docs/blog-archive/news-sentiment-showdown-who-checks-vibes-best.md``.

Assumptions:
    - Downloads are OPT-IN: :func:`load_financial_sentiment` reads the local
      cache by default and only touches the Hub when ``download=True`` (or when
      the ``CYBERNAUT_MINI_SENTIMENT_DOWNLOADS`` env gate is set, the pattern
      the gated tests use). A cold cache without the flag raises
      :class:`ConfigError` with the exact command to run, never a silent fetch.
    - The dataset's ``label`` strings map to the pool's integer classes as
      positive→1, neutral→0, negative→-1; unknown strings raise rather than
      default, because a silently mislabelled gold column poisons everything
      downstream.
    - The cache is JSONL written through ``canonical_dumps`` — one line per
      row, byte-stable across re-downloads of the same revision — matching the
      repo's rule that every JSON artifact goes through the canonical writer.
    - Fixture stories are the documents' real titles (CC-News headlines): the
      lexicon labelers and the distil harness need short, headline-shaped real
      text, and the title field is exactly that. No text is synthesised.

Alternatives rejected:
    - Loading via the repo's ``HuggingFaceDataset`` Kedro dataset: right tool
      inside a pipeline, but this loader must also work from tests and
      notebooks without a catalog; the pipeline wires the catalog around it.
    - Caching as parquet: smaller, but the repo's raw layer is JSONL and the
      canonical writer only speaks JSON; 100k short rows is ~30 MB either way.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from cybernaut_mini.config import ConfigError
from cybernaut_mini.models import canonical_dumps

__all__ = [
    "DOWNLOAD_ENV",
    "FINANCIAL_SENTIMENT_REPO",
    "LABEL_TO_INT",
    "SentimentStory",
    "default_cache_path",
    "downloads_enabled",
    "load_financial_sentiment",
    "load_fixture_stories",
]

FINANCIAL_SENTIMENT_REPO = "NOSIBLE/financial-sentiment"

#: Env gate for opt-in downloads, mirroring the fastText gate pattern in tests.
DOWNLOAD_ENV = "CYBERNAUT_MINI_SENTIMENT_DOWNLOADS"

LABEL_TO_INT: dict[str, int] = {"positive": 1, "neutral": 0, "negative": -1}

_FIXTURE_DOCUMENTS = Path("data/01_raw/fixtures/documents.jsonl")


@dataclass(frozen=True)
class SentimentStory:
    """One labelled story: the text the pool labels plus the dataset's gold label."""

    text: str
    label: int
    netloc: str = ""
    url: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"label": self.label, "netloc": self.netloc, "text": self.text, "url": self.url}


def downloads_enabled(environ: dict[str, str] | None = None) -> bool:
    """True when the opt-in download gate is set (non-empty)."""
    env = os.environ if environ is None else environ
    return bool(env.get(DOWNLOAD_ENV, "").strip())


def default_cache_path(data_dir: Path = Path("data/01_raw")) -> Path:
    return data_dir / "financial_sentiment" / "financial-sentiment.jsonl"


def _parse_label(value: str) -> int:
    key = value.strip().casefold()
    if key not in LABEL_TO_INT:
        msg = f"unknown sentiment label {value!r}; expected one of {sorted(LABEL_TO_INT)}"
        raise ConfigError(msg)
    return LABEL_TO_INT[key]


def _read_cache(path: Path, max_rows: int | None) -> list[SentimentStory]:
    stories: list[SentimentStory] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if max_rows is not None and len(stories) >= max_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            stories.append(
                SentimentStory(
                    text=str(row["text"]),
                    label=int(row["label"]),
                    netloc=str(row.get("netloc", "")),
                    url=str(row.get("url", "")),
                )
            )
    if not stories:
        msg = f"sentiment cache {path} is empty"
        raise ConfigError(msg)
    return stories


def load_financial_sentiment(
    *,
    cache_path: Path | None = None,
    download: bool | None = None,
    revision: str | None = None,
    max_rows: int | None = None,
) -> list[SentimentStory]:
    """Load NOSIBLE/financial-sentiment from cache, downloading only on opt-in.

    ``download=None`` defers to the ``CYBERNAUT_MINI_SENTIMENT_DOWNLOADS`` env
    gate; ``download=False`` never touches the network. The download is cached
    to *cache_path* as canonical JSONL so subsequent loads are offline.
    """
    path = cache_path or default_cache_path()
    if path.exists():
        return _read_cache(path, max_rows)

    allow = downloads_enabled() if download is None else download
    if not allow:
        msg = (
            f"sentiment dataset cache not found at {path} and downloads are opt-in.\n"
            f"  enable once  : {DOWNLOAD_ENV}=1 (or pass download=True)\n"
            f"  then rerun; the download is cached and later runs are offline.\n"
            f"  source       : https://huggingface.co/datasets/{FINANCIAL_SENTIMENT_REPO}"
        )
        raise ConfigError(msg)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        msg = (
            "downloading NOSIBLE/financial-sentiment needs the optional 'hf' extra "
            "(the `datasets` package); install it with `uv sync --extra hf`."
        )
        raise ConfigError(msg) from exc

    dataset = load_dataset(
        FINANCIAL_SENTIMENT_REPO, split="train", revision=revision
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            story = SentimentStory(
                text=str(row["text"]),
                label=_parse_label(str(row["label"])),
                netloc=str(row.get("netloc", "") or ""),
                url=str(row.get("url", "") or ""),
            )
            handle.write(canonical_dumps(story.as_dict()) + "\n")
    tmp_path.replace(path)
    return _read_cache(path, max_rows)


def load_fixture_stories(
    path: Path = _FIXTURE_DOCUMENTS, *, max_rows: int | None = 32
) -> list[str]:
    """Real headline-shaped stories: the fixture corpus documents' titles.

    Offline by construction — the fixture file is committed. Rows without a
    usable title are skipped, not invented.
    """
    if not path.exists():
        msg = f"fixture corpus not found at {path}; run from the repo root"
        raise ConfigError(msg)
    stories: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if max_rows is not None and len(stories) >= max_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            title = str(row.get("title", "")).strip()
            if title:
                stories.append(title)
    if not stories:
        msg = f"no titled documents found in {path}"
        raise ConfigError(msg)
    return stories
