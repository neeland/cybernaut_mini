"""Sentiment labeler pool: one column per labeler, one uniform interface.

Every labeler maps a list of financial-news stories to integer labels in
``{-1, 0, +1}`` (Negative, Neutral, Positive) and exposes the column name the
post's tables use (``TextBlob-0.15``, ``VADER-0.10``, ``FinBERT``, ...). The
pool is the input to the agreement matrices in :mod:`.benchmark` and the label
matrix that :mod:`.ensemble` selects over.

Blog ref: https://nosible.com/blog/news-sentiment-showdown-who-checks-vibes-best —
    "In this post we will compare TextBlob, VADER, Flair, SigmaFSA, FinBERT,
    FinBERT-Tone, PaLM-2 (Bison and Unicorn), Gemini-Pro, GPT-3.5, GPT-4, and
    GPT-4-Turbo", with TextBlob/VADER swept over polarity/compound thresholds,
    Flair "confidence-gated" to NEU, and every LLM run under the same verbatim
    nine-example few-shot prompt. Local copy:
    ``docs/blog-archive/news-sentiment-showdown-who-checks-vibes-best.md``.

Assumptions:
    - Labels are the integers ``{-1, 0, +1}`` rather than strings, because the
      ensemble in the follow-up post sums member labels row-wise and thresholds
      the sum at ±1 — integer columns make that a plain ``sum()`` exactly as the
      post's appendix code does.
    - The post's hosted LLMs (PaLM-2, GPT-4, Gemini) are replaced by any
      OpenAI-compatible endpoint — OpenRouter, OpenAI, or a local server
      (Ollama/vLLM at ``http://localhost:11434/v1``) — under the SAME verbatim
      prompt at ``temperature=0``. The prompt, not the vendor, is the recipe.
    - "When you are unsure ... you MUST reply with NEU" extends to us: a reply
      that parses to none of POS/NEU/NEG is recorded as NEU (0), never dropped,
      so every labeler column has a value for every row.
    - TextBlob and VADER at threshold ``t`` label ``+1`` when score ``>= t``,
      ``-1`` when score ``<= -t``, else ``0``. The post does not state boundary
      handling; inclusive bounds make ``t=0.0`` degenerate to sign(), which is
      the least surprising reading.
    - Flair is an optional import (not installed by default): constructing
      :class:`FlairLabeler` without the package raises :class:`ConfigError`
      with an install hint instead of failing at import time, mirroring how the
      repo treats sentence-transformers.
    - The two HF FinBERT checkpoints download weights on first use, so they are
      opt-in exactly like the repo's other model downloads; their differing
      label vocabularies ('positive' vs 'Positive') are folded by casefold.

Alternatives rejected:
    - Returning per-class probabilities instead of hard labels: richer, but the
      post's whole pipeline (agreement %, ensemble sums, teacher bands) is
      defined over hard three-class labels, and inventing a probability
      interface the source never had would be replica drift.
    - LangChain for the LLM calls (what the post used): one OpenAI-compatible
      client already covers every endpoint we can reach, and the repo has an
      established pattern for it in ``query/s1_language/translate.py``.
    - Reading the few-shot prompt from ``configs/sentiment/few_shot_prompt.txt``
      at import time: a wheel install has no ``configs/``; the constant lives
      here and the config file is asserted byte-identical by the tests instead.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from cybernaut_mini.config import ConfigError

__all__ = [
    "API_KEY_ENV_VARS",
    "DEFAULT_FLAIR_CONFIDENCES",
    "DEFAULT_TEXTBLOB_THRESHOLDS",
    "DEFAULT_VADER_THRESHOLDS",
    "FEW_SHOT_PROMPT",
    "FINBERT_COLUMN_NAMES",
    "FewShotLLMLabeler",
    "FinbertLabeler",
    "FlairLabeler",
    "Labeler",
    "RandomLabeler",
    "TextBlobLabeler",
    "VaderLabeler",
    "build_label_matrix",
    "build_pool",
    "parse_llm_reply",
]

#: Threshold sweep for TextBlob polarity. The post tested {0.15, 0.30, 0.45};
#: the replica extends the sweep across {0.10 .. 0.45} in 0.05 steps.
DEFAULT_TEXTBLOB_THRESHOLDS: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)

#: Threshold sweep for the VADER compound score ("We tested it with 3x thresholds").
DEFAULT_VADER_THRESHOLDS: tuple[float, ...] = (0.10, 0.20, 0.30)

#: Flair confidence gates: "We tested 3x values for this threshold - 70%, 80%,
#: and 90% confidence."
DEFAULT_FLAIR_CONFIDENCES: tuple[float, ...] = (0.70, 0.80, 0.90)

#: The two FinBERT checkpoints from the post, with the post's column names.
FINBERT_COLUMN_NAMES: Mapping[str, str] = {
    "ProsusAI/finbert": "FinBERT",
    "yiyanghkust/finbert-tone": "FinBERT-Tone",
}

#: Copied VERBATIM from the post's "LLM Prompt" section ("This prompt was used
#: for all of the evaluated LLMs with zero modifications"). The only slot is
#: ``{story}``. The canonical config copy lives in
#: ``configs/sentiment/few_shot_prompt.txt``; tests assert the two are identical.
FEW_SHOT_PROMPT = """You are a sentiment classification AI.

1. You are only able to reply with POS, NEU, or NEG.
2. When you are unsure of the sentiment, you MUST reply with NEU.

Here are some examples of correct replies:

Story: Unexpected demand for the new XYZ product is likely to boost earnings.
Reply: POS

Story: Sales of the new XYZ product are inline with projections.
Reply: NEU

Story: Company releases profit warning after the sales of XYZ disappoint.
Reply: NEG

Story: Following better than expected job numbers, the stock market rallied.
Reply: POS

Story: The stock market ended flat after job numbers came in as expected.
Reply: NEU

Story: A large spike in unemployment numbers sent the stock market into panic.
Reply: NEG

Story: XYZ stock soared after the FDA approved its new cancer treatment.
Reply: POS

Story: XYZ will announce results of its cancer treatment on the 15th of July.
Reply: NEU

Story: Following poor results, the FDA shuts down trials of XYZ cancer treatment.
Reply: NEG

Okay, now please classify the following news story:

Story: {story}
Reply:"""

#: Consulted in order at call time, mirroring ``query/s1_language/translate.py``.
API_KEY_ENV_VARS: tuple[str, ...] = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")

_REPLY_TO_LABEL: Mapping[str, int] = {"POS": 1, "NEU": 0, "NEG": -1}


class Labeler(Protocol):
    """One sentiment column: a name and a batch labeling function."""

    @property
    def name(self) -> str: ...

    def label(self, stories: Sequence[str]) -> list[int]: ...


def _threshold_label(score: float, threshold: float) -> int:
    if score >= threshold:
        return 1
    if score <= -threshold:
        return -1
    return 0


class TextBlobLabeler:
    """TextBlob pattern-lexicon polarity, thresholded to three classes."""

    def __init__(self, threshold: float) -> None:
        if threshold <= 0:
            msg = f"TextBlob threshold must be positive, got {threshold}"
            raise ConfigError(msg)
        self._threshold = threshold

    @property
    def name(self) -> str:
        return f"TextBlob-{self._threshold:.2f}"

    def label(self, stories: Sequence[str]) -> list[int]:
        from textblob import TextBlob  # type: ignore[import-untyped]

        labels: list[int] = []
        for story in stories:
            polarity = float(TextBlob(story).sentiment.polarity)
            labels.append(_threshold_label(polarity, self._threshold))
        return labels


class VaderLabeler:
    """VADER compound score, thresholded to three classes."""

    def __init__(self, threshold: float) -> None:
        if threshold <= 0:
            msg = f"VADER threshold must be positive, got {threshold}"
            raise ConfigError(msg)
        self._threshold = threshold
        self._analyzer: Any | None = None

    @property
    def name(self) -> str:
        return f"VADER-{self._threshold:.2f}"

    def label(self, stories: Sequence[str]) -> list[int]:
        if self._analyzer is None:
            from vaderSentiment.vaderSentiment import (  # type: ignore[import-untyped]
                SentimentIntensityAnalyzer,
            )

            self._analyzer = SentimentIntensityAnalyzer()
        labels: list[int] = []
        for story in stories:
            compound = float(self._analyzer.polarity_scores(story)["compound"])
            labels.append(_threshold_label(compound, self._threshold))
        return labels


class FlairLabeler:
    """Flair's two-class sentiment model, confidence-gated to NEU.

    The post: "To get Flair to output three classes we looked at the probability
    assigned to the best label and only accepted that label if the probability
    crossed a threshold." Flair is NOT a repo dependency; constructing this
    without the package installed raises :class:`ConfigError`.
    """

    def __init__(self, confidence: float) -> None:
        if not 0.0 < confidence < 1.0:
            msg = f"Flair confidence gate must be in (0, 1), got {confidence}"
            raise ConfigError(msg)
        self._confidence = confidence
        try:
            from flair.models import TextClassifier  # type: ignore[import-not-found]
            from flair.nn import Classifier  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            msg = (
                "labeler 'flair' needs the optional 'flair' package, which the "
                "default install does not include.\n"
                "  install it   : uv pip install flair\n"
                "  or drop the flair_confidences entry from the labeler pool config."
            )
            raise ConfigError(msg) from exc
        self._classifier = TextClassifier.load("en-sentiment")

    @property
    def name(self) -> str:
        return f"Flair-{self._confidence:.2f}"

    def label(self, stories: Sequence[str]) -> list[int]:
        from flair.data import Sentence  # type: ignore[import-not-found]

        labels: list[int] = []
        for story in stories:
            sentence = Sentence(story)
            self._classifier.predict(sentence)
            if not sentence.labels:
                labels.append(0)
                continue
            best = sentence.labels[0]
            if float(best.score) < self._confidence:
                labels.append(0)
            else:
                labels.append(1 if str(best.value).upper().startswith("POS") else -1)
        return labels


class FinbertLabeler:
    """A Hugging Face three-class financial-sentiment checkpoint.

    Covers both ``ProsusAI/finbert`` and ``yiyanghkust/finbert-tone`` (the post's
    FinBERT and FinBERT-Tone columns). Constructing this downloads weights, so
    it is opt-in — tests gate it behind ``CYBERNAUT_MINI_SENTIMENT_DOWNLOADS``.
    """

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            msg = (
                "labeler 'finbert' needs the optional 'st' extra (transformers + "
                "torch); install it with `uv sync --extra st`."
            )
            raise ConfigError(msg) from exc

        self._model_name = model_name
        self._pipeline = pipeline(
            "text-classification",
            model=model_name,
            revision=revision,
            token=os.environ.get("HF_TOKEN") or None,
        )

    @property
    def name(self) -> str:
        return FINBERT_COLUMN_NAMES.get(self._model_name, self._model_name.split("/")[-1])

    def label(self, stories: Sequence[str]) -> list[int]:
        results = self._pipeline(list(stories), truncation=True)
        labels: list[int] = []
        for result in results:
            value = str(result["label"]).casefold()
            if value.startswith("pos"):
                labels.append(1)
            elif value.startswith("neg"):
                labels.append(-1)
            else:
                labels.append(0)
        return labels


def parse_llm_reply(text: str) -> int:
    """Map a raw model reply onto ``{-1, 0, +1}``; anything unparseable is NEU (0).

    The prompt's own contract — "When you are unsure of the sentiment, you MUST
    reply with NEU" — is applied to the parser too: a reply that is not one of
    POS/NEU/NEG (after stripping whitespace, punctuation, and case) counts as
    unsure, hence 0.
    """
    match = re.search(r"\b(POS|NEU|NEG)\b", text.strip().upper())
    if match is None:
        return 0
    return _REPLY_TO_LABEL[match.group(1)]


class FewShotLLMLabeler:
    """An LLM behind any OpenAI-compatible endpoint under the verbatim prompt.

    The credential is re-resolved from the environment on every call and the
    client rebuilt only when it changes — the same discipline as
    :class:`cybernaut_mini.query.s1_language.translate.OpenAICompatibleTranslator`,
    and for the same reasons. ``client_factory`` is the seam tests use to prove
    the wiring (prompt, temperature 0, NEU-on-parse-failure) without a network.
    """

    def __init__(
        self,
        *,
        model: str,
        name: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
        timeout: float = 60.0,
        client_factory: Callable[[str, str], Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._name = name or model.split("/")[-1]
        self._api_key = api_key
        self._client_factory = client_factory
        self._environ = environ
        self._client: Any | None = None
        self._client_api_key: str | None = None

    @property
    def name(self) -> str:
        return self._name

    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        environ: Mapping[str, str] = os.environ if self._environ is None else self._environ
        for env_name in API_KEY_ENV_VARS:
            value = environ.get(env_name, "").strip()
            if value:
                return value
        names = " or ".join(API_KEY_ENV_VARS)
        msg = (
            f"No API key for {self.base_url}: set {names} in the environment or pass "
            f"api_key= to {type(self).__name__}."
        )
        raise ConfigError(msg)

    def _default_client_factory(self, api_key: str, base_url: str) -> Any:
        from openai import OpenAI

        return OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)

    def _get_client(self) -> Any:
        api_key = self._resolve_api_key()
        if self._client is None or self._client_api_key != api_key:
            factory = self._client_factory or self._default_client_factory
            self._client = factory(api_key, self.base_url)
            self._client_api_key = api_key
        return self._client

    def build_prompt(self, story: str) -> str:
        """The exact prompt sent for *story*, exposed so tests can pin it."""
        return FEW_SHOT_PROMPT.format(story=story)

    def label(self, stories: Sequence[str]) -> list[int]:
        client = self._get_client()
        labels: list[int] = []
        for story in stories:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self.build_prompt(story)}],
                temperature=0.0,
            )
            try:
                content = response.choices[0].message.content or ""
            except (AttributeError, IndexError, KeyError, TypeError):
                # An unreadable response is "unsure" by the prompt's own rule.
                content = ""
            labels.append(parse_llm_reply(content))
        return labels


class RandomLabeler:
    """Uniform random labels — the post's post-test-probability baseline.

    "Given a sentence the random classifier returns a -1 (Negative), 0
    (Neutral), or 1 (Positive) label with equal probability."
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    @property
    def name(self) -> str:
        return "Random"

    def label(self, stories: Sequence[str]) -> list[int]:
        rng = np.random.default_rng(self._seed)
        return [int(value) for value in rng.integers(-1, 2, size=len(stories))]


def build_pool(params: Mapping[str, Any]) -> list[Labeler]:
    """Construct the labeler pool a parameters block describes.

    Recognised keys (all optional; the default pool is the offline lexicon sweep):

    - ``textblob_thresholds``: list of floats (default the {0.10..0.45} sweep).
    - ``vader_thresholds``: list of floats (default {0.10, 0.20, 0.30}).
    - ``flair_confidences``: list of floats (default EMPTY — flair is optional).
    - ``finbert_models``: list of HF model ids (default EMPTY — downloads).
    - ``llm_models``: list of ``{model, name?, base_url?}`` dicts (default EMPTY).
    - ``random_seed``: int; include the Random baseline column (default 42;
      set to null to omit).
    """
    labelers: list[Labeler] = []
    textblob = params.get("textblob_thresholds", list(DEFAULT_TEXTBLOB_THRESHOLDS))
    labelers.extend(TextBlobLabeler(float(t)) for t in textblob)
    vader = params.get("vader_thresholds", list(DEFAULT_VADER_THRESHOLDS))
    labelers.extend(VaderLabeler(float(t)) for t in vader)
    for confidence in params.get("flair_confidences", []):
        labelers.append(FlairLabeler(float(confidence)))
    for model_name in params.get("finbert_models", []):
        labelers.append(FinbertLabeler(str(model_name)))
    for spec in params.get("llm_models", []):
        labelers.append(
            FewShotLLMLabeler(
                model=str(spec["model"]),
                name=spec.get("name"),
                base_url=str(spec.get("base_url", "https://openrouter.ai/api/v1")),
            )
        )
    seed = params.get("random_seed", 42)
    if seed is not None:
        labelers.append(RandomLabeler(int(seed)))
    return labelers


def build_label_matrix(
    stories: Sequence[str], labelers: Sequence[Labeler]
) -> dict[str, list[int]]:
    """Run every labeler over *stories*: one column per labeler, aligned by row."""
    if not stories:
        msg = "cannot label an empty story list"
        raise ConfigError(msg)
    matrix: dict[str, list[int]] = {}
    for labeler in labelers:
        if labeler.name in matrix:
            msg = f"duplicate labeler column {labeler.name!r}"
            raise ConfigError(msg)
        column = labeler.label(stories)
        if len(column) != len(stories):
            msg = (
                f"labeler {labeler.name!r} returned {len(column)} labels "
                f"for {len(stories)} stories"
            )
            raise ConfigError(msg)
        matrix[labeler.name] = [int(value) for value in column]
    return matrix
