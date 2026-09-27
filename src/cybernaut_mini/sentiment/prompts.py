"""The three decision-tree labeling prompts, verbatim, plus the task registry.

One :class:`TaskSpec` per classifier the post ships — financial sentiment,
forward-looking, prediction — bundling the verbatim decision-tree prompt, the
JSON response field it demands (``financial_sentiment`` / ``tense`` /
``causal``), the label vocabulary, the when-in-doubt default class, and the
strict-JSON reply parser :mod:`.vote` retries against.

Blog ref: https://nosible.com/blog/fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction —
    "Here are the prompts we used for each classification task": three prompts,
    each a materiality/task description, an ASCII decision tree ("Pay careful
    attention to the logic. Don't deviate."), a when-in-doubt default class,
    and a strict JSON response format ("You must respond with ONLY a valid
    JSON object ... DO NOT WRITE ANY PREAMBLE JUST RETURN JSON."). Local copy:
    ``docs/blog-archive/fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction.md``.

Assumptions:
    - The templates are stored exactly as the post's f-string bodies — ``{{``
      escapes, the ``{text}`` slot, curly typography, the "non-concreate" typo
      and all — so ``template.format(text=...)`` reproduces the post's f-string
      evaluation byte-for-byte. Canonical copies live in
      ``configs/sentiment/decision_tree_*.txt``; tests assert both the config
      copies AND the archived post match these constants.
    - The JSON reply is ``{rationale, <field>}`` where the field name differs
      per task (``financial_sentiment``, ``tense``, ``causal``); the parser
      accepts key order and whitespace variance (that is what "valid JSON
      object" means) but nothing else — a reply with the wrong field or an
      out-of-vocabulary label parses to ``None`` so the caller can retry, which
      is the seam :mod:`.vote` builds its retry-then-default policy on.
    - Some local models wrap JSON in a ```` ```json ```` fence despite the "NO
      PREAMBLE" instruction; the parser unwraps one fence (and one bare
      ``{...}`` object) before giving up, because discarding an otherwise valid
      label over markup would manufacture default-class labels the model never
      produced.
    - Each task's ``default_label`` is the prompt's own when-in-doubt class
      (neutral / not-forward / not-predictive); ties and parse failures
      downstream resolve to it, extending the prompt's contract to the caller
      exactly as :mod:`.labelers` does with NEU.
    - ``system_prompt`` carries the fine-tune "guard" system message; only the
      sentiment one appears verbatim in the post, the other two are minimal
      analogues (flagged as a deviation in the module tests and README notes).

Alternatives rejected:
    - One generic prompt template parameterized by label set: the post wrote
      three distinct prompts with distinct decision trees and disambiguation
      rules, and the whole point of copying them verbatim is not to paraphrase.
    - Pydantic models for the reply JSON: two string fields per task do not
      earn a schema class; the parser returns ``(label | None, rationale)`` and
      the vote table stores plain strings.
    - Normalising label synonyms ("pos", "bullish") like
      :func:`cybernaut_mini.sentiment.labelers.parse_llm_reply` does: these
      prompts demand exact vocabulary in strict JSON, so fuzzy matching would
      hide format non-compliance the retry loop is designed to surface.
"""

# ruff: noqa: RUF001 — the verbatim prompts contain the post's own curly quotes.
# ruff: noqa: E501 — the verbatim prompt bodies and the post's URL slug exceed the line limit.

from __future__ import annotations

import json
import re
from dataclasses import dataclass

__all__ = [
    "FINANCIAL_SENTIMENT_PROMPT",
    "FORWARD_LOOKING_PROMPT",
    "PREDICTION_PROMPT",
    "TASKS",
    "TaskSpec",
    "task_for",
]

#: Copied VERBATIM from the post's "Financial sentiment prompt" code block.
#: Canonical config copy: ``configs/sentiment/decision_tree_financial_sentiment.txt``.
FINANCIAL_SENTIMENT_PROMPT = """
# TASK DESCRIPTION

Read through the following snippet of text carefully and classify the **financial sentiment** as
either negative, neutral, or positive. You must also provide a short rationale for why you
assigned the financial sentiment you did.

For clarity here are the definitions negative, neutral, and positive sentiments:

   - **Negative**: The snippet describes an event or development that has had, is having, or is
       expected to have a material negative impact on the company's financial performance, share price, reputation,
       or outlook.

   - **Neutral**: The snippet is informational/descriptive and is not expected to have a material positive or
       negative impact on the company.

   - **Positive**: The snippet describes an event or development that has had, is having, or is expected to have
       a material positive impact on the company's financial performance, share price, reputation, or outlook.

Materiality note:
- “Material impact” includes likely effects on share price, revenue, costs, profitability, cash flow, guidance,
regulatory exposure, reputation, risk exposure, or competitive position.

# TASK GUIDELINES

For the avoidance of doubt here is a decision tree that you can follow to arrive at the most appropriate
sentiment classification for the snippet. Pay careful attention to the logic. Don't deviate.

START
│
├── Step 1: Carefully read and understand the snippet.
│
├── Step 2: Check for sentiment indicators:
│
├── Is the snippet clearly NEGATIVE?
│     (share price decline, losses, scandals, lawsuits, layoffs, product recalls, regulatory fines,
│      leadership resignations, declining sales, market-share losses, reputational damage etc.)
│    │
│    ├── YES → Classify as "negative"
│    │      └── Provide a rationale by summarizing WHY the snippet is negative.
│    │
│    └── NO → Continue below
│
├── Is the snippet clearly POSITIVE?
│     (share price increases, strong earnings, favorable partnerships, successful product launches, awards,
│      expansion plans, positive analyst coverage, reputational enhancement, etc.)
│    │
│    ├── YES → Classify as "positive"
│    │      └── Provide a rationale by summarizing WHY the snippet is positive.
│    │
│    └── NO → Continue below
│
└── If neither clearly positive nor negative → Classify as "neutral"
         (routine product announcements without performance implications, leadership appointments,
          scheduled reports, factual statements, general industry overviews, etc.)
       └── Provide a rationale by summarizing the WHY the snippet is neutral.

If there is conflicting sentiment in the snippet pick the most dominant one, otherwise default to **neutral**.

# RESPONSE FORMAT

You must respond with ONLY a valid JSON object formatted as follows. DO NOT WRITE ANY PREAMBLE JUST RETURN JSON.

{{
   "rationale": "A one-sentence rationale for your classification",
   "financial_sentiment": "either negative, neutral, or positive"
}}

# SNIPPET TO LABEL

Here is the snippet we would like you to assign a negative, neutral, or positive financial sentiment label to:

{text}

P.S. REMEMBER TO READ THE SNIPPET CAREFULLY AND FOLLOW THE GUIDELINES TO ARRIVE AT THE MOST APPROPRIATE
FINANCIAL SENTIMENT CLASSIFICATION. WHEN IN DOUBT, YOU SHOULD DEFER TO A "neutral" CLASSIFICATION FOR THE SNIPPET.
GOOD LUCK!
"""

#: Copied VERBATIM from the post's "Forward-looking prompt" code block.
#: Canonical config copy: ``configs/sentiment/decision_tree_forward_looking.txt``.
FORWARD_LOOKING_PROMPT = """
# TASK DESCRIPTION

You will be given a text snippet. Your task is to determine the **temporal
orientation** of the main event or topic in the text, classifying it as either
"forward" (forward-looking) or "not-forward" (backward-looking or neutral).

# GUIDELINES

For the avoidance of doubt here is a decision tree that you can follow to arrive
at the most appropriate temporal orientation classification for the snippet. Pay
careful attention to the logic. Don't deviate.

START
│
├── Step 1: Carefully read and identify the MAIN event or topic in the text.
│           (Ignore supporting details, commentary, or verb tenses of reporting)
│
├── Step 2: Determine the temporal orientation of this main event/topic:
│
├── Is the main event/topic FORWARD LOOKING?
│     (Will the event occur in the future or is it planned/expected?)
│     Examples: future launches, upcoming announcements, expansion plans, projections,
│              forecasts, guidance, targets, goals, roadmaps
│     Note: News about future plans (even if reported in past/present tense) = forward
│    │
│    ├── YES → Classify as "forward"
│    │      └── Provide a rationale explaining what future event the text focuses on.
│    │
│    └── NO → Classify as "not-forward"
│           └── This includes:
│               • Past events (announcements made yesterday, completed mergers, reported earnings)
│               • Current states (ongoing situations, present trading activity, existing conditions)
│               • Timeless facts or general statements
│               └── Provide a rationale explaining why the event is not forward-looking.
│
END

IMPORTANT: When uncertain about temporal orientation → Default to "not-forward"

# DISAMBIGUATION RULES

When the temporal orientation is unclear, apply these rules:

1. **Reporting Verb vs. Main Event Rule**
   - Ignore the tense of reporting verbs (said, announced, reported)
   - Focus on what is being reported about
   - Example: "CEO said yesterday the company will expand" → "forward" (expansion is future)

2. **Plans and Intentions Rule**
   - Any plans, intentions, targets, or forward guidance = "forward" (even if approved/decided in past)
   - Example: "Board approved new product launch" → "forward" (launch is future event)

3. **When in Doubt → Not-Forward**
   - If temporal orientation remains ambiguous → classify as "not-forward"

# TEXT SNIPPET TO LABEL
{text}

# RESPONSE FORMAT

You must respond with ONLY a valid JSON object formatted as follows. DO NOT WRITE ANY PREAMBLE JUST RETURN JSON.

{{
    "tense": "forward | not-forward",
    "rationale": "A one-sentence rationale for your classification"
}}

P.S. REMEMBER TO READ THE SNIPPET CAREFULLY AND FOLLOW THE GUIDELINES TO ARRIVE AT THE MOST APPROPRIATE
CLASSIFICATION. WHEN IN DOUBT, YOU SHOULD DEFER TO A "not-forward" CLASSIFICATION FOR THE SNIPPET. GOOD LUCK!
"""

#: Copied VERBATIM from the post's "Prediction prompt" code block.
#: Canonical config copy: ``configs/sentiment/decision_tree_prediction.txt``.
PREDICTION_PROMPT = """
# TASK DESCRIPTION

Read the following text snippet carefully and classify its **causal structure** as either
**predictive** or **not-predictive**. You must also provide a short rationale for
why you assigned the label you did.

For clarity, here are the definitions:

    1. Predictive: makes a concrete claim, forecast, prediction or estimate about a specific event.

    2. Not Predictive: only reports or explains past or present facts. It does not contain ANY
       predictions, estimates, or forecasts. Plans, schedules, hopes, retrospectives or non-concreate
       predictions or estimates means it is **not-predictive**.

# TASK GUIDELINES

For the avoidance of doubt, follow this decision tree exactly. Don’t deviate.

START
 │
 ├── Step 1: Carefully read and understand the snippet.
 │
 ├── Step 2: Check for causal structure indicators:
 │
 ├── Is the snippet clearly PREDICTIVE?
 │     (contains explicit forecasts or predictions about the future or
 │      numerical estimates about current or future events.)
 │    │
 │    ├── YES → Classify as "predictive"
 │    │      └── Provide a rationale summarizing WHAT future outcome or effect is being
 │    │          forecast or expected.
 │    │
 │    └── NO → Continue below
 │
 └── If it is not clearly predictive → Classify as "not-predictive"
        └── Provide a rationale summarizing WHY it is 'not-predictive'.

# DISAMBIGUATION RULES.

1. Predictive
    - If a snippet contains ANY predictive text, label it **"predictive"**.
    - Numerical estimates about CURRENT or FUTURE statistics are to be considered **predictive**.
    - Analyst estimates and ratings are **predictive** even if stated in the past tense.

2. Not predictive
    - Forward-looking plans or hopes without a claimed outcome remain **"not-predictive"**.
    - Schedules / announcements of events are not a forecast about an uncertain outcome or effect so **not-predictive**.
    - Retrospective narratives about past events should be classified as **not-predictive**.
    - Potential future actions without concrete predictions are **not-predictive**.
    - Future intent without a concrete prediction is **not-predictive**.

# RESPONSE FORMAT

You must respond with ONLY a valid JSON object formatted as follows. DO NOT WRITE ANY PREAMBLE JUST RETURN JSON.

{{
    "rationale": "A one-sentence rationale for your classification",
    "causal": "predictive | not-predictive"
}}

SNIPPET TO LABEL

Here is the snippet we would like you to label:

{text}

P.S. REMEMBER TO READ THE SNIPPET CAREFULLY AND FOLLOW THE GUIDELINES TO ARRIVE AT THE MOST APPROPRIATE
CAUSAL CLASSIFICATION. WHEN IN DOUBT, YOU SHOULD DEFER TO A "not-predictive" CLASSIFICATION FOR THE SNIPPET.
GOOD LUCK!
"""


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """One classification task: verbatim prompt, reply contract, label vocabulary.

    Attributes
    ----------
    name:
        Task key, also the filename stem for vote tables (``financial_sentiment``).
    template:
        The verbatim decision-tree prompt with the ``{text}`` slot unfilled.
    response_field:
        The JSON key carrying the label (``financial_sentiment``/``tense``/``causal``).
    labels:
        The exact label vocabulary the prompt allows, in prompt order.
    label_ints:
        Integer class per label, aligned with ``labels`` — the post's
        ``{positive: 1, neutral: 0, negative: -1}`` mapping and its binary analogues.
    default_label:
        The prompt's when-in-doubt class; ties and exhausted retries resolve here.
    system_prompt:
        The fine-tune/inference "guard" system message for this task.
    """

    name: str
    template: str
    response_field: str
    labels: tuple[str, ...]
    label_ints: tuple[int, ...]
    default_label: str
    system_prompt: str

    def build_prompt(self, text: str) -> str:
        """Render the verbatim template for *text* — the post's f-string evaluation."""
        return self.template.format(text=text)

    @property
    def label_to_int(self) -> dict[str, int]:
        return dict(zip(self.labels, self.label_ints, strict=True))

    @property
    def int_to_label(self) -> dict[int, str]:
        return dict(zip(self.label_ints, self.labels, strict=True))

    def parse_reply(self, raw: str) -> tuple[str | None, str]:
        """Parse a model reply under the strict-JSON contract.

        Returns ``(label, rationale)``; ``label`` is ``None`` when the reply is
        not a valid JSON object carrying this task's field with an allowed
        value — the caller decides whether to retry or fall back to
        :attr:`default_label`.
        """
        obj = _extract_json_object(raw)
        if not isinstance(obj, dict):
            return None, ""
        value = obj.get(self.response_field)
        rationale = obj.get("rationale")
        rationale_text = rationale.strip() if isinstance(rationale, str) else ""
        if not isinstance(value, str):
            return None, rationale_text
        label = value.strip().casefold()
        if label not in self.labels:
            return None, rationale_text
        return label, rationale_text


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(raw: str) -> object | None:
    """One strict parse, then one fence unwrap, then one bare ``{...}`` — no more."""
    candidates = [raw.strip()]
    fenced = _FENCE_RE.search(raw)
    if fenced is not None:
        candidates.append(fenced.group(1).strip())
    bare = _OBJECT_RE.search(raw)
    if bare is not None:
        candidates.append(bare.group(0))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed: object = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return parsed
    return None


#: The three tasks, keyed by name. Order follows the post.
TASKS: dict[str, TaskSpec] = {
    spec.name: spec
    for spec in (
        TaskSpec(
            name="financial_sentiment",
            template=FINANCIAL_SENTIMENT_PROMPT,
            response_field="financial_sentiment",
            labels=("negative", "neutral", "positive"),
            label_ints=(-1, 0, 1),
            default_label="neutral",
            # Verbatim from the post's fine-tuning script.
            system_prompt="Classify the financial sentiment as positive, negative, or neutral.",
        ),
        TaskSpec(
            name="forward_looking",
            template=FORWARD_LOOKING_PROMPT,
            response_field="tense",
            labels=("forward", "not-forward"),
            label_ints=(1, 0),
            default_label="not-forward",
            # The post only shows the sentiment system prompt; minimal analogue.
            system_prompt="Classify the temporal orientation as forward or not-forward.",
        ),
        TaskSpec(
            name="prediction",
            template=PREDICTION_PROMPT,
            response_field="causal",
            labels=("predictive", "not-predictive"),
            label_ints=(1, 0),
            default_label="not-predictive",
            # The post only shows the sentiment system prompt; minimal analogue.
            system_prompt="Classify the causal structure as predictive or not-predictive.",
        ),
    )
}


def task_for(name: str) -> TaskSpec:
    """Look up a task by name with an error that lists the valid ones."""
    try:
        return TASKS[name]
    except KeyError:
        from cybernaut_mini.config import ConfigError

        known = ", ".join(sorted(TASKS))
        msg = f"unknown sentiment task {name!r}; expected one of: {known}"
        raise ConfigError(msg) from None
