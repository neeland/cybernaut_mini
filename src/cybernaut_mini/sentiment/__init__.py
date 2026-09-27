"""Sentiment lab: labeler pool → benchmark → greedy ensemble → OLS distillation.

The four modules follow the two posts in order — :mod:`.labelers` and
:mod:`.benchmark` replicate "News Sentiment Showdown: Who Checks Vibes Best?",
:mod:`.ensemble` and :mod:`.distil` replicate "Ensemble and Distil"; see the
package README for the flow diagram and each module's docstring for the
assumptions and rejected alternatives.

Blog ref: https://nosible.com/blog/news-sentiment-showdown-who-checks-vibes-best (the
    labeler pool, the verbatim nine-example prompt, the agreement and timing
    findings) and https://nosible.com/blog/ensemble-and-distil (greedy iterative
    addition, 1,000-run bootstrap stability, OLS students on frozen embeddings).
    Local copies under ``docs/blog-archive/``.

Assumptions:
    - The re-exports below are the surface a survey script needs, not every helper
      in the four modules; the package is a lab, and the modules stay importable
      individually.
    - Optional members of the labeler pool — Flair, FinBERT, the few-shot LLMs — are
      opt-in, so the pool degrades to the always-available labelers rather than
      failing when an extra is not installed or no API key is present. That is what
      keeps the benchmark reproducible offline on the committed fixture.

Alternatives rejected: collapsing the four steps into one ``sentiment.py`` module.
    They have different inputs and different failure modes — a labeler pool, a
    scoring harness, a selection procedure, a regression fit — and the posts publish
    numbers for each separately, so one module would have to answer for all of them
    at once.
"""

from cybernaut_mini.sentiment.benchmark import agreement_matrix, cohens_kappa
from cybernaut_mini.sentiment.distil import build_sentences, distil_student, run_survey
from cybernaut_mini.sentiment.ensemble import (
    bootstrap_stability,
    greedy_forward_selection,
    sign_threshold,
)
from cybernaut_mini.sentiment.labelers import (
    FEW_SHOT_PROMPT,
    build_label_matrix,
    build_pool,
)

__all__ = [
    "FEW_SHOT_PROMPT",
    "agreement_matrix",
    "bootstrap_stability",
    "build_label_matrix",
    "build_pool",
    "build_sentences",
    "cohens_kappa",
    "distil_student",
    "greedy_forward_selection",
    "run_survey",
    "sign_threshold",
]
