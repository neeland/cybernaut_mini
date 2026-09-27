"""Corpus hygiene: inline-headline stripping and AI-site detection by embedding density.

Two curation lessons from the 2024 post, implemented as pure functions so they can
run as optional ingest nodes or ad hoc over any corpus slice. First, news bodies
carry cross-links to *other* stories ("READ MORE: ..."), and unless those sentences
are stripped before embedding, "just about every news story ended up being related to
Elon Musk". Second, AI-generated content farms cannot be caught story-by-story, but
their *domains* can: articles generated from one static prompt occupy a small, dense
region of embedding space, so a netloc whose documents have an unusually high mean
pairwise cosine — relative to the corpus background — plus unusually high volume is a
farm candidate. A completeness gate (``seo_gate``) drops rows too bare to be indexed
at all.

Blog ref: https://nosible.com/blog/using-vector-search-to-see-signals-in-company-news
    — "One early foible was not stripping those sentences out ... just about every
    news story ended up being related to Elon Musk"; "we look at the distribution of
    the embeddings of snippets 'written' by that domain ... AI-generated sites occupy
    a relatively small, densely packed region of the vector space"; "because the
    articles are generated from a *static* prompt plus some *dynamic* seed content
    the resulting articles are essentially 'constrained' to a small manifold in the
    vector space. When this metric is combined with a metric of site volume, you get
    a good predictor." Local copy:
    ``docs/blog-archive/using-vector-search-to-see-signals-in-company-news.md``.

Assumptions:
    - An inline headline is recognised structurally, not by a trained model: a line
      that starts with a cross-link marker ("READ MORE", "RELATED:", ...), or a short
      line (< 12 tokens) with no terminal punctuation that is title-case-heavy or
      verb-light. Real body sentences end with punctuation and contain verbs; wire
      headlines are noun phrases in title case. The token bound and both heuristics
      are parameters, and the whole check is per-line/per-sentence so a false
      positive costs one sentence, never a document.
    - Density is the mean pairwise cosine over a netloc's documents, computed exactly
      via the unit-vector identity ``sum_ij cos(i,j) = ||sum(v)||^2``, excluding
      self-pairs — no sampling, so the score is deterministic. The corpus background
      is the same statistic over all documents. What gets reported is the *excess*
      over background: embedding models place all news in a fairly tight cone, so an
      absolute threshold would not transfer across models but an excess does.
    - Volume enters as a z-score of per-domain articles/day across domains (the
      post's "metric of site volume"), and is ``None`` when no dates are available —
      a density-only flag is still useful on undated slices.
    - A domain needs ``min_docs`` documents before its density means anything; below
      that the estimator is dominated by chance pairs and the domain is not scored.
    - ``seo_gate`` checks the fields this repo's ``Document`` actually carries:
      title (non-empty by model contract), URL, publish date, and a publisher-ish
      metadata field (``publisher`` or ``sitename``). The post's fuller SEO-metadata
      gate (description, language rank) needs fields CC-News does not reliably have.

Alternatives rejected:
    - Per-story AI classifiers: the post is explicit that story-level identification
      "might not be possible at all"; the domain-level distributional signal is the
      lesson, so that is what is implemented.
    - Mean distance-to-centroid instead of mean pairwise cosine: monotonically
      related for unit vectors (``E[cos] = ||centroid||^2``-ish), but the post's own
      framing is the pairwise distribution of a domain's snippets, and the pairwise
      mean has the exact closed form used here — same cost, closer to the text.
    - pySBD for sentence splitting inside :func:`strip_inline_headlines`: already a
      repo dependency for stage 2, but inline headlines are usually *not* sentences
      (no terminal punctuation), which is precisely the signal; a line/period split
      keeps the structural cue a segmenter would normalise away.
    - langdetect-confidence source-language ranks from the laptop plan: needs the
      fastText model download in the default path, which would break the offline
      contract; left as an opt-in follow-up.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cybernaut_mini.models import Document

__all__ = [
    "DEFAULT_MAX_HEADLINE_TOKENS",
    "DomainDensity",
    "ai_domain_scores",
    "flag_ai_domains",
    "mean_pairwise_cosine",
    "seo_gate",
    "strip_inline_headlines",
]

FloatArray = npt.NDArray[np.float32]

#: Inline headlines are short; body sentences that matter rarely fit in 11 tokens.
DEFAULT_MAX_HEADLINE_TOKENS = 12

#: Case-insensitive prefixes that mark a line as a cross-link, not body text.
_CROSSLINK_PREFIXES: tuple[str, ...] = (
    "read more",
    "read next",
    "more:",
    "related:",
    "related articles",
    "also read",
    "see also",
    "watch:",
    "video:",
    "click here",
    "sign up",
    "subscribe",
)

#: Lowercase auxiliaries/copulas: a sentence containing one is not "verb-light".
_VERB_HINTS = frozenset(
    {
        "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had",
        "will", "would", "can", "could", "may", "might", "must", "shall", "should",
        "do", "does", "did",
        "says", "said", "say",
    }
)

# The curly quotes are deliberate: CC-News bodies end quoted sentences with them.
_SENTENCE_END = (".", "!", "?", '"', "”", "’", ";")  # noqa: RUF001


def _tokens(line: str) -> list[str]:
    return line.split()


def _title_case_share(tokens: Sequence[str]) -> float:
    alpha = [token for token in tokens if token[0].isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for token in alpha if token[0].isupper()) / len(alpha)


def _is_crosslink(line: str) -> bool:
    lowered = line.strip().lower().lstrip("*-•> ")
    return any(lowered.startswith(prefix) for prefix in _CROSSLINK_PREFIXES)


def _is_inline_headline(line: str, max_tokens: int) -> bool:
    """Short, unterminated, and either title-case-heavy or verb-light."""
    stripped = line.strip()
    if not stripped or stripped.endswith(_SENTENCE_END):
        return False
    tokens = _tokens(stripped)
    if len(tokens) >= max_tokens:
        return False
    title_heavy = _title_case_share(tokens) >= 0.6
    verb_light = not any(token.lower().strip(",.:;") in _VERB_HINTS for token in tokens)
    return title_heavy or verb_light


def strip_inline_headlines(
    text: str, *, max_tokens: int = DEFAULT_MAX_HEADLINE_TOKENS
) -> str:
    """Remove cross-link lines and inline headlines from a news body.

    Operates line by line: a line is dropped when it starts with a cross-link marker
    or looks like a bare headline (short, no terminal punctuation, title-case-heavy
    or verb-light). Surviving lines are joined back with single newlines with their
    internal text untouched, so the output embeds and snippets exactly as written.
    """
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _is_crosslink(stripped) or _is_inline_headline(stripped, max_tokens):
            continue
        kept.append(stripped)
    return "\n".join(kept)


def mean_pairwise_cosine(embeddings: FloatArray) -> float:
    """Exact mean cosine over all unordered pairs of rows, excluding self-pairs.

    Rows are L2-normalised first; then ``sum_{i,j} cos(i,j) = ||sum_i v_i||^2``,
    so the pairwise mean is ``(||sum||^2 - n) / (n * (n - 1))`` — one pass, no
    quadratic memory. Requires at least two rows.
    """
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        msg = f"need a (n>=2, dim) matrix, got shape {matrix.shape}"
        raise ValueError(msg)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = matrix / norms
    total = unit.sum(axis=0)
    n_rows = matrix.shape[0]
    return float((total @ total - n_rows) / (n_rows * (n_rows - 1)))


@dataclass(frozen=True)
class DomainDensity:
    """Embedding-density profile of one netloc.

    ``density_excess`` is the domain's mean pairwise cosine minus the corpus
    background; ``volume_zscore`` is the domain's articles/day z-scored across all
    scored domains, or ``None`` when no dates were supplied.
    """

    netloc: str
    n_docs: int
    mean_pairwise_cosine: float
    density_excess: float
    docs_per_day: float | None
    volume_zscore: float | None


def _docs_per_day(dates: list[dt.datetime | dt.date | None]) -> float | None:
    days = [d.date() if isinstance(d, dt.datetime) else d for d in dates if d is not None]
    if not days:
        return None
    span_days = (max(days) - min(days)).days + 1
    return len(days) / span_days


def ai_domain_scores(
    netlocs: Sequence[str],
    embeddings: FloatArray,
    *,
    dates: Sequence[dt.datetime | dt.date | None] | None = None,
    min_docs: int = 5,
) -> list[DomainDensity]:
    """Score every netloc with >= ``min_docs`` rows: density vs corpus background.

    ``netlocs`` and ``embeddings`` are row-aligned (``dates`` too, when given).
    Returns profiles sorted by descending ``density_excess`` then netloc, so farm
    candidates surface first. The background is the mean pairwise cosine over the
    *whole* row set, dense domains included — a conservative background that only
    dampens excesses, never inflates them.
    """
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(netlocs):
        msg = f"embeddings must be ({len(netlocs)}, dim), got {matrix.shape}"
        raise ValueError(msg)
    if dates is not None and len(dates) != len(netlocs):
        msg = f"dates has {len(dates)} rows, expected {len(netlocs)}"
        raise ValueError(msg)

    rows_by_domain: dict[str, list[int]] = defaultdict(list)
    for row, netloc in enumerate(netlocs):
        if netloc:
            rows_by_domain[netloc].append(row)
    scored = {
        domain: rows for domain, rows in rows_by_domain.items() if len(rows) >= min_docs
    }
    if not scored or matrix.shape[0] < 2:
        return []

    background = mean_pairwise_cosine(matrix)
    per_day: dict[str, float | None] = {}
    for domain, rows in scored.items():
        domain_dates = [dates[row] for row in rows] if dates is not None else []
        per_day[domain] = _docs_per_day(domain_dates) if dates is not None else None

    rates = [rate for rate in per_day.values() if rate is not None]
    rate_mean = float(np.mean(rates)) if rates else 0.0
    rate_std = float(np.std(rates)) if rates else 0.0

    profiles: list[DomainDensity] = []
    for domain, rows in scored.items():
        density = mean_pairwise_cosine(matrix[np.asarray(rows, dtype=np.int64)])
        rate = per_day[domain]
        if rate is None:
            zscore = None
        elif rate_std > 0.0:
            zscore = (rate - rate_mean) / rate_std
        else:
            zscore = 0.0
        profiles.append(
            DomainDensity(
                netloc=domain,
                n_docs=len(rows),
                mean_pairwise_cosine=density,
                density_excess=density - background,
                docs_per_day=rate,
                volume_zscore=zscore,
            )
        )
    profiles.sort(key=lambda p: (-p.density_excess, p.netloc))
    return profiles


def flag_ai_domains(
    profiles: Sequence[DomainDensity],
    *,
    density_margin: float = 0.15,
    volume_zscore_min: float | None = None,
) -> list[str]:
    """Netlocs whose density excess (and optionally volume z-score) cross thresholds.

    ``density_margin`` is how far above the corpus background a domain's mean
    pairwise cosine must sit. When ``volume_zscore_min`` is given, a domain must
    *also* publish unusually fast — the post's "combined with a metric of site
    volume" — and undated profiles are never flagged by the combined rule.
    """
    flagged = []
    for profile in profiles:
        if profile.density_excess < density_margin:
            continue
        if volume_zscore_min is not None and (
            profile.volume_zscore is None or profile.volume_zscore < volume_zscore_min
        ):
            continue
        flagged.append(profile.netloc)
    return sorted(flagged)


def seo_gate(
    documents: Sequence[Document],
) -> tuple[list[Document], list[Document]]:
    """Split documents into (kept, dropped) on metadata completeness.

    A document passes when it has a URL, a publish date, and a publisher-ish
    metadata field (``publisher`` or ``sitename``). Titles are non-empty by the
    ``Document`` model contract, so they are not re-checked. Order is preserved.
    """
    kept: list[Document] = []
    dropped: list[Document] = []
    for doc in documents:
        has_source = bool(
            str(doc.metadata.get("publisher", "") or "").strip()
            or str(doc.metadata.get("sitename", "") or "").strip()
        )
        if doc.url and doc.published_at is not None and has_source:
            kept.append(doc)
        else:
            dropped.append(doc)
    return kept, dropped
