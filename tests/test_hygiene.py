"""Corpus hygiene — planted cross-links use real fixture titles, density is pure math.

The inline-headline tests plant *real* corpus text inside *real* corpus text: the
title of ``ccn-174aa16cfff621d0`` ("The Ten Best Restaurants in Little Haiti", a
genuine short title-cased headline from the fixture) is inserted into the body of
another real article, once bare and once behind "READ MORE:". The assertion is the
gap matrix's contract: a planted cross-link never reaches the snippet store. Density
scoring is a pure function of numeric arrays, so its geometry tests use seeded
vectors (a tight cone vs diffuse directions — the post's static-prompt manifold in
miniature), and a real-data smoke test runs the scorer over the fixture's actual
publisher domains.

Blog ref: https://nosible.com/blog/using-vector-search-to-see-signals-in-company-news
    — inline headlines ("just about every news story ended up being related to Elon
    Musk") and AI-site detection ("a relatively small, densely packed region of the
    vector space ... combined with a metric of site volume"). Local copy:
    ``docs/blog-archive/using-vector-search-to-see-signals-in-company-news.md``.

Assumptions:
    - Fixture CC-News bodies are single-line paragraphs of punctuated sentences, so
      stripping a clean real article is a no-op — asserted on a specific committed
      document rather than all 200, because two fixture articles genuinely contain
      headline-like lines (which is the function working, not a bug).
    - The synthetic "farm" cone (base direction + small noise) is a numeric-array
      model of the post's static-prompt manifold, legitimate under the pure-math
      carve-out; no fabricated documents or events are created anywhere.

Alternatives rejected:
    - Generating real farm articles with a local LLM over rotating seed headlines
      (the laptop plan's validation): opt-in model download, minutes of generation —
      out of scope for the offline suite and listed as a follow-up.
    - Asserting which real fixture domains get flagged: the fixture has no actual
      content farms, so any such assertion would enshrine a false positive.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from cybernaut_mini.dedup import registered_domain
from cybernaut_mini.hygiene import (
    DomainDensity,
    ai_domain_scores,
    flag_ai_domains,
    mean_pairwise_cosine,
    seo_gate,
    strip_inline_headlines,
)
from cybernaut_mini.models import Document

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DOCS = REPO_ROOT / "data" / "01_raw" / "fixtures" / "documents.jsonl"
FIXTURE_EMBEDDINGS = REPO_ROOT / "artifacts" / "fixture" / "embeddings.npy"
FIXTURE_ROW_MAP = REPO_ROOT / "artifacts" / "fixture" / "row_map.json"

#: A real short, title-cased, verb-free headline committed in the fixture.
PLANTED_TITLE_DOC = "ccn-174aa16cfff621d0"
#: A real single-paragraph article to plant into.
HOST_DOC = "ccn-00b56c10586831d8"


@pytest.fixture(scope="module")
def fixture_rows() -> dict[str, dict[str, object]]:
    rows = [json.loads(line) for line in FIXTURE_DOCS.read_text().splitlines()]
    return {str(row["id"]): row for row in rows}


# ------------------------------------------------------------------ #
# strip_inline_headlines                                             #
# ------------------------------------------------------------------ #


def test_planted_crosslink_never_reaches_the_snippet_store(
    fixture_rows: dict[str, dict[str, object]],
) -> None:
    """A real title planted into a real body — bare and as READ MORE — is stripped."""
    planted_title = str(fixture_rows[PLANTED_TITLE_DOC]["title"])
    assert planted_title == "The Ten Best Restaurants in Little Haiti"
    host_text = str(fixture_rows[HOST_DOC]["text"])
    sentences = host_text.split(". ")
    contaminated = "\n".join(
        [
            ". ".join(sentences[:2]) + ".",
            planted_title,
            f"READ MORE: {planted_title}",
            ". ".join(sentences[2:]),
        ]
    )
    cleaned = strip_inline_headlines(contaminated)
    assert planted_title not in cleaned
    assert "READ MORE" not in cleaned
    # The real sentences on either side survive verbatim.
    assert sentences[0] in cleaned
    assert sentences[-1] in cleaned


def test_clean_real_article_is_untouched(fixture_rows: dict[str, dict[str, object]]) -> None:
    text = str(fixture_rows[HOST_DOC]["text"])
    assert strip_inline_headlines(text) == text.strip()


@pytest.mark.parametrize(
    "line",
    [
        "READ MORE: Puerto Rican crowned Miss Intercontinental 2016",
        "related: The Ten Best Restaurants in Little Haiti",
        "> Also read the follow-up coverage",
        "Sign up for our newsletter",
    ],
)
def test_crosslink_markers_are_dropped(line: str) -> None:
    assert strip_inline_headlines(line) == ""


def test_long_or_punctuated_lines_survive() -> None:
    # Punctuated sentence, even short and title-cased, is body text.
    kept = "The Scottish Thistle Awards Were Announced In Glasgow."
    assert strip_inline_headlines(kept) == kept
    # A short lowercase clause with a verb is body text.
    clause = "the winners will be announced in Glasgow"
    assert strip_inline_headlines(clause) == clause


def test_blank_lines_collapse() -> None:
    assert strip_inline_headlines("First sentence stays.\n\n\nSecond one stays.") == (
        "First sentence stays.\nSecond one stays."
    )


# ------------------------------------------------------------------ #
# mean_pairwise_cosine (pure math)                                   #
# ------------------------------------------------------------------ #


def test_mean_pairwise_cosine_matches_brute_force() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(9, 16)).astype(np.float32)
    unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    expected = np.mean(
        [float(unit[i] @ unit[j]) for i in range(9) for j in range(9) if i != j]
    )
    assert mean_pairwise_cosine(matrix) == pytest.approx(expected, abs=1e-5)


def test_mean_pairwise_cosine_bounds_and_errors() -> None:
    identical = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (4, 1))
    assert mean_pairwise_cosine(identical) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="n>=2"):
        mean_pairwise_cosine(identical[:1])


# ------------------------------------------------------------------ #
# ai_domain_scores / flag_ai_domains                                 #
# ------------------------------------------------------------------ #


def _cone_and_background() -> tuple[list[str], npt.NDArray[np.float32]]:
    """20 vectors in a tight cone (the 'farm') + 40 diffuse vectors (2 real-ish sites)."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=64)
    base /= np.linalg.norm(base)
    farm = base[None, :] + 0.05 * rng.normal(size=(20, 64))
    diffuse = rng.normal(size=(40, 64))
    netlocs = ["farm.example"] * 20 + ["broad-a.example"] * 20 + ["broad-b.example"] * 20
    matrix = np.vstack([farm, diffuse]).astype(np.float32)
    return netlocs, matrix


def test_static_prompt_cone_scores_far_above_background() -> None:
    netlocs, matrix = _cone_and_background()
    profiles = ai_domain_scores(netlocs, matrix, min_docs=5)
    by_domain = {profile.netloc: profile for profile in profiles}
    assert by_domain["farm.example"].density_excess > 0.5
    assert by_domain["broad-a.example"].density_excess < 0.2
    assert profiles[0].netloc == "farm.example"  # Sorted farm-first.
    flagged = flag_ai_domains(profiles, density_margin=0.3)
    assert flagged == ["farm.example"]


def test_volume_rule_requires_dates() -> None:
    netlocs, matrix = _cone_and_background()
    day = dt.date(2023, 7, 1)
    dates: list[dt.datetime | dt.date | None] = [day] * 20 + [None] * 40
    profiles = ai_domain_scores(netlocs, matrix, dates=dates, min_docs=5)
    by_domain = {profile.netloc: profile for profile in profiles}
    # 20 articles in one day: the farm publishes fastest of the dated domains.
    farm = by_domain["farm.example"]
    assert farm.docs_per_day == pytest.approx(20.0)
    # Undated domains are never flagged once the combined rule is on.
    assert flag_ai_domains(profiles, density_margin=0.3, volume_zscore_min=-1.0) == [
        "farm.example"
    ]
    undated = ai_domain_scores(netlocs, matrix, min_docs=5)
    assert flag_ai_domains(undated, density_margin=0.3, volume_zscore_min=-1.0) == []


def test_min_docs_gate_and_validation() -> None:
    netlocs, matrix = _cone_and_background()
    profiles = ai_domain_scores(netlocs[:20] + [""] * 40, matrix, min_docs=25)
    assert profiles == []  # 20 < 25 and blank netlocs are never scored.
    with pytest.raises(ValueError, match="embeddings"):
        ai_domain_scores(["a"], matrix)
    with pytest.raises(ValueError, match="dates"):
        ai_domain_scores(netlocs, matrix, dates=[None])


def test_real_fixture_domains_score_finite(
    fixture_rows: dict[str, dict[str, object]],
) -> None:
    """Smoke over the real corpus: every scored real publisher gets a finite profile."""
    ccn = [row for row in fixture_rows.values() if str(row["id"]).startswith("ccn-")]
    row_map: dict[str, int] = json.loads(FIXTURE_ROW_MAP.read_text())
    embeddings = np.load(FIXTURE_EMBEDDINGS).astype(np.float32)
    matrix = embeddings[np.asarray([row_map[str(row["id"])] for row in ccn], dtype=np.int64)]
    netlocs = [registered_domain(str(row["url"])) for row in ccn]
    profiles = ai_domain_scores(netlocs, matrix, min_docs=5)
    assert profiles  # The fixture has several publishers with >= 5 articles.
    for profile in profiles:
        assert isinstance(profile, DomainDensity)
        assert profile.n_docs >= 5
        assert np.isfinite(profile.mean_pairwise_cosine)
        assert np.isfinite(profile.density_excess)
    excesses = [profile.density_excess for profile in profiles]
    assert excesses == sorted(excesses, reverse=True)


# ------------------------------------------------------------------ #
# seo_gate                                                           #
# ------------------------------------------------------------------ #


def test_seo_gate_keeps_ccnews_and_drops_bare_miracl(
    fixture_rows: dict[str, dict[str, object]],
) -> None:
    """All 200 CC-News rows carry url+date+publisher; the 260 MIRACL rows do not."""
    documents = [Document.model_validate(row) for row in fixture_rows.values()]
    kept, dropped = seo_gate(documents)
    assert len(kept) + len(dropped) == len(documents)
    assert {doc.id[:4] for doc in kept} == {"ccn-"}
    assert len(kept) == 200
    assert all(doc.id.startswith("mir-") for doc in dropped)
    for doc in kept:
        assert doc.url
        assert doc.published_at is not None
