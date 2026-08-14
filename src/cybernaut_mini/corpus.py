"""Normalisation of an external corpus into validated :class:`Document` records.

The acquisition pipeline pulls rows in whatever shape the source publishes them
(the NOSIBLE Hugging Face datasets use ``text`` / ``label`` / ``netloc`` / ``url``)
and this module maps them onto the project's own schema. The field mapping is data,
not code, so a second source — the NOSIBLE Search API, a local dump — only needs a
new ``field_map`` in ``conf/``, never a new normaliser.

Determinism
-----------
Document ids are derived by hashing the source's natural key, so re-running the
pipeline over the same pinned revision yields the same ids, and therefore the same
shard assignments and the same byte-for-byte index.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — the "Documents" node
    that opens the build-time half of the architecture diagram, upstream of text
    processing and embedding. Local copy:
    ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - A public corpus contains junk. Rows that are empty or shorter than
      ``min_text_chars`` are skipped, not raised on, because one malformed row out
      of 100,000 must not fail a multi-hour production build.
    - The source's ``url`` is a natural key. Where it is absent the body text is
      hashed instead, so every row still gets a stable id rather than colliding on
      the empty string.
    - Passages arrive untitled. A title is derived from the first sentence because
      the shard manifest, the shard summary, and the ``title\\ntext`` embedding
      input all assume one exists.

Alternatives considered:
    - A normaliser class per source: the obvious object-oriented shape, and the
      right one once sources need genuinely different logic. Rejected while the
      difference between sources is only *which column holds what*, which a
      declarative ``field_map`` expresses without new code.
    - Sequential ids (``doc-000001``): shorter and readable. Rejected because they
      depend on row order, so re-fetching a corpus whose rows were reordered would
      renumber every document and silently invalidate the stored judgments.
    - Raising on unusable rows: better for a small curated corpus where every row is
      supposed to be valid. Rejected here for the reason in the assumptions above.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cybernaut_mini.models import Document, Judgment

#: Cap on a derived title. Long enough to be a useful routing signal, short enough
#: that it does not swamp the body text when the two are concatenated for embedding.
_TITLE_MAX_CHARS = 120

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


class CorpusSourceConfig(BaseModel):
    """How to turn raw source rows into :class:`Document` records."""

    model_config = ConfigDict(extra="forbid")

    #: Source column -> Document field. Values must name real Document fields.
    field_map: dict[str, str] = Field(
        default_factory=lambda: {
            "text": "text",
            "url": "url",
        }
    )
    #: Source columns copied verbatim into ``Document.metadata``.
    metadata_fields: list[str] = Field(default_factory=list)
    #: Applied when the source carries no per-row language column.
    default_language: str | None = "en"
    #: Prefix for generated ids, so documents from different sources cannot collide.
    id_prefix: str = "doc"
    #: Drop documents whose body is shorter than this (boilerplate, nav text).
    min_text_chars: int = Field(default=32, ge=0)


class CorpusNormalizeError(ValueError):
    """Raised when a source cannot be mapped onto the Document schema at all."""


def _clean(value: Any) -> str:
    """Collapse whitespace in a source value; non-strings become empty."""
    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", value).strip()


def derive_title(text: str) -> str:
    """Use the first sentence as the title, truncated on a word boundary.

    These corpora ship passages without titles, but the shard manifest, the shard
    summary, and the ``title\\ntext`` embedding input all assume one exists.
    """
    first = _SENTENCE_END.split(text, maxsplit=1)[0].strip()
    if not first:
        first = text.strip()
    if len(first) <= _TITLE_MAX_CHARS:
        return first
    clipped = first[:_TITLE_MAX_CHARS].rsplit(" ", 1)[0].strip()
    # A single token longer than the cap leaves nothing after rsplit; hard-cut it.
    return clipped or first[:_TITLE_MAX_CHARS]


def make_document_id(prefix: str, natural_key: str) -> str:
    """Stable id from the row's natural key, so ids survive a re-run."""
    digest = hashlib.blake2b(natural_key.encode("utf-8"), digest_size=8).hexdigest()
    return f"{prefix}-{digest}"


def normalize_rows(
    rows: list[dict[str, Any]],
    config: CorpusSourceConfig,
) -> list[Document]:
    """Map raw source rows onto validated, deduplicated :class:`Document` records.

    Rows that are unusable — no body text, or shorter than ``min_text_chars`` — are
    skipped rather than raised on: a 100k-row public corpus is expected to contain
    some junk, and one bad row must not fail a production build. Duplicate natural
    keys collapse to their first occurrence. Ordering is by document id so the
    result is independent of the source's row order.

    Raises :class:`CorpusNormalizeError` only for problems that make the *whole*
    source unusable, such as a ``field_map`` that never resolves a text column.
    """
    if not rows:
        return []

    text_sources = [src for src, dest in config.field_map.items() if dest == "text"]
    if not text_sources:
        msg = "field_map must map exactly one source column onto 'text'"
        raise CorpusNormalizeError(msg)
    text_source = text_sources[0]

    available = set(rows[0].keys())
    if text_source not in available:
        msg = (
            f"source column {text_source!r} (mapped to 'text') is absent from the "
            f"corpus; available columns: {sorted(available)}"
        )
        raise CorpusNormalizeError(msg)

    documents: dict[str, Document] = {}
    for row in rows:
        text = _clean(row.get(text_source))
        if len(text) < config.min_text_chars:
            continue

        fields: dict[str, Any] = {}
        for source_field, dest_field in config.field_map.items():
            if dest_field == "text":
                continue
            value = _clean(row.get(source_field))
            if value:
                fields[dest_field] = value

        metadata = {
            name: row[name] for name in config.metadata_fields if row.get(name) is not None
        }

        # Prefer the source url as the natural key; fall back to the body so rows
        # without one still get a stable id rather than colliding on the empty string.
        natural_key = fields.get("url") or text
        doc_id = make_document_id(config.id_prefix, str(natural_key))
        if doc_id in documents:
            continue

        documents[doc_id] = Document(
            id=doc_id,
            title=fields.get("title") or derive_title(text),
            text=text,
            url=fields.get("url"),
            language=fields.get("language") or config.default_language,
            published_at=fields.get("published_at"),
            metadata=metadata,
        )

    return [documents[key] for key in sorted(documents)]


# ------------------------------------------------------------------ #
# MIRACL qrels → Judgment loader                                      #
# ------------------------------------------------------------------ #

_MIRACL_ID_PREFIX = "mir"


def load_miracl_judgments(rows: list[dict[str, Any]]) -> list[Judgment]:
    """Map structured MIRACL en-dev rows onto :class:`~cybernaut_mini.models.Judgment` records.

    The catalog entry for ``miracl_en_dev_source`` is backed by
    ``HuggingFaceDataset("miracl/miracl", config="en", split="dev")``, which returns
    rows with four fields: ``query_id`` (str), ``query`` (str), ``positive_passages``
    (list of ``{docid, title, text}`` with relevance grade=1), and
    ``negative_passages`` (list of ``{docid, title, text}`` with relevance grade=0).

    Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — the evaluation
        substrate used to record per-mode recall/nDCG/MRR baselines. Local copy:
        ``data/00_reference/the-road-to-cybernaut-1.md``.

    Assumptions:
        - ``relevant_document_ids`` holds ALL graded docids (both 0 and 1), not only
          the positives. The Judgment model stores ``dict[str, int]`` so both grades
          survive; downstream eval nodes compute nDCG correctly from the full set.
        - Docids are keyed as ``make_document_id("mir", passage["docid"])`` to match
          the ``id_prefix="mir"`` used by the MIRACL corpus ingest branch. A judgment
          whose docids use bare MIRACL strings would never match any Document.id.
        - A query row with zero assessments (empty positive AND negative lists) is
          silently skipped rather than raised on; MIRACL en-dev has none of these, but
          robustness beats a fragile length check.

    Alternatives rejected:
        - Accepting raw TSV bytes (``{query_id}\\t{question}`` + TREC qrel format) as
          two separate function arguments. Rejected because the structured
          ``miracl/miracl`` Hub dataset gives the same information already parsed, so
          separate TSV parsing adds code without adding insight.
        - Including only grade=1 docids. Rejected because nDCG requires the full
          relevance pool to distinguish true negatives (grade=0 but retrieved) from
          un-judged documents.

    Raises :class:`ValueError` when ``rows`` is empty (a configuration error — an
    empty MIRACL source is never valid).
    """
    if not rows:
        msg = "MIRACL source returned zero rows; cannot build judgments from an empty dataset"
        raise ValueError(msg)

    judgments: list[Judgment] = []
    for row in rows:
        query_id = str(row["query_id"])
        question = str(row.get("query", ""))

        relevant: dict[str, int] = {}
        for passage in row.get("positive_passages", []):
            doc_id = make_document_id(_MIRACL_ID_PREFIX, str(passage["docid"]))
            relevant[doc_id] = 1
        for passage in row.get("negative_passages", []):
            doc_id = make_document_id(_MIRACL_ID_PREFIX, str(passage["docid"]))
            relevant[doc_id] = 0

        if not relevant:
            # A query with no assessed passages contributes nothing to evaluation.
            continue

        judgments.append(
            Judgment(
                query_id=query_id,
                question=question,
                relevant_document_ids=relevant,
            )
        )

    return judgments


def validate_judgments_against_corpus(
    judgments: list[Judgment],
    corpus_doc_ids: set[str],
) -> None:
    """Assert every judgment docid exists in the built corpus.

    Raises :class:`ValueError` listing up to 20 missing docids so the operator
    can diagnose which MIRACL passages were dropped by the corpus filter step.

    Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — acceptance
        criterion: "Every docid in miracl_en_dev_judgments.jsonl exists in the
        built index." A judgment over a missing doc silently caps recall.

    Assumptions: the corpus ids are passed as a ``set`` so the ``in`` check is O(1)
        per docid rather than O(n). Building the set once before calling this is the
        caller's responsibility.

    Alternatives rejected: returning the list of missing ids and leaving it to the
        caller to decide whether to raise. Rejected because a missing docid is never
        acceptable — it would silently cap recall — so failing loudly is the only
        correct behaviour.
    """
    missing: list[str] = []
    for judgment in judgments:
        for doc_id in judgment.relevant_document_ids:
            if doc_id not in corpus_doc_ids:
                missing.append(doc_id)

    if missing:
        sample = sorted(missing)[:20]
        msg = (
            f"validate_judgments: {len(missing)} judgment docid(s) absent from corpus "
            f"(first 20): {sample}. "
            f"Ensure the MIRACL corpus branch snapshots all qrels docids before selection."
        )
        raise ValueError(msg)
