"""Pure-function Kedro nodes for the corpus_ingest pipeline.

No node here opens a file or a socket. The source is supplied by the catalog
(``raw_corpus_source``) and every output is handed back as plain data for the
catalog to persist, which is what keeps the raw -> intermediate -> primary layers
inspectable and swappable per environment.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — produces the
    "Documents" input to the build-time flow, upstream of text processing,
    embedding, and shard learning. Local copy:
    ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - An empty result is a configuration error, not a valid outcome. Each node
      raises rather than passing an empty list downstream, so a wrong ``field_map``
      or an over-tight filter surfaces here instead of as a confusing failure
      several nodes later.
    - Selection belongs after normalisation. Narrowing the corpus must never
      require re-downloading it, so the raw snapshot always stays complete.

Alternatives considered:
    - Filtering at fetch time (a smaller ``max_rows``, a server-side filter): less
      data to store and faster runs. Rejected because it couples the snapshot to the
      current build's parameters, so re-tuning the corpus means re-fetching it.
    - Folding normalise and select into one node: fewer nodes, one less
      intermediate artifact. Rejected because the intermediate layer is what lets
      you inspect *why* a document was dropped without re-running acquisition.
"""

from __future__ import annotations

from typing import Any

from cybernaut_mini.corpus import (
    CorpusSourceConfig,
    load_miracl_judgments,
    normalize_rows,
    validate_judgments_against_corpus,
)
from cybernaut_mini.models import Document


def snapshot_raw_corpus(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist the source rows verbatim into the raw layer.

    Deliberately a pass-through: the value is the *snapshot*. Once written to
    ``data/01_raw`` the build no longer depends on the Hub being reachable, and the
    exact bytes a shard was built from stay on disk for audit.
    """
    if not rows:
        msg = "corpus source returned zero rows; refusing to snapshot an empty corpus"
        raise ValueError(msg)
    return rows


def normalize_corpus(
    rows: list[dict[str, Any]],
    corpus_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Map raw source rows onto the Document schema (intermediate layer)."""
    config = CorpusSourceConfig.model_validate(corpus_params)
    documents = normalize_rows(rows, config)
    if not documents:
        msg = (
            f"normalisation produced zero documents from {len(rows)} source rows; "
            f"check field_map and min_text_chars"
        )
        raise ValueError(msg)
    return [doc.model_dump(mode="json") for doc in documents]


def build_miracl_judgments(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert structured MIRACL en-dev rows to serialisable Judgment dicts.

    A Kedro node wrapper around :func:`cybernaut_mini.corpus.load_miracl_judgments`.
    The catalog entry ``miracl_en_dev_source`` supplies the rows; the output is
    written to the ``judgments`` catalog entry for downstream evaluation.
    """
    judgments = load_miracl_judgments(rows)
    return [j.model_dump(mode="json") for j in judgments]


def validate_judgments(
    judgments: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assert every judgment docid exists in the corpus; pass judgments through.

    A Kedro node wrapper around
    :func:`cybernaut_mini.corpus.validate_judgments_against_corpus`. Wires the
    corpus pipeline's ``documents`` output to the judgment pipeline so validation
    runs after both branches complete.
    """
    from cybernaut_mini.models import Judgment

    corpus_ids = {d["id"] for d in documents}
    parsed = [Judgment.model_validate(j) for j in judgments]
    validate_judgments_against_corpus(parsed, corpus_ids)
    return judgments


def merge_documents(
    ccnews_docs: list[dict[str, Any]],
    miracl_docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine normalised CC-News and MIRACL corpus documents into one list.

    The two sources use disjoint id prefixes (``ccn-*`` for CC-News, ``mir-*``
    for MIRACL) so id collisions should never occur in a correctly configured
    pipeline. Defensive deduplication by id is applied regardless: the first
    occurrence wins, and CC-News precedes MIRACL in argument order, so a
    ccn- record beats a mir- record on an accidental collision.

    Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — the fidelity
        requirement that every qrels-positive MIRACL passage is present in the
        corpus mandates a *separate* MIRACL corpus branch rather than a single
        CC-News source, which would miss the MIRACL passages entirely. Local
        copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

    Assumptions:
        - ``ccnews_docs`` and ``miracl_docs`` have already been normalised by
          :func:`normalize_corpus` with their respective ``CorpusSourceConfig``
          values, so every record has an ``id`` field.
        - The MIRACL branch is expected to be smaller (~8 k passages) and all of
          its documents survive into the merged list before :func:`select_documents`
          applies the language filter and ``max_documents`` cap. This guarantees
          the fidelity rule: qrels-positive passages are never dropped by the cap.

    Alternatives rejected:
        - A single source with a unified ``field_map``: CC-News and MIRACL use
          different column names (``plain_text`` vs ``text``, ``requested_url`` vs
          ``docid``), so a shared config would need conditional logic — the
          definition of what this class exists to avoid.
        - Merging inside ``select_documents``: that node already has one
          responsibility (filtering + capping); adding a join would hide a
          structurally important step from ``kedro viz`` lineage.

    Raises :exc:`ValueError` when both branches produce zero documents.
    """
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for doc in (*ccnews_docs, *miracl_docs):
        doc_id = doc["id"]
        if doc_id not in seen:
            seen.add(doc_id)
            merged.append(doc)
    if not merged:
        msg = (
            "merge_documents: both ccnews and miracl branches produced zero "
            "documents; check the source catalog entries and their field_maps"
        )
        raise ValueError(msg)
    return merged


def select_documents(
    documents: list[dict[str, Any]],
    selection_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter and cap the normalised corpus into the build-ready primary layer.

    Selection happens here rather than at fetch time so the raw snapshot stays
    complete: narrowing the build corpus never requires re-downloading.

    ``selection_params`` keys:
      ``languages``          — keep only these ``Document.language`` values
      ``metadata_equals``    — keep only documents whose metadata matches every pair
      ``max_documents``      — cap the corpus, keeping the lowest ids for determinism
    """
    languages: list[str] | None = selection_params.get("languages")
    metadata_equals: dict[str, Any] | None = selection_params.get("metadata_equals")
    max_documents: int | None = selection_params.get("max_documents")

    selected: list[dict[str, Any]] = []
    for raw in documents:
        doc = Document.model_validate(raw)
        if languages is not None and doc.language not in languages:
            continue
        if metadata_equals is not None and any(
            doc.metadata.get(key) != expected for key, expected in metadata_equals.items()
        ):
            continue
        selected.append(raw)

    # Sort by id before capping so the retained subset is independent of input order.
    selected.sort(key=lambda raw: str(raw["id"]))
    if max_documents is not None:
        selected = selected[:max_documents]

    if not selected:
        msg = (
            "document selection matched zero documents; relax languages / "
            "metadata_equals in the corpus_selection parameters"
        )
        raise ValueError(msg)
    return selected
