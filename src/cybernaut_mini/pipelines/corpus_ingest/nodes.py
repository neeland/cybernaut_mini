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

from cybernaut_mini.corpus import CorpusSourceConfig, normalize_rows
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
