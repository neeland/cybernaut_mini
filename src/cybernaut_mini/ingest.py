"""Document ingestion with strict validation.

Validates each record through :class:`~cybernaut_mini.models.Document`; raises
:class:`IngestError` on any structural problem so callers can surface actionable
messages without catching generic exceptions.

Parsing and validation are separate so both entry points share one rule set: the
Kedro catalog hands nodes already-parsed records, while :func:`load_documents`
still reads a JSONL file directly for the CLI and for tests.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — ingestion is upstream of
    stage 1: the post's pipeline starts from already-parsed documents, so this module
    is the strict door between raw corpus rows and the ``Document`` objects every
    later stage assumes. Local copy:
    ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - Validation is fail-fast and whole-file: one bad row, one duplicate id, or one
      missing required field raises ``IngestError`` and no partial document list is
      returned. Silent skipping is the failure mode the real-data rule forbids.
    - Parsing and validation are separate functions so both callers share one rule
      set: ``load_documents`` reads JSONL for the CLI and tests, while the Kedro
      catalog hands already-parsed dicts to ``validate_records``.
    - Blank lines are skipped silently; every other line is expected to be a JSON
      object.
    - Error messages carry the position and, when it can be read, the record ``id``,
      so a 460-document fixture failure points at a locatable row.
    - Duplicate detection is on the exact ``Document.id`` string, not a normalised
      form, and happens after field validation.

Alternatives considered:
    - ``pandas.read_json`` or ``datasets.load_dataset``: rejected because neither
      reports the offending line number, which is the only thing that makes a broken
      corpus fixable.
    - Letting pydantic's ``ValidationError`` escape: rejected in favour of
      ``IngestError`` with the line and id attached; callers catch one domain error
      instead of ``Exception``.
    - Lazy streaming ingestion that yields documents one at a time: rejected for now
      because every consumer embeds the full corpus anyway and the duplicate check
      needs the whole id set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cybernaut_mini.models import Document


class IngestError(ValueError):
    """Raised for malformed or duplicate input records during ingestion."""


def validate_records(
    numbered_records: list[tuple[int, Any]],
    *,
    label: str = "line",
) -> list[Document]:
    """Validate ``(position, record)`` pairs into :class:`Document` objects.

    ``label`` names the position unit in error messages ("line" for a file,
    "record" for catalog-supplied data) so the message always points at something
    the caller can actually locate.

    Raises :class:`IngestError` on invalid fields or duplicate ids.
    """
    documents: list[Document] = []
    seen_ids: set[str] = set()

    for position, record in numbered_records:
        # Extract id for error messages before full validation.
        doc_id: Any = record.get("id") if isinstance(record, dict) else None

        try:
            doc = Document.model_validate(record)
        except Exception as exc:  # pydantic ValidationError
            msg = (
                f"{label} {position} (id={doc_id!r}): {exc}"
                if doc_id
                else f"{label} {position}: {exc}"
            )
            raise IngestError(msg) from exc

        if doc.id in seen_ids:
            msg = f"{label} {position}: duplicate id {doc.id!r}"
            raise IngestError(msg)

        seen_ids.add(doc.id)
        documents.append(doc)

    return documents


def load_documents(path: Path) -> list[Document]:
    """Read a JSONL file and return validated :class:`Document` objects.

    Skips blank lines. Raises :class:`IngestError` naming the offending record
    (line number and id when known) on: invalid JSON, missing/empty required
    fields, or duplicate ids.
    """
    numbered_records: list[tuple[int, Any]] = []

    with path.open(encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"line {lineno}: invalid JSON — {exc}"
                raise IngestError(msg) from exc

            numbered_records.append((lineno, record))

    return validate_records(numbered_records, label="line")
