"""Document ingestion with strict validation.

Validates each record through :class:`~cybernaut_mini.models.Document`; raises
:class:`IngestError` on any structural problem so callers can surface actionable
messages without catching generic exceptions.

Parsing and validation are separate so both entry points share one rule set: the
Kedro catalog hands nodes already-parsed records, while :func:`load_documents`
still reads a JSONL file directly for the CLI and for tests.
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
