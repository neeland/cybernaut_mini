"""Document ingestion from JSONL files with strict validation.

Validates each record through :class:`~cybernaut_mini.models.Document`; raises
:class:`IngestError` on any structural problem so callers can surface actionable
messages without catching generic exceptions.
"""

from __future__ import annotations

import json
from pathlib import Path

from cybernaut_mini.models import Document


class IngestError(ValueError):
    """Raised for malformed or duplicate input records during ingestion."""


def load_documents(path: Path) -> list[Document]:
    """Read a JSONL file and return validated :class:`Document` objects.

    Skips blank lines. Raises :class:`IngestError` naming the offending record
    (line number and id when known) on: invalid JSON, missing/empty required
    fields, or duplicate ids.
    """
    documents: list[Document] = []
    seen_ids: set[str] = set()

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

            # Extract id for error messages before full validation.
            doc_id: str | None = record.get("id") if isinstance(record, dict) else None

            try:
                doc = Document.model_validate(record)
            except Exception as exc:  # pydantic ValidationError
                msg = (
                    f"line {lineno} (id={doc_id!r}): {exc}"
                    if doc_id
                    else f"line {lineno}: {exc}"
                )
                raise IngestError(msg) from exc

            if doc.id in seen_ids:
                msg = f"line {lineno}: duplicate id {doc.id!r}"
                raise IngestError(msg)

            seen_ids.add(doc.id)
            documents.append(doc)

    return documents
