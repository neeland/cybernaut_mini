"""Dump the MIRACL en-dev qrel docid set for stream-time corpus filtering.

Reads the pinned ``miracl/miracl`` en-dev topics + qrels via
:class:`cybernaut_mini.datasets.MiraclTsvDataset` and writes every referenced
raw docid (positive and negative, one per line, sorted) to:

    data/01_raw/miracl_en_dev_docids.txt

``conf/prod/catalog.yml`` points ``raw_miracl_source.filter_values_file`` at this
file so the MIRACL corpus snapshot keeps only the ~7.9k qrel-referenced passages
instead of streaming all 32.9M English passages into memory.

Run:
    uv run python scripts/dump_miracl_docids.py

The script is idempotent: the docid set is fully determined by the pinned
revision, so re-running overwrites the file with byte-identical output.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when run from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cybernaut_mini.datasets import MiraclTsvDataset

# Same SHA as conf/prod/catalog.yml `miracl_en_dev_source.revision`.
_MIRACL_TOPICS_SHA = "5be20db9509754dadad47689368639fcec739c00"

_OUT_PATH = Path("data/01_raw/miracl_en_dev_docids.txt")


def main() -> None:
    rows = MiraclTsvDataset(
        repo_id="miracl/miracl",
        revision=_MIRACL_TOPICS_SHA,
        language="en",
        split="dev",
    ).load()

    docids: set[str] = set()
    for row in rows:
        for passage in (*row["positive_passages"], *row["negative_passages"]):
            docids.add(str(passage["docid"]))

    if not docids:
        msg = "qrels produced zero docids; refusing to write an empty filter file"
        raise SystemExit(msg)

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text("\n".join(sorted(docids)) + "\n", encoding="utf-8")
    print(f"wrote {len(docids):,} docids from {len(rows):,} queries to {_OUT_PATH}")


if __name__ == "__main__":
    main()
