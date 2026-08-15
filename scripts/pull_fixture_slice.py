"""Pull a small real fixture slice for cybernaut-mini offline tests.

Downloads:
  - First 25 MIRACL en-dev queries and every passage they reference (positive +
    negative) from the MIRACL corpus — the passages whose docids appear in qrels.
  - First 200 English CC-News articles via streaming.

Writes pre-normalised Document objects:
  data/01_raw/fixtures/documents.jsonl  — MIRACL passages + CC-News articles
  data/01_raw/fixtures/judgments.jsonl  — 25 graded MIRACL judgments

Pinned SHAs (immutable; the same SHAs live in conf/prod/catalog.yml):
  CC-News (stanford-oval/ccnews):         d733e654c9a506df519e1a166a86c118c7657ce4
  MIRACL corpus (miracl/miracl-corpus):   d921ec7e349ce0d28daf30b2da9da5ee698bef0d
  MIRACL topics+qrels (miracl/miracl):    5be20db9509754dadad47689368639fcec739c00

Run:
    uv run python scripts/pull_fixture_slice.py

The script is idempotent: re-running it overwrites the fixture files with
byte-identical output (deterministic normalisation + same rows in same order).

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — the fixture is the
    smallest slice that exercises real-data retrieval: MIRACL as the evaluation
    substrate (qrels-positive passages must be recoverable) and CC-News as the
    broad-coverage corpus spine.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

# Make the package importable when run from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cybernaut_mini.corpus import CorpusSourceConfig, make_document_id, normalize_rows
from cybernaut_mini.models import Judgment, canonical_dumps

# ── Pinned SHAs ───────────────────────────────────────────────────────────────
_CCNEWS_SHA = "d733e654c9a506df519e1a166a86c118c7657ce4"
_MIRACL_CORPUS_SHA = "d921ec7e349ce0d28daf30b2da9da5ee698bef0d"
_MIRACL_TOPICS_SHA = "5be20db9509754dadad47689368639fcec739c00"

# ── Slice sizes ───────────────────────────────────────────────────────────────
#: Number of MIRACL en-dev queries to pull (first N from the topics TSV).
_N_MIRACL_TOPICS = 25
#: Number of English CC-News articles to include.
_N_CCNEWS = 200

# ── Corpus source configs (mirror conf/base/parameters.yml) ───────────────────
_MIRACL_CONFIG = CorpusSourceConfig(
    field_map={"text": "text", "title": "title", "docid": "url"},
    metadata_fields=["docid"],
    id_prefix="mir",
    default_language="en",
    min_text_chars=32,
)

_CCNEWS_CONFIG = CorpusSourceConfig(
    field_map={
        "plain_text": "text",
        "requested_url": "url",
        "title": "title",
        "language": "language",
        "published_date": "published_at",
    },
    metadata_fields=["publisher", "sitename", "author"],
    id_prefix="ccn",
    default_language="en",
    min_text_chars=32,
)


# ── Download helpers ──────────────────────────────────────────────────────────


def _read_miracl_topics_and_qrels(
    fs: object,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[str]], set[str]]:
    """Return (topics, positives, negatives, all_docids) for the first N topics.

    Reads the TSV files directly via HfFileSystem (load_dataset is blocked by the
    deprecated miracl.py loading script).
    """
    rev = _MIRACL_TOPICS_SHA
    base = f"datasets/miracl/miracl@{rev}/miracl-v1.0-en"

    topics: dict[str, str] = {}
    with fs.open(  # type: ignore[union-attr]
        f"{base}/topics/topics.miracl-v1.0-en-dev.tsv", "r", encoding="utf-8"
    ) as fh:
        for line in fh:
            if len(topics) >= _N_MIRACL_TOPICS:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                topics[parts[0]] = parts[1]

    target_qids = set(topics.keys())
    positives: dict[str, list[str]] = {}
    negatives: dict[str, list[str]] = {}
    all_docids: set[str] = set()

    with fs.open(  # type: ignore[union-attr]
        f"{base}/qrels/qrels.miracl-v1.0-en-dev.tsv", "r", encoding="utf-8"
    ) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            qid, docid, grade = parts[0], parts[2], parts[3].strip()
            if qid not in target_qids:
                continue
            all_docids.add(docid)
            if grade == "1":
                positives.setdefault(qid, []).append(docid)
            else:
                negatives.setdefault(qid, []).append(docid)

    return topics, positives, negatives, all_docids


def _fetch_miracl_corpus_passages(
    fs: object, target_docids: set[str]
) -> list[dict[str, object]]:
    """Stream MIRACL corpus shards and collect rows whose docid is in target_docids.

    Stops early once all docids have been found.  Prints progress so a human
    watching the terminal can see it's not hung.
    """
    rev = _MIRACL_CORPUS_SHA
    pattern = (
        f"datasets/miracl/miracl-corpus@{rev}/miracl-corpus-v1.0-en/*.jsonl.gz"
    )
    shard_paths: list[str] = sorted(fs.glob(pattern))  # type: ignore[union-attr]
    print(
        f"  Scanning {len(shard_paths)} corpus shards for {len(target_docids)} docids…"
    )

    remaining = set(target_docids)
    rows: list[dict[str, object]] = []

    for i, shard_path in enumerate(shard_paths):
        if not remaining:
            print(f"  All docids found after shard {i} — stopping early.")
            break
        found_in_shard = 0
        try:
            with fs.open(shard_path, "rb") as raw_fh:  # type: ignore[union-attr]
                with gzip.open(raw_fh, "rt", encoding="utf-8") as gz:
                    for line in gz:
                        line = line.strip()
                        if not line:
                            continue
                        row: dict[str, object] = json.loads(line)
                        docid = str(row.get("docid", ""))
                        if docid in remaining:
                            rows.append(row)
                            remaining.discard(docid)
                            found_in_shard += 1
        except Exception as exc:
            print(f"  Shard {i:02d}: SKIPPED after repeated network error ({exc!r})")
            continue
        if found_in_shard:
            print(
                f"  Shard {i:02d}: found {found_in_shard}"
                f" (remaining: {len(remaining)})"
            )

    if remaining:
        print(
            f"  WARNING: {len(remaining)} docids not found in any shard:"
            f" {sorted(remaining)[:5]}…"
        )
    return rows


def _fetch_ccnews(n: int) -> list[dict[str, object]]:
    """Stream first n English rows from CC-News via the streaming loader.

    CC-News has a deprecated loading script too, but streaming=True still works
    (it uses the parquet shards directly rather than the Python loading script).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        msg = "Pull script needs `uv sync --extra hf`"
        raise SystemExit(msg) from exc

    ds = load_dataset(
        "stanford-oval/ccnews",
        revision=_CCNEWS_SHA,
        split="train",
        streaming=True,
    )
    rows: list[dict[str, object]] = []
    for row in ds:
        if row.get("language") != "en":
            continue
        rows.append(dict(row))
        if len(rows) >= n:
            break
        if len(rows) % 50 == 0:
            print(f"  CC-News: {len(rows)}/{n}…")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as exc:
        msg = "Pull script needs `uv sync --extra hf`"
        raise SystemExit(msg) from exc

    out_dir = Path(__file__).parent.parent / "data" / "01_raw" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = HfFileSystem()

    # 1. MIRACL topics + qrels
    print(f"[1/4] Reading first {_N_MIRACL_TOPICS} MIRACL en-dev topics + qrels…")
    topics, positives, negatives, all_docids = _read_miracl_topics_and_qrels(fs)
    print(
        f"      {len(topics)} topics, {len(all_docids)} unique docids"
        f" ({sum(len(v) for v in positives.values())} positives,"
        f" {sum(len(v) for v in negatives.values())} negatives)"
    )

    # 2. MIRACL corpus passages
    print("[2/4] Fetching MIRACL corpus passages…")
    miracl_rows = _fetch_miracl_corpus_passages(fs, all_docids)
    print(f"      Retrieved {len(miracl_rows)} corpus rows")

    # 3. CC-News docs
    print(f"[3/4] Fetching {_N_CCNEWS} CC-News en docs…")
    ccnews_rows = _fetch_ccnews(_N_CCNEWS)
    print(f"      Retrieved {len(ccnews_rows)} CC-News rows")

    # 4. Normalise + merge
    print("[4/4] Normalising and writing fixtures…")
    miracl_docs = normalize_rows(miracl_rows, _MIRACL_CONFIG)
    ccnews_docs = normalize_rows(ccnews_rows, _CCNEWS_CONFIG)

    # Stable dedup by id: sort so the file is byte-identical on reruns.
    seen_ids: set[str] = set()
    all_docs = []
    for doc in sorted(miracl_docs + ccnews_docs, key=lambda d: d.id):
        if doc.id not in seen_ids:
            seen_ids.add(doc.id)
            all_docs.append(doc)

    corpus_ids = {doc.id for doc in all_docs}

    # Build judgments, keeping only docids present in the fixture corpus.
    # (A docid might be absent if normalize_rows dropped a too-short passage.)
    judgments: list[Judgment] = []
    for qid, question in topics.items():
        rel: dict[str, int] = {}
        for docid in positives.get(qid, []):
            did = make_document_id("mir", docid)
            if did in corpus_ids:
                rel[did] = 1
        for docid in negatives.get(qid, []):
            did = make_document_id("mir", docid)
            if did in corpus_ids:
                rel[did] = 0
        if rel:
            judgments.append(
                Judgment(query_id=qid, question=question, relevant_document_ids=rel)
            )

    # Write documents
    docs_path = out_dir / "documents.jsonl"
    docs_path.write_text(
        "".join(
            canonical_dumps(doc.model_dump(mode="json")) + "\n" for doc in all_docs
        ),
        encoding="utf-8",
    )

    # Write judgments
    judg_path = out_dir / "judgments.jsonl"
    judg_path.write_text(
        "".join(
            canonical_dumps(j.model_dump(mode="json")) + "\n" for j in judgments
        ),
        encoding="utf-8",
    )

    # Report
    missing_count = sum(
        1
        for j in judgments
        for did in j.relevant_document_ids
        if did not in corpus_ids
    )
    print()
    print("── Fixture written ─────────────────────────────────────────────────")
    print(f"  {docs_path}: {len(all_docs)} documents")
    print(f"    MIRACL passages : {len(miracl_docs)}")
    print(f"    CC-News articles: {len(ccnews_docs)}")
    print(f"  {judg_path}: {len(judgments)} judgments")
    if missing_count:
        print(
            f"  WARNING: {missing_count} judgment docids absent from fixture corpus"
            " (passages too short, dropped by normaliser)"
        )
    else:
        print("  All judgment docids present in fixture corpus ✓")
    print()
    print("  SHA provenance:")
    print(f"    CC-News:             {_CCNEWS_SHA}")
    print(f"    MIRACL corpus:       {_MIRACL_CORPUS_SHA}")
    print(f"    MIRACL topics+qrels: {_MIRACL_TOPICS_SHA}")


if __name__ == "__main__":
    main()
