"""Simulate streaming days of ingest and print the nDCG-vs-corpus-size curve.

Replays a real JSONL corpus (default: the committed MIRACL + CC-News fixture)
through :mod:`cybernaut_mini.entities.ingest_stream`: bootstrap-cluster a base
slice, then append the rest in simulated days routed to the nearest existing
centroid, streaming every day's documents through the entity loop's sampled
discovery and importance-weighted flushes, and re-running the evaluation harness
after each increment. Writes the growth curve as canonical JSON and prints a table.

Run (fully offline — hash embedder, fixture corpus):
    uv run python scripts/simulate_ingest.py
    uv run python scripts/simulate_ingest.py --base-size 200 --docs-per-day 100 \
        --days 2 --n-shards 8 --out data/08_reporting/ingest_curve.json

The gap analysis scales the post's ~20M pages/day to 5-10k passages per simulated
day; against the 460-document fixture the defaults use proportionally smaller days
(same code path, smaller constant). Pass --docs-per-day 5000 with a larger corpus
for the full-scale laptop run.

Blog ref: https://nosible.com/blog/introducing-cybernaut-1-agentic-search-with-mcts
    — quality stability "even as we continue expanding our web coverage (currently
    growing at ~20 million webpages per day)". Local copy:
    ``docs/blog-archive/introducing-cybernaut-1-agentic-search-with-mcts.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the package importable when run from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cybernaut_mini.config import AppConfig
from cybernaut_mini.entities.flush import FlushBuffer
from cybernaut_mini.entities.ingest_stream import EntityStreamState, dump_curve, simulate_stream
from cybernaut_mini.entities.store import EntityStore
from cybernaut_mini.entities.tagger import CapitalizedSpanDiscovery
from cybernaut_mini.ingest import load_documents
from cybernaut_mini.models import Judgment
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.text import TextProcessor


def _load_judgments(path: Path) -> list[Judgment]:
    judgments: list[Judgment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            judgments.append(Judgment.model_validate(json.loads(line)))
    return judgments


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate streaming ingest days and print the nDCG-vs-corpus-size curve."
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("data/01_raw/fixtures/documents.jsonl"),
        help="JSONL corpus to replay (default: the committed real fixture slice)",
    )
    parser.add_argument(
        "--judgments",
        type=Path,
        default=Path("data/01_raw/fixtures/judgments.jsonl"),
        help="JSONL graded judgments for the fixed evaluation set",
    )
    parser.add_argument("--base-size", type=int, default=200, help="day-0 bootstrap corpus size")
    parser.add_argument("--docs-per-day", type=int, default=100, help="documents per simulated day")
    parser.add_argument("--days", type=int, default=None, help="max days (default: drain corpus)")
    parser.add_argument("--n-shards", type=int, default=8, help="shard count for the bootstrap")
    parser.add_argument("--seed", type=int, default=42, help="clustering / sampling seed")
    parser.add_argument("--dim", type=int, default=64, help="hash embedder dimensions")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["hybrid"],
        choices=["lexical", "dense", "hybrid", "agent"],
        help="evaluation modes per increment",
    )
    parser.add_argument(
        "--sample-rate", type=float, default=0.075, help="entity-loop discovery sampling X"
    )
    parser.add_argument(
        "--no-entity-loop",
        action="store_true",
        help="skip the entity loop (pure index-growth curve)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/08_reporting/ingest_curve.json"),
        help="where to write the canonical-JSON growth curve",
    )
    args = parser.parse_args()

    documents = load_documents(args.documents)
    judgments = _load_judgments(args.judgments)
    entity_state = None
    if not args.no_entity_loop:
        entity_state = EntityStreamState(
            store=EntityStore(),
            buffer=FlushBuffer(),
            discovery=CapitalizedSpanDiscovery(),
            sample_rate=args.sample_rate,
        )

    reports = simulate_stream(
        documents,
        judgments,
        provider=HashEmbedder(dim=args.dim),
        processor=TextProcessor(use_spacy=False),
        config=AppConfig(seed=args.seed),
        base_size=args.base_size,
        docs_per_day=args.docs_per_day,
        n_shards=args.n_shards,
        seed=args.seed,
        days=args.days,
        modes=tuple(args.modes),
        entity_state=entity_state,
    )

    header = f"{'day':>4} {'corpus':>7} {'added':>6} {'sampled':>8}"
    header += "".join(f" {'ndcg@10/' + mode:>16}" for mode in args.modes)
    print(header)
    for report in reports:
        row = (
            f"{report.day:>4} {report.corpus_size:>7} {report.added:>6} "
            f"{report.sampled_chunks:>8}"
        )
        row += "".join(f" {report.ndcg_at_10(mode):>16.4f}" for mode in args.modes)
        print(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump_curve(reports) + "\n", encoding="utf-8")
    print(f"curve written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
