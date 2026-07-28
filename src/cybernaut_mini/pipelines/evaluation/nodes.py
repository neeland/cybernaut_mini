"""Pure-function Kedro nodes for the evaluation pipeline.

Each function accepts and returns plain Python objects so Kedro can pass
them through its in-memory dataset layer without custom pickling.
Heavy objects (LoadedIndex, TextProcessor, EmbeddingProvider) are constructed
inside the node so they are never serialised.

Node sequence
-------------
load_index_node   -> loaded_index_path (passes the path string through)
load_judgments_node -> judgments_list (list of Judgment dicts)
evaluate_node     -> metrics_list (list of ModeMetrics dicts)
report_node       -> report_dict (structured summary)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_index_node(index_path: str) -> str:
    """Validate that the index at ``index_path`` is loadable and return the path.

    Raises :class:`~cybernaut_mini.indexing.IndexLoadError` if the index is
    incomplete (missing ``_VALID`` marker or required artifact files).
    """
    from cybernaut_mini.indexing import LoadedIndex

    # Eagerly validate by loading — this also warms up the BM25 index.
    # We return the path string so Kedro doesn't need to serialise LoadedIndex.
    LoadedIndex.load(Path(index_path))
    return index_path


def load_judgments_node(judgments_path: str) -> list[dict[str, Any]]:
    """Read a JSONL file of :class:`~cybernaut_mini.models.Judgment` records.

    Returns a list of validated Judgment model dicts.
    Raises :exc:`ValueError` naming the offending line on any validation error.
    """
    from pydantic import ValidationError

    from cybernaut_mini.models import Judgment

    path = Path(judgments_path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"judgments line {lineno}: invalid JSON — {exc}"
                raise ValueError(msg) from exc
            try:
                j = Judgment.model_validate(raw)
            except (ValidationError, ValueError) as exc:
                qid = raw.get("query_id") if isinstance(raw, dict) else None
                label = f"query_id={qid!r}" if qid else f"line {lineno}"
                msg = f"judgments {label}: {exc}"
                raise ValueError(msg) from exc
            records.append(j.model_dump(mode="json"))
    return records


def evaluate_node(
    index_path: str,
    judgments_list: list[dict[str, Any]],
    embedding_params: dict[str, Any],
    rrf_params: dict[str, Any],
    agent_params: dict[str, Any],
    seed: int,
    offline: bool,
    modes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run all four retrieval modes over judgments and return metrics dicts.

    Constructs ``LoadedIndex``, ``TextProcessor``, and ``EmbeddingProvider``
    inside the node. Uses ``evals.evaluate`` for metric computation.
    """
    from cybernaut_mini.config import AgentConfig, AppConfig, EmbeddingConfig, RRFConfig
    from cybernaut_mini.evals import evaluate
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.models import Judgment
    from cybernaut_mini.retrieval import provider_from_meta
    from cybernaut_mini.text import TextProcessor

    index = LoadedIndex.load(Path(index_path))
    judgments = [Judgment.model_validate(j) for j in judgments_list]

    embedding_config = EmbeddingConfig.model_validate(embedding_params)
    rrf_config = RRFConfig.model_validate(rrf_params)
    agent_config = AgentConfig.model_validate(agent_params)
    app_config = AppConfig(
        seed=seed,
        embedding=embedding_config,
        rrf=rrf_config,
        agent=agent_config,
    )

    # Use provider_from_meta so the embedder matches the index's embedding model.
    provider = provider_from_meta(index.meta, offline=offline)
    processor = TextProcessor(use_spacy=None)

    eval_modes: tuple[str, ...] = tuple(modes) if modes else ("lexical", "dense", "hybrid", "agent")
    metrics = evaluate(
        index,
        judgments,
        config=app_config,
        processor=processor,
        provider=provider,
        modes=eval_modes,
    )
    return [m.as_dict() for m in metrics]


def report_node(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise metrics into a human-readable report dict."""
    return {
        "metrics": metrics_list,
        "n_modes": len(metrics_list),
        "best_mode_ndcg": max(
            (m["mode"] for m in metrics_list),
            key=lambda mode: next(
                (m["ndcg_at_10"] for m in metrics_list if m["mode"] == mode), 0.0
            ),
        )
        if metrics_list
        else None,
    }
