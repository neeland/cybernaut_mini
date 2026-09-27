"""Pure-function Kedro nodes for the evaluation pipeline.

The index arrives already loaded from ``ShardIndexDataset`` and the judgments from
a ``JsonlDataset``, so no node opens a path. Derived objects that are expensive but
cheap to rebuild (``TextProcessor``, ``EmbeddingProvider``) are constructed inside
the node rather than passed between them, so nothing large crosses a dataset boundary.

Node sequence
-------------
validate_judgments -> judgments_list (list of Judgment dicts)
evaluate_node      -> metrics_list (list of ModeMetrics dicts)
report_node        -> eval_report (structured summary, persisted by the catalog)

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — the eight-stage
    pipeline's own quality is measured against human judgments. The post mentions
    per-shard "Evals"; the build guide's step 10 reads that as whole-corpus
    nDCG/recall@k against gold judgments [inferred]. Local copy:
    ``data/00_reference/the-road-to-cybernaut-1.md``. The real-data rule that
    makes these numbers meaningful is in ``data/README.md``.

Assumptions:
    - Judgments are real MIRACL qrels, never synthetic. ``validate_judgments``
      rejects an empty file and names the offending record on a schema failure,
      because a truncated or fabricated qrels file would otherwise produce a
      plausible-looking metric instead of an error.
    - Nodes are pure and open no paths: the index arrives already loaded from
      ``ShardIndexDataset`` and the judgments from a ``JsonlDataset``.
      ``TextProcessor`` and the embedding provider are constructed *inside*
      ``evaluate_node`` rather than passed as datasets — they are cheap to rebuild
      and Kedro's in-memory layer deepcopies whatever it is handed.
    - The provider is built with ``provider_from_meta(index.meta, ...)`` so the
      evaluation embedder is the one the index was built with. A config mismatch
      would otherwise surface as a mysterious dense-mode collapse.
    - ``shard_recall_at_n`` means different things per mode and is never averaged
      across them: for lexical/dense/hybrid it measures router quality from one
      ``route()`` call shared by all three, while agent mode measures the shards
      the search actually explored. ``report_node`` keeps them in a per-mode map
      for exactly that reason.
    - ``modes`` defaults to all four (lexical, dense, hybrid, agent). The node
      signature exposes the subset knob so a direct caller or a test can narrow
      it, but the pipeline does not pass it.

Alternatives considered:
    - Folding evaluation into ``index_build``: fewer pipelines, but a build is
      reproducible from a corpus and a seed while an evaluation is reproducible
      only against a pinned qrels file. A shared DAG would make it impossible to
      re-evaluate an existing index without rebuilding it.
    - Passing the ``LoadedIndex`` between nodes as a dataset: works, but Kedro's
      ``MemoryDataset`` deepcopy was measured at 21.6M calls and ~32s on a
      16k-document build. Loading once inside ``evaluate_node`` is cheaper and
      keeps exactly one node responsible for the index.
    - Computing the metrics inside ``report_node``: would remove a dataset from
      the catalog, but the metrics list is the artifact worth persisting. The
      report is a view of it, so it can be regenerated without re-running
      retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cybernaut_mini.indexing import LoadedIndex


def validate_judgments(judgments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate catalog-supplied judgment records.

    Raises :exc:`ValueError` naming the offending record on any validation error.
    """
    from pydantic import ValidationError

    from cybernaut_mini.models import Judgment

    if not judgments:
        msg = "judgments file contained no records; cannot evaluate"
        raise ValueError(msg)

    records: list[dict[str, Any]] = []
    for position, raw in enumerate(judgments, start=1):
        try:
            j = Judgment.model_validate(raw)
        except (ValidationError, ValueError) as exc:
            qid = raw.get("query_id") if isinstance(raw, dict) else None
            label = f"query_id={qid!r}" if qid else f"record {position}"
            msg = f"judgments {label}: {exc}"
            raise ValueError(msg) from exc
        records.append(j.model_dump(mode="json"))
    return records


def evaluate_node(
    index: LoadedIndex,
    judgments_list: list[dict[str, Any]],
    embedding_params: dict[str, Any],
    rrf_params: dict[str, Any],
    agent_params: dict[str, Any],
    seed: int,
    offline: bool,
    shard_beam_n: int = 100,
    modes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run all four retrieval modes over judgments and return metrics dicts.

    ``shard_beam_n`` controls how many shards the router may select per query
    when computing ``shard_recall@N`` for static modes (lexical/dense/hybrid).
    Defaults to 100, matching ``DEFAULT_CANDIDATE_DEPTH`` in
    ``query.s5_select`` — the width the system actually runs with.
    """
    from cybernaut_mini.config import AgentConfig, AppConfig, EmbeddingConfig, RRFConfig
    from cybernaut_mini.evals import evaluate
    from cybernaut_mini.models import Judgment
    from cybernaut_mini.retrieval import provider_from_meta
    from cybernaut_mini.text import TextProcessor

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
        shard_beam_n=shard_beam_n,
    )
    return [m.as_dict() for m in metrics]


def report_node(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise metrics into a human-readable report dict.

    ``shard_recall_at_n`` is reported per mode. Static modes (lexical/dense/
    hybrid) measure *router quality*; agent mode measures *exploration coverage*.
    Do not average or compare the two — see :func:`cybernaut_mini.evals.evaluate`.
    """
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
        "shard_recall_by_mode": {
            m["mode"]: m["shard_recall_at_n"] for m in metrics_list
        },
    }
