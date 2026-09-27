"""Typer CLI. Commands are added milestone by milestone; M1 ships the app shell.

Two entry points share this module:

``app``
    The Typer application behind the ``cybernaut-mini`` console script — the
    task-oriented interface (build, search, eval).
``cli``
    An empty Click group that the ``kedro`` executable discovers here. Kedro
    requires a Click group at ``<package>.cli:cli`` and merges its own commands
    into it; without one, ``kedro run`` / ``catalog`` / ``registry`` / ``viz``
    all abort with "Cannot load commands". Keeping it empty means Kedro's
    commands stay exactly as Kedro defines them.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — the CLI is the whole
    eight-stage pipeline behind four verbs: ``build`` (stages 1-6), ``search``
    (stages 5-8, plus the Cybernaut-1 agent), ``inspect-shards`` (stage 5's output)
    and ``eval``. Local copy: ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - ``build`` shells into a real ``KedroSession`` and runs ``index_build``, so a CLI
      build and ``kedro run --pipeline index_build`` write identical artifacts. It is
      not a private re-implementation of the pipeline nodes.
    - ``search`` and ``eval`` do not open a Kedro session: agent search branches at
      runtime with a shared budget counter, which a static DAG cannot express, so
      they call ``run_agent_search`` / ``evaluate`` directly.
    - ``--json`` writes only to stdout and progress logging goes to stderr, so
      ``cybernaut-mini search ... --json | jq`` stays parseable.
    - ``--mode`` is validated before the index is loaded, so a typo fails in
      milliseconds rather than after a multi-gigabyte ``embeddings.npy`` mmap.
    - ``--offline`` is enforced by ``AppConfig.require_offline_compatible`` for
      ``build``, ``search --mode agent`` and ``eval``; plain static search has no
      config to check because it has no model-backed component.

Alternatives considered:
    - Calling the index-build nodes directly from ``build``: fewer moving parts, but
      it makes the CLI a second build entry point that can drift from the Kedro
      pipeline and disappears from ``kedro viz`` lineage. Rejected.
    - Routing ``eval`` through the Kedro ``evaluation`` pipeline: rejected for the CLI
      because the extra session costs more than it adds for a single call; the
      pipeline stays registered for ``kedro run --pipeline evaluation``.
    - A separate module for the Kedro Click group: rejected because ``kedro``
      discovers ``cybernaut_mini.cli:cli`` and a second file would only split one
      interface across two.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import click
import typer

from cybernaut_mini import __version__

app = typer.Typer(name="cybernaut-mini", no_args_is_help=True, add_completion=False)


@click.group(name="cybernaut-mini")
def cli() -> None:
    """Project command group discovered by the ``kedro`` executable."""


@app.callback()
def main() -> None:
    """cybernaut-mini: sharded hybrid retrieval with a three-stage search agent."""
    # Progress logging for long-running commands (eval, search). Stderr, never
    # stdout: --json output must stay pipeable. No-op when a handler already
    # exists (e.g. Kedro configures its own inside `build`).
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def build(
    input: Annotated[Path, typer.Option("--input", help="Path to input JSONL corpus")],
    index: Annotated[Path, typer.Option("--index", help="Path to write the index")],
    config: Annotated[
        Path | None, typer.Option("--config", help="Optional YAML config file")
    ] = None,
    offline: Annotated[bool, typer.Option("--offline", help="Offline mode")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON summary")] = False,
) -> None:
    """Build a sharded hybrid index from a JSONL corpus."""
    from cybernaut_mini.config import ConfigError, load_config
    from cybernaut_mini.ingest import IngestError

    try:
        app_config = load_config(config, overrides={})
        if offline:
            app_config.require_offline_compatible()
    except (ConfigError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    runtime_params: dict[str, object] = {
        "input_path": str(input),
        "index_path": str(index),
        "seed": app_config.seed,
        "embedding": app_config.embedding.model_dump(mode="json"),
        "index": app_config.index.model_dump(mode="json"),
        "rrf": app_config.rrf.model_dump(mode="json"),
        "agent": app_config.agent.model_dump(mode="json"),
        "offline": offline,
    }

    try:
        from kedro.framework.session import KedroSession
        from kedro.framework.startup import bootstrap_project

        bootstrap_project(Path.cwd())
        with KedroSession.create(
            project_path=Path.cwd(),
            runtime_params=runtime_params,
        ) as session:
            kedro_session: KedroSession = session  # type: ignore[assignment]
            kedro_session.run(pipeline_names=["index_build"])

    except Exception as exc:
        # Unwrap Kedro pipeline exceptions to surface the original message.
        original = _unwrap_exception(exc)
        if isinstance(original, (IngestError, ConfigError, ValueError)):
            typer.echo(str(original), err=True)
            raise typer.Exit(1) from exc
        # Re-raise unexpected errors.
        raise

    # Read index meta for the summary.
    try:
        meta_path = index / "index_meta.json"
        meta_data: dict[str, object] = json.loads(meta_path.read_text(encoding="utf-8"))
        n_documents = meta_data.get("n_documents", 0)
        n_shards = meta_data.get("n_shards", 0)
    except Exception:
        n_documents = 0
        n_shards = 0

    if as_json:
        typer.echo(
            json.dumps(
                {"index": str(index), "n_documents": n_documents, "n_shards": n_shards},
                sort_keys=True,
            )
        )
    else:
        typer.echo(
            f"Index built at {index}: {n_documents} documents in {n_shards} shards."
        )


@app.command(name="inspect-shards")
def inspect_shards(
    index: Annotated[Path, typer.Option("--index", help="Path to index directory")],
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
) -> None:
    """Inspect shard manifests in a built index."""
    from cybernaut_mini.indexing import IndexLoadError, LoadedIndex

    try:
        loaded = LoadedIndex.load(index)
    except (IndexLoadError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    if as_json:
        shards_data = [
            manifest.model_dump(mode="json")
            for manifest in sorted(loaded.manifests.values(), key=lambda m: m.shard_id)
        ]
        typer.echo(json.dumps({"shards": shards_data}, sort_keys=True))
    else:
        for shard_id, manifest in sorted(loaded.manifests.items()):
            top5_keywords = ", ".join(kw.term for kw in manifest.keywords[:5])
            typer.echo(
                f"shard {shard_id:03d}: {manifest.document_count} docs | "
                f"{manifest.title!r} | keywords: [{top5_keywords}]"
            )


@app.command()
def search(
    index: Annotated[Path, typer.Option("--index", help="Path to index directory")],
    question: Annotated[str, typer.Option("--question", help="Search question")],
    mode: Annotated[
        str,
        typer.Option("--mode", help="Retrieval mode: lexical, dense, or hybrid"),
    ] = "hybrid",
    top_k: Annotated[int, typer.Option("--top-k", help="Number of results to return")] = 10,
    filter_json: Annotated[
        str | None, typer.Option("--filter", help="Metadata filter as JSON object")
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Optional YAML config (agent mode)")
    ] = None,
    trace_out: Annotated[
        Path | None, typer.Option("--trace-out", help="Write the agent trace to this JSON file")
    ] = None,
    offline: Annotated[bool, typer.Option("--offline", help="Offline mode")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
) -> None:
    """Search a built index with lexical, dense, hybrid, or agent retrieval."""
    from cybernaut_mini.config import ConfigError, RRFConfig, load_config
    from cybernaut_mini.indexing import IndexLoadError, LoadedIndex
    from cybernaut_mini.models import MetadataFilter
    from cybernaut_mini.retrieval import provider_from_meta, retrieve
    from cybernaut_mini.text import TextProcessor

    valid_modes = ("lexical", "dense", "hybrid", "agent")
    if mode not in valid_modes:
        typer.echo(f"Invalid mode {mode!r}; must be one of: {', '.join(valid_modes)}", err=True)
        raise typer.Exit(1)

    metadata_filter: MetadataFilter | None = None
    if filter_json is not None:
        try:
            raw_filter = json.loads(filter_json)
        except json.JSONDecodeError as exc:
            typer.echo(f"--filter is not valid JSON: {exc}", err=True)
            raise typer.Exit(1) from exc
        try:
            from pydantic import ValidationError

            metadata_filter = MetadataFilter.model_validate(raw_filter)
        except (ValidationError, ValueError) as exc:
            typer.echo(f"--filter validation error: {exc}", err=True)
            raise typer.Exit(1) from exc

    try:
        loaded = LoadedIndex.load(index)
    except (IndexLoadError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    try:
        provider = provider_from_meta(loaded.meta, offline=offline)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    processor = TextProcessor(use_spacy=None)

    if mode == "agent":
        from cybernaut_mini.agent.search import run_agent_search

        try:
            app_config = load_config(config, overrides={})
            if offline:
                app_config.require_offline_compatible()
        except (ConfigError, ValueError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc

        result, _ = run_agent_search(
            loaded,
            question,
            config=app_config,
            processor=processor,
            provider=provider,
            metadata_filter=metadata_filter,
            output_top_k=top_k,
        )
        if trace_out is not None:
            trace_out.write_text(
                result.trace.model_dump_json() + "\n", encoding="utf-8"
            )
        if as_json:
            typer.echo(
                json.dumps(
                    {
                        "question": question,
                        "mode": mode,
                        "best_query": result.best_query,
                        "shard_ids": result.shard_ids,
                        "stop_reason": result.trace.stop_reason,
                        "hits": [h.model_dump(mode="json") for h in result.hits],
                    },
                    sort_keys=True,
                )
            )
        else:
            typer.echo(
                f"agent: query={result.best_query!r} shards={result.shard_ids} "
                f"calls={result.trace.retrieval_calls} stop={result.trace.stop_reason}"
            )
            for hit in result.hits:
                typer.echo(
                    f"{hit.rank}. [{hit.score:.4f}] {hit.document.id}"
                    f" shard={hit.shard_id} {hit.document.title}"
                )
                typer.echo(f"  {hit.snippet[:120]}")
        return

    rrf_config = RRFConfig()
    hits = retrieve(
        loaded,
        question,
        mode=mode,  # type: ignore[arg-type]
        processor=processor,
        provider=provider,
        metadata_filter=metadata_filter,
        rrf_config=rrf_config,
        top_k=top_k,
    )

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "question": question,
                    "mode": mode,
                    "hits": [h.model_dump(mode="json") for h in hits],
                },
                sort_keys=True,
            )
        )
    else:
        for hit in hits:
            title = hit.document.title
            typer.echo(
                f"{hit.rank}. [{hit.score:.4f}] {hit.document.id}"
                f" shard={hit.shard_id} {title}"
            )
            typer.echo(f"  {hit.snippet[:120]}")


@app.command(name="eval")
def eval_cmd(
    index: Annotated[Path, typer.Option("--index", help="Path to built index directory")],
    judgments: Annotated[Path, typer.Option("--judgments", help="Path to judgments JSONL file")],
    config: Annotated[
        Path | None, typer.Option("--config", help="Optional YAML config file")
    ] = None,
    offline: Annotated[bool, typer.Option("--offline", help="Offline mode")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
) -> None:
    """Evaluate retrieval quality over all four modes using graded judgments."""
    from cybernaut_mini.config import ConfigError, load_config
    from cybernaut_mini.evals import evaluate
    from cybernaut_mini.indexing import IndexLoadError, LoadedIndex
    from cybernaut_mini.models import Judgment
    from cybernaut_mini.retrieval import provider_from_meta
    from cybernaut_mini.text import TextProcessor

    # Load index.
    try:
        loaded = LoadedIndex.load(index)
    except (IndexLoadError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    # Load judgments with per-record error reporting.
    judgment_list: list[Judgment] = []
    try:
        import json as _json

        with judgments.open(encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    raw = _json.loads(line)
                except _json.JSONDecodeError as exc:
                    typer.echo(f"judgments line {lineno}: invalid JSON — {exc}", err=True)
                    raise typer.Exit(1) from exc
                try:
                    from pydantic import ValidationError

                    j = Judgment.model_validate(raw)
                except (ValidationError, ValueError) as exc:
                    qid = raw.get("query_id") if isinstance(raw, dict) else None
                    label = f"query_id={qid!r}" if qid else f"line {lineno}"
                    typer.echo(f"judgments {label}: {exc}", err=True)
                    raise typer.Exit(1) from exc
                judgment_list.append(j)
    except typer.Exit:
        raise
    except OSError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    # Build provider.
    try:
        provider = provider_from_meta(loaded.meta, offline=offline)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    # Load config.
    try:
        app_config = load_config(config, overrides={})
        if offline:
            app_config.require_offline_compatible()
    except (ConfigError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    processor = TextProcessor(use_spacy=None)

    # Run evaluation directly (no Kedro session needed for CLI — the Kedro
    # pipeline is defined and registered for `kedro run --pipeline evaluation`
    # but the CLI calls evals.evaluate directly to avoid Kedro session overhead).
    metrics = evaluate(
        loaded,
        judgment_list,
        config=app_config,
        processor=processor,
        provider=provider,
    )

    if as_json:
        typer.echo(json.dumps({"metrics": [m.as_dict() for m in metrics]}, sort_keys=True))
    else:
        # Table output.
        col_w = 12
        header = (
            f"{'Mode':<10} {'Recall@5':>{col_w}} {'Recall@10':>{col_w}}"
            f" {'MRR@10':>{col_w}} {'nDCG@10':>{col_w}}"
            f" {'Ret calls':>{col_w}} {'LLM calls':>{col_w}}"
            f" {'Wall (s) (informational)':>26}"
        )
        typer.echo(header)
        typer.echo("-" * len(header))
        for m in metrics:
            typer.echo(
                f"{m.mode:<10} {m.recall_at_5:>{col_w}.4f} {m.recall_at_10:>{col_w}.4f}"
                f" {m.mrr_at_10:>{col_w}.4f} {m.ndcg_at_10:>{col_w}.4f}"
                f" {m.mean_retrieval_calls:>{col_w}.1f} {m.mean_llm_calls:>{col_w}.1f}"
                f" {m.wall_clock_seconds:>26.2f}"
            )


@app.command()
def explain(
    stage: Annotated[
        int | None,
        typer.Option("--stage", help="Blog stage to explain, 1-8. Omit to print all eight."),
    ] = None,
) -> None:
    """Explain a stage of the query pipeline: what it does, assumed, and rejected.

    The text is read from `configs/learn/curriculum.yaml`, not from this module, so
    the teaching content can be edited without touching code.
    """
    from cybernaut_mini.config import ConfigError
    from cybernaut_mini.curriculum import format_lesson, load_curriculum

    try:
        curriculum = load_curriculum()
        lessons = (curriculum.by_stage(stage),) if stage is not None else curriculum.ordered
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    for position, lesson in enumerate(lessons):
        if position:
            typer.echo("")
        typer.echo(format_lesson(lesson))


def _unwrap_exception(exc: BaseException) -> BaseException:
    """Recursively unwrap exception chains to find the original cause."""
    seen: set[int] = set()
    current = exc
    while current is not None:
        if id(current) in seen:
            break
        seen.add(id(current))
        cause = current.__cause__ or current.__context__
        if cause is None:
            return current
        current = cause
    return exc


if __name__ == "__main__":
    app()
