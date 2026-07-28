"""CLI tests for ``search --mode agent`` and trace output (M6). Offline throughout."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from cybernaut_mini.cli import app
from cybernaut_mini.trace import AgentTrace

runner = CliRunner()


def test_agent_mode_exit_zero(built_index_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(built_index_path),
            "--question", "gene editing immune response",
            "--mode", "agent",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "agent:" in result.output


def test_agent_mode_json_is_single_object(built_index_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(built_index_path),
            "--question", "gene editing immune response",
            "--mode", "agent",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "agent"
    assert "hits" in payload and "best_query" in payload


def test_agent_trace_out_writes_valid_trace(built_index_path: Path, tmp_path: Path) -> None:
    trace_path = tmp_path / "run.json"
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(built_index_path),
            "--question", "solar panel efficiency",
            "--mode", "agent",
            "--trace-out", str(trace_path),
        ],
    )
    assert result.exit_code == 0, result.output
    trace = AgentTrace.model_validate_json(trace_path.read_text(encoding="utf-8"))
    assert trace.retrieval_calls <= 18
    assert {n.stage for n in trace.nodes} == {"explore", "refine", "exploit"}


def test_invalid_mode_exits_nonzero(built_index_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(built_index_path),
            "--question", "q",
            "--mode", "bogus",
        ],
    )
    assert result.exit_code != 0
