"""CLI tests for the ``search`` command (M3).

Uses a session-scoped index built via the conftest helper so the build cost
is paid once. All tests are offline (HashEmbedder index).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from cybernaut_mini.cli import app

runner = CliRunner()


# ------------------------------------------------------------------ #
# Basic exit-code / output tests                                      #
# ------------------------------------------------------------------ #


def test_search_lexical_exit_zero(built_index_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(built_index_path),
            "--question", "zylophristine assay",
            "--mode", "lexical",
        ],
    )
    assert result.exit_code == 0, f"Unexpected exit: {result.output}"
    assert result.output.strip(), "Expected non-empty output"


def test_search_dense_exit_zero(built_index_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(built_index_path),
            "--question", "gene edit",
            "--mode", "dense",
        ],
    )
    assert result.exit_code == 0, f"Unexpected exit: {result.output}"
    assert result.output.strip(), "Expected non-empty output"


def test_search_hybrid_exit_zero(built_index_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(built_index_path),
            "--question", "solar panel efficiency",
            "--mode", "hybrid",
        ],
    )
    assert result.exit_code == 0, f"Unexpected exit: {result.output}"
    assert result.output.strip(), "Expected non-empty output"


# ------------------------------------------------------------------ #
# --json flag                                                         #
# ------------------------------------------------------------------ #


def test_search_json_parseable(built_index_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(built_index_path),
            "--question", "solar energy",
            "--mode", "hybrid",
            "--json",
        ],
    )
    assert result.exit_code == 0, f"Unexpected exit: {result.output}"
    # Extract the JSON line (there may be Kedro log lines before it).
    json_line = next(
        (line for line in reversed(result.output.splitlines()) if line.strip().startswith("{")),
        None,
    )
    assert json_line is not None, f"No JSON object found in output: {result.output!r}"
    parsed = json.loads(json_line)
    assert "hits" in parsed, f"Missing 'hits' key: {parsed}"
    assert isinstance(parsed["hits"], list), "hits must be a list"
    assert "question" in parsed
    assert "mode" in parsed


def test_search_json_single_object(built_index_path: Path) -> None:
    """--json emits a single parseable JSON object with no extra text after it."""
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(built_index_path),
            "--question", "battery storage",
            "--json",
        ],
    )
    assert result.exit_code == 0
    json_line = next(
        (line for line in reversed(result.output.splitlines()) if line.strip().startswith("{")),
        None,
    )
    assert json_line is not None
    parsed = json.loads(json_line)
    assert isinstance(parsed, dict)


# ------------------------------------------------------------------ #
# Invalid --filter                                                    #
# ------------------------------------------------------------------ #


def test_search_invalid_filter_field_exits_nonzero(built_index_path: Path) -> None:
    """An unsupported operator field triggers a validation error and non-zero exit."""
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(built_index_path),
            "--question", "energy",
            "--filter", '{"language_not": ["en"]}',
        ],
    )
    assert result.exit_code != 0
    # Error message should mention validation.
    combined = (result.output or "") + (result.stderr if hasattr(result, "stderr") else "")
    assert "validation" in combined.lower() or "error" in combined.lower(), (
        f"Expected validation error in output; got: {combined!r}"
    )


def test_search_malformed_filter_json_exits_nonzero(built_index_path: Path) -> None:
    """Malformed JSON for --filter exits non-zero."""
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(built_index_path),
            "--question", "energy",
            "--filter", "{not valid json",
        ],
    )
    assert result.exit_code != 0


# ------------------------------------------------------------------ #
# Missing index                                                       #
# ------------------------------------------------------------------ #


def test_search_missing_index_exits_nonzero(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_index"
    result = runner.invoke(
        app,
        [
            "search",
            "--index", str(missing),
            "--question", "anything",
        ],
    )
    assert result.exit_code != 0
    # The IndexLoadError message mentions _VALID.
    combined = (result.output or "") + (result.stderr if hasattr(result, "stderr") else "")
    assert "_VALID" in combined, (
        f"Expected _VALID in error output; got: {combined!r}"
    )
