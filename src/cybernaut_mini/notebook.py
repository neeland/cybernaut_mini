"""Kedro session bootstrap for notebooks.

``kedro jupyter lab`` injects ``catalog``, ``context``, ``session`` and ``pipelines``
into the kernel automatically. Nothing else does — not ``jupyter lab``, not VS Code's
notebook editor, and not the headless executor the test suite uses. A notebook that
relied on those injected names would therefore only run one way, and would silently
stop being verifiable.

This module gives notebooks one entry point that works everywhere, so the same file a
reader opens in JupyterLab is the file pytest executes.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — supports the notebooks
    that make the build-time claims (shard coherence, uniformity, hybrid fusion)
    measurable. Local copy: ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - Catalog paths are relative to the project root, so the helpers chdir there.
      A notebook opened from ``notebooks/`` would otherwise resolve
      ``data/sample/documents.jsonl`` against the wrong directory.
    - Building the sample index is cheap (hash embedder, 63 documents, well under a
      second), so a notebook may build it on demand rather than requiring the reader
      to have run ``make build-sample`` first.
    - Rebuilding is avoided when ``_VALID`` is present, so a notebook run never
      clobbers an index the reader built with different settings.

Alternatives considered:
    - Requiring ``kedro jupyter lab``: the officially blessed path, and the reason
      ``scripts/lab.sh`` exists. Rejected as the *only* path because it cannot be
      driven headlessly, so notebooks could not be executed in the test suite — which
      is the mechanism that stops them drifting from the pipeline.
    - ``%load_ext kedro.ipython`` in each notebook: the standard magic, and it does
      work under plain Jupyter. Rejected because it still assumes an IPython kernel
      with the extension importable and the cwd already correct, and it returns
      nothing a plain ``python -c`` could use.
    - A conftest fixture injecting the catalog into the kernel: keeps notebooks
      free of bootstrap code. Rejected because the notebook would then be unrunnable
      by a human opening it, which defeats the point of shipping notebooks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_MARKER = "pyproject.toml"
_bootstrapped = False


def project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` to the directory holding the Kedro ``pyproject.toml``."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        marker = candidate / _MARKER
        if marker.is_file() and "[tool.kedro]" in marker.read_text(encoding="utf-8"):
            return candidate
    msg = f"no Kedro project root found above {current}"
    raise RuntimeError(msg)


def _bootstrap() -> Path:
    """Make the project importable to Kedro and chdir to its root (idempotent)."""
    global _bootstrapped

    root = project_root()
    if os.getcwd() != str(root):
        os.chdir(root)
    if not _bootstrapped:
        from kedro.framework.startup import bootstrap_project

        bootstrap_project(root)
        _bootstrapped = True
    return root


def kedro_session(env: str = "local", **runtime_params: Any) -> Any:
    """Return a fresh :class:`KedroSession` for the project.

    ``runtime_params`` are the same values the Typer CLI forwards, so a notebook can
    point the catalog at a different corpus or index exactly as ``--input`` does.
    """
    from kedro.framework.session import KedroSession

    root = _bootstrap()
    return KedroSession.create(
        project_path=root,
        env=env,
        runtime_params=runtime_params or None,
    )


def kedro_catalog(env: str = "local", **runtime_params: Any) -> Any:
    """Return the Kedro ``DataCatalog`` — the notebook's only data entry point.

    Every notebook loads through this rather than opening paths, so a renamed or
    repointed dataset surfaces as a failure in the notebook run instead of a
    notebook that quietly reads a stale file.
    """
    with kedro_session(env, **runtime_params) as session:
        return session.load_context().catalog


def run_pipeline(name: str, env: str = "local", **runtime_params: Any) -> dict[str, Any]:
    """Run a registered pipeline from a notebook and return its free outputs."""
    with kedro_session(env, **runtime_params) as session:
        result: dict[str, Any] = session.run(pipeline_name=name)
        return result


#: Settings that reproduce ``make build-sample`` (configs/tiny.yaml): the hash
#: embedder, so a notebook never needs a model download or network access.
SAMPLE_BUILD_PARAMS: dict[str, Any] = {
    "input_path": "data/sample/documents.jsonl",
    "index_path": "artifacts/sample",
    "judgments_path": "data/sample/judgments.jsonl",
    "embedding": {"provider": "hash", "dim": 256},
    "index": {
        "n_shards": 8,
        "max_keywords": 30,
        "max_entities": 30,
        "cooccurrence_window": 5,
        "min_edge_count": 2,
    },
    "seed": 42,
    "offline": True,
}


def ensure_sample_index(index_path: str = "artifacts/sample") -> Path:
    """Build the sample index if it is absent, then return its path.

    ``_VALID`` is written last by the index writer, so its presence means a complete
    index and the build is skipped. That keeps a notebook run from overwriting an
    index the reader built themselves.
    """
    root = _bootstrap()
    target = root / index_path
    if (target / "_VALID").exists():
        return target

    params = dict(SAMPLE_BUILD_PARAMS)
    params["index_path"] = index_path
    run_pipeline("index_build", **params)
    return target
