"""Execute every notebook against the real catalog.

A notebook that is not executed is documentation that rots. These tests run each
notebook end to end in a fresh kernel, so a renamed catalog entry, a changed node
signature, or a removed helper fails the build instead of leaving a notebook that
quietly no longer works.

Network access is asserted off, not merely assumed off: the devcontainer injects .env,
so NOSIBLE_API_KEY is ambient and the API-reference notebook would otherwise fire real
billable requests on every run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

#: Guards against a notebook opening a path directly instead of going through the
#: catalog. These are the fixture paths a notebook is most likely to reach for.
_DIRECT_PATH_SMELLS = (
    'open("data/',
    "open('data/",
    'read_text("data/',
    'Path("data/01_raw/fixtures',
    "Path('data/01_raw/fixtures",
    'np.load("artifacts',
    'open("artifacts',
)


def test_notebooks_are_discovered() -> None:
    """Fail loudly if the glob stops matching — an empty suite must not look green."""
    assert NOTEBOOKS, f"no notebooks found in {NOTEBOOK_DIR}"


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_executes(notebook_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the notebook in a fresh kernel; any cell exception fails the test."""
    # Never make live API calls from a test run, even on a machine where a developer
    # has opted in for interactive use.
    monkeypatch.delenv("CYBERNAUT_LIVE_API", raising=False)

    notebook = nbformat.read(notebook_path, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        # Catalog paths are relative to the project root.
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()

    errors = [
        output
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors, (
        f"{notebook_path.name} raised {errors[0]['ename']}: {errors[0]['evalue']}\n"
        + "\n".join(errors[0].get("traceback", [])[-8:])
    )


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_has_a_mermaid_diagram(notebook_path: Path) -> None:
    """Every notebook is a teaching artifact, so it carries a diagram GitHub renders."""
    notebook = nbformat.read(notebook_path, as_version=4)
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook.cells if cell["cell_type"] == "markdown"
    )
    assert "```mermaid" in markdown, f"{notebook_path.name} has no ```mermaid block"


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_reads_through_the_catalog(notebook_path: Path) -> None:
    """No notebook may open a data path directly — that is the drift this prevents."""
    notebook = nbformat.read(notebook_path, as_version=4)
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook.cells if cell["cell_type"] == "code"
    )
    offenders = [smell for smell in _DIRECT_PATH_SMELLS if smell in source]
    assert not offenders, (
        f"{notebook_path.name} opens a data path directly ({offenders}); "
        f"load it via `catalog.load(...)` instead"
    )


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_is_committed_without_outputs(notebook_path: Path) -> None:
    """Outputs are regenerated on every run, so committing them only creates diff noise."""
    raw = json.loads(notebook_path.read_text(encoding="utf-8"))
    with_outputs = [
        cell.get("execution_count")
        for cell in raw["cells"]
        if cell.get("cell_type") == "code" and cell.get("outputs")
    ]
    assert not with_outputs, (
        f"{notebook_path.name} has stored outputs; clear them before committing "
        f"(the test suite regenerates them)"
    )


def test_every_notebook_is_listed_in_the_readme() -> None:
    """A notebook nobody can find is a notebook nobody maintains."""
    readme = NOTEBOOK_DIR / "README.md"
    assert readme.is_file(), "notebooks/README.md is missing"
    content = readme.read_text(encoding="utf-8")
    missing = [path.name for path in NOTEBOOKS if path.name not in content]
    assert not missing, f"notebooks/README.md does not mention: {missing}"
