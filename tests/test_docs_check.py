"""Tests for ``tools/docs_check.py`` — the repository's documentation contract.

The checker is what keeps this repo teachable, so it is itself tested. The failure
mode worth guarding is a checker that passes everything: that converts "docs are
enforced" into a green badge over nothing. These tests prove the checker
discriminates, by running the real script against throwaway git repositories that
deliberately do and do not violate the contract.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 (whole-repo policy).
Local copy: ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - The checker resolves its root from its own location, so a throwaway repo
      needs a copy of the script inside it rather than just a different cwd.
    - Only *tracked* files are inspected, so fixtures must be ``git add``-ed
      before the checker will consider them.
    - ``git`` is available. It is a prerequisite of working in this repo at all.

Alternatives considered:
    - Importing ``tools/docs_check.py`` as a module and calling individual check
      functions: rejected because the interesting behaviour is the composition —
      git plumbing, README discovery, docstring parsing and the exit code. Testing
      the functions in isolation would not have caught the bug this file's
      ``test_repository_satisfies_contract`` now guards, where the checker flagged
      a path quoted as an example in its own docstring.
    - Asserting on ``BLOG_REF_EXEMPT`` membership: rejected as a test of
      implementation detail. The contract-level assertion covers it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKER = REPO_ROOT / "tools" / "docs_check.py"

#: Minimal README satisfying the "every directory has a README with a diagram" rule.
_COMPLIANT_README = """# Fixture repo

```mermaid
graph LR
    A --> B
```
"""

#: A module that satisfies all three required docstring sections. It deliberately
#: cites no local reference path: the checker resolves those against the repository
#: it runs in, and the throwaway fixture repo has no ``data/00_reference/``.
_COMPLIANT_MODULE = '''"""A compliant fixture module.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1, stage 3 of 8.

Assumptions:
    - This fixture exists only to prove the checker accepts a good module.

Alternatives considered:
    - No module docstring at all: rejected, that is the violation under test.
"""
'''


def _make_repo(tmp_path: Path, module_source: str) -> Path:
    """Build a throwaway git repo containing the real checker and one module."""
    (tmp_path / "tools").mkdir()
    shutil.copyfile(CHECKER, tmp_path / "tools" / "docs_check.py")
    (tmp_path / "README.md").write_text(_COMPLIANT_README, encoding="utf-8")
    # `tools/` is one of the directories the checker always requires a README for.
    (tmp_path / "tools" / "README.md").write_text(_COMPLIANT_README, encoding="utf-8")

    # No package __init__.py: that would make src/ a package directory and drag in
    # the README requirement, which is not what these tests are about.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "fixture_module.py").write_text(module_source, encoding="utf-8")

    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, capture_output=True, check=True)
    return tmp_path


def _run_checker(repo: Path) -> subprocess.CompletedProcess[str]:
    """Run the checker exactly as a developer or hook would."""
    return subprocess.run(
        [sys.executable, "tools/docs_check.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def test_checker_rejects_a_module_without_a_docstring(tmp_path: Path) -> None:
    """A module with no docstring must fail the contract."""
    repo = _make_repo(tmp_path, "x = 1\n")

    result = _run_checker(repo)

    assert result.returncode == 1, f"checker passed a module with no docstring:\n{result.stdout}"
    assert "module-docstring" in result.stdout
    assert "fixture_module.py" in result.stdout


def test_checker_rejects_a_module_missing_required_sections(tmp_path: Path) -> None:
    """Having *a* docstring is not enough — the three sections are the contract."""
    repo = _make_repo(tmp_path, '"""Just a summary, no contract sections."""\n\nx = 1\n')

    result = _run_checker(repo)

    assert result.returncode == 1, f"checker accepted a bare docstring:\n{result.stdout}"
    assert "Blog ref:" in result.stdout
    assert "Alternatives" in result.stdout


def test_checker_accepts_a_compliant_module(tmp_path: Path) -> None:
    """The same fixture with a compliant docstring must pass.

    Paired with the rejection tests above, this proves the checker discriminates on
    the docstring contract rather than failing for an unrelated reason.
    """
    repo = _make_repo(tmp_path, _COMPLIANT_MODULE)

    result = _run_checker(repo)

    assert result.returncode == 0, f"checker rejected a compliant module:\n{result.stdout}"
    assert "PASS" in result.stdout


def test_repository_satisfies_contract() -> None:
    """This repository passes every check in the contract.

    This is the assertion that makes the contract binding rather than aspirational:
    if it fails, ``make check`` fails, and the documentation gap is a build break
    rather than something a reviewer has to notice.

    It also exercises the two self-exemption rules, which exist because the checker
    would otherwise flag the credential shapes and reference paths it quotes as
    examples in its own docstrings.
    """
    result = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "the repository violates its own documentation contract:\n" + result.stdout
    )
