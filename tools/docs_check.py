#!/usr/bin/env python3
"""Enforce the cybernaut-mini documentation and real-data contract.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 (whole-repo policy;
    not a single stage). Local copy: ``data/00_reference/the-road-to-cybernaut-1.md``.

This repository exists to *teach* the architecture in that blog post. Documentation
is therefore a functional requirement, not a nicety: a module without its blog
reference, stated assumptions, and rejected alternatives has failed at its only job.
Prose conventions rot silently, so the convention is executable and this script is
wired into ``make check`` and a git ``pre-commit`` hook.

Assumptions:
    - Stdlib only. This runs inside a git hook where the project virtualenv may not
      be active, so it must never import a third-party package.
    - Only *tracked* files are inspected. Untracked scratch files are the author's
      business; anything git will carry is the project's business.
    - Checks are textual, not semantic. This verifies that a module *claims* a blog
      reference, not that the claim is accurate — that is a reviewer's job.

Alternatives considered:
    - ``pre-commit`` framework: the standard choice, and better at multi-language
      repos and tool pinning. Rejected because it needs a network install to
      bootstrap and hides the rules inside YAML; a single readable script is a
      better teaching artifact for a repo whose subject *is* the code.
    - ``interrogate`` / ``pydocstyle`` / ``ruff``'s pydocstyle rules (D1xx): these
      enforce that a docstring *exists*. None can express "must cite the blog and
      list rejected alternatives", which is the actual requirement here.
    - ``detect-secrets`` / ``gitleaks`` for the secret scan: strictly better at
      secret detection (entropy analysis, far broader rule sets). Deliberately not
      adopted as a dependency for a handful of patterns; if this repo ever handles
      real credentials, switch to gitleaks rather than extending ``SECRET_PATTERNS``.
    - A CI-only check: rejected outright. The requirement was enforcement at *every
      commit*, and CI runs after the commit exists.

Usage:
    python3 tools/docs_check.py            # check everything, exit 1 on failure
    python3 tools/docs_check.py --staged   # only files staged for commit (hook mode)
    python3 tools/docs_check.py --summary  # counts only, no per-violation detail
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------

#: Directories that must carry a README.md with a mermaid diagram. Any Python
#: package under src/ is added dynamically on top of these.
REQUIRED_README_DIRS = (
    ".",
    "configs",
    "data",
    "docs",
    "notebooks",
    "scripts",
    "tests",
    "tools",
)

#: A README must contain a mermaid fence so the folder's role renders as a diagram
#: on GitHub. ```mermaid is the only form GitHub renders natively.
MERMAID_PATTERN = re.compile(r"^```\s*mermaid\s*$", re.MULTILINE)

#: Sections every substantive module docstring must carry. These are the three
#: things the repo promises a reader for each component.
REQUIRED_DOCSTRING_SECTIONS = ("Blog ref:", "Assumptions:", "Alternatives")

#: A module may opt out with this marker plus a reason, e.g.
#: ``docs-check: exempt(pure re-export, no behaviour)``.
EXEMPT_PATTERN = re.compile(r"docs-check:\s*exempt\(([^)]+)\)")

#: Files whose existence violates the "no dummy data" rule. The user was explicit:
#: "THIS MUST NEVER USE DUMMY DATA." Synthetic corpora and fabricated relevance
#: judgments silently poison every downstream metric, so their return is a hard
#: failure rather than a warning.
BANNED_PATHS = (
    "data/sample/",
    "scripts/generate_sample_corpus.py",
)

BANNED_PATH_REASON = (
    "synthetic corpora and fabricated judgments are banned; this repo evaluates "
    "against real human relevance judgments only (see data/README.md)"
)

#: Credential shapes worth catching before they reach history. Deliberately narrow:
#: each requires enough trailing entropy that documentation placeholders such as
#: ``nos_sk_...`` do not trip them.
SECRET_PATTERNS = (
    ("NOSIBLE api key", re.compile(r"nos_sk_[A-Za-z0-9]{20,}")),
    ("OpenAI api key", re.compile(r"\bsk-[A-Za-z0-9]{32,}")),
    ("Anthropic api key", re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{32,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("HuggingFace token", re.compile(r"\bhf_[A-Za-z0-9]{34,}")),
    ("Generic private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)

#: Paths exempt from the secret scan: the reference material quotes the blog, and
#: this file necessarily contains the patterns themselves.
SECRET_SCAN_EXEMPT = ("tools/docs_check.py",)

TEXT_SUFFIXES = {
    ".py", ".md", ".yaml", ".yml", ".json", ".jsonl", ".toml", ".txt",
    ".ipynb", ".sh", ".cfg", ".ini", ".html",
}


# --------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------


@dataclass
class Violation:
    """One failed rule, anchored to a file so editors can jump to it."""

    check: str
    path: str
    message: str
    line: int | None = None

    def render(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else self.path
        return f"  {location}\n      {self.message}"


@dataclass
class Report:
    violations: list[Violation] = field(default_factory=list)
    checked: dict[str, int] = field(default_factory=dict)

    def add(self, violation: Violation) -> None:
        self.violations.append(violation)

    def note(self, check: str, count: int) -> None:
        self.checked[check] = self.checked.get(check, 0) + count

    @property
    def ok(self) -> bool:
        return not self.violations


# --------------------------------------------------------------------------------
# Git helpers
# --------------------------------------------------------------------------------


def _git(*args: str) -> list[str]:
    """Run a git command and return non-empty stdout lines."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def tracked_files(staged_only: bool) -> list[Path]:
    """Return repo-relative paths of tracked (or staged) files that still exist."""
    if staged_only:
        names = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    else:
        names = _git("ls-files")
    paths = []
    for name in names:
        path = REPO_ROOT / name
        if path.is_file():
            paths.append(Path(name))
    return paths


def read_text(path: Path) -> str | None:
    try:
        return (REPO_ROOT / path).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


# --------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------


def python_package_dirs() -> list[Path]:
    """Every directory under src/ that is an importable package."""
    src = REPO_ROOT / "src"
    if not src.is_dir():
        return []
    return sorted(
        {init.parent.relative_to(REPO_ROOT) for init in src.rglob("__init__.py")}
    )


def check_readmes(report: Report) -> None:
    """Every documented directory has a README.md containing a mermaid diagram.

    Rationale: the user asked for a mermaid diagram in every folder README so the
    repo can be navigated visually on GitHub. A README without a diagram satisfies
    the letter of that and not the point of it, so both are checked together.
    """
    required = [Path(d) for d in REQUIRED_README_DIRS]
    required.extend(python_package_dirs())

    seen: set[Path] = set()
    for directory in required:
        if directory in seen:
            continue
        seen.add(directory)

        abs_dir = REPO_ROOT / directory
        if not abs_dir.is_dir():
            continue

        readme = directory / "README.md"
        if not (REPO_ROOT / readme).is_file():
            report.add(
                Violation(
                    check="readme-present",
                    path=str(readme),
                    message=(
                        "missing README.md — every package and top-level directory "
                        "must explain its role and cite the blog section it implements"
                    ),
                )
            )
            continue

        content = read_text(readme)
        if content is None:
            continue
        if not MERMAID_PATTERN.search(content):
            report.add(
                Violation(
                    check="readme-mermaid",
                    path=str(readme),
                    message=(
                        "no ```mermaid block — every folder README needs a diagram "
                        "that renders on GitHub"
                    ),
                )
            )

    report.note("readme", len(seen))


def _module_docstring(source: str) -> str | None:
    """Extract a module docstring without importing or fully parsing the module.

    Uses ``ast`` rather than a regex because a triple-quoted string inside a
    function is not a module docstring, and only the parser knows the difference.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    return ast.get_docstring(tree)


def check_module_docs(report: Report, files: list[Path]) -> None:
    """Every substantive module cites the blog, its assumptions, and alternatives.

    This is the core learning contract. ``__init__.py`` files are skipped when they
    are pure re-exports; anything else must either comply or declare an exemption
    with a reason.
    """
    count = 0
    for path in files:
        if path.suffix != ".py":
            continue
        if not str(path).startswith(("src/", "tools/")):
            continue

        source = read_text(path)
        if source is None:
            continue

        # A trivial __init__.py (no statements beyond imports/docstring) is exempt.
        if path.name == "__init__.py" and len(source.strip()) < 400:
            continue

        exemption = EXEMPT_PATTERN.search(source)
        if exemption:
            count += 1
            continue

        docstring = _module_docstring(source)
        if not docstring:
            report.add(
                Violation(
                    check="module-docstring",
                    path=str(path),
                    line=1,
                    message=(
                        "no module docstring — needs 'Blog ref:', 'Assumptions:' and "
                        "'Alternatives considered:' sections, or a "
                        "'docs-check: exempt(reason)' marker"
                    ),
                )
            )
            count += 1
            continue

        missing = [s for s in REQUIRED_DOCSTRING_SECTIONS if s not in docstring]
        if missing:
            report.add(
                Violation(
                    check="module-docstring",
                    path=str(path),
                    line=1,
                    message=(
                        f"module docstring missing section(s): {', '.join(missing)}. "
                        "Each module states which blog stage it implements, what it "
                        "assumes, and what was rejected and why."
                    ),
                )
            )
        count += 1

    report.note("module-docstring", count)


def check_blog_references(report: Report, files: list[Path]) -> None:
    """Blog references must point at something that exists.

    Catches the common rot: a module cites ``data/00_reference/foo.md`` after the
    file is renamed, or cites a stage number outside 1-8.
    """
    local_ref = re.compile(r"data/00_reference/[\w.\-]+\.md")
    stage_ref = re.compile(r"[Bb]log [Ss]tage (\d+)")
    count = 0

    for path in files:
        if path.suffix not in {".py", ".md"}:
            continue
        content = read_text(path)
        if content is None:
            continue

        for match in local_ref.finditer(content):
            target = match.group(0)
            if not (REPO_ROOT / target).is_file():
                report.add(
                    Violation(
                        check="blog-reference",
                        path=str(path),
                        line=content[: match.start()].count("\n") + 1,
                        message=f"references {target}, which does not exist",
                    )
                )

        for match in stage_ref.finditer(content):
            stage = int(match.group(1))
            if not 1 <= stage <= 8:
                report.add(
                    Violation(
                        check="blog-reference",
                        path=str(path),
                        line=content[: match.start()].count("\n") + 1,
                        message=(
                            f"cites blog stage {stage}, but the pipeline has stages 1-8"
                        ),
                    )
                )
        count += 1

    report.note("blog-reference", count)


def check_banned_paths(report: Report, files: list[Path]) -> None:
    """No synthetic corpora or fabricated judgments, ever."""
    for path in files:
        text = str(path)
        for banned in BANNED_PATHS:
            if text.startswith(banned):
                report.add(
                    Violation(
                        check="no-dummy-data",
                        path=text,
                        message=BANNED_PATH_REASON,
                    )
                )
    report.note("no-dummy-data", len(files))


def check_secrets(report: Report, files: list[Path]) -> None:
    """Block credential shapes before they reach git history.

    This repo already had one live API key committed to a local branch, which is
    exactly the failure mode this catches.
    """
    count = 0
    for path in files:
        if str(path) in SECRET_SCAN_EXEMPT:
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        content = read_text(path)
        if content is None:
            continue
        for label, pattern in SECRET_PATTERNS:
            match = pattern.search(content)
            if match:
                report.add(
                    Violation(
                        check="secret-scan",
                        path=str(path),
                        line=content[: match.start()].count("\n") + 1,
                        message=(
                            f"possible {label} committed. Rotate it, then remove it "
                            "from the file (use an environment variable instead)."
                        ),
                    )
                )
        count += 1
    report.note("secret-scan", count)


# --------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------

#: Display order for the report. Any check name not listed here is still printed,
#: appended alphabetically — a report that silently drops violations is worse than
#: no report at all.
CHECK_ORDER = (
    "secret-scan",
    "no-dummy-data",
    "readme-present",
    "readme-mermaid",
    "module-docstring",
    "blog-reference",
)


def run(staged_only: bool = False) -> Report:
    report = Report()
    files = tracked_files(staged_only)

    # README coverage is always whole-repo: deleting a README elsewhere still breaks
    # the contract even if this commit does not touch it.
    check_readmes(report)
    check_module_docs(report, files)
    check_blog_references(report, files)
    check_banned_paths(report, files)
    check_secrets(report, files)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce the cybernaut-mini documentation and real-data contract."
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="only inspect files staged for commit (used by the pre-commit hook)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print counts only, omitting per-violation detail",
    )
    args = parser.parse_args()

    report = run(staged_only=args.staged)

    if report.ok:
        scanned = ", ".join(f"{k}={v}" for k, v in sorted(report.checked.items()))
        print(f"docs-check: PASS ({scanned})")
        return 0

    by_check: dict[str, list[Violation]] = {}
    for violation in report.violations:
        by_check.setdefault(violation.check, []).append(violation)

    # Ordered checks first, then anything unrecognised, so a newly added check can
    # never vanish from the report just because it was not registered above.
    ordered = [c for c in CHECK_ORDER if c in by_check]
    ordered += sorted(c for c in by_check if c not in CHECK_ORDER)

    print(f"docs-check: FAIL — {len(report.violations)} violation(s)\n")
    shown = 0
    for check in ordered:
        items = by_check[check]
        shown += len(items)
        print(f"[{check}] {len(items)} violation(s)")
        if not args.summary:
            for violation in items:
                print(violation.render())
        print()

    assert shown == len(report.violations), "report dropped violations"
    print("The documentation contract is defined in AGENTS.md.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
