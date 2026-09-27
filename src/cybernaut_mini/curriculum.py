"""The learning curriculum: one entry per stage of the 8-stage query pipeline.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 (stages 1-8).
Local copy: ``data/00_reference/the-road-to-cybernaut-1.md``.

This module backs ``cybernaut-mini explain``. It answers the question a reader
arrives with — *where is blog stage 3 in this codebase, and what did the replica
have to decide?* — without making them read the whole tree first.

The curriculum is data (``configs/learn/curriculum.yaml``), not prose duplicated
in the CLI, so the map can be edited without touching code. ``tests/test_curriculum.py``
asserts every declared module path exists and that stages 1-8 are all present, so
the map cannot silently rot away from the code it describes.

Assumptions:
    - Exactly one entry per stage, covering stages 1-8 and nothing else. The
      pipeline is a fixed eight-stage design; a ninth entry would describe a
      different pipeline.
    - Each entry cites the module that *implements* the stage, not every file that
      participates in it. The full inventory belongs in the package README.

Alternatives considered:
    - Hardcoding the teaching text in ``cli.py``: rejected. It puts course content
      in the argument-parsing layer, where it cannot be edited without a code
      change and cannot be asserted by a test that does not shell out to the CLI.
    - Rendering straight from module docstrings: rejected. Those docstrings are
      written for a reader already inside the file; parsing them would make the
      curriculum's content depend on docstring formatting, and the ``Blog ref:``
      contract is checked textually rather than parsed.
    - Shipping the YAML inside the package as package data: rejected to match the
      repository convention that configuration lives under ``configs/``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from cybernaut_mini.config import ConfigError

#: Repository root, resolved from this file so the curriculum is found no matter
#: what the caller's working directory is.
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CURRICULUM_PATH = REPO_ROOT / "configs" / "learn" / "curriculum.yaml"

#: The pipeline is an eight-stage design; the blog numbers them 1-8.
STAGE_NUMBERS = frozenset(range(1, 9))


class StageLesson(BaseModel):
    """One blog stage, as this replica implements it."""

    model_config = ConfigDict(extra="forbid")

    stage: int = Field(ge=1, le=8, description="Blog stage number, 1-8.")
    title: str = Field(description="Short human title, e.g. 'Shard reranking'.")
    blog_ref: str = Field(description="URL or local reference for this stage.")
    modules: list[str] = Field(
        min_length=1,
        description=(
            "Repository-relative paths implementing this stage. Asserted to exist "
            "by tests/test_curriculum.py."
        ),
    )
    disclosed: str = Field(description="What NOSIBLE stated publicly about this stage.")
    assumptions: str = Field(description="What this replica had to decide, and why.")
    alternatives: str = Field(description="An approach that was rejected, and why.")

    def module_paths(self) -> tuple[Path, ...]:
        """Absolute paths of this stage's modules, for existence checks."""
        return tuple(REPO_ROOT / module for module in self.modules)


class Curriculum(BaseModel):
    """The whole ladder: stages 1-8, each present exactly once."""

    model_config = ConfigDict(extra="forbid")

    stages: list[StageLesson] = Field(min_length=1)

    @model_validator(mode="after")
    def _covers_every_stage_once(self) -> Curriculum:
        seen = [lesson.stage for lesson in self.stages]
        duplicates = sorted({s for s in seen if seen.count(s) > 1})
        if duplicates:
            raise ValueError(f"duplicate stage entries: {duplicates}")
        missing = sorted(STAGE_NUMBERS - set(seen))
        if missing:
            raise ValueError(f"curriculum is missing stage(s): {missing}")
        return self

    @property
    def ordered(self) -> tuple[StageLesson, ...]:
        """Lessons sorted by stage number, which is the order a reader follows."""
        return tuple(sorted(self.stages, key=lambda lesson: lesson.stage))

    def by_stage(self, stage: int) -> StageLesson:
        """Return one stage, or raise :class:`ConfigError` naming the valid range."""
        for lesson in self.stages:
            if lesson.stage == stage:
                return lesson
        raise ConfigError(f"no curriculum entry for stage {stage}; valid stages are 1-8")


def _flatten(text: str) -> str:
    """Collapse YAML folded-scalar line breaks into single spaces."""
    return " ".join(text.split())


def format_lesson(lesson: StageLesson) -> str:
    """Render one lesson for a terminal.

    Lives here rather than in ``cli.py`` so a test can assert the rendered text
    without invoking the command.
    """
    lines = [
        f"Stage {lesson.stage} — {lesson.title}",
        f"Blog ref: {lesson.blog_ref}",
        "",
        "Disclosed:",
        f"  {_flatten(lesson.disclosed)}",
        "",
        "This replica assumes:",
        f"  {_flatten(lesson.assumptions)}",
        "",
        "Rejected:",
        f"  {_flatten(lesson.alternatives)}",
        "",
        "Code:",
    ]
    lines.extend(f"  {module}" for module in lesson.modules)
    return "\n".join(lines)


def load_curriculum(path: Path | None = None) -> Curriculum:
    """Load and validate the curriculum, defaulting to ``configs/learn/``."""
    resolved = path or DEFAULT_CURRICULUM_PATH
    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read curriculum at {resolved}: {exc}") from exc

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {resolved}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ConfigError(f"{resolved} must contain a mapping with a 'stages' list")

    try:
        return Curriculum.model_validate(parsed)
    except ValidationError as exc:
        raise ConfigError(f"invalid curriculum in {resolved}: {exc}") from exc
