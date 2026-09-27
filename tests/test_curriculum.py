"""Tests for the learning curriculum behind ``cybernaut-mini explain``.

The curriculum is a map from blog stage to code. A map that points at modules which
no longer exist is worse than no map: it teaches a reader a layout the repository
does not have. These tests are what keep it honest.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 (stages 1-8).
Local copy: ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - The curriculum file is part of the repository, not packaging data, so the tests
      read it from ``configs/learn/`` via the loader's own default path.
    - Stage numbers are exactly 1-8. The pipeline is a fixed eight-stage design.

Alternatives considered:
    - Asserting on the YAML text directly: rejected. It would not catch a module path
      that stopped existing, which is the failure this file exists to catch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cybernaut_mini.config import ConfigError
from cybernaut_mini.curriculum import (
    DEFAULT_CURRICULUM_PATH,
    Curriculum,
    StageLesson,
    format_lesson,
    load_curriculum,
)


@pytest.fixture(scope="module")
def curriculum() -> Curriculum:
    return load_curriculum()


def test_default_curriculum_path_exists() -> None:
    """The loader's default must point at a committed file."""
    assert DEFAULT_CURRICULUM_PATH.is_file(), f"missing {DEFAULT_CURRICULUM_PATH}"


def test_curriculum_covers_every_stage_exactly_once(curriculum: Curriculum) -> None:
    stages = [lesson.stage for lesson in curriculum.stages]
    assert sorted(stages) == list(range(1, 9))


def test_ordered_is_sorted_by_stage(curriculum: Curriculum) -> None:
    assert [lesson.stage for lesson in curriculum.ordered] == list(range(1, 9))


def test_every_declared_module_exists(curriculum: Curriculum) -> None:
    """The map must not point at files the repository does not have."""
    missing = {
        lesson.stage: [str(path) for path in lesson.module_paths() if not path.is_file()]
        for lesson in curriculum.stages
    }
    assert not any(missing.values()), f"curriculum cites missing modules: {missing}"


def test_every_lesson_carries_the_three_teaching_sections(curriculum: Curriculum) -> None:
    """A lesson with an empty section teaches nothing about that section."""
    for lesson in curriculum.ordered:
        assert lesson.disclosed.strip(), f"stage {lesson.stage} has no disclosure"
        assert lesson.assumptions.strip(), f"stage {lesson.stage} has no assumptions"
        assert lesson.alternatives.strip(), f"stage {lesson.stage} has no alternatives"


def test_format_lesson_renders_the_contract(curriculum: Curriculum) -> None:
    """The rendered lesson must expose the blog ref, all three sections, and the code."""
    lesson = curriculum.by_stage(3)
    rendered = format_lesson(lesson)

    assert "Stage 3" in rendered
    assert "Blog ref:" in rendered
    assert "Disclosed:" in rendered
    assert "This replica assumes:" in rendered
    assert "Rejected:" in rendered
    for module in lesson.modules:
        assert module in rendered


def test_by_stage_unknown_stage_raises_config_error(curriculum: Curriculum) -> None:
    with pytest.raises(ConfigError, match="stage 9"):
        curriculum.by_stage(9)


def _lesson(stage: int) -> StageLesson:
    return StageLesson(
        stage=stage,
        title=f"Stage {stage}",
        blog_ref="https://nosible.com/blog/the-road-to-cybernaut-1",
        modules=["README.md"],
        disclosed="d",
        assumptions="a",
        alternatives="x",
    )


def test_duplicate_stages_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate stage"):
        Curriculum(stages=[_lesson(1), _lesson(1)] + [_lesson(s) for s in range(2, 9)])


def test_missing_stage_is_rejected() -> None:
    """Seven stages is not a curriculum with a hole in it; it is invalid."""
    with pytest.raises(ValidationError, match="missing stage"):
        Curriculum(stages=[_lesson(stage) for stage in range(1, 8)])


def test_stage_number_outside_the_pipeline_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _lesson(9)


def test_lesson_requires_at_least_one_module() -> None:
    with pytest.raises(ValidationError):
        StageLesson(
            stage=1,
            title="Empty",
            blog_ref="https://nosible.com/blog/the-road-to-cybernaut-1",
            modules=[],
            disclosed="d",
            assumptions="a",
            alternatives="x",
        )


def test_unreadable_curriculum_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read curriculum"):
        load_curriculum(tmp_path / "absent.yaml")


def test_malformed_curriculum_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "curriculum.yaml"
    path.write_text("stages: not-a-list\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid curriculum"):
        load_curriculum(path)
