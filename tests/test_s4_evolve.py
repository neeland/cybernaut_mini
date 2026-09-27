"""Tests for stage-4 instruction evolution: the LLM writer and the hill-climb loop.

The loop is tested with stub chat functions and stub scorers so it is fast,
deterministic and offline — the plan explicitly forbids running a real evolution
in tests. The nDCG scorer is exercised against the session index with real
MIRACL dev judgments from the committed fixture file (real qrels, never
fabricated); its value is not asserted beyond its contract because those
judgments reference the MIRACL corpus, not the test corpus.

Every test runs offline. No API key, no model download, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from cybernaut_mini.config import AppConfig, EmbeddingConfig
from cybernaut_mini.models import Judgment
from cybernaut_mini.providers.embeddings import HashEmbedder
from cybernaut_mini.query.s4_instruct.evolve import (
    EvolutionResult,
    LLMInstructionWriter,
    evolve_template,
    llm_mutator,
    make_ndcg_scorer,
    write_artifact,
)
from cybernaut_mini.query.s4_instruct.templates import DISCLOSED_TEMPLATE, InstructionError
from cybernaut_mini.query.s4_instruct.writer import DefaultInstructionWriter, InstructionRequest
from cybernaut_mini.text import TextProcessor

if TYPE_CHECKING:
    from cybernaut_mini.indexing import LoadedIndex

JUDGMENTS_PATH = Path("data/01_raw/fixtures/judgments.jsonl")

REQUEST = InstructionRequest(question="what is gene editing", language="English")


def _judgments(n: int) -> list[Judgment]:
    lines = JUDGMENTS_PATH.read_text(encoding="utf-8").splitlines()[:n]
    return [Judgment.model_validate(json.loads(line)) for line in lines]


# ------------------------- LLMInstructionWriter ------------------------ #


def test_llm_writer_returns_the_model_instruction() -> None:
    writer = LLMInstructionWriter(chat_fn=lambda system, user: "  Retrieve gene facts. \n")
    assert writer.write(REQUEST) == "Retrieve gene facts."
    assert writer.identifier == "llm:injected"


def test_llm_writer_falls_back_on_a_crash() -> None:
    def broken(system: str, user: str) -> str:
        raise RuntimeError("model exploded")

    writer = LLMInstructionWriter(chat_fn=broken)
    assert writer.write(REQUEST) == DefaultInstructionWriter().write(REQUEST)


@pytest.mark.parametrize("bad", ["", "   \n  ", "x" * 500])
def test_llm_writer_falls_back_on_blank_or_overlong_output(bad: str) -> None:
    writer = LLMInstructionWriter(chat_fn=lambda system, user: bad)
    assert writer.write(REQUEST) == DefaultInstructionWriter().write(REQUEST)


# ------------------------------ mutator -------------------------------- #


def test_llm_mutator_splits_lines_and_drops_blanks() -> None:
    mutate = llm_mutator(lambda system, user: "one {Language}\n\n  two {Language}  \n")
    assert mutate("seed {Language}", 5, 0) == ["one {Language}", "two {Language}"]


def test_llm_mutator_survives_a_crashed_generation() -> None:
    def broken(system: str, user: str) -> str:
        raise RuntimeError("model exploded")

    assert llm_mutator(broken)("seed {Language}", 5, 0) == []


# ----------------------------- hill climb ------------------------------ #


def test_hill_climb_keeps_the_strictly_best_candidate() -> None:
    proposals = {
        0: ["Find {Language} answers.", "Broken template {Other}."],
        1: ["Find better {Language} answers.", "Find {Language} answers."],
    }
    scores = {
        "Find {Language} answers.": 0.5,
        "Find better {Language} answers.": 0.7,
    }

    def mutate(parent: str, n: int, generation: int) -> list[str]:
        return proposals[generation]

    def score(text: str) -> float:
        return scores.get(text, 0.1)

    result = evolve_template(
        mutate=mutate, score=score, seed_template="Seed {Language} template.", generations=2
    )
    assert result.seed.origin == "seed"
    assert result.seed.score == pytest.approx(0.1)
    assert result.winner.text == "Find better {Language} answers."
    assert result.winner.score == pytest.approx(0.7)
    assert len(result.generations) == 2
    # The malformed proposal never reached the scorer's candidate list.
    gen0_texts = [candidate.text for candidate in result.generations[0].candidates]
    assert gen0_texts == ["Find {Language} answers."]


def test_hill_climb_ties_keep_the_incumbent() -> None:
    result = evolve_template(
        mutate=lambda parent, n, generation: ["Another {Language} template."],
        score=lambda text: 0.4,
        seed_template="Seed {Language} template.",
        generations=3,
    )
    assert result.winner.text == "Seed {Language} template."
    assert result.winner.origin == "seed"


def test_hill_climb_survives_generations_with_no_valid_proposals() -> None:
    result = evolve_template(
        mutate=lambda parent, n, generation: ["{Wrong} placeholder", ""],
        score=lambda text: 0.4,
        seed_template="Seed {Language} template.",
        generations=2,
    )
    assert result.winner.text == "Seed {Language} template."
    assert all(not generation.candidates for generation in result.generations)


def test_the_disclosed_template_is_a_valid_seed() -> None:
    result = evolve_template(
        mutate=lambda parent, n, generation: [],
        score=lambda text: 0.0,
        seed_template=DISCLOSED_TEMPLATE,
        generations=1,
    )
    assert result.winner.text == " ".join(DISCLOSED_TEMPLATE.split())


@pytest.mark.parametrize("bad_seed", ["", "no placeholder at all", "two {Language} {Extra}"])
def test_invalid_seed_is_rejected(bad_seed: str) -> None:
    with pytest.raises(InstructionError):
        evolve_template(
            mutate=lambda parent, n, generation: [], score=lambda text: 0.0,
            seed_template=bad_seed,
        )


def test_zero_generations_are_rejected() -> None:
    with pytest.raises(InstructionError):
        evolve_template(
            mutate=lambda parent, n, generation: [],
            score=lambda text: 0.0,
            generations=0,
        )


# ------------------------------ artifact ------------------------------- #


def test_artifact_is_canonical_json(tmp_path: Path) -> None:
    result = evolve_template(
        mutate=lambda parent, n, generation: ["Improved {Language} template."],
        score=lambda text: 1.0 if "Improved" in text else 0.5,
        seed_template="Seed {Language} template.",
        generations=1,
    )
    path = tmp_path / "artifacts" / "evolved_template.json"
    write_artifact(path, result)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["winner"]["text"] == "Improved {Language} template."
    assert payload["improvement"] == pytest.approx(0.5)
    assert len(payload["generations"]) == 1
    # Canonical: byte-identical to the result's own canonical rendering.
    assert path.read_text(encoding="utf-8") == result.canonical_json() + "\n"
    assert isinstance(result, EvolutionResult)


# ---------------------------- nDCG scorer ------------------------------ #


def test_ndcg_scorer_runs_real_judgments_through_the_live_path(
    built_index: LoadedIndex,
) -> None:
    config = AppConfig(embedding=EmbeddingConfig(provider="hash", dim=64))
    scorer = make_ndcg_scorer(
        built_index,
        _judgments(2),
        processor=TextProcessor(use_spacy=False),
        provider=HashEmbedder(dim=64),
        rrf_config=config.rrf,
        top_k=5,
    )
    first = scorer(DISCLOSED_TEMPLATE)
    assert 0.0 <= first <= 1.0
    assert scorer(DISCLOSED_TEMPLATE) == first  # deterministic


def test_ndcg_scorer_rejects_zero_judgments(built_index: LoadedIndex) -> None:
    config = AppConfig(embedding=EmbeddingConfig(provider="hash", dim=64))
    with pytest.raises(InstructionError):
        make_ndcg_scorer(
            built_index,
            [],
            processor=TextProcessor(use_spacy=False),
            provider=HashEmbedder(dim=64),
            rrf_config=config.rrf,
        )
