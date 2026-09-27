"""Stage 4's LLM half: a bespoke-instruction writer and template evolution.

Two things the post describes and this package only had seams for:

* :class:`LLMInstructionWriter` — an :class:`~cybernaut_mini.query.s4_instruct
  .writer.InstructionWriter` that asks a small local LLM to write a bespoke
  one-sentence instruction for the question ("getting a small LLM to write a
  bespoke one"), reusing the agent's local generator provider
  (:meth:`~cybernaut_mini.providers.query_generator.LLMQueryGenerator.chat`)
  rather than loading a second model, and falling back to the deterministic
  template writer whenever the model's output is unusable;
* :func:`evolve_template` — the hill-climb loop that "evolved" the templates:
  seed with the disclosed template, have the LLM propose N mutations per
  generation, score each candidate by real retrieval quality (nDCG@10 against
  MIRACL qrels via :func:`make_ndcg_scorer`), keep the winner, repeat. The full
  score history persists as a canonical-JSON artifact.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 4: "A while
    back we used an LLM to generate and 'evolve' optimal instruction templates
    for E5 … In our internal evals, we have seen that optimizing the instruction
    - or getting a small LLM to write a bespoke one - can yield a free 1-5%
    improvement in search precision and recall." Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - The post gives neither population size nor generation count; N=5 mutations
      per generation for 5 generations is the build plan's laptop-scale setting
      and both are parameters.
    - [inferred] evolution is restricted to templates whose only placeholder is
      ``{Language}`` — the disclosed seed's shape. Mutating a template into
      requiring ``{Entities}`` would make its score depend on an entity extractor
      that the offline eval queries do not exercise, so such mutations are
      discarded as invalid rather than scored misleadingly.
    - Scoring uses **real qrels only**: :func:`make_ndcg_scorer` takes
      :class:`~cybernaut_mini.models.Judgment` objects (MIRACL dev judgments in
      this repo's fixtures) and runs the real retrieval path with the candidate
      instruction as the embedding input. No synthetic relevance anywhere.
    - Hill climbing replaces the incumbent only on a strictly better score, so
      ties keep the earlier (more disclosed-like) template and the loop is
      deterministic given a deterministic mutator and scorer.

Alternatives rejected:
    - A population-based GA (crossover, tournament selection): the post says
      "evolve" but discloses one winning template, not a population; hill climbing
      is the smallest loop that reproduces the claim and its artifact.
    - Scoring by embedding-space similarity to the disclosed template: circular
      and free of retrieval evidence; the 1-5% claim is about precision/recall.
    - Running the evolution at import or build time: it costs dozens of retrieval
      sweeps; it is a tool invoked deliberately, and its unit tests stub both the
      mutator and the scorer.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from cybernaut_mini.models import canonical_dumps
from cybernaut_mini.query.s4_instruct.e5 import format_e5_instruct
from cybernaut_mini.query.s4_instruct.templates import (
    DISCLOSED_TEMPLATE,
    InstructionError,
    language_name,
)
from cybernaut_mini.query.s4_instruct.writer import (
    DefaultInstructionWriter,
    InstructionRequest,
    InstructionWriter,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from cybernaut_mini.config import RRFConfig
    from cybernaut_mini.indexing import LoadedIndex
    from cybernaut_mini.models import Judgment
    from cybernaut_mini.providers.embeddings import EmbeddingProvider
    from cybernaut_mini.text import TextProcessor

__all__ = [
    "DEFAULT_CHILDREN",
    "DEFAULT_GENERATIONS",
    "EvolutionResult",
    "Generation",
    "LLMInstructionWriter",
    "ScoredTemplate",
    "evolve_template",
    "llm_mutator",
    "make_ndcg_scorer",
    "write_artifact",
]

#: The post gives no numbers; these are the laptop-scale loop from the build plan.
DEFAULT_GENERATIONS = 5
DEFAULT_CHILDREN = 5

#: The only placeholder an evolvable template may carry — the disclosed seed's shape.
_ALLOWED_PLACEHOLDERS = frozenset({"Language"})

_WRITER_SYSTEM_PROMPT = (
    "You write one-sentence retrieval instructions for a multilingual E5 embedding "
    "model. Answer with the instruction sentence only: no quotes, no numbering, no "
    "explanations."
)

_MUTATOR_SYSTEM_PROMPT = (
    "You improve instruction templates for a multilingual E5 retrieval model. Each "
    "template is a single English sentence containing the literal placeholder "
    "{Language}. Answer with the rewritten templates only, one per line: no "
    "numbering, no bullets, no quotes, no explanations. Every line must contain "
    "{Language} exactly as written."
)


class ChatFn(Protocol):
    """One chat turn: ``(system, user) -> assistant text``.

    :meth:`cybernaut_mini.providers.query_generator.LLMQueryGenerator.chat`
    satisfies this, which is how both the writer and the mutator reuse the
    agent's local model.
    """

    def __call__(self, system: str, user: str) -> str: ...


# ------------------------------------------------------------------ #
# Bespoke instruction writer                                          #
# ------------------------------------------------------------------ #


@dataclass(frozen=True, slots=True)
class LLMInstructionWriter:
    """Instruction writer backed by a small local LLM, template fallback built in.

    Satisfies the :class:`InstructionWriter` protocol. A blank, crashed or
    over-long generation degrades to :class:`DefaultInstructionWriter`'s rendered
    template — the selector never sees an empty instruction, and a broken model
    demotes retrieval to the tested template baseline instead of failing it.
    """

    chat_fn: ChatFn
    fallback: InstructionWriter = field(default_factory=DefaultInstructionWriter)
    model_label: str = "injected"
    #: A generation longer than this is the model explaining itself, not an
    #: instruction; the disclosed template is 214 characters.
    max_chars: int = 400

    @property
    def identifier(self) -> str:
        return f"llm:{self.model_label}"

    def write(self, request: InstructionRequest) -> str:
        try:
            raw = self.chat_fn(_WRITER_SYSTEM_PROMPT, self._user_prompt(request))
        except Exception:
            return self.fallback.write(request)
        instruction = " ".join(raw.split()).strip("\"'` ")
        if not instruction or len(instruction) > self.max_chars:
            return self.fallback.write(request)
        return instruction

    @staticmethod
    def _user_prompt(request: InstructionRequest) -> str:
        lines = [
            f"Question: {request.question}",
            f"Results language: {request.language}",
        ]
        if request.entities:
            lines.append("Named entities: " + ", ".join(request.entities))
        if request.region:
            lines.append(f"Region: {request.region}")
        if request.topic:
            lines.append(f"Topic: {request.topic}")
        lines.append(
            "Write one sentence instructing the retriever what to retrieve for "
            "this question."
        )
        return "\n".join(lines)


# ------------------------------------------------------------------ #
# Template validation and mutation                                    #
# ------------------------------------------------------------------ #


def _placeholders(template: str) -> tuple[str, ...] | None:
    """Placeholder names in *template*, or ``None`` when the braces are malformed."""
    names: list[str] = []
    try:
        fields = list(string.Formatter().parse(template))
    except ValueError:
        return None
    for _literal, field_name, _spec, _conv in fields:
        if field_name is None:
            continue
        if not field_name or field_name.isdigit():
            return None
        if field_name not in names:
            names.append(field_name)
    return tuple(names)


def _valid_candidate(text: str) -> str | None:
    """Normalise a proposed template; ``None`` when it cannot be evolved.

    Valid means: one non-empty line once whitespace-collapsed, well-formed braces,
    and exactly the ``{Language}`` placeholder (see the module assumptions).
    """
    cleaned = " ".join(text.split()).strip("\"'` ")
    if not cleaned:
        return None
    found = _placeholders(cleaned)
    if found is None or set(found) != _ALLOWED_PLACEHOLDERS:
        return None
    return cleaned


def llm_mutator(chat_fn: ChatFn) -> Callable[[str, int, int], list[str]]:
    """A mutation operator backed by the local generator provider.

    Returns ``mutate(parent_text, n, generation) -> proposals`` (raw; the loop
    validates). A crashed generation proposes nothing — the incumbent survives
    the generation, it does not kill the run.
    """

    def mutate(parent: str, n: int, generation: int) -> list[str]:
        user = (
            f"Current best template (generation {generation}):\n{parent}\n"
            f"Write {n} improved variants of this template, one per line. Keep each "
            "a single sentence and keep the {Language} placeholder."
        )
        try:
            raw = chat_fn(_MUTATOR_SYSTEM_PROMPT, user)
        except Exception:
            return []
        return [line for line in (ln.strip() for ln in raw.splitlines()) if line]

    return mutate


# ------------------------------------------------------------------ #
# Scoring                                                             #
# ------------------------------------------------------------------ #


def make_ndcg_scorer(
    index: LoadedIndex,
    judgments: Sequence[Judgment],
    *,
    processor: TextProcessor,
    provider: EmbeddingProvider,
    rrf_config: RRFConfig,
    language: str = "en",
    top_k: int = 10,
    mode: str = "hybrid",
) -> Callable[[str], float]:
    """Score a template by mean nDCG@``top_k`` over real qrels.

    For each judgment the candidate template is rendered (``{Language}`` filled
    with the English name of *language*), composed into the E5-instruct wire
    format with the judged question, and handed to the real retrieval path as
    the embedding input. Deterministic for deterministic providers.
    """
    from cybernaut_mini.evals import ndcg_at_k
    from cybernaut_mini.retrieval import retrieve

    if not judgments:
        raise InstructionError("cannot build a scorer from zero judgments")
    resolved_language = language_name(language)

    def score(template_text: str) -> float:
        cleaned = _valid_candidate(template_text)
        if cleaned is None:
            raise InstructionError(f"template is not evolvable: {template_text!r}")
        instruction = cleaned.format(Language=resolved_language)
        total = 0.0
        for judgment in judgments:
            wire = format_e5_instruct(instruction, judgment.question)
            hits = retrieve(
                index,
                judgment.question,
                mode=mode,  # type: ignore[arg-type]
                processor=processor,
                provider=provider,
                rrf_config=rrf_config,
                top_k=top_k,
                embedding_text=wire,
            )
            ranked = [hit.document.id for hit in hits]
            total += ndcg_at_k(ranked, judgment.relevant_document_ids, top_k)
        return total / len(judgments)

    return score


# ------------------------------------------------------------------ #
# The hill-climb loop and its artifact                                #
# ------------------------------------------------------------------ #


@dataclass(frozen=True, slots=True)
class ScoredTemplate:
    """One evaluated template."""

    text: str
    score: float
    origin: str  # "seed" | "mutation"

    def to_payload(self) -> dict[str, Any]:
        return {"text": self.text, "score": self.score, "origin": self.origin}


@dataclass(frozen=True, slots=True)
class Generation:
    """One generation's candidates and the incumbent after it."""

    index: int
    candidates: tuple[ScoredTemplate, ...]
    best: ScoredTemplate

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "candidates": [candidate.to_payload() for candidate in self.candidates],
            "best": self.best.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    """The whole run: seed, winner, and the per-generation score history."""

    seed: ScoredTemplate
    winner: ScoredTemplate
    generations: tuple[Generation, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed.to_payload(),
            "winner": self.winner.to_payload(),
            "generations": [generation.to_payload() for generation in self.generations],
            "improvement": self.winner.score - self.seed.score,
        }

    def canonical_json(self) -> str:
        return canonical_dumps(self.to_payload())


def evolve_template(
    *,
    mutate: Callable[[str, int, int], list[str]],
    score: Callable[[str], float],
    seed_template: str = DISCLOSED_TEMPLATE,
    generations: int = DEFAULT_GENERATIONS,
    children: int = DEFAULT_CHILDREN,
) -> EvolutionResult:
    """Hill-climb *seed_template* for *generations* rounds of *children* mutations.

    Each generation asks ``mutate`` for ``children`` proposals of the incumbent,
    validates them (single line, ``{Language}`` only), deduplicates them
    case-insensitively against each other and the incumbent, scores the
    survivors, and keeps the best strictly-improving candidate. A generation
    whose proposals are all invalid or all worse leaves the incumbent standing.
    """
    if generations < 1 or children < 1:
        raise InstructionError("generations and children must both be >= 1")
    seed_clean = _valid_candidate(seed_template)
    if seed_clean is None:
        raise InstructionError(f"seed template is not evolvable: {seed_template!r}")

    best = ScoredTemplate(text=seed_clean, score=score(seed_clean), origin="seed")
    seed = best
    history: list[Generation] = []

    for generation in range(generations):
        seen: set[str] = {best.text.casefold()}
        candidates: list[ScoredTemplate] = []
        for proposal in mutate(best.text, children, generation):
            cleaned = _valid_candidate(proposal)
            if cleaned is None or cleaned.casefold() in seen:
                continue
            seen.add(cleaned.casefold())
            candidates.append(
                ScoredTemplate(text=cleaned, score=score(cleaned), origin="mutation")
            )
            if len(candidates) >= children:
                break
        for candidate in candidates:
            # Strictly better only: ties keep the earlier, more disclosed-like text.
            if candidate.score > best.score:
                best = candidate
        history.append(
            Generation(index=generation, candidates=tuple(candidates), best=best)
        )

    return EvolutionResult(seed=seed, winner=best, generations=tuple(history))


def write_artifact(path: Path, result: EvolutionResult) -> None:
    """Persist the evolution history as a canonical-JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.canonical_json() + "\n", encoding="utf-8")
