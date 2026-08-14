"""The stage-4 entry point: question in, E5 instruction out.

The post says the templates were written and "evolved" by an LLM, and that "getting a
small LLM to write a bespoke one" is worth the same 1-5% as a good template. So the
writing step is a seam, not a function: :class:`InstructionWriter` is a Protocol, and
:class:`DefaultInstructionWriter` — deterministic, template-based, offline — is the
implementation this repo ships and tests. An LLM-backed writer can be dropped into
:class:`InstructionSelector` later without touching anything else in the pipeline.

No LLM is called from this module. Nothing here reaches the network.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 4: "A while back we
    used an LLM to generate and 'evolve' optimal instruction templates for E5 … In our
    internal evals, we have seen that optimizing the instruction - or getting a small
    LLM to write a bespoke one - can yield a free 1-5% improvement in search precision
    and recall." Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - The instruction is written in English regardless of the question's language.
      The disclosed template is English and interpolates the language as a *name*
      inside an English sentence ("relevant {Language} Headlines"), which only parses
      if the surrounding sentence is English too.
    - Language is required, not defaulted per query. It comes from stage 1 (language
      detection), so a caller that has not run stage 1 must say so explicitly by
      constructing the selector with a ``default_language``.
    - A writer that returns a blank instruction is a bug in the writer. The selector
      refuses the result instead of embedding a bare query, so a broken LLM writer
      surfaces at once rather than as a slow retrieval regression.

Alternatives rejected:
    - A single ``write_instruction()`` function with an ``llm=`` flag: fewer moving
      parts, but it would put a network call behind a keyword argument in a module
      whose whole value is being deterministic.
    - An abstract base class instead of a Protocol: forces an import dependency on
      this package for anyone implementing a writer. A Protocol keeps the seam
      structural, matching ``providers/embeddings.py`` in this repo.
    - Caching rendered instructions: they are microseconds of string formatting; a
      cache would add state to an otherwise pure module for no measurable gain.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from cybernaut_mini.query.s4_instruct.e5 import compose_query, format_e5_instruct
from cybernaut_mini.query.s4_instruct.selection import select_template
from cybernaut_mini.query.s4_instruct.templates import (
    InstructionError,
    InstructionTemplate,
    format_entities,
    get_template,
    language_name,
    normalise_text,
)

__all__ = [
    "DefaultInstructionWriter",
    "InstructionRequest",
    "InstructionSelector",
    "InstructionWriter",
]


@dataclass(frozen=True, slots=True)
class InstructionRequest:
    """Everything a writer may condition on. Normalised at construction.

    ``language`` is resolved to an English language name here (``"ja"`` ->
    ``"Japanese"``), so every writer — template-based or LLM-backed — sees the same
    value and cannot leak a raw ISO code into a prompt.
    """

    question: str
    language: str
    entities: tuple[str, ...] = ()
    region: str | None = None
    topic: str | None = None

    def __init__(
        self,
        question: str,
        language: str,
        entities: Iterable[str] = (),
        region: str | None = None,
        topic: str | None = None,
    ) -> None:
        text = normalise_text(question, "question")
        if not text:
            raise InstructionError("question must not be empty")
        cleaned_entities = tuple(
            item for item in (normalise_text(e, "entity") for e in entities) if item
        )
        cleaned_region = normalise_text(region, "region") if region is not None else ""
        cleaned_topic = normalise_text(topic, "topic") if topic is not None else ""
        object.__setattr__(self, "question", text)
        object.__setattr__(self, "language", language_name(language))
        object.__setattr__(self, "entities", cleaned_entities)
        object.__setattr__(self, "region", cleaned_region or None)
        object.__setattr__(self, "topic", cleaned_topic or None)

    def placeholder_values(self) -> dict[str, str]:
        """The dimension bag, keyed by placeholder name, blanks omitted."""
        values: dict[str, str] = {"Language": self.language}
        entities = format_entities(self.entities)
        if entities:
            values["Entities"] = entities
        if self.region:
            values["Region"] = self.region
        if self.topic:
            values["Topic"] = self.topic
        return values


@runtime_checkable
class InstructionWriter(Protocol):
    """Writes the one-sentence instruction E5 conditions on.

    Implementations must be pure with respect to the request: the same
    :class:`InstructionRequest` yields the same string. An LLM-backed writer meets
    this by pinning the model and sampling at temperature 0, or by caching.
    """

    @property
    def identifier(self) -> str:
        """Stable name for traces and eval reports, e.g. ``"template:v1"``."""
        ...

    def write(self, request: InstructionRequest) -> str:
        """Return the instruction text for *request*."""
        ...


@dataclass(frozen=True, slots=True)
class DefaultInstructionWriter:
    """Template-based writer: select a template, render it, done.

    The tested default and the only writer in this package. ``force_template`` pins a
    registry entry — what an ablation ("does the region variant actually help?") and a
    regression test both need.
    """

    force_template: str | None = None

    @property
    def identifier(self) -> str:
        suffix = self.force_template or "auto"
        return f"template:{suffix}"

    def choose(self, request: InstructionRequest) -> InstructionTemplate:
        """Return the template this writer would use for *request*."""
        if self.force_template is not None:
            return get_template(self.force_template)
        return select_template(
            entities=request.entities,
            region=request.region,
            topic=request.topic,
        )

    def write(self, request: InstructionRequest) -> str:
        return self.choose(request).render(request.placeholder_values())


@dataclass(frozen=True, slots=True)
class InstructionSelector:
    """Stage 4's entry point.

    ``select()`` returns the rendered instruction; ``embedding_input()`` returns the
    full E5 query-side string, expansions included. Construct it once and reuse it —
    it holds no mutable state.
    """

    writer: InstructionWriter = field(default_factory=DefaultInstructionWriter)
    default_language: str = "English"

    def request(
        self,
        question: str,
        *,
        language: str | None = None,
        entities: Iterable[str] = (),
        region: str | None = None,
        topic: str | None = None,
    ) -> InstructionRequest:
        """Build a normalised :class:`InstructionRequest` for these inputs."""
        return InstructionRequest(
            question=question,
            language=(
                language
                if language is not None and normalise_text(language, "language")
                else self.default_language
            ),
            entities=entities,
            region=region,
            topic=topic,
        )

    def select(
        self,
        question: str,
        *,
        language: str | None = None,
        entities: Iterable[str] = (),
        region: str | None = None,
        topic: str | None = None,
    ) -> str:
        """Return the rendered instruction string for this question."""
        return self.write(
            self.request(
                question,
                language=language,
                entities=entities,
                region=region,
                topic=topic,
            )
        )

    def write(self, request: InstructionRequest) -> str:
        """Run the configured writer and reject an unusable result."""
        instruction = self.writer.write(request)
        if not instruction or not instruction.strip():
            raise InstructionError(
                f"instruction writer {self.writer.identifier!r} returned an empty instruction"
            )
        return " ".join(instruction.split())

    def embedding_input(
        self,
        question: str,
        *,
        language: str | None = None,
        entities: Iterable[str] = (),
        region: str | None = None,
        topic: str | None = None,
        expansions: Sequence[str] = (),
    ) -> str:
        """Return the exact string to hand to a ``multilingual-e5-*-instruct`` encoder.

        ``"Instruct: {instruction}\\nQuery: {question}; {expansions…}"`` — the
        instruction from the configured writer, the query from
        :func:`~cybernaut_mini.query.s4_instruct.e5.compose_query`.
        """
        request = self.request(
            question,
            language=language,
            entities=entities,
            region=region,
            topic=topic,
        )
        instruction = self.write(request)
        return format_e5_instruct(instruction, compose_query(request.question, expansions))
