"""Stage 4 of Hybrid-3 — Instruction Tuning: writing E5's instruction for a question.

The query pipeline's fourth stage generates an instruction for the question and sends
it, with the stage-3 expansions, to ``intfloat/multilingual-e5-large-instruct``. This
package is the *instruction* half of that stage: templates, a strict renderer, a
deterministic template chooser, the exact E5 wire format, and a pluggable writer seam
for the LLM-written instructions the post describes. The encoders themselves live in
``cybernaut_mini.providers.embeddings``; nothing here loads a model or opens a socket.

Typical use::

    selector = InstructionSelector()
    selector.select("Who is buying ASML machines?", language="en", region="Taiwan")
    selector.embedding_input("...", language="en", expansions=["euv lithography"])

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 4, "Instruction
    Tuning and Embedding": one template is disclosed verbatim, plus the fact that
    templates carry placeholders for named entities, geographic regions, topics of
    interest and other dimensions, that an LLM evolved them, and that a good
    instruction is worth 1-5% in precision and recall. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions: the split between "write the instruction" and "encode the text" is this
    repo's, not the post's — the post treats stage 4 as one step. Splitting it keeps
    the instruction logic testable with no model weights and no network.
Alternatives rejected: putting instruction rendering inside the embedding provider
    (fewer files, but then every instruction test needs an encoder), and shipping the
    templates as configuration (see ``templates.py`` for why they live in code).
"""

from __future__ import annotations

from cybernaut_mini.query.s4_instruct.e5 import (
    E5_INSTRUCT_WIRE_FORMAT,
    compose_query,
    format_e5_document,
    format_e5_instruct,
)
from cybernaut_mini.query.s4_instruct.selection import (
    DIMENSION_WEIGHTS,
    DIMENSIONS,
    query_dimensions,
    select_template,
)
from cybernaut_mini.query.s4_instruct.templates import (
    DEFAULT_TEMPLATE_NAME,
    DISCLOSED_TEMPLATE,
    LANGUAGE_NAMES,
    TEMPLATES,
    InstructionError,
    InstructionTemplate,
    MissingPlaceholderError,
    Provenance,
    TemplateDefinitionError,
    UnknownLanguageError,
    format_entities,
    get_template,
    language_name,
    normalise_text,
)
from cybernaut_mini.query.s4_instruct.writer import (
    DefaultInstructionWriter,
    InstructionRequest,
    InstructionSelector,
    InstructionWriter,
)

__all__ = [
    "DEFAULT_TEMPLATE_NAME",
    "DIMENSIONS",
    "DIMENSION_WEIGHTS",
    "DISCLOSED_TEMPLATE",
    "E5_INSTRUCT_WIRE_FORMAT",
    "LANGUAGE_NAMES",
    "TEMPLATES",
    "DefaultInstructionWriter",
    "InstructionError",
    "InstructionRequest",
    "InstructionSelector",
    "InstructionTemplate",
    "InstructionWriter",
    "MissingPlaceholderError",
    "Provenance",
    "TemplateDefinitionError",
    "UnknownLanguageError",
    "compose_query",
    "format_e5_document",
    "format_e5_instruct",
    "format_entities",
    "get_template",
    "language_name",
    "normalise_text",
    "query_dimensions",
    "select_template",
]
