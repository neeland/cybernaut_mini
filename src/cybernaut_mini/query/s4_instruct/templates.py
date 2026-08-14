"""Instruction templates for E5, and the strict renderer that fills them.

Stage 4 of the Hybrid-3 query pipeline is *instruction tuning*: before the question
is embedded, an instruction is written for it. This module holds the instruction
half — the template objects, the registry, and a formatter that fails loudly rather
than shipping a half-filled prompt to the encoder. Nothing here touches an encoder;
the embedding providers live elsewhere.

The registry's default entry is the post's own template, quoted character for
character and tagged ``Provenance.DISCLOSED``. Every other entry is
``Provenance.INFERRED``: the post says the templates carry placeholders for named
entities, geographic regions and topics of interest, but prints only the one, so
the variants below are this repo's reconstruction of that shape, not NOSIBLE's text.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 4, "Instruction
    Tuning and Embedding": "we generate an appropriate instruction for the question
    and submit it, along with the expansions, to multilingual-e5-large-instruct …
    We have templates that have placeholders for named entities, geographic regions,
    topics of interest, and other dimensions." Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - The disclosed template's ``{Language}`` is a language *name* in English
      ("English", "Japanese"), not an ISO code. The post shows the placeholder but
      never a filled example; a rendered "relevant ja Headlines" would be a silently
      degraded prompt, so :func:`language_name` maps codes to names and refuses
      unknown codes instead of guessing.
    - A blank placeholder value is an error, not an empty string. Rendering
      "focus on the named entities , and provide substantive answers" is worse than
      raising, because it is invisible in the retrieval metrics.
    - The variants keep the disclosed template's skeleton ("Given a question, please
      retrieve any relevant {Language} Headlines, Leads, Passages, and Source URLs …
      and provide substantive answers to the question") and vary only the focus
      clause. The post attributes its gains to *evolved* templates, so drifting the
      whole sentence would discard the one datapoint we have about what works.
    - Placeholders are capitalised (``{Entities}``, ``{Region}``, ``{Topic}``) to
      match the disclosed ``{Language}``.

Alternatives rejected:
    - ``str.format`` straight on the template: one missing key raises ``KeyError``
      with no context, and — worse — ``string.Template.safe_substitute``-style
      leniency would emit a literal "{Language}" into the prompt. The renderer here
      pre-checks the required set and names every missing field.
    - Free-text templates stored in ``conf/``: better for tuning without a deploy,
      and where this would go in production. Rejected here because the templates are
      part of the replica's *claim* about the post — they belong next to the citation
      that justifies them, under test, not in an editable YAML file.
    - Generating variants combinatorially from clause fragments: fewer literals, but
      the rendered English degrades at the joins and no single string is reviewable
      against the post. Each template is written out in full instead.
"""

from __future__ import annotations

import string
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "DEFAULT_TEMPLATE_NAME",
    "DISCLOSED_TEMPLATE",
    "LANGUAGE_NAMES",
    "TEMPLATES",
    "InstructionError",
    "InstructionTemplate",
    "MissingPlaceholderError",
    "Provenance",
    "TemplateDefinitionError",
    "UnknownLanguageError",
    "format_entities",
    "get_template",
    "language_name",
    "normalise_text",
]


class InstructionError(ValueError):
    """Base class for every error this stage raises."""


class TemplateDefinitionError(InstructionError):
    """A template's declared placeholders disagree with its text."""


class MissingPlaceholderError(InstructionError):
    """A required placeholder was absent or blank at render time."""

    def __init__(self, template_name: str, missing: Iterable[str]) -> None:
        self.template_name = template_name
        self.missing = tuple(sorted(missing))
        joined = ", ".join(self.missing)
        super().__init__(
            f"template {template_name!r} cannot be rendered: "
            f"missing or blank placeholder(s): {joined}"
        )


class UnknownLanguageError(InstructionError):
    """A language code was not recognised, so no English name could be filled in."""


class Provenance(StrEnum):
    """Whether a template is quoted from the post or reconstructed by this repo."""

    DISCLOSED = "disclosed"
    INFERRED = "inferred"


_FORMATTER = string.Formatter()


def _parse_placeholders(template: str) -> tuple[str, ...]:
    """Return the placeholder names in *template*, in order of first appearance.

    Raises :class:`TemplateDefinitionError` for malformed braces and for positional
    or auto-numbered fields (``{}`` / ``{0}``), which have no place in a named
    instruction template.
    """
    seen: list[str] = []
    try:
        fields = list(_FORMATTER.parse(template))
    except ValueError as exc:  # unbalanced or stray braces
        raise TemplateDefinitionError(f"malformed template: {exc}") from exc
    for _literal, field_name, _spec, _conv in fields:
        if field_name is None:
            continue
        if field_name == "" or field_name.isdigit():
            raise TemplateDefinitionError(
                f"positional placeholder {{{field_name}}} is not allowed; use a name"
            )
        # Strip attribute/index access: {Topic.title} would not survive review anyway.
        name = field_name.split(".")[0].split("[")[0]
        if name != field_name:
            raise TemplateDefinitionError(
                f"placeholder {{{field_name}}} uses attribute or index access; use a plain name"
            )
        if name not in seen:
            seen.append(name)
    return tuple(seen)


@dataclass(frozen=True, slots=True)
class InstructionTemplate:
    """One E5 instruction template.

    Attributes
    ----------
    name:
        Registry key. Stable: it is what :func:`select_template` returns and what a
        trace or an eval report names.
    template:
        The instruction text, with ``{Name}`` placeholders.
    required_placeholders:
        The placeholders the caller must supply. Declared explicitly rather than
        derived, so a typo in the text (``{Langauge}``) fails at import time instead
        of at the first query that needs it.
    provenance:
        :attr:`Provenance.DISCLOSED` only for text quoted from the post.
    dimensions:
        The query dimensions this template is *for* — the routing key
        :func:`select_template` matches against. Empty for the default template.
    source:
        Human-readable citation, printed in ``__str__`` of errors during review.
    """

    name: str
    template: str
    required_placeholders: tuple[str, ...]
    provenance: Provenance
    dimensions: frozenset[str] = frozenset()
    source: str = "docs/blog-archive/the-road-to-cybernaut-1.md — stage 4"

    def __post_init__(self) -> None:
        found = _parse_placeholders(self.template)
        declared = tuple(self.required_placeholders)
        if set(found) != set(declared):
            raise TemplateDefinitionError(
                f"template {self.name!r} declares placeholders {sorted(declared)} "
                f"but its text contains {sorted(found)}"
            )
        if len(set(declared)) != len(declared):
            raise TemplateDefinitionError(
                f"template {self.name!r} declares a duplicate placeholder: {sorted(declared)}"
            )
        if not self.template.strip():
            raise TemplateDefinitionError(f"template {self.name!r} is empty")

    def render(self, values: Mapping[str, str] | None = None, /, **overrides: str) -> str:
        """Fill the template, raising rather than emitting an unfilled placeholder.

        Values may be passed as a mapping, as keyword arguments, or both (keywords
        win). Keys the template does not use are ignored, so a caller may pass the
        whole dimension bag to any template. A required value that is missing, or
        that is blank once stripped, raises :class:`MissingPlaceholderError` naming
        every offending placeholder at once.
        """
        merged: dict[str, str] = dict(values or {})
        merged.update(overrides)

        missing = [
            key
            for key in self.required_placeholders
            if key not in merged or not str(merged[key]).strip()
        ]
        if missing:
            raise MissingPlaceholderError(self.name, missing)

        supplied = {key: str(merged[key]).strip() for key in self.required_placeholders}
        return self.template.format(**supplied)


# --------------------------------------------------------------------------------
# The registry.
#
# ``DISCLOSED_TEMPLATE`` below is the post's sentence typed out in full — the one
# string in this package that is quoted rather than invented. ``test_s4_instruct.py``
# greps the archived post for it, so it cannot drift from its source silently.
#
# ``_SKELETON`` is that same sentence with its focus clause factored out; it exists
# only so the [inferred] variants keep the disclosed shape at the joins. It is
# asserted equal to ``DISCLOSED_TEMPLATE`` at import time.
# --------------------------------------------------------------------------------

#: [disclosed] — "Here is one of the most successful templates:" (stage 4 of the post),
#: quoted character for character. Local copy:
#: ``docs/blog-archive/the-road-to-cybernaut-1.md``.
DISCLOSED_TEMPLATE = (
    "Given a question, please retrieve any relevant {Language} Headlines, Leads, "
    "Passages, and Source URLs that focus on the same named entities as the "
    "question, and provide substantive answers to the question."
)

#: The disclosed sentence with its focus clause replaced by ``{focus}``. Used only to
#: build the [inferred] variants below.
_SKELETON = (
    "Given a question, please retrieve any relevant {Language} Headlines, Leads, "
    "Passages, and Source URLs that {focus}, and provide substantive answers to "
    "the question."
)


def _build(focus: str) -> str:
    return _SKELETON.replace("{focus}", focus)


if _build("focus on the same named entities as the question") != DISCLOSED_TEMPLATE:
    # The variants would silently stop matching the disclosed shape. Fail at import.
    raise TemplateDefinitionError(
        "the instruction skeleton no longer reproduces the disclosed template"
    )

#: The default: the one template the post prints, quoted verbatim. Cited [disclosed].
DEFAULT_TEMPLATE_NAME = "entity_focus"

_TEMPLATE_LIST: tuple[InstructionTemplate, ...] = (
    InstructionTemplate(
        name=DEFAULT_TEMPLATE_NAME,
        template=DISCLOSED_TEMPLATE,
        required_placeholders=("Language",),
        provenance=Provenance.DISCLOSED,
        dimensions=frozenset(),
        source="the-road-to-cybernaut-1.md, stage 4, quoted verbatim",
    ),
    InstructionTemplate(
        name="named_entities",
        # [inferred] — the post names "named entities" as a placeholder dimension.
        template=_build("focus on the named entities {Entities}"),
        required_placeholders=("Language", "Entities"),
        provenance=Provenance.INFERRED,
        dimensions=frozenset({"entities"}),
    ),
    InstructionTemplate(
        name="geographic_region",
        # [inferred] — "geographic regions" is a named placeholder dimension.
        template=_build("report on events in {Region}"),
        required_placeholders=("Language", "Region"),
        provenance=Provenance.INFERRED,
        dimensions=frozenset({"region"}),
    ),
    InstructionTemplate(
        name="topic",
        # [inferred] — "topics of interest" is a named placeholder dimension.
        template=_build("cover the topic of {Topic}"),
        required_placeholders=("Language", "Topic"),
        provenance=Provenance.INFERRED,
        dimensions=frozenset({"topic"}),
    ),
    InstructionTemplate(
        name="entities_in_region",
        # [inferred] — two disclosed dimensions combined.
        template=_build("focus on the named entities {Entities} as reported in {Region}"),
        required_placeholders=("Language", "Entities", "Region"),
        provenance=Provenance.INFERRED,
        dimensions=frozenset({"entities", "region"}),
    ),
    InstructionTemplate(
        name="entities_on_topic",
        # [inferred] — two disclosed dimensions combined.
        template=_build("focus on the named entities {Entities} in the context of {Topic}"),
        required_placeholders=("Language", "Entities", "Topic"),
        provenance=Provenance.INFERRED,
        dimensions=frozenset({"entities", "topic"}),
    ),
    InstructionTemplate(
        name="topic_in_region",
        # [inferred] — two disclosed dimensions combined.
        template=_build("cover the topic of {Topic} as reported in {Region}"),
        required_placeholders=("Language", "Topic", "Region"),
        provenance=Provenance.INFERRED,
        dimensions=frozenset({"topic", "region"}),
    ),
    InstructionTemplate(
        name="entities_on_topic_in_region",
        # [inferred] — all three disclosed dimensions at once.
        template=_build(
            "focus on the named entities {Entities} in the context of {Topic} "
            "as reported in {Region}"
        ),
        required_placeholders=("Language", "Entities", "Topic", "Region"),
        provenance=Provenance.INFERRED,
        dimensions=frozenset({"entities", "topic", "region"}),
    ),
)

#: Name -> template. Insertion-ordered, so iteration is deterministic.
TEMPLATES: Mapping[str, InstructionTemplate] = {t.name: t for t in _TEMPLATE_LIST}


def get_template(name: str) -> InstructionTemplate:
    """Return the registry entry *name*, or raise with the list of valid names."""
    try:
        return TEMPLATES[name]
    except KeyError:
        known = ", ".join(sorted(TEMPLATES))
        raise InstructionError(f"unknown instruction template {name!r}; known: {known}") from None


# --------------------------------------------------------------------------------
# Placeholder value helpers.
# --------------------------------------------------------------------------------

#: ISO 639-1 -> English language name, covering the languages this repo's stage-1
#: language detector reports most often. Deliberately not exhaustive: an unknown
#: code raises rather than leaking a two-letter code into the prompt.
LANGUAGE_NAMES: Mapping[str, str] = {
    "ar": "Arabic",
    "bn": "Bengali",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


def language_name(value: str) -> str:
    """Return the English language name for *value*.

    Accepts an ISO 639-1 code (``"en"``, ``"zh-Hans"`` — the subtag is dropped) or a
    name that is already spelled out (``"English"``, returned unchanged). Raises
    :class:`UnknownLanguageError` for a short token that is not in
    :data:`LANGUAGE_NAMES`, because filling ``{Language}`` with a raw code produces a
    grammatically broken instruction that no metric would flag.
    """
    text = normalise_text(value, "language")
    if not text:
        raise UnknownLanguageError("language must not be blank")
    primary = text.replace("_", "-").split("-")[0]
    mapped = LANGUAGE_NAMES.get(primary.lower())
    if mapped is not None:
        return mapped
    if len(primary) <= 3 and primary.isascii() and primary.isalpha():
        known = ", ".join(sorted(LANGUAGE_NAMES))
        raise UnknownLanguageError(
            f"unknown language code {value!r}; pass a spelled-out name or one of: {known}"
        )
    # Already a name ("English", "Brazilian Portuguese", "Português"): trust the caller.
    return text


def normalise_text(value: object, field: str) -> str:
    """Whitespace-collapse *value*, refusing anything that is not a string.

    ``str(value)`` is deliberately *not* used: coercing ``None`` gives the literal
    ``"None"``, which renders as "focus on the named entities None" and is invisible
    in every retrieval metric. A wrong type is a caller bug, so it raises.
    """
    if not isinstance(value, str):
        raise InstructionError(f"{field} must be a string, got {type(value).__name__}: {value!r}")
    return " ".join(value.split())


def format_entities(entities: Iterable[str]) -> str:
    """Render an entity list for a prompt: ``"Nvidia, TSMC and ASML"``.

    Order is the caller's; duplicates are dropped case-insensitively keeping the
    first spelling, so the output depends only on the input sequence — never on set
    iteration order. A non-string entity raises :class:`InstructionError`.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in entities:
        item = normalise_text(raw, "entity")
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return f"{', '.join(cleaned[:-1])} and {cleaned[-1]}"
