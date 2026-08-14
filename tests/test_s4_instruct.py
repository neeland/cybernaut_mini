"""Tests for blog stage 4 — instruction tuning (``cybernaut_mini.query.s4_instruct``).

Offline and key-free by construction: this stage is string handling, and the one
external fact it depends on (E5's wire format) is asserted against a literal, not a
download. The disclosed template is asserted against the archived post itself, so the
replica cannot drift away from its source without a red test.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from cybernaut_mini.query.s4_instruct import (
    DEFAULT_TEMPLATE_NAME,
    DISCLOSED_TEMPLATE,
    E5_INSTRUCT_WIRE_FORMAT,
    TEMPLATES,
    DefaultInstructionWriter,
    InstructionError,
    InstructionRequest,
    InstructionSelector,
    InstructionTemplate,
    InstructionWriter,
    MissingPlaceholderError,
    Provenance,
    TemplateDefinitionError,
    UnknownLanguageError,
    compose_query,
    format_e5_document,
    format_e5_instruct,
    format_entities,
    get_template,
    language_name,
    query_dimensions,
    select_template,
)

BLOG = Path(__file__).resolve().parents[1] / "docs/blog-archive/the-road-to-cybernaut-1.md"

#: The post's template, retyped independently of the source module so a test can pin
#: the registry against a literal here as well as against the archive file. If someone
#: "tidies" the template in ``templates.py``, this copy is what goes red.
POST_TEMPLATE = (
    "Given a question, please retrieve any relevant {Language} Headlines, Leads, "
    "Passages, and Source URLs that focus on the same named entities as the "
    "question, and provide substantive answers to the question."
)

RENDERED_ENGLISH = POST_TEMPLATE.replace("{Language}", "English")


# ---------------------------------------------------------------------------
# The disclosed template
# ---------------------------------------------------------------------------


def test_disclosed_template_matches_the_archived_post() -> None:
    """The default template is the post's, character for character."""
    text = BLOG.read_text(encoding="utf-8")
    quoted = [
        line.strip().lstrip("> ").strip().strip('"')
        for line in text.splitlines()
        if "Given a question, please retrieve" in line
    ]
    assert quoted, "the archived post no longer contains the stage-4 template"
    assert TEMPLATES[DEFAULT_TEMPLATE_NAME].template == quoted[0]


def test_default_template_is_the_disclosed_literal() -> None:
    assert TEMPLATES[DEFAULT_TEMPLATE_NAME].template == POST_TEMPLATE
    assert DISCLOSED_TEMPLATE == POST_TEMPLATE
    assert TEMPLATES[DEFAULT_TEMPLATE_NAME].provenance is Provenance.DISCLOSED


def test_disclosed_template_renders_verbatim_for_english() -> None:
    rendered = TEMPLATES[DEFAULT_TEMPLATE_NAME].render(Language="English")
    assert rendered == RENDERED_ENGLISH
    assert "{" not in rendered and "}" not in rendered


def test_exactly_one_template_is_disclosed() -> None:
    """The post prints one template. Anything else in the registry is inferred."""
    disclosed = [t.name for t in TEMPLATES.values() if t.provenance is Provenance.DISCLOSED]
    assert disclosed == [DEFAULT_TEMPLATE_NAME]


# ---------------------------------------------------------------------------
# Registry health
# ---------------------------------------------------------------------------


def test_every_entry_is_tagged_and_renders() -> None:
    """Every registry entry is disclosed-or-inferred and renders without error."""
    values = {
        "Language": "English",
        "Entities": "Nvidia and TSMC",
        "Region": "Taiwan",
        "Topic": "export controls",
    }
    assert TEMPLATES, "registry must not be empty"
    for name, template in TEMPLATES.items():
        assert template.name == name
        assert template.provenance in (Provenance.DISCLOSED, Provenance.INFERRED)
        assert template.source
        rendered = template.render(values)
        assert "{" not in rendered and "}" not in rendered
        assert rendered.endswith("provide substantive answers to the question.")
        # Only placeholders the template declares may appear in its text.
        assert set(re.findall(r"{(\w+)}", template.template)) == set(template.required_placeholders)
        # Every declared placeholder is one the dimension bag can actually fill.
        assert set(template.required_placeholders) <= set(values)


def test_template_dimensions_are_distinct_so_selection_is_unambiguous() -> None:
    seen = [tuple(sorted(t.dimensions)) for t in TEMPLATES.values()]
    assert len(seen) == len(set(seen))


def test_get_template_rejects_unknown_names() -> None:
    with pytest.raises(InstructionError, match="unknown instruction template"):
        get_template("no_such_template")


# ---------------------------------------------------------------------------
# The strict renderer
# ---------------------------------------------------------------------------


def test_missing_placeholder_raises_instead_of_leaking_braces() -> None:
    with pytest.raises(MissingPlaceholderError) as excinfo:
        TEMPLATES[DEFAULT_TEMPLATE_NAME].render()
    assert excinfo.value.missing == ("Language",)
    assert "Language" in str(excinfo.value)


def test_blank_placeholder_is_treated_as_missing() -> None:
    with pytest.raises(MissingPlaceholderError):
        TEMPLATES["named_entities"].render(Language="English", Entities="   ")


def test_all_missing_placeholders_are_reported_at_once() -> None:
    with pytest.raises(MissingPlaceholderError) as excinfo:
        TEMPLATES["entities_on_topic_in_region"].render(Language="English")
    assert excinfo.value.missing == ("Entities", "Region", "Topic")


def test_render_ignores_extra_values_and_strips_supplied_ones() -> None:
    rendered = TEMPLATES[DEFAULT_TEMPLATE_NAME].render(
        {"Language": "  English  ", "Topic": "unused"}
    )
    assert rendered == RENDERED_ENGLISH


def test_keyword_overrides_beat_the_mapping() -> None:
    rendered = TEMPLATES[DEFAULT_TEMPLATE_NAME].render({"Language": "French"}, Language="English")
    assert rendered == RENDERED_ENGLISH


def test_declared_placeholders_must_match_the_text() -> None:
    with pytest.raises(TemplateDefinitionError):
        InstructionTemplate(
            name="typo",
            template="Retrieve {Langauge} documents.",
            required_placeholders=("Language",),
            provenance=Provenance.INFERRED,
        )


def test_positional_placeholders_are_rejected() -> None:
    with pytest.raises(TemplateDefinitionError):
        InstructionTemplate(
            name="positional",
            template="Retrieve {} documents.",
            required_placeholders=(),
            provenance=Provenance.INFERRED,
        )


def test_malformed_braces_are_rejected() -> None:
    with pytest.raises(TemplateDefinitionError):
        InstructionTemplate(
            name="malformed",
            template="Retrieve {Language documents.",
            required_placeholders=("Language",),
            provenance=Provenance.INFERRED,
        )


# ---------------------------------------------------------------------------
# The E5 wire format
# ---------------------------------------------------------------------------


def test_e5_wire_format_is_exact() -> None:
    """``Instruct: {instruction}\\nQuery: {query}`` — the model card's own helper."""
    assert E5_INSTRUCT_WIRE_FORMAT == "Instruct: {instruction}\nQuery: {query}"
    out = format_e5_instruct("Given a question, retrieve passages.", "who owns ASML?")
    assert out == "Instruct: Given a question, retrieve passages.\nQuery: who owns ASML?"
    assert out.count("\n") == 1
    instruction_line, query_line = out.split("\n")
    assert instruction_line.startswith("Instruct: ")
    assert query_line.startswith("Query: ")


def test_e5_wire_format_collapses_a_multiline_instruction() -> None:
    out = format_e5_instruct("Given a question,\n  retrieve   passages.", "q")
    assert out == "Instruct: Given a question, retrieve passages.\nQuery: q"


@pytest.mark.parametrize(
    ("instruction", "query"),
    [("", "q"), ("   ", "q"), ("i", ""), ("i", "\n ")],
)
def test_e5_wire_format_refuses_empty_halves(instruction: str, query: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        format_e5_instruct(instruction, query)


def test_documents_get_no_prefix() -> None:
    """The model card: 'No need to add instruction for retrieval documents.'"""
    assert format_e5_document("  ASML shipped 40 EUV machines.\n") == (
        "ASML shipped 40 EUV machines."
    )


def test_compose_query_appends_expansions_deterministically() -> None:
    out = compose_query("safer gene editing", ["gene-editing medicine", "CRISPR"])
    assert out == "safer gene editing; gene-editing medicine; CRISPR"
    assert compose_query("safer gene editing") == "safer gene editing"
    # Expansions already present in the question, and duplicates, are dropped.
    assert compose_query("CRISPR trials", ["crispr", "trials", "  ", "gene therapy"]) == (
        "CRISPR trials; gene therapy"
    )
    assert compose_query("a b", ["c"]) == compose_query("  a   b  ", ["c"])


def test_compose_query_refuses_an_empty_question() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compose_query("   ", ["x"])


# ---------------------------------------------------------------------------
# Template selection
# ---------------------------------------------------------------------------


def test_query_dimensions_reads_only_non_blank_values() -> None:
    assert query_dimensions() == frozenset()
    assert query_dimensions(entities=["", "  "]) == frozenset()
    assert query_dimensions(entities=["ASML"], region=" ", topic="trade") == frozenset(
        {"entities", "topic"}
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, DEFAULT_TEMPLATE_NAME),
        ({"entities": ["ASML"]}, "named_entities"),
        ({"region": "Taiwan"}, "geographic_region"),
        ({"topic": "export controls"}, "topic"),
        ({"entities": ["ASML"], "region": "Taiwan"}, "entities_in_region"),
        ({"entities": ["ASML"], "topic": "export controls"}, "entities_on_topic"),
        ({"region": "Taiwan", "topic": "export controls"}, "topic_in_region"),
        (
            {"entities": ["ASML"], "region": "Taiwan", "topic": "export controls"},
            "entities_on_topic_in_region",
        ),
    ],
)
def test_selection_is_driven_by_the_dimensions_the_query_carries(
    kwargs: dict[str, object], expected: str
) -> None:
    assert select_template(**kwargs).name == expected  # type: ignore[arg-type]


def test_selection_is_deterministic_across_repeated_calls() -> None:
    calls = [
        select_template(entities=["ASML", "TSMC"], region="Taiwan", topic="chips")
        for _ in range(25)
    ]
    assert len({t.name for t in calls}) == 1


def test_selected_template_never_needs_a_value_the_query_lacks() -> None:
    """Selection can never produce a template the caller cannot render."""
    dimension_placeholder = {"entities": "Entities", "region": "Region", "topic": "Topic"}
    for entities in ([], ["ASML"]):
        for region in (None, "Taiwan"):
            for topic in (None, "export controls"):
                template = select_template(entities=entities, region=region, topic=topic)
                present = query_dimensions(entities=entities, region=region, topic=topic)
                available = {"Language"} | {dimension_placeholder[d] for d in present}
                assert set(template.required_placeholders) <= available


def test_selection_falls_back_to_the_disclosed_template_for_a_narrow_registry() -> None:
    narrow = {DEFAULT_TEMPLATE_NAME: TEMPLATES[DEFAULT_TEMPLATE_NAME]}
    chosen = select_template(entities=["ASML"], region="Taiwan", registry=narrow)
    assert chosen.name == DEFAULT_TEMPLATE_NAME


def test_selection_raises_when_a_custom_registry_has_no_usable_entry() -> None:
    odd = {"only": TEMPLATES["named_entities"]}
    with pytest.raises(InstructionError, match="no template for dimensions"):
        select_template(registry=odd)


# ---------------------------------------------------------------------------
# Language handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("en", "English"),
        ("EN", "English"),
        ("  ja  ", "Japanese"),
        ("zh-Hans", "Chinese"),
        ("pt_BR", "Portuguese"),
        ("English", "English"),
        ("Brazilian Portuguese", "Brazilian Portuguese"),
    ],
)
def test_language_name_maps_codes_to_names(value: str, expected: str) -> None:
    assert language_name(value) == expected


@pytest.mark.parametrize("value", ["xx", "qqq", "  "])
def test_language_name_refuses_unknown_codes(value: str) -> None:
    with pytest.raises(UnknownLanguageError):
        language_name(value)


def test_format_entities_is_order_preserving_and_deduplicated() -> None:
    assert format_entities([]) == ""
    assert format_entities(["ASML"]) == "ASML"
    assert format_entities(["ASML", "TSMC"]) == "ASML and TSMC"
    assert format_entities(["ASML", "TSMC", "Nvidia"]) == "ASML, TSMC and Nvidia"
    assert format_entities(["ASML", "asml", "  ", "TSMC"]) == "ASML and TSMC"


# ---------------------------------------------------------------------------
# The request, the writer seam, and the selector entry point
# ---------------------------------------------------------------------------


def test_request_normalises_its_inputs() -> None:
    request = InstructionRequest(
        question="  Who   buys ASML machines? ",
        language="ja",
        entities=["  ASML ", "", "TSMC"],
        region="  Taiwan ",
        topic="   ",
    )
    assert request.question == "Who buys ASML machines?"
    assert request.language == "Japanese"
    assert request.entities == ("ASML", "TSMC")
    assert request.region == "Taiwan"
    assert request.topic is None
    assert request.placeholder_values() == {
        "Language": "Japanese",
        "Entities": "ASML and TSMC",
        "Region": "Taiwan",
    }


def test_request_refuses_an_empty_question() -> None:
    with pytest.raises(InstructionError, match="question must not be empty"):
        InstructionRequest(question="  ", language="en")


def test_default_writer_is_the_tested_default_and_is_a_writer() -> None:
    writer = DefaultInstructionWriter()
    assert isinstance(writer, InstructionWriter)
    assert writer.identifier == "template:auto"
    assert DefaultInstructionWriter(force_template="topic").identifier == "template:topic"


def test_selector_returns_the_disclosed_instruction_for_a_bare_question() -> None:
    selector = InstructionSelector()
    assert selector.select("What did the Fed decide?", language="en") == RENDERED_ENGLISH


def test_selector_is_deterministic() -> None:
    selector = InstructionSelector()
    outputs = {
        selector.select(
            "What did the Fed decide?",
            language="en",
            entities=["Federal Reserve"],
            topic="monetary policy",
        )
        for _ in range(25)
    }
    assert len(outputs) == 1
    assert "Federal Reserve" in outputs.pop()


def test_selector_defaults_the_language_when_stage_one_did_not_run() -> None:
    assert InstructionSelector().select("q") == RENDERED_ENGLISH
    assert "French" in InstructionSelector(default_language="fr").select("q")


def test_selector_can_pin_a_template() -> None:
    selector = InstructionSelector(writer=DefaultInstructionWriter(force_template="topic"))
    out = selector.select("q", language="en", topic="export controls")
    assert "cover the topic of export controls" in out


def test_pinned_template_still_requires_its_values() -> None:
    selector = InstructionSelector(writer=DefaultInstructionWriter(force_template="topic"))
    with pytest.raises(MissingPlaceholderError):
        selector.select("q", language="en")


def test_selector_builds_the_full_embedding_input() -> None:
    selector = InstructionSelector()
    out = selector.embedding_input(
        "safer gene editing",
        language="en",
        expansions=["gene-editing medicine", "CRISPR"],
    )
    assert out == (
        f"Instruct: {RENDERED_ENGLISH}\nQuery: safer gene editing; gene-editing medicine; CRISPR"
    )


class _StubWriter:
    """Stands in for the LLM-backed writer the post describes. Calls nothing."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[InstructionRequest] = []

    @property
    def identifier(self) -> str:
        return "stub"

    def write(self, request: InstructionRequest) -> str:
        self.calls.append(request)
        return self._text


def test_a_custom_writer_can_be_plugged_in() -> None:
    writer = _StubWriter("  Retrieve\n documents about ASML.  ")
    selector = InstructionSelector(writer=writer)
    out = selector.select("who buys ASML machines?", language="en", region="Taiwan")
    # The selector normalises whatever the writer returns to a single line.
    assert out == "Retrieve documents about ASML."
    assert len(writer.calls) == 1
    assert writer.calls[0].region == "Taiwan"
    assert writer.calls[0].language == "English"


def test_a_writer_returning_nothing_is_rejected_not_embedded() -> None:
    selector = InstructionSelector(writer=_StubWriter("   "))
    with pytest.raises(InstructionError, match="empty instruction"):
        selector.select("q", language="en")


# ---------------------------------------------------------------------------
# Regressions: silent prompt corruption from non-string inputs
#
# Every one of these used to pass ``str(value)`` and end up embedding the literal
# text "None" (or "123") in a prompt, or crashing with a bare AttributeError from
# inside a dataclass __init__. Neither is visible in a retrieval metric.
# ---------------------------------------------------------------------------


def test_a_none_question_is_refused_not_embedded_as_the_word_none() -> None:
    selector = InstructionSelector()
    with pytest.raises(InstructionError, match="question must be a string"):
        selector.embedding_input(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [None, 123, object()])
def test_a_non_string_entity_is_refused(bad: object) -> None:
    with pytest.raises(InstructionError, match="entity must be a string"):
        format_entities([bad, "ASML"])  # type: ignore[list-item]
    with pytest.raises(InstructionError, match="entity must be a string"):
        InstructionSelector().select("q", language="en", entities=[bad])  # type: ignore[list-item]


@pytest.mark.parametrize(("field", "kwargs"), [("region", {"region": 5}), ("topic", {"topic": 5})])
def test_a_non_string_region_or_topic_raises_instruction_error_not_attribute_error(
    field: str, kwargs: dict[str, object]
) -> None:
    with pytest.raises(InstructionError, match=f"{field} must be a string"):
        InstructionSelector().select("q", language="en", **kwargs)  # type: ignore[arg-type]


def test_a_non_string_language_raises_instruction_error() -> None:
    with pytest.raises(InstructionError, match="language must be a string"):
        InstructionSelector().select("q", language=7)  # type: ignore[arg-type]


def test_the_wire_format_survives_a_newline_in_the_query() -> None:
    """A second ``\\n`` would split the query into a field the encoder reads apart."""
    out = format_e5_instruct("retrieve passages", "line one\nline two")
    assert out == "Instruct: retrieve passages\nQuery: line one line two"
    assert out.count("\n") == 1


# ---------------------------------------------------------------------------
# Selection: the branches the shipped registry does not reach on its own
# ---------------------------------------------------------------------------


def test_selection_is_independent_of_registry_iteration_order() -> None:
    """The choice must not depend on how the registry dict happens to be ordered."""
    rng = random.Random(20240814)
    items = list(TEMPLATES.items())
    baseline = select_template(entities=["ASML"], region="Taiwan", topic="chips").name
    for _ in range(25):
        rng.shuffle(items)
        shuffled = dict(items)
        chosen = select_template(
            entities=["ASML"], region="Taiwan", topic="chips", registry=shuffled
        )
        assert chosen.name == baseline


def test_dimension_weights_break_ties_entities_first() -> None:
    """With no exact match available, entities outrank region outrank topic.

    ``TEMPLATES`` carries an exact template for all eight dimension subsets, so this
    tie-break never fires against the shipped registry; a partial registry is the
    only way to exercise the rule the module documents.
    """
    partial = {
        name: TEMPLATES[name]
        for name in ("entity_focus", "named_entities", "geographic_region", "topic")
    }
    # Query carries all three; no template covers more than one, so weights decide.
    chosen = select_template(entities=["ASML"], region="Taiwan", topic="chips", registry=partial)
    assert chosen.name == "named_entities"
    # Drop the entity template and region must win over topic.
    del partial["named_entities"]
    assert (
        select_template(entities=["ASML"], region="Taiwan", topic="chips", registry=partial).name
        == "geographic_region"
    )
