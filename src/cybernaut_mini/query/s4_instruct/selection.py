"""Choosing which instruction template a question gets.

The post does not describe the selection rule — only that templates exist "that have
placeholders for named entities, geographic regions, topics of interest, and other
dimensions". The rule here is therefore [inferred] and deliberately dumb: a template
declares the dimensions it needs, the query carries some subset of those dimensions,
and the best-covering template wins. No model, no scoring against the question text.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 4, "Instruction
    Tuning and Embedding". Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - A dimension is "present" iff the caller supplied a non-blank value for it.
      Upstream stages decide what an entity or a region *is*; this stage does not
      re-detect them, so that stage 4 stays testable without an NER model.
    - Richer conditioning is better conditioning: given both entities and a region,
      the template that names both is preferred over either single-dimension one.
      Unverified — the post reports a 1-5% band for instruction quality overall, not
      a ranking of dimensions — so it is stated as a rule, not a finding.
    - When no template matches, the disclosed template is the answer. It is the one
      variant with evidence behind it, so it is the floor, never a fallback that
      throws.

Alternatives rejected:
    - Detecting the dimensions here (NER, gazetteer, topic classifier): duplicates
      stages 2-3 and drags a model into a module that should stay pure and instant.
    - Scoring every template by embedding similarity to the question: circular — it
      needs the encoder that this stage's output configures — and non-deterministic
      across model versions.
    - Random or round-robin template choice for online exploration: the right shape
      for the A/B test the post's "evolved" templates imply, but it breaks the
      repo-wide rule that the same query yields the same result.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from cybernaut_mini.query.s4_instruct.templates import (
    DEFAULT_TEMPLATE_NAME,
    TEMPLATES,
    InstructionError,
    InstructionTemplate,
)

__all__ = ["DIMENSIONS", "DIMENSION_WEIGHTS", "query_dimensions", "select_template"]

#: The dimensions the post names, in the order it names them.
DIMENSIONS: tuple[str, ...] = ("entities", "region", "topic")

#: Tie-break weights when two templates cover the same number of dimensions.
#: Entities outrank region outranks topic, following the post's own ordering and the
#: fact that its one disclosed template is the entity-focused one.
DIMENSION_WEIGHTS: Mapping[str, int] = {"entities": 4, "region": 2, "topic": 1}


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())


def query_dimensions(
    *,
    entities: Iterable[str] = (),
    region: str | None = None,
    topic: str | None = None,
) -> frozenset[str]:
    """Return the set of dimensions the query actually carries."""
    present: set[str] = set()
    if any(_has_text(item) for item in entities):
        present.add("entities")
    if _has_text(region):
        present.add("region")
    if _has_text(topic):
        present.add("topic")
    return frozenset(present)


def select_template(
    *,
    entities: Iterable[str] = (),
    region: str | None = None,
    topic: str | None = None,
    registry: Mapping[str, InstructionTemplate] = TEMPLATES,
) -> InstructionTemplate:
    """Pick the template for a query carrying these dimensions.

    Rule, applied in order:

    1. A template whose declared ``dimensions`` equal the query's dimensions exactly.
    2. Otherwise the template covering the most of the query's dimensions without
       requiring any it lacks (so rendering can never fail for want of a value).
    3. Ties broken by :data:`DIMENSION_WEIGHTS`, then by template name — total and
       deterministic, independent of dict iteration order.
    4. Otherwise the disclosed default, :data:`DEFAULT_TEMPLATE_NAME`.

    Raises :class:`InstructionError` only if a custom *registry* is passed that
    contains neither a match nor the default template.
    """
    present = query_dimensions(entities=entities, region=region, topic=topic)

    candidates = sorted(registry.items())
    for _name, template in candidates:
        if template.dimensions == present:
            return template

    # ``candidates`` is sorted by name, so a strict ``>`` keeps the
    # lexicographically first template on an exact tie — no dict-order dependence.
    best: InstructionTemplate | None = None
    best_key: tuple[int, int] = (-1, -1)
    for _name, template in candidates:
        if not template.dimensions <= present:
            continue
        key = (
            len(template.dimensions),
            sum(DIMENSION_WEIGHTS.get(dim, 0) for dim in template.dimensions),
        )
        if key > best_key:
            best, best_key = template, key
    if best is not None:
        return best

    default = registry.get(DEFAULT_TEMPLATE_NAME)
    if default is None:
        raise InstructionError(
            f"registry has no template for dimensions {sorted(present)} and no "
            f"{DEFAULT_TEMPLATE_NAME!r} fallback"
        )
    return default
