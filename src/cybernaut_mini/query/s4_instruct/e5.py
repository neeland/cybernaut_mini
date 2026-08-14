"""The E5-instruct wire format: how an instruction is attached to a query.

``intfloat/multilingual-e5-large-instruct`` does not take the instruction as a
separate argument. It is *prepended to the query string itself*, in one exact shape,
and the model card's own helper is the specification::

    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\\nQuery: {query}'

Two details are easy to get wrong and neither fails loudly — retrieval just gets
quietly worse:

1. The separator is a single ``\\n``. Not ``". "``, not ``"\\n\\n"``.
2. **Documents get no prefix at all.** The card: "No need to add instruction for
   retrieval documents." Prefixing the corpus side with ``Instruct: …`` — or with
   the ``query:``/``passage:`` prefixes that the *non*-instruct mE5 checkpoints
   require — misaligns the two sides of the cosine.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 4: "we generate an
    appropriate instruction for the question and submit it, along with the
    expansions, to multilingual-e5-large-instruct". Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.
Wire-format ref: https://huggingface.co/intfloat/multilingual-e5-large-instruct
    (model card, "Usage" — ``get_detailed_instruct``); the format originates with
    E5-mistral-7b-instruct, arXiv:2401.00368, and carries over to the multilingual
    checkpoints, arXiv:2402.05672.

Assumptions:
    - The post says the expansions are submitted "along with" the instruction but
      never how. :func:`compose_query` puts them on the query side, appended to the
      question, because the instruction slot is documented as "a one-sentence
      instruction that describes the task" — a bag of terms does not belong there.
      This join is [inferred] and is the most challengeable line in this stage.
    - An instruction is a single line. Internal newlines are collapsed rather than
      rejected, so a future LLM-backed writer that returns a wrapped sentence cannot
      corrupt the wire format by accident.

Alternatives rejected:
    - Formatting the wire string at the call site with an f-string: one character of
      drift (``"\\n\\n"``, a stray space after the colon) would be invisible in
      review and unmeasurable in evals. It is a named, tested function instead.
    - Interleaving expansions into the instruction ("… that focus on X, Y, Z"): it
      reads well, but it moves lexical expansion terms into the slot the model uses
      for task conditioning, which is a different job.
    - Rejecting multi-line instructions outright: strictly safer, but it would turn a
      cosmetic LLM formatting quirk into a hard query failure at serve time.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "E5_INSTRUCT_WIRE_FORMAT",
    "compose_query",
    "format_e5_document",
    "format_e5_instruct",
]

#: The documented wire format, kept as data so a test can assert on it directly.
E5_INSTRUCT_WIRE_FORMAT = "Instruct: {instruction}\nQuery: {query}"


def _collapse(text: str) -> str:
    """Collapse all runs of whitespace (newlines included) to single spaces."""
    return " ".join(text.split())


def format_e5_instruct(instruction: str, query: str) -> str:
    """Return the query-side input for a ``multilingual-e5-*-instruct`` encoder.

    Produces exactly ``"Instruct: {instruction}\\nQuery: {query}"``. *Both* halves are
    whitespace-collapsed to a single line: a newline anywhere in the query would put a
    second ``\\n`` into a two-line wire format, so the encoder would read the tail of
    the query as a field of its own. Callers who want their own spacing preserved are
    asking for a format this model does not have.

    Raises
    ------
    ValueError
        If either the instruction or the query is empty once stripped. An empty
        instruction would send ``"Instruct: \\nQuery: …"``, which is measurably worse
        than sending the bare query, so it is refused rather than tolerated.
    """
    cleaned_instruction = _collapse(instruction)
    cleaned_query = _collapse(query)
    if not cleaned_instruction:
        raise ValueError("instruction must not be empty")
    if not cleaned_query:
        raise ValueError("query must not be empty")
    return E5_INSTRUCT_WIRE_FORMAT.format(instruction=cleaned_instruction, query=cleaned_query)


def format_e5_document(text: str) -> str:
    """Return the corpus-side input for an instruct checkpoint: the text, unprefixed.

    A one-line function that exists to be greppable. The model card is explicit that
    documents take no instruction, and the non-instruct mE5 checkpoints' ``passage:``
    prefix does *not* apply here; anyone reaching for a document prefix should land
    on this docstring first.
    """
    return text.strip()


def compose_query(question: str, expansions: Iterable[str] = ()) -> str:
    """Join the question with its stage-3 expansions into one query string.

    [inferred] — see the module docstring. Expansions are appended after the
    question, separated by ``"; "``, in the order given; blanks are dropped and
    duplicates are removed case-insensitively (keeping the first spelling), as is any
    expansion already contained in the question. Same input, same output.
    """
    base = " ".join(question.split())
    if not base:
        raise ValueError("question must not be empty")
    haystack = base.casefold()
    kept: list[str] = []
    seen: set[str] = set()
    for raw in expansions:
        term = " ".join(str(raw).split())
        if not term:
            continue
        key = term.casefold()
        if key in seen or key in haystack:
            continue
        seen.add(key)
        kept.append(term)
    if not kept:
        return base
    return f"{base}; {'; '.join(kept)}"
