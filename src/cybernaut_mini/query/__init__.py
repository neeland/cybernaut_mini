"""The eight-stage query-time retrieval pipeline.

Every question NOSIBLE receives is described in the blog as passing through eight
numbered stages. This package holds one subpackage per stage, named so the numbering
survives contact with the filesystem:

===========================  ==========================================================
``s1_language``              Language detection and translation
``s2_tokenize``              Multilingual tokenization
``s3_intents``               Search intent prediction
``s4_instruct``              Instruction tuning and embedding
``s5_select``                Shard selection
``s6_rerank``                Shard reranking
``s7_expand``                Shard-based question expansion
``s8_retrieve``              Retrieval using map-reduce
===========================  ==========================================================

Build-time concerns (sharding, feature extraction, index writing) deliberately live
outside this package: these modules run per *question*, not per corpus.

This file exists so ``cybernaut_mini.query`` is a real package rather than an implicit
namespace package. Every other package in this repo carries one, and tooling that
enumerates packages rather than walking directories does not see namespace packages.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — "Every question that
    comes into NOSIBLE goes through a pretty sophisticated eight-stage retrieval
    pipeline." Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    The post names the eight stages but not their module boundaries. Splitting one
    stage per subpackage is our choice, made so that a reader can hold the post and
    the source side by side. It also splits the repo's original ``routing.py``, which
    fused stages 5 and 6 into a single function.

Alternatives rejected:
    A single flat ``query.py`` — the stages carry roughly 3,000 lines between them and
    the post's own structure is the most defensible seam available.
    Naming without numbers (``language/``, ``tokenize/``) — loses the ordering, which
    is the one piece of structure the post is unambiguous about.
"""

from __future__ import annotations

#: Stage subpackage names in pipeline order. Kept as data so the ``explain`` CLI and
#: the docs checker can enumerate stages without hardcoding the list a third time.
STAGES: tuple[str, ...] = (
    "s1_language",
    "s2_tokenize",
    "s3_intents",
    "s4_instruct",
    "s5_select",
    "s6_rerank",
    "s7_expand",
    "s8_retrieve",
)

__all__ = ["STAGES"]
