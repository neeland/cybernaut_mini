"""Query generation behind a protocol; heuristic by default, LLM opt-in.

``HeuristicQueryGenerator`` proposes up to five variant shapes (original,
expanded, missing-keyword, keyword-only, entity-preserving), deduplicated on
lexical form. Deterministic and free — the default.

``LLMQueryGenerator`` is the model-backed generator (``agent.query_generator:
llm``): a small instruct model from the Hugging Face Hub rewrites the query from
the same :class:`StateSummary` the heuristic sees — stage, shard keywords the
query is missing, entities, and evidence terms from the current best hits. It
runs in-process on MPS on Apple silicon (CUDA/CPU elsewhere), decodes greedily
so runs stay reproducible per device, and falls back to the heuristic whenever
the model output parses to nothing — a bad generation degrades the search to
the old behaviour instead of failing it.

Blog ref: https://nosible.com/blog/introducing-cybernaut-1-agentic-search-with-mcts —
    "LLM-guided" expansion: a model proposes the next candidate refinements from
    the current search state. The candidate schema, the prompt and the fallback
    are this replica's [inferred]; the post names no model and prints no prompt.
    Local copy:
    ``docs/blog-archive/introducing-cybernaut-1-agentic-search-with-mcts.md``.

Assumptions:
    - ``HeuristicQueryGenerator`` is the default (zero downloads, zero model
      calls) and always emits the current query as candidate one. Its five shapes
      — original, plus-expansions, plus-first-missing-shard-keyword,
      content-tokens-only, entity-preserving — are a stand-in for the
      undisclosed proposal policy, not a reconstruction of it.
    - ``LLMQueryGenerator`` decodes greedily by default, so one (model, device)
      pair reproduces the same rewrites. When the agent sets ``temperature`` the
      generator samples with ``torch.manual_seed`` re-applied before every call,
      so the annealed schedule stays replayable per (model, device, temperature,
      prompt) tuple even though sampling is stochastic.
    - The current query is always candidate one and the model's rewrites are
      topped up with heuristic variants when it under-produces, so a garbled or
      crashed generation degrades the search to the previous behaviour instead of
      failing the turn.
    - ``parse_generated_queries`` is strict: it strips bullets, numbering and
      quotes, drops ``<think>`` blocks and any line over 120 characters, and
      deduplicates by lexical form. Small instruct models routinely answer with
      an explanation; a line that long is treated as prose, not a query.
    - ``build_generator_prompt`` is pure and pinned by tests. Its prompt includes
      the retriever's own signal table (``providers/signals.py``), which is this
      repo's reading of the post's "high-trust" LLM components.

Alternatives considered:
    - A hosted chat API (OpenRouter/Gemma, as stage 1 uses for translation):
      rejected because the default install must run offline with zero model calls
      and no key; a local causal LM behind the optional ``st`` extra keeps the
      free path free and the model path opt-in.
    - Beam search or unseeded sampling for the rewrites: would raise diversity,
      but a non-reproducible generator makes a trace's candidate list
      unverifiable; seeded sampling at a set temperature is the compromise.
    - Letting the model emit candidate queries together with their intended
      actions or shard sets: closer to an end-to-end planner, but it would couple
      the generator to the action union and let a model propose a state the
      executor cannot represent. The generator proposes text; the executor
      derives the action.
    - Trusting the model's output verbatim: rejected because instruct models pad
      answers with numbering and commentary, so every trace would carry query
      strings the retriever never intended. The line filter plus heuristic top-up
      is cheaper than a retry loop.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Protocol

from cybernaut_mini.agent.state import StateSummary
from cybernaut_mini.config import AgentConfig, ConfigError
from cybernaut_mini.models import QueryCandidate
from cybernaut_mini.text import TextProcessor, lexical_form

_SYSTEM_PROMPT = (
    "You rewrite search queries for a sharded hybrid (BM25 + dense) news search "
    "engine. Answer with the rewritten queries only, one per line: no numbering, "
    "no bullets, no quotes, no explanations."
)

#: Lines longer than this are almost always the model explaining itself.
_MAX_QUERY_CHARS = 120

_STAGE_GOAL = {
    "explore": "Propose diverse rewrites that probe different aspects of the question.",
    "refine": "Sharpen the current query using the missing keywords and evidence terms.",
    "exploit": "Make small precision edits to the current query; keep what already works.",
}


class QueryGenerator(Protocol):
    def generate(
        self, question: str, state_summary: StateSummary, n: int
    ) -> list[QueryCandidate]: ...


class HeuristicQueryGenerator:
    def __init__(self, processor: TextProcessor) -> None:
        self._processor = processor

    def generate(
        self, question: str, state_summary: StateSummary, n: int
    ) -> list[QueryCandidate]:
        query = state_summary.current_query
        candidates: list[QueryCandidate] = [
            QueryCandidate(text=query, origin="original")
        ]

        if state_summary.expansions:
            candidates.append(
                QueryCandidate(
                    text=query,
                    origin="expanded",
                    expansions=list(state_summary.expansions),
                )
            )

        if state_summary.missing_keywords:
            keyword = state_summary.missing_keywords[0]
            candidates.append(
                QueryCandidate(text=f"{query} {keyword}", origin="keyword")
            )

        content = self._processor.content_tokens(query)
        if content:
            candidates.append(
                QueryCandidate(text=" ".join(content), origin="keyword_only")
            )

        if state_summary.entities:
            entity_text = " ".join(state_summary.entities)
            remaining = [tok for tok in content if tok not in set(entity_text.split())]
            candidates.append(
                QueryCandidate(
                    text=" ".join([entity_text, *remaining]).strip(),
                    origin="entity_preserving",
                )
            )

        deduped: list[QueryCandidate] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for candidate in candidates:
            key = (lexical_form(candidate.text), tuple(candidate.expansions))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped[:n]


def build_generator_prompt(question: str, summary: StateSummary, n: int) -> str:
    """Render the state summary into the user turn; pure so tests can pin it.

    When the summary carries hits, their per-hit signal bundle (bm25/dense/rerank/
    rrf) is rendered as a compact table — the "high-trust" evidence the post says
    the LLM components are shown, instead of prose alone.
    """
    from cybernaut_mini.providers.signals import render_signal_table

    lines = [
        f"Question: {question}",
        f"Current query: {summary.current_query}",
        f"Stage: {summary.stage.value}. {_STAGE_GOAL[summary.stage.value]}",
    ]
    if summary.missing_keywords:
        lines.append(
            "Shard keywords missing from the query: "
            + ", ".join(summary.missing_keywords[:8])
        )
    if summary.entities:
        lines.append("Entities to preserve verbatim: " + ", ".join(summary.entities))
    if summary.evidence_terms:
        lines.append(
            "Title terms from current top hits: " + ", ".join(summary.evidence_terms)
        )
    if summary.top_hits:
        table = render_signal_table(summary.top_hits)
        if table:
            lines.append("Ranking signals for the current top hits:")
            lines.append(table)
    if summary.expansions:
        lines.append(
            "Term-graph expansion candidates: " + ", ".join(summary.expansions)
        )
    lines.append(f"Write {n} alternative search queries, one per line.")
    return "\n".join(lines)


def parse_generated_queries(text: str) -> list[str]:
    """Extract clean query strings from raw model output.

    Strips numbering/bullets/quotes, drops blank lines, chain-of-thought blocks,
    over-long lines (the model explaining itself), and duplicates by lexical form.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    queries: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", line)
        line = line.strip("\"'` ")
        if not line or len(line) > _MAX_QUERY_CHARS:
            continue
        key = lexical_form(line)
        if not key or key in seen:
            continue
        seen.add(key)
        queries.append(line)
    return queries


class LLMQueryGenerator:
    """Hugging Face causal-LM query rewriting on MPS/CUDA/CPU.

    Generation is greedy (``do_sample=False``) so a given model + device pair
    reproduces the same rewrites; the same MPS-vs-CPU caveat as embeddings
    applies across devices. ``calls`` counts model invocations for the trace.

    The current query is always candidate one (origin ``original``) — it is
    already scored history and grounds the beam — followed by the model's
    rewrites (origin ``llm``), topped up with heuristic variants when the model
    under-produces.

    ``temperature`` is the agent's per-stage annealing knob (1.0 wide -> 0.3
    narrow). ``None`` (the default) keeps the original greedy decode. A set
    temperature switches to seeded sampling: ``torch.manual_seed`` is re-applied
    before every generate call, so a given (model, device, temperature, prompt)
    still reproduces its output — annealing does not give up replayability.
    """

    def __init__(
        self,
        processor: TextProcessor,
        *,
        model_name: str,
        revision: str | None = None,
        device: str = "auto",
        max_new_tokens: int = 192,
        chat_fn: Callable[[str, str], str] | None = None,
        sample_seed: int = 42,
    ) -> None:
        self._processor = processor
        self._heuristic = HeuristicQueryGenerator(processor)
        self.calls = 0
        #: Per-stage sampling temperature; None keeps greedy decoding.
        self.temperature: float | None = None
        self._sample_seed = sample_seed
        if chat_fn is not None:
            self._chat = chat_fn
            self.device = "injected"
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            msg = (
                "agent query generator 'llm' needs the optional 'st' extra "
                "(transformers + torch), which the default install does not "
                "include.\n"
                "  install it   : uv sync --extra st --extra mps   (Mac; drop "
                "--extra mps elsewhere)\n"
                "  or stay local: agent.query_generator: heuristic"
            )
            raise ConfigError(msg) from exc
        from cybernaut_mini.accel import resolve_device

        self.device = resolve_device(device)  # type: ignore[arg-type]
        token = os.environ.get("HF_TOKEN") or None
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, token=token)
        # float16 halves memory and is enough for short greedy rewrites; MPS has
        # no bfloat16 on older Macs and CPU float16 is slower than float32.
        dtype = torch.float16 if self.device in {"mps", "cuda"} else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, dtype=dtype, token=token
        )
        # transformers 5 types ``.to`` through a decorator wrapper that confuses
        # mypy; the runtime call is the ordinary nn.Module device move.
        model = model.to(self.device).eval()  # type: ignore[arg-type]

        def chat(system: str, user: str) -> str:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            sampling: dict[str, object] = {"do_sample": False}
            if self.temperature is not None and self.temperature > 0.0:
                # Seeded before every call: annealed sampling stays replayable
                # for a fixed (model, device, temperature, prompt) tuple.
                torch.manual_seed(self._sample_seed)
                sampling = {"do_sample": True, "temperature": float(self.temperature)}
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    **sampling,
                )
            decoded = tokenizer.decode(
                output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            # ``decode`` is typed ``str | list[str]`` in transformers 5; a single
            # sequence always decodes to one string.
            return decoded if isinstance(decoded, str) else " ".join(decoded)

        self._chat = chat

    def chat(self, system: str, user: str) -> str:
        """Run one chat turn against the underlying model, counting the call.

        Public seam: the stage-4 instruction writer/evolver reuses this local
        generator provider instead of loading a second model.
        """
        self.calls += 1
        return self._chat(system, user)

    def generate(
        self, question: str, state_summary: StateSummary, n: int
    ) -> list[QueryCandidate]:
        candidates: list[QueryCandidate] = [
            QueryCandidate(text=state_summary.current_query, origin="original")
        ]
        try:
            raw = self._chat(
                _SYSTEM_PROMPT, build_generator_prompt(question, state_summary, n)
            )
            self.calls += 1
            rewrites = parse_generated_queries(raw)
        except Exception:
            # A crashed or garbled generation must not kill the search turn;
            # the heuristic top-up below restores the old behaviour.
            rewrites = []
        candidates.extend(QueryCandidate(text=text, origin="llm") for text in rewrites)
        if len(candidates) < n:
            candidates.extend(self._heuristic.generate(question, state_summary, n))

        deduped: list[QueryCandidate] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for candidate in candidates:
            key = (lexical_form(candidate.text), tuple(candidate.expansions))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped[:n]


def create_query_generator(config: AgentConfig, processor: TextProcessor) -> QueryGenerator:
    """Build the generator the config asks for; the heuristic needs no downloads."""
    if config.query_generator == "heuristic":
        return HeuristicQueryGenerator(processor)
    return LLMQueryGenerator(
        processor,
        model_name=config.generator_model_name,
        revision=config.generator_revision,
        device=config.device,
        max_new_tokens=config.generator_max_new_tokens,
    )
