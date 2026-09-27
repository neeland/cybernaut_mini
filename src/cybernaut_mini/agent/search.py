"""Three-stage Explore -> Refine -> Exploit search agent.

The tree is a staged beam search with UCT-ordered budget allocation in Refine
(honestly not full MCTS — see the README). One retrieval call is one candidate
query executed across its selected shard set; the shared budget defaults to 18,
matching the default stage schedule 5 + 9 + 4.

The schedule itself is now a value (:mod:`cybernaut_mini.agent.schedule`), which
adds the post's wide-to-narrow annealing knobs on top of the legacy numbers:
per-stage generator temperature, hybrid-weight variant branches
(:class:`~cybernaut_mini.agent.actions.AdjustHybridWeights`), a shard-reranker
toggle branch (:class:`~cybernaut_mini.agent.actions.ToggleRerankers`), the
retained user filter surfacing as :class:`~cybernaut_mini.agent.actions
.RetainFilter`, and a cheap wide judge for Explore when the configured judge is
model-backed. Per-phase accounting (retrieval / embedding / LLM calls and nodes
per stage) is recorded into the trace under ``config["phase_accounting"]``.

Blog ref: https://nosible.com/blog/introducing-cybernaut-1-agentic-search-with-mcts —
    the LLM-guided MCTS layer over Hybrid-3; and
    https://nosible.com/blog/the-road-to-cybernaut-1 — the eight-stage pipeline
    this agent searches (stages 5-8: shard selection, shard reranking, expansion,
    map-reduce retrieval). Local copies:
    ``docs/blog-archive/introducing-cybernaut-1-agentic-search-with-mcts.md`` and
    ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - This is a staged beam search, not full MCTS, and says so. UCT orders which
      *parent* to expand inside Refine only; Explore always branches from the
      root and Exploit never branches. The post discloses no tree shape, so the
      three-stage wide-to-narrow skeleton is an inferred reading.
    - One retrieval call is one candidate query executed across its selected shard
      set. ``max_retrieval_calls`` (default 18) is the shared budget for all three
      stages, and the default schedule spends 5 + 9 + 4 of it. The seed routing
      call before Explore is not charged as a candidate.
    - Duplicate candidates are filtered on ``(lexical_form(text), sorted
      expansions)``, so a rewrite differing only in punctuation cannot burn a
      retrieval call. Exhausting candidates sets a stop reason; it is not an
      error.
    - Rewards from different stages are compared directly: survivors are ranked by
      ``mean_value`` regardless of which stage produced them, because the judge
      and the reward are stage-independent. A stage policy changes what is
      *proposed* and how deep retrieval goes, not how a result set is scored.
    - The winner is the highest-mean-value node among Exploit's nodes, falling
      back to Refine then Explore, so a budget-limited run still returns its best
      evidence rather than nothing.
    - ``replay()`` re-executes a node's stored ``RetrievalPlan`` instead of
      returning cached hits. That is what makes a trace a reproducible artifact
      rather than a log.

Alternatives considered:
    - Full four-phase MCTS (selection, expansion, simulation, backpropagation)
      per the build guide: rejected because a simulation is a chain of retrieval
      calls, so rollouts and real search would compete for the same 18-call
      budget; the post's emphasis on inference cost points the other way.
    - Letting UCT select across all three stages: closer to textbook MCTS, and it
      would let a promising Explore branch keep its budget. Rejected because the
      disclosed behaviour is wide-then-narrow, and one shared UCT frontier can
      starve the deep stage entirely on a cheap-looking early branch.
    - Making the budget a soft penalty instead of a hard stop: simpler to reason
      about than a stop reason, but call counts would become unpredictable and
      every trace would need to explain why it overspent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from cybernaut_mini.agent.actions import (
    Action,
    AddExpansions,
    AdjustHybridWeights,
    NarrowShards,
    RetainFilter,
    RewriteQuery,
    ToggleRerankers,
    describe_action,
)
from cybernaut_mini.agent.node import SearchNode, select_child
from cybernaut_mini.agent.policy import compute_reward
from cybernaut_mini.agent.schedule import (
    DEFAULT_SCHEDULE,
    StagePolicy,
    StageSchedule,
    schedule_from_config,
)
from cybernaut_mini.agent.state import SearchState, Stage, StateSummary
from cybernaut_mini.config import AppConfig, RRFConfig
from cybernaut_mini.expansion import expand_query
from cybernaut_mini.indexing import LoadedIndex
from cybernaut_mini.models import MetadataFilter, QueryCandidate, SearchHit
from cybernaut_mini.providers.embeddings import EmbeddingProvider, FloatArray
from cybernaut_mini.providers.judge import HeuristicJudge, Judge, create_judge
from cybernaut_mini.providers.query_generator import QueryGenerator, create_query_generator
from cybernaut_mini.providers.signals import signal_summary
from cybernaut_mini.retrieval import retrieve
from cybernaut_mini.routing import RoutingSignals, route
from cybernaut_mini.text import TextProcessor, lexical_form
from cybernaut_mini.trace import (
    AgentTrace,
    NodeTrace,
    StageTiming,
    StopReason,
    make_trace_id,
    routing_to_dict,
)

#: Legacy view of the default schedule, kept for external readers of the old
#: module dict. The agent itself consumes ``StagePolicy`` objects.
STAGE_SCHEDULE: dict[Stage, dict[str, int]] = {
    stage: {
        key: value
        for key, value in {
            "candidates": policy.candidates,
            "per_survivor": policy.per_survivor,
            "shards": policy.shards,
            "hits_per_shard": policy.hits_per_shard,
            "survivors": policy.survivors,
        }.items()
        if value is not None
    }
    for stage, policy in DEFAULT_SCHEDULE.items()
}


@dataclass
class Counters:
    max_retrieval_calls: int
    retrieval_calls: int = 0
    embedding_calls: int = 0
    llm_calls: int = 0

    def budget_available(self) -> bool:
        return self.retrieval_calls < self.max_retrieval_calls


class _CountingProvider:
    """Wraps an embedding provider to count query-embedding invocations."""

    def __init__(self, inner: EmbeddingProvider, counters: Counters) -> None:
        self._inner = inner
        self._counters = counters

    @property
    def identifier(self) -> str:
        return self._inner.identifier

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed_documents(self, texts: list[str]) -> FloatArray:
        return self._inner.embed_documents(texts)

    def embed_queries(self, texts: list[str]) -> FloatArray:
        self._counters.embedding_calls += 1
        return self._inner.embed_queries(texts)


@dataclass(frozen=True)
class RetrievalPlan:
    query: str
    shard_ids: tuple[int, ...]
    expansions: tuple[str, ...]
    lexical_weight: float
    dense_weight: float
    per_shard_limit: int
    top_k: int


@dataclass(frozen=True)
class _KnobVariant:
    """A refine-stage branch that re-runs the parent's query with a knob turned.

    ``lexical_weight``/``dense_weight`` are multipliers on the configured RRF
    weights (the {0.5, 1.0, 1.5} grid); ``rerank=False`` routes with the stage-6
    shard rerankers switched off. Spends an ordinary candidate slot.
    """

    lexical_weight: float
    dense_weight: float
    rerank: bool = True


@dataclass
class AgentResult:
    question: str
    best_query: str
    expansions: list[str]
    metadata_filter: MetadataFilter | None
    shard_ids: list[int]
    hits: list[SearchHit]
    trace: AgentTrace
    plan: RetrievalPlan = field(
        repr=False, default_factory=lambda: RetrievalPlan("", (), (), 1.0, 1.0, 0, 0)
    )
    #: Per-stage resource usage; also embedded in ``trace.config["phase_accounting"]``.
    phase_accounting: dict[str, dict[str, float]] = field(default_factory=dict)


def _normalize_candidate(candidate: QueryCandidate) -> tuple[str, tuple[str, ...]]:
    return (lexical_form(candidate.text), tuple(sorted(candidate.expansions)))


class SearchAgent:
    def __init__(
        self,
        index: LoadedIndex,
        *,
        processor: TextProcessor,
        provider: EmbeddingProvider,
        generator: QueryGenerator,
        judge: Judge,
        rrf_config: RRFConfig,
        exploration_constant: float = 1.2,
        max_expansions: int = 5,
        max_retrieval_calls: int = 18,
        seed: int = 42,
        output_top_k: int = 10,
        stage_schedule: StageSchedule | None = None,
        wide_judge: Judge | None = None,
    ) -> None:
        self._index = index
        self._processor = processor
        self._counters = Counters(max_retrieval_calls=max_retrieval_calls)
        self._provider = _CountingProvider(provider, self._counters)
        self._generator = generator
        self._judge = judge
        self._wide_judge = wide_judge
        self._rrf_config = rrf_config
        self._c = exploration_constant
        self._max_expansions = max_expansions
        self._seed = seed
        self._output_top_k = output_top_k
        self._schedule: StageSchedule = stage_schedule or DEFAULT_SCHEDULE

        self._node_counter = 0
        self._node_traces: list[NodeTrace] = []
        self._seen: set[tuple[str, tuple[str, ...]]] = set()
        self._all_nodes: list[SearchNode] = []
        self._budget_hit = False
        self._duplicate_hit = False
        self._phase_accounting: dict[str, dict[str, float]] = {}

    # --------------------------- helpers ---------------------------- #

    def _next_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def _n_shards_total(self) -> int:
        return len(self._index.manifests)

    def _top_keywords_not_in_query(self, shard_ids: tuple[int, ...], query: str) -> list[str]:
        query_tokens = set(self._processor.content_tokens(query))
        missing: list[str] = []
        seen: set[str] = set()
        for sid in shard_ids[:3]:
            for kw in self._index.manifests[sid].keywords:
                if kw.term not in query_tokens and kw.term not in seen:
                    seen.add(kw.term)
                    missing.append(kw.term)
        return missing

    def _summary_for(
        self,
        query: str,
        stage: Stage,
        shard_ids: tuple[int, ...],
        evidence: tuple[str, ...],
        hits: tuple[SearchHit, ...] = (),
    ) -> StateSummary:
        expansions = tuple(
            expand_query(
                self._index,
                query,
                shard_ids=list(shard_ids),
                processor=self._processor,
                max_terms=self._max_expansions,
            )
        )
        missing = tuple(self._top_keywords_not_in_query(shard_ids, query))
        entities = tuple(self._processor.entities(query))
        return StateSummary(
            current_query=query,
            stage=stage,
            top_shard_ids=shard_ids,
            expansions=expansions,
            missing_keywords=missing,
            entities=entities,
            evidence_terms=evidence,
            top_hits=tuple(hits[:5]),
        )

    def _route(
        self, query: str, n_shards: int, *, rerank: bool = True
    ) -> tuple[list[int], RoutingSignals]:
        return route(
            self._index,
            query,
            processor=self._processor,
            provider=self._provider,
            rrf_config=self._rrf_config,
            top_n=n_shards,
            rerank=rerank,
        )

    def _candidate_action(
        self,
        candidate: QueryCandidate,
        shard_ids: tuple[int, ...],
        metadata_filter: MetadataFilter | None,
    ) -> Action:
        if candidate.expansions:
            return AddExpansions(tuple(candidate.expansions))
        if metadata_filter is not None and candidate.origin == "original":
            # The agent never invents a filter; re-executing the user's own is the
            # post's RetainFilter action, and this is where it surfaces in traces.
            return RetainFilter(metadata_filter)
        if shard_ids:
            return NarrowShards(shard_ids)
        return RewriteQuery(candidate.text)

    def _judge_for(self, policy: StagePolicy) -> Judge:
        """The cheap wide judge for shallow stages, the configured one for deep."""
        if not policy.use_model_judge and self._wide_judge is not None:
            return self._wide_judge
        return self._judge

    def _apply_temperature(self, policy: StagePolicy) -> None:
        """Anneal the generator when it exposes a temperature knob."""
        if hasattr(self._generator, "temperature"):
            self._generator.temperature = policy.temperature

    def _usage(self) -> dict[str, float]:
        return {
            "retrieval_calls": float(self._counters.retrieval_calls),
            "embedding_calls": float(self._counters.embedding_calls),
            "llm_calls": float(
                getattr(self._generator, "calls", 0)
                + getattr(self._judge, "calls", 0)
                + getattr(self._wide_judge, "calls", 0)
            ),
            "nodes": float(len(self._all_nodes)),
        }

    def _account_stage(self, policy: StagePolicy, stage: Stage, before: dict[str, float]) -> None:
        after = self._usage()
        delta = {key: after[key] - before[key] for key in after}
        delta["temperature"] = policy.temperature
        self._phase_accounting[stage.value] = delta

    def _evidence_from_hits(self, hits: list[SearchHit]) -> tuple[str, ...]:
        terms: list[str] = []
        seen: set[str] = set()
        for hit in hits[:3]:
            for token in self._processor.content_tokens(hit.document.title):
                if token not in seen:
                    seen.add(token)
                    terms.append(token)
        return tuple(terms[:10])

    def _execute(
        self,
        parent: SearchNode,
        candidate: QueryCandidate,
        stage: Stage,
        policy: StagePolicy,
        metadata_filter: MetadataFilter | None,
        *,
        lexical_weight: float | None = None,
        dense_weight: float | None = None,
        rerank: bool = True,
        action: Action | None = None,
    ) -> SearchNode | None:
        """Route, retrieve, judge, and score one candidate. Returns None if budget-blocked."""
        if not self._counters.budget_available():
            self._budget_hit = True
            return None

        query = candidate.text
        query_tokens = self._processor.content_tokens(query)
        if not query_tokens:
            query_tokens = self._processor.tokenize(query)
        if not query_tokens and query.isascii():
            return None

        weight_lex = parent.state.lexical_weight if lexical_weight is None else lexical_weight
        weight_dense = parent.state.dense_weight if dense_weight is None else dense_weight

        routed, signals = self._route(
            query, min(policy.shards, self._n_shards_total()), rerank=rerank
        )
        expansions = tuple(candidate.expansions)

        self._counters.retrieval_calls += 1
        plan = RetrievalPlan(
            query=query,
            shard_ids=tuple(routed),
            expansions=expansions,
            lexical_weight=weight_lex,
            dense_weight=weight_dense,
            per_shard_limit=policy.hits_per_shard,
            top_k=max(policy.hits_per_shard * len(routed), self._output_top_k),
        )
        hits = self._run_retrieval(plan, metadata_filter)

        judge = self._judge_for(policy)
        judge_score = judge.score(
            parent.state.question, query, hits, signals=signal_summary(hits)
        )
        reward, components = compute_reward(judge_score, hits)

        state = SearchState(
            question=parent.state.question,
            query=query,
            stage=stage,
            shard_ids=tuple(routed),
            expansions=expansions,
            metadata_filter=metadata_filter,
            lexical_weight=weight_lex,
            dense_weight=weight_dense,
            routing_signals=signals,
            hits=tuple(hits),
            evidence_terms=self._evidence_from_hits(hits),
            parent_action=(
                action
                if action is not None
                else self._candidate_action(candidate, tuple(routed), metadata_filter)
            ),
        )
        node = parent.add_child(self._next_id(), state)
        node.reward = reward
        node.reward_components = components
        node.backpropagate(reward)
        self._all_nodes.append(node)

        self._node_traces.append(
            NodeTrace(
                node_id=node.id,
                parent_id=parent.id,
                stage=stage.value,
                action=describe_action(state.parent_action) if state.parent_action else None,
                query=query,
                shard_ids=list(routed),
                expansions=list(expansions),
                lexical_weight=state.lexical_weight,
                dense_weight=state.dense_weight,
                routing=routing_to_dict(signals),
                hit_ids=[hit.document.id for hit in hits],
                reward=reward,
                reward_components=components,
                visits=node.visits,
                cumulative_value=node.total_value,
                mean_value=node.mean_value,
                uct=node.uct_score(self._c),
                judge_reason=judge_score.reason,
            )
        )
        return node

    def _run_retrieval(
        self, plan: RetrievalPlan, metadata_filter: MetadataFilter | None
    ) -> list[SearchHit]:
        rrf_config = self._rrf_config
        if plan.lexical_weight != 1.0 or plan.dense_weight != 1.0:
            # The plan's weights are multipliers on the configured RRF weights —
            # this is what makes AdjustHybridWeights a real knob, not trace décor.
            rrf_config = rrf_config.model_copy(
                update={
                    "lexical_weight": rrf_config.lexical_weight * plan.lexical_weight,
                    "dense_weight": rrf_config.dense_weight * plan.dense_weight,
                }
            )
        return retrieve(
            self._index,
            plan.query,
            mode="hybrid",
            processor=self._processor,
            provider=self._provider,
            metadata_filter=metadata_filter,
            rrf_config=rrf_config,
            top_k=plan.top_k,
            shard_ids=list(plan.shard_ids),
            per_shard_limit=plan.per_shard_limit,
            expansions=list(plan.expansions),
            query_variant=plan.query,
        )

    def _generate(
        self, question: str, summary: StateSummary, n: int
    ) -> list[QueryCandidate]:
        candidates = self._generator.generate(question, summary, n)
        fresh: list[QueryCandidate] = []
        for candidate in candidates:
            key = _normalize_candidate(candidate)
            if key in self._seen:
                continue
            self._seen.add(key)
            fresh.append(candidate)
        if not fresh:
            self._duplicate_hit = True
        return fresh

    # --------------------------- stages ----------------------------- #

    def run(
        self,
        question: str,
        metadata_filter: MetadataFilter | None,
        config_snapshot: dict[str, object],
    ) -> AgentResult:
        timings: list[StageTiming] = []
        root_state = SearchState(question=question, query=question, stage=Stage.EXPLORE)
        root = SearchNode(id=0, state=root_state)

        explore_policy = self._schedule[Stage.EXPLORE]
        refine_policy = self._schedule[Stage.REFINE]
        exploit_policy = self._schedule[Stage.EXPLOIT]

        seed_routed, _ = self._route(
            question, min(explore_policy.shards, self._n_shards_total())
        )

        # ---- Explore ----
        start = time.perf_counter()
        usage = self._usage()
        self._apply_temperature(explore_policy)
        summary = self._summary_for(question, Stage.EXPLORE, tuple(seed_routed), ())
        explore_candidates = self._generate(question, summary, explore_policy.candidates)
        explore_nodes = self._run_candidates(
            root, explore_candidates, Stage.EXPLORE, explore_policy, metadata_filter
        )
        timings.append(StageTiming(stage=Stage.EXPLORE.value, seconds=time.perf_counter() - start))
        self._account_stage(explore_policy, Stage.EXPLORE, usage)
        survivors = self._survivors(explore_nodes, explore_policy.survivors)

        # ---- Refine (UCT-ordered budget allocation) ----
        start = time.perf_counter()
        usage = self._usage()
        self._apply_temperature(refine_policy)
        refine_nodes = self._run_refine(question, survivors, refine_policy, metadata_filter)
        timings.append(StageTiming(stage=Stage.REFINE.value, seconds=time.perf_counter() - start))
        self._account_stage(refine_policy, Stage.REFINE, usage)
        refine_survivors = self._survivors(refine_nodes, refine_policy.survivors)
        if not refine_survivors:
            refine_survivors = survivors

        # ---- Exploit ----
        start = time.perf_counter()
        usage = self._usage()
        self._apply_temperature(exploit_policy)
        exploit_nodes = self._run_exploit(
            question, refine_survivors, exploit_policy, metadata_filter
        )
        timings.append(StageTiming(stage=Stage.EXPLOIT.value, seconds=time.perf_counter() - start))
        self._account_stage(exploit_policy, Stage.EXPLOIT, usage)

        winner = self._winner(exploit_nodes or refine_nodes or explore_nodes)
        return self._finalize(
            question,
            metadata_filter,
            config_snapshot,
            root,
            winner,
            timings,
            completed_exploit=bool(exploit_nodes),
        )

    def _run_candidates(
        self,
        parent: SearchNode,
        candidates: list[QueryCandidate],
        stage: Stage,
        policy: StagePolicy,
        metadata_filter: MetadataFilter | None,
    ) -> list[SearchNode]:
        nodes: list[SearchNode] = []
        for candidate in candidates:
            node = self._execute(parent, candidate, stage, policy, metadata_filter)
            if node is None:
                break
            nodes.append(node)
        return nodes

    def _knob_variants(self, policy: StagePolicy) -> list[_KnobVariant]:
        """The knob-turning branches this stage's policy declares, in fixed order."""
        variants = [
            _KnobVariant(lexical_weight=lex, dense_weight=dense)
            for lex, dense in policy.weight_variants
        ]
        if policy.try_rerank_off:
            variants.append(_KnobVariant(1.0, 1.0, rerank=False))
        return variants

    def _execute_variant(
        self,
        parent: SearchNode,
        variant: _KnobVariant,
        stage: Stage,
        policy: StagePolicy,
        metadata_filter: MetadataFilter | None,
    ) -> SearchNode | None:
        """Re-run the parent's query with one knob turned. Spends a candidate slot."""
        action: Action
        if not variant.rerank:
            action = ToggleRerankers(enabled=False)
        else:
            action = AdjustHybridWeights(
                lexical_weight=variant.lexical_weight, dense_weight=variant.dense_weight
            )
        candidate = QueryCandidate(text=parent.state.query, origin="knob_variant")
        return self._execute(
            parent,
            candidate,
            stage,
            policy,
            metadata_filter,
            lexical_weight=variant.lexical_weight,
            dense_weight=variant.dense_weight,
            rerank=variant.rerank,
            action=action,
        )

    def _run_refine(
        self,
        question: str,
        survivors: list[SearchNode],
        policy: StagePolicy,
        metadata_filter: MetadataFilter | None,
    ) -> list[SearchNode]:
        per = policy.per_survivor or 1
        max_candidates = policy.candidates
        pending: dict[int, list[QueryCandidate | _KnobVariant]] = {}
        by_id: dict[int, SearchNode] = {}
        for survivor in survivors:
            summary = self._summary_for(
                survivor.state.query,
                Stage.REFINE,
                survivor.state.shard_ids,
                survivor.state.evidence_terms,
                survivor.state.hits,
            )
            pending[survivor.id] = list(self._generate(question, summary, per))
            by_id[survivor.id] = survivor
        if survivors:
            # Knob variants branch from the best explore survivor: same query,
            # different hybrid weights / reranker setting. They queue behind the
            # survivor's query rewrites and spend ordinary candidate slots.
            pending[survivors[0].id].extend(self._knob_variants(policy))

        executed: list[SearchNode] = []
        while len(executed) < max_candidates and self._counters.budget_available():
            available = [by_id[sid] for sid, cands in pending.items() if cands]
            if not available:
                break
            chosen = select_child(available, self._c)
            item = pending[chosen.id].pop(0)
            if isinstance(item, _KnobVariant):
                node = self._execute_variant(chosen, item, Stage.REFINE, policy, metadata_filter)
            else:
                node = self._execute(chosen, item, Stage.REFINE, policy, metadata_filter)
            if node is None:
                break
            executed.append(node)
        return executed

    def _run_exploit(
        self,
        question: str,
        survivors: list[SearchNode],
        policy: StagePolicy,
        metadata_filter: MetadataFilter | None,
    ) -> list[SearchNode]:
        per = policy.per_survivor or 1
        max_candidates = policy.candidates
        executed: list[SearchNode] = []
        for survivor in survivors:
            if len(executed) >= max_candidates or not self._counters.budget_available():
                break
            summary = self._summary_for(
                survivor.state.query,
                Stage.EXPLOIT,
                survivor.state.shard_ids,
                survivor.state.evidence_terms,
                survivor.state.hits,
            )
            candidates = self._generate(question, summary, per)
            for candidate in candidates:
                if len(executed) >= max_candidates:
                    break
                node = self._execute(survivor, candidate, Stage.EXPLOIT, policy, metadata_filter)
                if node is None:
                    break
                executed.append(node)
        return executed

    def _survivors(self, nodes: list[SearchNode], count: int) -> list[SearchNode]:
        return sorted(nodes, key=lambda node: (-node.mean_value, node.id))[:count]

    def _winner(self, nodes: list[SearchNode]) -> SearchNode | None:
        candidates = nodes or self._all_nodes
        if not candidates:
            return None
        return max(candidates, key=lambda node: (node.mean_value, -node.id))

    def _finalize(
        self,
        question: str,
        metadata_filter: MetadataFilter | None,
        config_snapshot: dict[str, object],
        root: SearchNode,
        winner: SearchNode | None,
        timings: list[StageTiming],
        *,
        completed_exploit: bool,
    ) -> AgentResult:
        # The stop reason reports why the run ended or was limited. Duplicate exhaustion
        # is only surfaced when it actually prevented the final stage from running.
        if winner is None or not winner.state.hits:
            stop_reason: str | None = StopReason.NO_RESULTS.value
        elif self._budget_hit:
            stop_reason = StopReason.BUDGET_EXHAUSTED.value
        elif self._duplicate_hit and not completed_exploit:
            stop_reason = StopReason.DUPLICATE_CANDIDATES.value
        else:
            stop_reason = None

        # Model-backed providers count their own invocations; heuristics have none.
        self._counters.llm_calls = getattr(self._generator, "calls", 0) + getattr(
            self._judge, "calls", 0
        )

        if winner is None:
            hits: list[SearchHit] = []
            best_query = question
            expansions: list[str] = []
            shard_ids: list[int] = []
            selected_path = [root.id]
            plan = RetrievalPlan(question, (), (), 1.0, 1.0, 0, 0)
        else:
            hits = list(winner.state.hits[: self._output_top_k])
            best_query = winner.state.query
            expansions = list(winner.state.expansions)
            shard_ids = list(winner.state.shard_ids)
            selected_path = [node.id for node in winner.path_from_root()]
            winner_policy = self._schedule[winner.state.stage]
            plan = RetrievalPlan(
                query=winner.state.query,
                shard_ids=winner.state.shard_ids,
                expansions=winner.state.expansions,
                lexical_weight=winner.state.lexical_weight,
                dense_weight=winner.state.dense_weight,
                per_shard_limit=winner_policy.hits_per_shard,
                top_k=max(
                    winner_policy.hits_per_shard * len(winner.state.shard_ids),
                    self._output_top_k,
                ),
            )

        # Per-phase accounting rides in the trace's free-form config dict; the
        # trace id stays a function of the *input* configuration only, so two runs
        # of one question keep one id while their fingerprints still cover the
        # (deterministic) accounting.
        trace_config: dict[str, object] = dict(config_snapshot)
        trace_config["phase_accounting"] = {
            stage: dict(values) for stage, values in self._phase_accounting.items()
        }

        trace = AgentTrace(
            trace_id=make_trace_id(question, self._seed, config_snapshot),
            question=question,
            normalized_tokens=self._processor.content_tokens(question),
            seed=self._seed,
            config=trace_config,
            nodes=self._node_traces,
            selected_path=selected_path,
            final_query=best_query,
            final_expansions=expansions,
            final_shard_ids=shard_ids,
            stop_reason=stop_reason,
            stage_timings=timings,
            embedding_calls=self._counters.embedding_calls,
            retrieval_calls=self._counters.retrieval_calls,
            llm_calls=self._counters.llm_calls,
        )
        return AgentResult(
            question=question,
            best_query=best_query,
            expansions=expansions,
            metadata_filter=metadata_filter,
            shard_ids=shard_ids,
            hits=hits,
            trace=trace,
            plan=plan,
            phase_accounting={
                stage: dict(values) for stage, values in self._phase_accounting.items()
            },
        )

    def replay(
        self, plan: RetrievalPlan, metadata_filter: MetadataFilter | None
    ) -> list[SearchHit]:
        """Reproduce a node's hits from its stored retrieval plan."""
        return self._run_retrieval(plan, metadata_filter)[: self._output_top_k]


def run_agent_search(
    index: LoadedIndex,
    question: str,
    *,
    config: AppConfig,
    processor: TextProcessor,
    provider: EmbeddingProvider,
    metadata_filter: MetadataFilter | None = None,
    output_top_k: int = 10,
) -> tuple[AgentResult, SearchAgent]:
    """Construct the configured providers and run the agent for one question.

    ``config.agent.judge`` / ``config.agent.query_generator`` select between the
    free heuristics (default) and the Hugging Face model-backed providers, which
    run on MPS/CUDA/CPU per ``config.agent.device``. The cross-encoder judge
    reuses the session's embedding provider for semantic redundancy; when it is
    configured, Explore still judges with the free heuristic (wide-and-cheap) and
    only Refine/Exploit pay the model cost (narrow-and-deep).

    The stage schedule is config-driven through the optional
    ``config.agent.stage_schedule`` attribute (``"default"``, ``"annealed"``, or a
    per-stage override mapping — see
    :func:`cybernaut_mini.agent.schedule.schedule_from_config`); a config model
    without that field runs the default schedule.
    """
    generator = create_query_generator(config.agent, processor)
    judge = create_judge(config.agent, processor, embedder=provider)
    wide_judge: Judge | None = None
    if config.agent.judge != "heuristic":
        wide_judge = HeuristicJudge(processor)
    raw_schedule = getattr(config.agent, "stage_schedule", None)
    agent = SearchAgent(
        index,
        processor=processor,
        provider=provider,
        generator=generator,
        judge=judge,
        rrf_config=config.rrf,
        exploration_constant=config.agent.exploration_constant,
        max_expansions=config.agent.max_expansions,
        max_retrieval_calls=config.agent.max_retrieval_calls,
        seed=config.seed,
        output_top_k=output_top_k,
        stage_schedule=schedule_from_config(raw_schedule),
        wide_judge=wide_judge,
    )
    config_snapshot = {
        "seed": config.seed,
        "embedding": config.embedding.model_dump(mode="json"),
        "rrf": config.rrf.model_dump(mode="json"),
        "agent": config.agent.model_dump(mode="json"),
    }
    result = agent.run(question, metadata_filter, config_snapshot)
    return result, agent
