"""Three-stage Explore -> Refine -> Exploit search agent.

The tree is a staged beam search with UCT-ordered budget allocation in Refine
(honestly not full MCTS — see the README). One retrieval call is one candidate
query executed across its selected shard set; the shared budget defaults to 18,
matching the stage schedule 5 + 9 + 4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from cybernaut_mini.agent.actions import (
    Action,
    AddExpansions,
    NarrowShards,
    RewriteQuery,
    describe_action,
)
from cybernaut_mini.agent.node import SearchNode, select_child
from cybernaut_mini.agent.policy import compute_reward
from cybernaut_mini.agent.state import SearchState, Stage, StateSummary
from cybernaut_mini.config import AppConfig, RRFConfig
from cybernaut_mini.expansion import expand_query
from cybernaut_mini.indexing import LoadedIndex
from cybernaut_mini.models import MetadataFilter, QueryCandidate, SearchHit
from cybernaut_mini.providers.embeddings import EmbeddingProvider, FloatArray
from cybernaut_mini.providers.judge import HeuristicJudge, Judge
from cybernaut_mini.providers.query_generator import HeuristicQueryGenerator, QueryGenerator
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

# Per-stage budget: candidate queries, shards routed, hits per shard, survivors.
STAGE_SCHEDULE = {
    Stage.EXPLORE: {"candidates": 5, "shards": 12, "hits_per_shard": 3, "survivors": 3},
    Stage.REFINE: {
        "candidates": 9,
        "per_survivor": 3,
        "shards": 5,
        "hits_per_shard": 10,
        "survivors": 2,
    },
    Stage.EXPLOIT: {
        "candidates": 4,
        "per_survivor": 2,
        "shards": 3,
        "hits_per_shard": 40,
        "survivors": 1,
    },
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
    ) -> None:
        self._index = index
        self._processor = processor
        self._counters = Counters(max_retrieval_calls=max_retrieval_calls)
        self._provider = _CountingProvider(provider, self._counters)
        self._generator = generator
        self._judge = judge
        self._rrf_config = rrf_config
        self._c = exploration_constant
        self._max_expansions = max_expansions
        self._seed = seed
        self._output_top_k = output_top_k

        self._node_counter = 0
        self._node_traces: list[NodeTrace] = []
        self._seen: set[tuple[str, tuple[str, ...]]] = set()
        self._all_nodes: list[SearchNode] = []
        self._budget_hit = False
        self._duplicate_hit = False

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
        self, query: str, stage: Stage, shard_ids: tuple[int, ...], evidence: tuple[str, ...]
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
        )

    def _route(self, query: str, n_shards: int) -> tuple[list[int], RoutingSignals]:
        return route(
            self._index,
            query,
            processor=self._processor,
            provider=self._provider,
            rrf_config=self._rrf_config,
            top_n=n_shards,
            rerank=True,
        )

    def _candidate_action(self, candidate: QueryCandidate, shard_ids: tuple[int, ...]) -> Action:
        if candidate.expansions:
            return AddExpansions(tuple(candidate.expansions))
        if shard_ids:
            return NarrowShards(shard_ids)
        return RewriteQuery(candidate.text)

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
        n_shards: int,
        hits_per_shard: int,
        metadata_filter: MetadataFilter | None,
    ) -> SearchNode | None:
        """Route, retrieve, judge, and score one candidate. Returns None if budget-blocked."""
        if not self._counters.budget_available():
            self._budget_hit = True
            return None

        query = candidate.text
        query_tokens = self._processor.content_tokens(query)
        if not query_tokens:
            query_tokens = self._processor.tokenize(query)
        if not query_tokens:
            return None

        routed, signals = self._route(query, min(n_shards, self._n_shards_total()))
        expansions = tuple(candidate.expansions)

        self._counters.retrieval_calls += 1
        plan = RetrievalPlan(
            query=query,
            shard_ids=tuple(routed),
            expansions=expansions,
            lexical_weight=parent.state.lexical_weight,
            dense_weight=parent.state.dense_weight,
            per_shard_limit=hits_per_shard,
            top_k=max(hits_per_shard * len(routed), self._output_top_k),
        )
        hits = self._run_retrieval(plan, metadata_filter)

        judge_score = self._judge.score(parent.state.question, query, hits)
        reward, components = compute_reward(judge_score, hits)

        state = SearchState(
            question=parent.state.question,
            query=query,
            stage=stage,
            shard_ids=tuple(routed),
            expansions=expansions,
            metadata_filter=metadata_filter,
            lexical_weight=parent.state.lexical_weight,
            dense_weight=parent.state.dense_weight,
            routing_signals=signals,
            hits=tuple(hits),
            evidence_terms=self._evidence_from_hits(hits),
            parent_action=self._candidate_action(candidate, tuple(routed)),
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
        return retrieve(
            self._index,
            plan.query,
            mode="hybrid",
            processor=self._processor,
            provider=self._provider,
            metadata_filter=metadata_filter,
            rrf_config=self._rrf_config,
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

        explore_n = int(STAGE_SCHEDULE[Stage.EXPLORE]["shards"])
        seed_routed, _ = self._route(question, min(explore_n, self._n_shards_total()))

        # ---- Explore ----
        start = time.perf_counter()
        summary = self._summary_for(question, Stage.EXPLORE, tuple(seed_routed), ())
        explore_candidates = self._generate(
            question, summary, int(STAGE_SCHEDULE[Stage.EXPLORE]["candidates"])
        )
        explore_nodes = self._run_candidates(
            root, explore_candidates, Stage.EXPLORE, metadata_filter
        )
        timings.append(StageTiming(stage=Stage.EXPLORE.value, seconds=time.perf_counter() - start))
        survivors = self._survivors(explore_nodes, int(STAGE_SCHEDULE[Stage.EXPLORE]["survivors"]))

        # ---- Refine (UCT-ordered budget allocation) ----
        start = time.perf_counter()
        refine_nodes = self._run_refine(question, survivors, metadata_filter)
        timings.append(StageTiming(stage=Stage.REFINE.value, seconds=time.perf_counter() - start))
        refine_survivors = self._survivors(
            refine_nodes, int(STAGE_SCHEDULE[Stage.REFINE]["survivors"])
        )
        if not refine_survivors:
            refine_survivors = survivors

        # ---- Exploit ----
        start = time.perf_counter()
        exploit_nodes = self._run_exploit(question, refine_survivors, metadata_filter)
        timings.append(StageTiming(stage=Stage.EXPLOIT.value, seconds=time.perf_counter() - start))

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
        metadata_filter: MetadataFilter | None,
    ) -> list[SearchNode]:
        schedule = STAGE_SCHEDULE[stage]
        nodes: list[SearchNode] = []
        for candidate in candidates:
            node = self._execute(
                parent,
                candidate,
                stage,
                int(schedule["shards"]),
                int(schedule["hits_per_shard"]),
                metadata_filter,
            )
            if node is None:
                break
            nodes.append(node)
        return nodes

    def _run_refine(
        self,
        question: str,
        survivors: list[SearchNode],
        metadata_filter: MetadataFilter | None,
    ) -> list[SearchNode]:
        schedule = STAGE_SCHEDULE[Stage.REFINE]
        per = int(schedule["per_survivor"])
        max_candidates = int(schedule["candidates"])
        pending: dict[int, list[QueryCandidate]] = {}
        by_id: dict[int, SearchNode] = {}
        for survivor in survivors:
            summary = self._summary_for(
                survivor.state.query,
                Stage.REFINE,
                survivor.state.shard_ids,
                survivor.state.evidence_terms,
            )
            pending[survivor.id] = self._generate(question, summary, per)
            by_id[survivor.id] = survivor

        executed: list[SearchNode] = []
        while len(executed) < max_candidates and self._counters.budget_available():
            available = [by_id[sid] for sid, cands in pending.items() if cands]
            if not available:
                break
            chosen = select_child(available, self._c)
            candidate = pending[chosen.id].pop(0)
            node = self._execute(
                chosen,
                candidate,
                Stage.REFINE,
                int(schedule["shards"]),
                int(schedule["hits_per_shard"]),
                metadata_filter,
            )
            if node is None:
                break
            executed.append(node)
        return executed

    def _run_exploit(
        self,
        question: str,
        survivors: list[SearchNode],
        metadata_filter: MetadataFilter | None,
    ) -> list[SearchNode]:
        schedule = STAGE_SCHEDULE[Stage.EXPLOIT]
        per = int(schedule["per_survivor"])
        max_candidates = int(schedule["candidates"])
        executed: list[SearchNode] = []
        for survivor in survivors:
            if len(executed) >= max_candidates or not self._counters.budget_available():
                break
            summary = self._summary_for(
                survivor.state.query,
                Stage.EXPLOIT,
                survivor.state.shard_ids,
                survivor.state.evidence_terms,
            )
            candidates = self._generate(question, summary, per)
            for candidate in candidates:
                if len(executed) >= max_candidates:
                    break
                node = self._execute(
                    survivor,
                    candidate,
                    Stage.EXPLOIT,
                    int(schedule["shards"]),
                    int(schedule["hits_per_shard"]),
                    metadata_filter,
                )
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
            schedule = STAGE_SCHEDULE[winner.state.stage]
            plan = RetrievalPlan(
                query=winner.state.query,
                shard_ids=winner.state.shard_ids,
                expansions=winner.state.expansions,
                lexical_weight=winner.state.lexical_weight,
                dense_weight=winner.state.dense_weight,
                per_shard_limit=int(schedule["hits_per_shard"]),
                top_k=max(
                    int(schedule["hits_per_shard"]) * len(winner.state.shard_ids),
                    self._output_top_k,
                ),
            )

        trace = AgentTrace(
            trace_id=make_trace_id(question, self._seed, config_snapshot),
            question=question,
            normalized_tokens=self._processor.content_tokens(question),
            seed=self._seed,
            config=config_snapshot,
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
    """Construct heuristic providers and run the agent for one question."""
    generator = HeuristicQueryGenerator(processor)
    judge = HeuristicJudge(processor)
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
    )
    config_snapshot = {
        "seed": config.seed,
        "embedding": config.embedding.model_dump(mode="json"),
        "rrf": config.rrf.model_dump(mode="json"),
        "agent": config.agent.model_dump(mode="json"),
    }
    result = agent.run(question, metadata_filter, config_snapshot)
    return result, agent
