"""Config-driven stage schedule with wide-to-narrow annealing for the search agent.

The agent's three stages used to be governed by a hard-coded module dict. This
module makes the schedule a value: a mapping from :class:`~cybernaut_mini.agent
.state.Stage` to a frozen :class:`StagePolicy`, buildable from configuration, with
two shipped instances —

* :data:`DEFAULT_SCHEDULE`, byte-identical in behaviour to the legacy dict for the
  numbers it shared (candidates 5/9/4, shards 12/5/3, hits 3/10/40, survivors
  3/2/1), extended with the knobs the legacy dict lacked: per-stage generator
  temperature (1.0 -> 0.6 -> 0.3), refine-stage hybrid-weight variants, and the
  wide-vs-deep judge split;
* :data:`ANNEALED_SCHEDULE`, the post's wide-shallow-to-narrow-deep shape stated
  as numbers: branching ``k_children`` decays 4 -> 2 -> 1, per-node ``top_k``
  (hits per shard) grows 10 -> 20 -> 40, and generator temperature anneals
  1.0 -> 0.3, inside the same 18-call retrieval budget.

Blog ref: https://nosible.com/blog/introducing-cybernaut-1-agentic-search-with-mcts
    — the search "starts wide and shallow, then narrows and deepens": early
    iterations branch broadly with cheap, shallow retrievals; late iterations
    commit to few branches with deep result pulls; and the agent "tunes every
    knob", including the hybrid lexical/dense balance and the rerankers. Local
    copy: ``docs/blog-archive/introducing-cybernaut-1-agentic-search-with-mcts.md``.

Assumptions:
    - [inferred] the post gives the annealing *direction*, not its constants. The
      endpoints here (4 -> 1 children, 10 -> 40 hits, 1.0 -> 0.3 temperature) are
      taken from the build plan and sized so the annealed schedule's 12 candidate
      executions fit the shared 18-call budget with headroom for weight variants.
    - [inferred] "tunes every knob" is expressed as *variant branches*: a refine
      step may re-execute a survivor's query with the lexical RRF weight scaled by
      0.5 or 1.5 (dense held at 1.0, the {0.5, 1.0, 1.5} grid relative to the
      configured weights) or with the stage-6 shard rerankers switched off. Those
      branches spend ordinary candidate slots, so enabling them never exceeds the
      budget the schedule already declares.
    - ``use_model_judge=False`` on Explore is the cheap-wide/expensive-deep split:
      when a model-backed judge is configured, Explore falls back to the heuristic
      judge and only Refine/Exploit pay for the cross-encoder. With the default
      heuristic judge the flag changes nothing.
    - Backward compatibility is a hard constraint: :data:`DEFAULT_SCHEDULE` keeps
      the legacy candidate/shard/hit/survivor numbers exactly, so existing traces,
      budgets and stop reasons are unchanged unless a caller opts into
      :data:`ANNEALED_SCHEDULE` or a config override.

Alternatives rejected:
    - Putting the schedule into ``AgentConfig`` directly: that pydantic model is a
      shared file owned by the config workstream. :func:`schedule_from_config`
      accepts the plain mapping a future ``agent.stage_schedule`` field would
      carry, so the wiring is one line when that field lands.
    - True MCTS-style continuous annealing (per-iteration decay instead of three
      plateaus): closer to the post's prose, but this agent is a staged beam
      search (see ``search.py``); per-stage plateaus are the honest granularity.
    - Making the weight variants full grid search over {0.5, 1.0, 1.5}^2: nine
      branches per survivor would starve query rewrites inside an 18-call budget.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace

from cybernaut_mini.agent.state import Stage

__all__ = [
    "ANNEALED_SCHEDULE",
    "DEFAULT_SCHEDULE",
    "StagePolicy",
    "StageSchedule",
    "schedule_from_config",
]


@dataclass(frozen=True, slots=True)
class StagePolicy:
    """Every knob one agent stage runs with.

    ``candidates`` caps how many retrieval-backed nodes the stage may execute;
    ``per_survivor`` (``k_children``) is how many children each surviving node may
    branch into (``None`` for Explore, which branches from the root only);
    ``hits_per_shard`` is the per-node retrieval depth (``top_k``);
    ``temperature`` is handed to the query generator when it supports sampling;
    ``weight_variants`` are ``(lexical, dense)`` RRF weight multipliers tried as
    :class:`~cybernaut_mini.agent.actions.AdjustHybridWeights` branches;
    ``try_rerank_off`` adds one branch with the stage-6 shard rerankers disabled
    (:class:`~cybernaut_mini.agent.actions.ToggleRerankers`);
    ``use_model_judge`` selects the configured (possibly model-backed) judge over
    the cheap wide judge for this stage.
    """

    candidates: int
    shards: int
    hits_per_shard: int
    survivors: int
    per_survivor: int | None = None
    temperature: float = 1.0
    weight_variants: tuple[tuple[float, float], ...] = ()
    try_rerank_off: bool = False
    use_model_judge: bool = True


#: A full schedule: one policy per stage.
StageSchedule = Mapping[Stage, StagePolicy]

#: The {0.5, 1.0, 1.5} lexical-weight grid relative to the configured RRF weights
#: (dense held at 1.0; (1.0, 1.0) is the parent branch itself and is not repeated).
_REFINE_WEIGHT_VARIANTS: tuple[tuple[float, float], ...] = ((0.5, 1.0), (1.5, 1.0))

#: Legacy numbers, now with temperature annealing, refine-stage knob variants and
#: the wide-vs-deep judge split. Candidate/shard/hit/survivor counts are unchanged.
DEFAULT_SCHEDULE: StageSchedule = {
    Stage.EXPLORE: StagePolicy(
        candidates=5,
        shards=12,
        hits_per_shard=3,
        survivors=3,
        temperature=1.0,
        use_model_judge=False,
    ),
    Stage.REFINE: StagePolicy(
        candidates=9,
        shards=5,
        hits_per_shard=10,
        survivors=2,
        per_survivor=3,
        temperature=0.6,
        weight_variants=_REFINE_WEIGHT_VARIANTS,
        try_rerank_off=True,
    ),
    Stage.EXPLOIT: StagePolicy(
        candidates=4,
        shards=3,
        hits_per_shard=40,
        survivors=1,
        per_survivor=2,
        temperature=0.3,
    ),
}

#: The post's wide-shallow-to-narrow-deep annealing stated as numbers:
#: k_children 4 -> 2 -> 1, top_k 10 -> 20 -> 40, temperature 1.0 -> 0.6 -> 0.3.
#: 4 + 6 + 2 = 12 candidate executions, leaving 6 calls of headroom in the
#: 18-call budget for knob-variant branches and the seed routing call.
ANNEALED_SCHEDULE: StageSchedule = {
    Stage.EXPLORE: StagePolicy(
        candidates=4,
        shards=12,
        hits_per_shard=10,
        survivors=3,
        temperature=1.0,
        use_model_judge=False,
    ),
    Stage.REFINE: StagePolicy(
        candidates=6,
        shards=5,
        hits_per_shard=20,
        survivors=2,
        per_survivor=2,
        temperature=0.6,
        weight_variants=_REFINE_WEIGHT_VARIANTS,
        try_rerank_off=True,
    ),
    Stage.EXPLOIT: StagePolicy(
        candidates=2,
        shards=3,
        hits_per_shard=40,
        survivors=1,
        per_survivor=1,
        temperature=0.3,
    ),
}

_NAMED: Mapping[str, StageSchedule] = {
    "default": DEFAULT_SCHEDULE,
    "annealed": ANNEALED_SCHEDULE,
}

_POLICY_FIELDS = frozenset(f.name for f in fields(StagePolicy))


def schedule_from_config(raw: str | Mapping[str, object] | None) -> StageSchedule:
    """Build a schedule from configuration.

    ``None`` and ``"default"`` return :data:`DEFAULT_SCHEDULE`; ``"annealed"``
    returns :data:`ANNEALED_SCHEDULE`. A mapping is a per-stage override on top of
    the default, e.g. ``{"refine": {"hits_per_shard": 20, "temperature": 0.5}}``;
    unknown stages or fields raise :class:`ValueError` so a typo in a config file
    fails at load rather than silently running the default.
    """
    if raw is None:
        return DEFAULT_SCHEDULE
    if isinstance(raw, str):
        named = _NAMED.get(raw.strip().lower())
        if named is None:
            known = ", ".join(sorted(_NAMED))
            msg = f"unknown stage schedule {raw!r}; known: {known}"
            raise ValueError(msg)
        return named

    result: dict[Stage, StagePolicy] = dict(DEFAULT_SCHEDULE)
    for stage_name, overrides in raw.items():
        try:
            stage = Stage(str(stage_name).strip().lower())
        except ValueError:
            known = ", ".join(s.value for s in Stage)
            msg = f"unknown agent stage {stage_name!r}; known: {known}"
            raise ValueError(msg) from None
        if not isinstance(overrides, Mapping):
            msg = f"stage {stage.value!r} overrides must be a mapping, got {overrides!r}"
            raise ValueError(msg)
        unknown = set(overrides) - _POLICY_FIELDS
        if unknown:
            msg = (
                f"unknown stage-policy field(s) for {stage.value!r}: {sorted(unknown)}; "
                f"known: {sorted(_POLICY_FIELDS)}"
            )
            raise ValueError(msg)
        coerced: dict[str, object] = dict(overrides)
        raw_variants = coerced.get("weight_variants")
        if raw_variants is not None:
            if not isinstance(raw_variants, Iterable) or isinstance(raw_variants, (str, bytes)):
                msg = f"weight_variants for {stage.value!r} must be a list of pairs"
                raise ValueError(msg)
            variants: list[tuple[float, float]] = []
            for pair in raw_variants:
                if not isinstance(pair, Sequence) or len(pair) != 2:
                    msg = f"weight_variants for {stage.value!r} must be (lexical, dense) pairs"
                    raise ValueError(msg)
                variants.append((float(pair[0]), float(pair[1])))
            coerced["weight_variants"] = tuple(variants)
        result[stage] = replace(result[stage], **coerced)  # type: ignore[arg-type]
    return result
