# AGENTS.md

## Purpose
This repo uses a planner-builder-reviewer workflow.

- The primary model is the orchestrator.
- Planning and review are high-value tasks.
- Implementation should be delegated to cheaper coding agents whenever possible.
- Keep behavior simple, interpretable, and file-based.

## Roles

### Planner
Use the strongest reasoning model available for:
- Clarifying requirements.
- Producing implementation plans.
- Identifying risks, assumptions, and edge cases.
- Reviewing completed work against acceptance criteria.

The planner must not start coding unless explicitly instructed.

### Builder
Use cheaper coding agents for:
- Writing code.
- Editing files.
- Refactoring narrow scopes.
- Adding tests.
- Running verification commands.
- Updating docs tied to the change.

### Reviewer
Use the planner or a dedicated review agent to:
- Compare output to the agreed plan.
- Check for regressions.
- Check whether tests actually prove the change.
- Decide whether the next step is fix, continue, or stop.

## Core workflow

1. Restate the task briefly.
2. Identify ambiguities and ask only blocking questions.
3. Write a plan before making significant changes.
4. Break work into small milestones.
5. For each milestone, delegate implementation to a builder agent.
6. After implementation, review against acceptance criteria.
7. Only then move to the next milestone.

## Planning output format
Before implementation, produce:

- Goal.
- Constraints.
- Assumptions.
- Files likely to change.
- Ordered implementation steps.
- Verification steps.
- Acceptance criteria.
- Risks or open questions.

Keep plans concrete and short.

## Delegation rules

- Delegate implementation by default.
- Delegate any task that is mostly mechanical or localized.
- Keep each builder task narrow and explicit.
- Give the builder the exact milestone, files, and done criteria.
- Do not let the builder redesign the approach unless it finds a blocker.
- If a blocker appears, return to planning mode.

## Builder rules

- Change the minimum number of files needed.
- Prefer existing patterns over new abstractions.
- Do not introduce large frameworks or indirection unless required.
- Keep functions, classes, and modules easy for humans to read.
- Preserve backward compatibility unless the task explicitly allows breaking changes.
- Add or update tests for changed behavior.
- Report what changed, why, and how it was verified.

## Review rules

On review, check:

- Does the implementation match the plan?
- Are acceptance criteria satisfied?
- Are edge cases covered?
- Are tests relevant and sufficient?
- Is the code simpler than the obvious alternative?
- Were unnecessary files or abstractions added?

If not acceptable, return a short fix list and re-delegate.

## Coding principles

- Favor simple, explicit code over clever code.
- Prefer small diffs.
- Reuse existing project conventions.
- Avoid speculative generalization.
- Avoid hidden magic.
- Keep naming literal and predictable.
- Keep comments for intent, not narration.

## File and change discipline

- Read relevant files before editing.
- Do not rewrite unrelated code.
- Do not rename files unless needed.
- Do not make drive-by formatting changes.
- Call out risky migrations before performing them.

## Testing and verification

Always run the smallest meaningful verification first, then broader checks if needed.

Preferred order:
1. Targeted unit or feature test.
2. Type check or lint for touched code.
3. Broader test suite only when justified.

If you cannot run a command, say so clearly and explain the impact.

## Communication style

- Be concise.
- Show plans as bullets.
- Show progress milestone by milestone.
- Surface blockers early.
- Distinguish facts from assumptions.
- Do not pretend verification happened if it did not.

## Decision policy

Once a plan is approved:
- Treat architecture and scope decisions as fixed.
- Do not reopen settled decisions unless a real blocker appears.
- If a blocker appears, explain it briefly and propose the smallest plan change.

## Subagent patterns

Preferred builder agent types:
- coder: implements a scoped change.
- tester: adds or fixes tests.
- refactorer: simplifies code without changing behavior.
- researcher: inspects the repo and finds relevant files or patterns.
- reviewer: checks output against plan and acceptance criteria.

Use separate agents when context isolation helps.

## When not to delegate

Do not delegate when:
- The task is tiny enough to finish directly in less effort than delegation.
- The task is mostly architectural planning.
- The task involves sensitive repo-wide decisions that need central reasoning.

## Definition of done

A task is done only when:
- The requested behavior is implemented.
- Acceptance criteria are met.
- Relevant tests pass, or test limitations are clearly stated.
- The diff is minimal and understandable.
- Any follow-up work is listed separately, not mixed into the change.

## Default instruction for planner
Use the best reasoning model for planning and review.
Use cheaper agents for implementation.
Do not code during planning.
Do not let builders rethink approved decisions.
Return to planning only for blockers, failed verification, or scope changes.
