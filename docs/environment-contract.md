# Runtime environment contract probe

ReMemAgent keeps external benchmark packages outside the research core, but a normalized adapter still needs a runtime check before an expensive benchmark run. `remem.environments.validation.validate_environment_contract` provides that check.

## What it validates

The probe always calls `reset(**reset_kwargs)` and requires the returned observation to be a string. When `probe_action` is supplied, it additionally calls `step(probe_action)` and verifies that:

- the result is a `StepResult`;
- the next observation is a string;
- the reward is finite;
- `terminated` and `truncated` are booleans.

The validator closes the environment in a `finally` block, including when validation fails.

## Why the action is explicit

There is no safe universal no-op action across ALFWorld, WebShop, and arbitrary caller-owned adapters. The validator therefore never invents an action. A caller that wants step-level validation must provide an action appropriate for the concrete environment.

A reset-only probe remains useful for factories whose first valid action requires model inference or task-specific state inspection.

## Scope

This is an integration sanity check, not a benchmark. It does not load a model, score a task, or establish scientific performance. It should be run against the same concrete adapter and environment configuration that will be used for the subsequent benchmark.
