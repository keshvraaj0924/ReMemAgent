# External policy contract

ReMemAgent keeps learned policy ownership outside the research core. External integrations provide a `module:attribute` factory that constructs a policy for an episode seed; ReMemAgent supplies the normalized observation and consumes a string action.

## Runtime preflight

`validate_external_benchmark_runtime()` performs two independent checks before measured execution:

1. Constructs the configured benchmark environment through the same normalized adapter used by the benchmark runner and validates `reset()` plus an optional caller-selected step action.
2. Constructs the configured policy with the same probe seed and an isolated `MemoryStore`, then calls it once with the normalized reset observation.

The policy probe validates the actual `seed -> policy -> observation -> action` boundary. If the policy factory loads a checkpoint or model, that loading occurs during the probe because model ownership remains with the caller.

The probe is not benchmark data. It is a fail-fast integration check and its output is not included in measured reports.

## Contract

A complete policy factory has the shape:

```text
(seed: int, store: MemoryStore) -> (observation: str) -> action: str
```

For an `action_policy_factory`, the integration layer first composes it with `MemoryGuidedPolicy`; the caller-owned factory receives only the deterministic episode seed and returns its own guided-action callable.

Policies must return a non-empty string action. ReMemAgent does not assume a universal no-op action because external environments may have different action spaces.

## Why this boundary matters

A successful environment reset alone does not establish that a benchmark is runnable. A misconfigured checkpoint, tokenizer, model wrapper, or action decoder can fail only when the first observation reaches the learned component. Probing the policy immediately after the environment reset catches that class of integration failure before the measured episode suite starts.

The core library remains free of ALFWorld, WebShop, model SDK, tokenizer, and checkpoint dependencies.
