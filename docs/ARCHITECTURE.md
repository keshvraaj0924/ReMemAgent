# ReMemAgent Architecture

## Decision principle

Memory is treated as evidence, not truth. Retrieval produces candidates; reconstruction aligns a candidate with the current state; counterfactual routing estimates whether the memory path is better than self-reasoning; outcome attribution updates the memory lifecycle.

## Memory lifecycle

```text
Episode -> Episodic Memory -> Reuse Evidence -> Consolidation -> Semantic/Procedural Memory
                                      |
                                      +-> Failure Evidence -> Avoidance Memory
```

Every record carries empirical success, transferability, confidence, and lifecycle state. This makes memory quality observable and allows stale or low-transfer memories to be retired.

## Routing

The router compares an estimated memory-guided score with a memory-free baseline. It can select:

- `memory`: memory is expected to provide meaningful benefit.
- `hybrid`: benefit is small; retain memory as evidence while relying on self-reasoning.
- `self`: expected memory benefit is below the configured threshold.

The production implementation will replace heuristic score components with learned value estimates during GRPO training.

## Research hypothesis

If an agent explicitly estimates the counterfactual value of retrieved experience, it should reduce negative transfer while preserving useful transfer. We will evaluate this directly using matched memory-on / memory-off trajectories and report negative-transfer rate, acceptance precision, OOD transfer, latency, and token cost.
