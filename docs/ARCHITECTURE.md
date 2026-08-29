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

## Execution and guidance boundary

Benchmark environments implement `EnvironmentAdapter`, which normalizes reset and step behavior into deterministic `StepResult` values. `EpisodeRunner` executes a policy without importing benchmark-specific packages. `EpisodeMemoryRecorder` converts the resulting trajectory into typed episodic memories, and `EpisodeMemoryIngestor` deduplicates those memories before storing them.

`MemoryGuidedPolicy` is the composition boundary between that stored evidence and an action policy. For each state it retrieves and reconstructs the strongest trusted candidate, then supplies the resulting guidance to an injected action policy. The action policy remains responsible for deciding the final environment action; the deterministic memory layer never executes reconstructed text as an action.

```text
EnvironmentAdapter
        |
  EpisodeRunner
        |
   EpisodeResult
        |
 EpisodeMemoryRecorder
        |
 EpisodeMemoryIngestor
        |
    MemoryStore
        |
 MemoryGuidedPolicy -> learned / heuristic action policy
        |
      action
```

This separation allows learned policy components to be introduced later without coupling the memory engine to ALFWorld, WebShop, an LLM provider, or a training framework.

## Routing

The repository currently contains two routing contracts for different research stages. `adaptive_router.py` exposes the richer `memory` / `hybrid` / `self` decision with explicit memory and reconstruction quality features. `counterfactual.py` exposes an injected evaluator contract for matched memory-on / memory-off utility estimates. They should remain separate until the learned routing experiment establishes a stable replacement contract.

The production research path will replace heuristic score components with learned value estimates during GRPO training while retaining the same explicit counterfactual interface.

## Research hypothesis

If an agent explicitly estimates the counterfactual value of retrieved experience, it should reduce negative transfer while preserving useful transfer. We will evaluate this directly using matched memory-on / memory-off trajectories and report negative-transfer rate, acceptance precision, OOD transfer, latency, and token cost.
