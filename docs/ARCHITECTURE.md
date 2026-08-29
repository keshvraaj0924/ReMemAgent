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

`MemoryGuidedPolicy` is the composition boundary between that stored evidence and an action policy. `select_guidance()` returns a `MemoryGuidanceDecision` containing the selected memory identifier, retrieval similarity, trust confidence, and reconstructed guidance. The callable policy path then passes only the guidance text to the injected action policy. The action policy remains responsible for deciding the final environment action; the deterministic memory layer never executes reconstructed text as an action.

When an action is evaluated, `MemoryTransferRecorder` can attribute the observed result to the selected memory. A self-reasoning decision with no selected memory is deliberately ignored, while a selected memory increments both ordinary use statistics and transfer-attempt statistics. This keeps transferability evidence tied to actual memory-guided decisions rather than retrieval alone.

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
 MemoryGuidedPolicy -- select_guidance() --> traceable memory decision
        |                                      |
 injected action policy                        |
        |                               MemoryTransferRecorder
      action                                     |
        |                               observed success/failure
        +----------------------------------------+
```

A service-level integration test exercises the stored-memory loop: one episode creates a memory, a later episode uses `MemoryGuidedPolicy`, retrieval reconstructs the prior experience, and the injected policy receives that guidance. Transfer attribution is intentionally a separate operation so experiments can define success without coupling the core memory layer to benchmark-specific reward semantics.

This separation allows learned policy components to be introduced later without coupling the memory engine to ALFWorld, WebShop, an LLM provider, or a training framework.

## Routing

The repository currently contains two routing contracts for different research stages. `adaptive_router.py` exposes the richer `memory` / `hybrid` / `self` decision with explicit memory and reconstruction quality features. `counterfactual.py` exposes an injected evaluator contract for matched memory-on / memory-off utility estimates. They should remain separate until the learned routing experiment establishes a stable replacement contract.

The production research path will replace heuristic score components with learned value estimates during GRPO training while retaining the same explicit counterfactual interface.

## Research hypothesis

If an agent explicitly estimates the counterfactual value of retrieved experience, it should reduce negative transfer while preserving useful transfer. We will evaluate this directly using matched memory-on / memory-off trajectories and report negative-transfer rate, acceptance precision, OOD transfer, latency, and token cost.
