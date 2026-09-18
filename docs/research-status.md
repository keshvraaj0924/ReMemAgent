# Research status and claims

ReMemAgent is a research framework for adaptive, reconstructive memory in language-model agents. This document separates implemented engineering capability from empirical claims that still require experiments.

## Research hypothesis

The central hypothesis is that retrieved experience should be treated as fallible evidence rather than replayed context. A memory can be relevant yet harmful when its assumptions do not transfer to the current state. ReMemAgent therefore decomposes memory use into retrieval, trust/transferability estimation, reconstruction, counterfactual routing, outcome attribution, and lifecycle management.

## Implemented scope

The repository contains deterministic and testable components for the memory domain model and storage, retrieval and deduplication, reconstruction, trust/transferability policies, counterfactual routing, failure memory, consolidation/lifecycle behavior, controlled negative-transfer evaluation, ablations and metrics, external-environment adapter boundaries, GRPO/verl-agent integration boundaries, reproducibility/provenance support, CI quality gates, and lightweight in-process observability.

These implementations establish an experimental framework. Their existence does not establish that ReMemAgent improves task success, sample efficiency, latency, or negative-transfer rate against competing memory systems.

## Evidence levels

Use the following language when reporting results:

1. **Unit-tested behavior** means a software contract is exercised by automated tests. It is not an empirical research result.
2. **Synthetic experimental result** means a metric was produced by a controlled synthetic benchmark with recorded configuration and seed. It supports conclusions only about that benchmark.
3. **Environment result** means an ALFWorld or WebShop experiment was actually executed with its environment/model configuration preserved. Adapter tests alone do not qualify.
4. **Training result** means a GRPO/verl-agent training run completed with reproducible configuration, checkpoints or sufficient provenance, and evaluation. Integration code alone does not qualify.
5. **Research claim** should require repeated runs, suitable baselines, uncertainty reporting, and enough provenance for independent reproduction.

## Claims not currently made

Until corresponding experiments are executed and preserved, the project does not claim:

- state-of-the-art performance;
- statistically significant task-success improvements;
- reduced token usage or latency in deployed LLM agents;
- superior ALFWorld or WebShop performance;
- successful GRPO policy improvement;
- general negative-transfer reduction outside controlled evaluation;
- production readiness.

## Evaluation protocol

For empirical comparisons, keep the task set, model, prompt contract, retrieval budget, memory budget, random seeds, and environment version fixed across policies. At minimum compare self-reasoning/no-memory, replay-style memory, reconstructive memory without routing, and the full routing policy when those variants are supported by the experiment.

Report task success and memory-specific outcomes separately. Memory evaluation should include acceptance/rejection behavior, negative-transfer events, routing decisions, memory growth or retirement, and latency/token overhead when measurable. Aggregate metrics must not hide per-seed failures.

Every reported table should identify the exact repository revision and experiment configuration that generated it. Failed or incomplete runs should remain distinguishable from zero-valued results.

## Reproducibility and observability

The reproducibility and runtime-provenance documentation defines how deterministic experiment identity and execution metadata are recorded. The observability module supplies dependency-free counters and duration aggregates for instrumentation; it is infrastructure rather than a benchmark result.

See `docs/reproducibility.md`, `docs/runtime-provenance.md`, `docs/policy-contract.md`, and `docs/ARCHITECTURE.md` for the corresponding contracts.

## Next empirical milestone

The highest-value next milestone is to execute the controlled benchmark and external-environment integrations under the documented reproducibility contract, archive raw outputs, and only then write quantitative conclusions. Learned routing/training claims should follow after the deterministic baselines have reproducible evidence.