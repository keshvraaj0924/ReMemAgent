# Research status

This document records the implemented research surface without implying experimental conclusions that have not been measured.

## Verified engineering baseline

The `feat/core-memory-engine` branch has a previously verified green GitHub `Quality` workflow at commit `e9f02fc` (run `305`). The workflow reported the repository test, formatting, lint, and type-check stages as successful.

The framework currently provides deterministic implementations for:

- typed episodic memory and lifecycle management;
- retrieval, deduplication, reconstruction, trust, and transferability;
- counterfactual routing and failure-memory capture;
- consolidation and retirement policies;
- synthetic negative-transfer evaluation and ablation reporting;
- normalized ALFWorld/WebShop environment boundaries;
- an explicit external benchmark callable boundary and concrete ALFWorld/WebShop adapter-factory bridge;
- a caller-owned model-policy composition factory that injects ReMemAgent memory guidance without taking ownership of model loading or inference;
- GRPO batch normalization and an async verl agent-loop bridge;
- reproducible single-run and explicit multi-seed experiment execution with provenance-preserving JSON serialization;
- backend-neutral observability primitives.

Integration callable loading is centralized in `remem.integrations.loading`, so benchmark environment factories and external policy/evaluator factories share one explicit `module:attribute` resolution contract.

External benchmark reports retain the declared benchmark configuration alongside measured episode results, including episode count, step limit, seed, and the exact callable specifications used for the environment, policy, success evaluator, and optional transfer evaluator. The runner rejects supplied provenance metadata when its benchmark name, episode count, step limit, or seed differs from the actual invocation. This prevents a measured artifact from silently carrying stale run metadata.

The model-policy composition boundary accepts a caller-owned factory that receives the deterministic episode seed and returns the action-generation callable expected by `MemoryGuidedPolicy`. This keeps learned components, checkpoint loading, tokenization, and inference outside the research heuristic layer while making memory-guided benchmark policies directly composable. The external benchmark CLI now exposes this mode through `--action-policy-factory` and `--minimum-trust`; callers may alternatively provide a complete `--policy-factory` when they already own policy composition.

These capabilities are intentionally separated from model SDKs and external benchmark packages where practical.

## What has not been established

No benchmark improvement, transfer advantage, statistical significance, or production-readiness claim is made by this repository status. Those claims require executed experiments with fixed configurations, recorded outputs, and independently reproducible runs.

In particular, the following remain experimental integration work rather than verified end-to-end claims:

1. ALFWorld and WebShop runs against their real upstream environments and model checkpoints.
2. Full GRPO/verl training runs with real model checkpoints and distributed infrastructure.
3. Statistical analysis across repeated seeds for the negative-transfer benchmark and ablations.
4. External observability exporters and deployment-specific telemetry.

## Reproducibility contract

Research results should be recorded with:

- the benchmark configuration;
- random seed(s);
- code revision;
- environment/dependency versions;
- serialized per-run metrics;
- enough input metadata to reconstruct the run.

The repository provides a multi-seed runner that gives each synthetic case set an isolated random generator and preserves per-run case and experiment fingerprints. The external benchmark runner accepts an optional run seed: each episode receives `seed + episode_index` through both environment and policy factories, while the report records the run-level seed and declared callable configuration. Benchmark integrations provide a typed factory that resolves a caller-owned raw environment factory and wraps each created environment with the correct ALFWorld or WebShop adapter. The model-policy factory preserves the same episode seed when creating a caller-owned action policy, allowing deterministic checkpoint/model selection without embedding model-specific logic in ReMemAgent. Shared callable loading keeps those integration boundaries consistent without introducing benchmark-package dependencies into the core library. Provenance supplied directly to the suite runner is validated against the invocation before execution. The framework does not calculate or imply statistical significance, and it does not fabricate or infer missing experimental evidence.

## Engineering gates

Before promoting a research result, run the complete local quality suite and then execute the corresponding benchmark or integration with its declared configuration. CI success is evidence of software correctness for the covered tests; it is not evidence of scientific effectiveness.

## Next milestone

The next highest-value milestone is **real benchmark execution and reproducible reporting**: connect real caller-owned ALFWorld/WebShop environment factories and model policies to the external benchmark CLI, execute controlled multi-seed runs, record the required provenance, and publish only measured results. The CLI now supports direct memory-guided composition around an action-policy factory, removing one remaining manual assembly step. The next implementation should add a repository-owned deterministic smoke environment/policy module that exercises this complete path without requiring third-party benchmark packages, then use that smoke path as the regression gate before real upstream integrations. The same provenance boundary should then be applied to full GRPO/verl runs, with observability attached at the integration boundary.
