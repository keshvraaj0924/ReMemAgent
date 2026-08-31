# Research status

This document records the implemented research surface without implying experimental conclusions that have not been measured.

## Verified engineering baseline

The `feat/core-memory-engine` branch currently has a green GitHub `Quality` workflow at commit `e9f02fc` (run `305`). The workflow reports the repository test, formatting, lint, and type-check stages as successful.

The framework currently provides deterministic implementations for:

- typed episodic memory and lifecycle management;
- retrieval, deduplication, reconstruction, trust, and transferability;
- counterfactual routing and failure-memory capture;
- consolidation and retirement policies;
- synthetic negative-transfer evaluation and ablation reporting;
- normalized ALFWorld/WebShop environment boundaries;
- GRPO batch normalization and an async verl agent-loop bridge;
- reproducible experiment configuration/result serialization;
- backend-neutral observability primitives.

These capabilities are intentionally separated from model SDKs and external benchmark packages where practical.

## What has not been established

No benchmark improvement, transfer advantage, or production-readiness claim is made by this repository status. Those claims require executed experiments with fixed configurations, recorded outputs, and independently reproducible runs.

In particular, the following remain experimental integration work rather than verified end-to-end claims:

1. ALFWorld and WebShop runs against their real upstream environments.
2. Full GRPO/verl training runs with real model checkpoints and distributed infrastructure.
3. Statistical comparison across repeated seeds for the negative-transfer benchmark and ablations.
4. External observability exporters and deployment-specific telemetry.

## Reproducibility contract

Research results should be recorded with:

- the benchmark configuration;
- random seed(s);
- code revision;
- environment/dependency versions;
- serialized metrics;
- enough input metadata to reconstruct the run.

The repository's reproducibility helpers provide deterministic JSON-compatible configuration/result serialization. They do not fabricate or infer missing experimental evidence.

## Engineering gates

Before promoting a research result, run the complete local quality suite and then execute the corresponding benchmark or integration with its declared configuration. CI success is evidence of software correctness for the covered tests; it is not evidence of scientific effectiveness.

## Next milestone

The next highest-value milestone is **real benchmark execution and reproducible reporting**: wire the existing deterministic harnesses to concrete ALFWorld/WebShop and GRPO/verl environments, run controlled multi-seed experiments, and publish only measured results. Observability should then be connected at the integration boundary rather than added to the core memory model.
