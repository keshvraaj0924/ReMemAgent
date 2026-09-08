# External benchmark environment setup

ReMemAgent keeps ALFWorld and WebShop out of the core dependency set. The adapters accept already-created upstream environments, which keeps the research framework testable without forcing heavyweight benchmark stacks into every installation.

## Dependency discovery

Use `check_benchmark_dependencies()` from `remem.environments.dependencies` before constructing an external adapter. It uses Python import discovery and does not import either benchmark package, so a missing optional environment can be reported without triggering its startup side effects.

For a shell-level preflight, run `remem-check-environments`. It reports both supported benchmarks. By default it exits with status `1` if either dependency is unavailable; use `--require alfworld` or `--require webshop` when only one external benchmark is required for a run.

A missing package is an environment-setup condition, not a benchmark result. Do not record a benchmark run as measured until the upstream environment has been constructed successfully and the runtime preflight passes.

## ALFWorld

The official ALFWorld project documents installation with `pip install alfworld[full]` for the full stack and `alfworld-download` for its task data. Text-only evaluation can use the base package; the visual stack has additional requirements. See the official repository for the version-specific setup and data download instructions.

The ReMemAgent `AlfWorldAdapter` expects the initialized, batch-size-one environment returned by the upstream API and normalizes its batched observation/reward/terminal values into the framework's scalar `StepResult` contract.

## WebShop

WebShop is intentionally treated as a source-installed external environment rather than a pinned core dependency. Install and initialize the exact WebShop revision used by an experiment, then inject its environment into `WebShopAdapter`. Keep the environment in a dedicated virtual environment when its dependency constraints conflict with the ReMemAgent development environment.

## Evidence boundary

External benchmark execution requires all of the following:

1. The corresponding optional dependency is discoverable.
2. The caller successfully constructs the upstream environment.
3. The adapter runtime preflight succeeds for every requested seed.
4. The measured run persists its protocol, runtime provenance, seed set, and experiment identity.

This repository does not claim ALFWorld/WebShop benchmark results until those steps have been executed against the real upstream environments.
