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
- concrete lazy factory bridges for the upstream ALFWorld text API and WebShop Gym text API, without adding those third-party packages to core dependencies;
- a caller-owned model-policy composition factory that injects ReMemAgent memory guidance without taking ownership of model loading or inference;
- GRPO batch normalization and an async verl agent-loop bridge;
- deterministic JSON Lines writers for framework-neutral GRPO and verl training artifacts;
- integrity manifests with row counts and SHA-256 digests for persisted training artifacts;
- reproducible single-run and explicit multi-seed experiment execution with provenance-preserving JSON serialization;
- seed-level descriptive benchmark statistics without pooling independent episodes;
- paired treatment-minus-baseline descriptive deltas aligned by independent seed;
- versioned benchmark report artifacts;
- backend-neutral observability primitives with deterministic, atomically persisted local snapshots and additive multi-worker snapshot merging;
- import-only and runtime environment preflight checks for external benchmark launches.

Integration callable loading is centralized in `remem.integrations.loading`, so benchmark environment factories and external policy/evaluator factories share one explicit `module:attribute` resolution contract.

External benchmark reports retain the declared benchmark configuration alongside measured episode results, including episode count, step limit, seed, the exact callable specifications used for the environment, policy, success evaluator, and optional transfer evaluator, and the configured minimum trust threshold. The runner rejects supplied provenance metadata when its benchmark name, episode count, step limit, or seed differs from the actual invocation. This prevents a measured artifact from silently carrying stale run metadata; trust thresholds are now part of the persisted configuration rather than being implicit CLI state.

Benchmark report JSON is explicitly versioned with `BENCHMARK_REPORT_SCHEMA_VERSION`. Both single-run and repeated-run artifacts carry the schema version, while repeated artifacts also preserve the shared configuration fingerprint. This creates a stable compatibility boundary for future readers and prevents schema evolution from being confused with changes to measured benchmark semantics.

The model-policy composition boundary accepts a caller-owned factory that receives the deterministic episode seed and returns the action-generation callable expected by `MemoryGuidedPolicy`. This keeps learned components, checkpoint loading, tokenization, and inference outside the research heuristic layer while making memory-guided benchmark policies directly composable. The external benchmark CLI exposes this mode through `--action-policy-factory` and `--minimum-trust`; callers may alternatively provide a complete `--policy-factory` when they already own policy composition.

A repository-owned deterministic smoke fixture exercises the complete external path without third-party benchmark dependencies: a batch-shaped ALFWorld-compatible environment is adapted, a seed-aware action policy is memory-guided after its warm-up episode, successful trajectories are ingested, and the resulting report preserves callable provenance. This fixture is an engineering regression gate only and is not a scientific benchmark.

The concrete official benchmark bridges now expose `build_alfworld_text_environment_factory` and `build_webshop_text_environment_factory`. ALFWorld construction follows the upstream `get_environment(...).init_env(batch_size=1)` text-environment API; WebShop construction follows the upstream `gym.make("WebAgentTextEnv-v0", observation_mode="text", ...)` interface. Imports remain lazy, so the repository's dependency-free core and CI smoke tests do not require either external benchmark package. The returned raw factories are still passed through ReMemAgent's normalized benchmark adapters rather than embedding benchmark-specific logic in the research runner.

Repeated external execution has an explicit multi-seed boundary. `run_repeated_external_benchmarks` rejects empty or duplicate seed lists, creates independent benchmark runs for each seed, and preserves the seed in every report's configuration. `save_repeated_benchmark_reports` persists those reports as one JSON artifact while rejecting duplicate seeds, mixed benchmark names, and inconsistent experimental configuration, preventing ambiguous aggregation of runs with different episode counts, step limits, callable implementations, or trust thresholds.

The benchmark statistics module summarizes independent seed runs without pooling their episodes. It reports descriptive mean, sample standard deviation, standard error, and a 95% normal-approximation confidence interval for success rate, mean reward, and transfer success rate. A single seed reports zero estimated uncertainty rather than implying that a sampling distribution was measured. These statistics are descriptive and do not establish significance. Paired condition comparisons now additionally compute treatment-minus-baseline deltas per shared seed before summarizing the deltas, preserving the pairing induced by the experimental seed design without introducing a hypothesis test.

The external benchmark CLI now supports `--seeds` as an ordered, comma-separated set of unique independent seeds. Multi-seed CLI runs persist both the individual reports and the computed seed-level descriptive statistics in the same deterministic JSON artifact. The existing `--seed` single-run mode remains available.

Runtime provenance is captured at the benchmark artifact boundary. `collect_runtime_provenance` records the Python version, platform, installed ReMemAgent package version, Git revision when available, and a deterministic fingerprint plus normalized versions of installed Python distributions. CI/container callers can provide `REMEM_GIT_COMMIT` explicitly; local checkouts fall back to `git rev-parse HEAD`. If no revision can be resolved, the artifact records `unknown` rather than inventing a revision. Dependency metadata is captured from the runtime's installed distributions rather than inferred from the project manifest, so an experiment records the environment that actually executed it. The benchmark CLI attaches this metadata to its JSON report, while the serializer remains backward-compatible for callers that do not provide runtime metadata.

Observability snapshots expose a deterministic JSON-compatible representation with sorted metric keys. `write_observation_snapshot` additionally persists a snapshot atomically after flushing and syncing a temporary file in the destination directory, creating parent directories as needed. This keeps local telemetry suitable for experiment artifacts without coupling the research core to an external telemetry backend. `merge_observation_snapshots` provides a deterministic additive aggregation boundary for independent worker snapshots and rejects invalid aggregate values rather than silently carrying corrupt telemetry forward.

The training integration layer can persist validated GRPO or verl batches as deterministic JSON Lines. Writers create parent directories, preserve row ordering, sort JSON keys, and avoid introducing tokenizer or trainer dependencies. Each persisted training artifact can now be accompanied by an integrity manifest containing its row count and SHA-256 digest; verification re-parses the JSONL and checks the exact bytes, so downstream training can fail closed on mutation or malformed data. GRPO batch construction now also rejects singleton groups before dataset persistence, preventing a superficially valid batch from producing only zero comparative advantages.

The CI quality workflow now also checks dependency consistency, compiles all repository Python sources, and builds the distributable package in addition to tests, formatting, linting, and type checking. Package-build validation uses the same development dependency set as the quality job, so packaging regressions are caught before a research artifact is published.

The external benchmark launcher now has two explicit preflight levels. `--preflight` resolves every configured callable without constructing environments. `--runtime-preflight` constructs the real configured environment through the same normalized adapter used by measured runs and validates its reset contract; supplying `--probe-action` additionally validates one concrete step. The runtime probe uses the configured run seed (or zero only when no seed was supplied) and closes the environment on every path. This provides a fail-fast integration check without pretending that a reset or one-step probe is a benchmark result.

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

The repository provides a multi-seed runner that gives each synthetic case set an isolated random generator and preserves per-run case and experiment fingerprints. The external benchmark runner accepts an optional run seed: each episode receives `seed + episode_index` through both environment and policy factories, while the report records the run-level seed and declared callable configuration. Repeated external execution independently invokes that same contract for each requested seed and rejects duplicate seeds. Repeated report persistence additionally verifies that every run shares the same declared experimental configuration apart from its independent seed. Benchmark integrations provide typed factories that resolve caller-owned raw environment factories and wrap each created environment with the correct ALFWorld or WebShop adapter. The official benchmark bridges provide a concrete construction path for the upstream APIs while keeping third-party imports optional. The model-policy factory preserves the same episode seed when creating a caller-owned action policy, allowing deterministic checkpoint/model selection without embedding model-specific logic in ReMemAgent. Shared callable loading keeps those integration boundaries consistent without introducing benchmark-package dependencies into the core library. Provenance supplied directly to the suite runner is validated against the invocation before execution, and the memory-guidance trust threshold is retained alongside that provenance. Single and repeated benchmark reports can be serialized with deterministic JSON settings; every serialized report declares the current report schema version, repeated artifacts preserve seed order and the shared configuration fingerprint, and mixed benchmark families or mismatched configurations are rejected. The deterministic smoke fixture validates this full contract locally through the same external runner and demonstrates memory guidance crossing the policy boundary. Seed-level statistics operate on per-run metrics rather than pooled episodes, while paired condition comparisons operate on matched seeds and summarize treatment-minus-baseline deltas without testing significance. Benchmark CLI artifacts additionally record runtime provenance, including the installed distribution versions and their deterministic fingerprint, when available. Observability snapshots can be serialized deterministically for inclusion in future experiment artifacts and can now be persisted atomically as local JSON files; independent worker snapshots can be merged additively without mutating their sources. GRPO/verl batches can be persisted as deterministic JSON Lines after validation, preserving reward/advantage alignment and memory provenance. Training artifact manifests can certify those exact JSONL bytes and row counts before external ingestion. GRPO batch construction rejects singleton groups so persisted training data cannot silently contain only zero comparative advantages. The runtime benchmark preflight can now exercise the actual configured adapter before measured execution, including one caller-selected valid action when supplied. The framework does not calculate or imply statistical significance, and it does not fabricate or infer missing experimental evidence.

## Engineering gates

Before promoting a research result, run the complete local quality suite and then execute the corresponding benchmark or integration with its declared configuration. CI success is evidence of software correctness for the covered tests; it is not evidence of scientific effectiveness.

## Next milestone

The next highest-value milestone is **real benchmark execution and reproducible reporting**: use the concrete upstream factory bridges with real ALFWorld/WebShop installations and model policies, run the runtime preflight against those same configurations, execute controlled multi-seed runs through the `--seeds` CLI path, persist the per-seed reports and descriptive statistics, record code and dependency provenance, and publish only measured results. Paired condition comparisons should be used for baseline-versus-memory reporting when both conditions share the same independent seeds, while any inferential significance analysis must remain a separately justified statistical layer. The smoke fixture should remain a dependency-free regression gate. The same provenance boundary should then be applied to full GRPO/verl runs, with observability attached at the integration boundary.
