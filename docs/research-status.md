# Research status

This document records what the repository has implemented and verified without turning engineering checks into scientific claims.

## Verified engineering baseline

The `feat/core-memory-engine` branch has a verified green GitHub `Quality` workflow at commit `6d1f888d48cc571b14fce8a218b6936f4e15fc95` (run `1278`) on September 12, 2026. Both Python 3.11 and 3.12 completed the configured quality pipeline successfully. The run collected **917 pytest tests** and also passed Ruff formatting and linting, mypy type checking, dependency validation, Python compilation, wheel/source-distribution builds, isolated distribution smoke tests, and artifact upload.

This is an engineering verification statement. It does **not** establish benchmark improvement, transfer advantage, statistical significance, model-training effectiveness, or production readiness.

## Implemented research surface

The framework currently provides tested implementations for the planned research layers:

- typed episodic memory, storage, ingestion, lifecycle state, and failure evidence;
- retrieval, deduplication, reconstruction, trust, and transferability estimation;
- explicit heuristic and counterfactual routing boundaries;
- consolidation, staleness, validation, and retirement policies;
- synthetic negative-transfer evaluation and ablation reporting;
- seed-aware benchmark statistics and paired baseline-versus-treatment analysis;
- exact paired sign-flip tests, Holm correction, and paired effect-size utilities;
- normalized ALFWorld and WebShop adapters plus lazy upstream construction bridges;
- caller-owned learned-policy composition so model loading and inference remain separate from memory heuristics;
- framework-neutral GRPO data contracts and an async verl agent-loop boundary;
- deterministic JSONL training artifacts and SHA-256 integrity manifests;
- single-run, repeated multi-seed, and paired external benchmark execution;
- runtime and source-checkout preflight before controlled measurement;
- exact runtime dependency, code revision, working-tree, and external source revision admission contracts;
- identity-bound persistence of the exact runtime/source snapshots admitted before measurement;
- deterministic benchmark report serialization, experiment identities, and exact-byte integrity manifests;
- artifact verification that fails closed on malformed schemas, stale identities, source/runtime evidence drift, or byte tampering;
- deterministic local observability snapshots, deltas, checkpoint persistence, additive worker merging, and explicit success/failure accounting;
- CI quality, package build, installed-package import, and wheel/source-distribution smoke checks.

## Strict paired reproducibility

`remem-paired-benchmark --strict-reproducibility` is the fail-closed entry point for paired experiments intended to become research evidence. Before external measurement starts, strict mode requires:

1. at least two independent seeds;
2. an exact ReMemAgent revision;
3. a clean ReMemAgent working tree;
4. at least one exact installed dependency version;
5. at least one exact external source-checkout revision;
6. clean external source checkouts with no dirty-tree exception; and
7. an explicit integrity-manifest destination.

The strict gate validates the **declared contract**. It does not pretend that the machine satisfies that contract. Controlled preflight separately collects the actual runtime/source state immediately before measurement, validates it against the declaration, probes the configured environment, and only then permits paired execution. The admitted snapshots are carried forward into artifact identity and persistence instead of being recollected after measurement.

See [`strict-reproducibility.md`](strict-reproducibility.md) for the command-line contract and example shape.

## Reproducibility and evidence boundary

External benchmark evidence can record the benchmark protocol, ordered independent seeds, code revision, working-tree state, installed dependency versions, source-checkout revisions, callable specifications, policy trust threshold, per-seed reports, paired statistics, experiment identities, preflight provenance, and exact-byte artifact manifests.

Repeated execution does not pool episodes across independent seeds. Paired execution aligns baseline and treatment by shared seed and preserves explicit protocol configuration. Statistical utilities are available for descriptive and paired analysis, but their existence is not itself evidence that ReMemAgent improves an agent.

The deterministic ALFWorld/WebShop fixtures exercised in CI are engineering regression gates. They verify adapter and execution contracts without claiming to be official benchmark measurements. Likewise, GRPO/verl integration tests verify data and agent-loop boundaries without claiming that a real checkpoint has been trained successfully.

## What has not been established

The repository still does **not** have verified end-to-end scientific evidence for:

- real ALFWorld benchmark results using a pinned upstream checkout and fixed model-policy runtime;
- real WebShop benchmark results using a pinned upstream checkout and fixed model-policy runtime;
- a demonstrated baseline-versus-memory performance improvement across repeated independent seeds;
- full GRPO/verl optimization using a real checkpoint and distributed training infrastructure;
- production deployment reliability or operational scalability.

No upstream Git revision, dependency version, model checkpoint, metric, or benchmark outcome should be inferred when it has not been measured. The framework deliberately records `unknown` or rejects the run instead of fabricating missing provenance.

## Engineering gates for publishable experiments

A candidate external result should pass the repository quality workflow, strict reproducibility declaration, runtime/source admission, environment preflight, measured execution, identity-bound artifact persistence, and exact-byte artifact verification. The resulting artifact should then be analyzed using the stored per-seed results and explicit protocol rather than reconstructed from console output.

CI success proves only that the covered software contracts passed. It does not substitute for executing the real benchmark. A historical green workflow also does not certify later code changes; each implementation increment must earn its own successful quality run before it is described as verified.

## Next milestone

The next highest-value scientific milestone is to execute **real explicitly pinned ALFWorld/WebShop multi-seed paired experiments** through `--strict-reproducibility`. The execution environment must supply the actual upstream source revision, installed dependency versions, model-policy implementation/checkpoint, valid benchmark configuration, and artifact destination. Those values should be measured and recorded from the environment chosen for the experiment rather than invented in repository documentation.

After the real external benchmark evidence is established and independently reproducible, the same admission, provenance, artifact-integrity, and reporting discipline should be applied to full GRPO/verl training runs and subsequent learned routing experiments.
