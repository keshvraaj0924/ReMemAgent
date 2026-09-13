# External research experiment protocol

This document is the canonical execution protocol for turning ReMemAgent's engineering framework into scientific benchmark evidence. It deliberately separates **engineering verification**, **experimental admission**, **measured evidence**, and **research claims**.

## Evidence levels

ReMemAgent uses four evidence levels. A higher level does not retroactively strengthen a lower-level artifact; each level has its own contract.

| Level | Evidence | What it establishes | What it does not establish |
|---|---|---|---|
| E0 | Green repository Quality workflow | Tested software contracts, formatting, linting, typing, packaging, and distribution smoke checks passed for one commit | Benchmark quality, model effectiveness, production readiness |
| E1 | Verified preflight/readiness evidence | Declared runtime, source checkout, callable, seed, and protocol requirements matched the controlled environment before measurement | Any task-performance result |
| E2 | Verified measured benchmark bundle | The persisted benchmark report and retained evidence passed schema, identity, manifest, provenance, and optional observability verification | Generalization beyond the executed protocol or statistical significance |
| E3 | Repeated paired experiment analysis | Baseline and treatment were executed under matched controlled conditions and analyzed across independent seeds | Production reliability or causal claims outside the tested benchmark/runtime |

A paper, report, README, or presentation should state the highest evidence level actually achieved and retain the artifacts required to verify it.

## Freeze the experiment before measurement

Before running ALFWorld or WebShop, record the following values from the actual environment. Do not insert guessed placeholders into a result that will later be presented as evidence.

- ReMemAgent commit SHA and clean-working-tree requirement;
- upstream benchmark checkout path, commit SHA, and clean-tree requirement;
- Python and exact dependency versions required by the policy/runtime;
- benchmark name, episode count, maximum steps, and ordered independent seeds;
- environment and success-evaluator callable specifications;
- baseline and treatment policy-factory callable specifications;
- model/provider identifier and checkpoint revision where applicable;
- decoding/inference parameters controlled by the caller-owned policy;
- ReMemAgent trust threshold and any treatment-specific memory configuration;
- output, manifest, readiness-evidence, observability, distribution, and attestation paths.

Changing any value that can alter the measured behavior starts a new experiment identity. Do not silently replace a failed or inconvenient run while keeping the same claimed protocol.

## Phase 1: engineering gate

Run the repository Quality workflow at the exact ReMemAgent revision that will be measured. The workflow is an engineering prerequisite, not a benchmark result.

The configured gate covers pytest, Ruff formatting/lint, mypy, dependency validation, compilation, wheel/source-distribution builds, and isolated installed-package smoke tests on supported Python versions.

## Phase 2: readiness evidence

Use `remem-paired-benchmark --preflight-only` with the same controlled values intended for measurement. Strict experiments should require an exact ReMemAgent revision, a clean working tree, exact dependency versions, and exact clean external source revisions.

Example shape:

```bash
remem-paired-benchmark \
  --benchmark webshop \
  --episodes 100 \
  --max-steps 50 \
  --seeds 11,17,29,43,71 \
  --environment-factory your_package.environments:build_webshop \
  --success-evaluator your_package.metrics:is_success \
  --baseline-policy-factory your_package.policies:build_baseline \
  --treatment-policy-factory your_package.policies:build_remem \
  --strict-reproducibility \
  --require-code-revision <REMEM_COMMIT> \
  --require-clean-working-tree \
  --require-dependency-version torch==<PINNED_VERSION> \
  --source-checkout webshop=/path/to/webshop \
  --require-source-revision webshop=<WEBSHOP_COMMIT> \
  --preflight-only \
  --preflight-evidence artifacts/webshop-readiness.json
```

Verify the resulting readiness object before measurement:

```bash
remem-verify-preflight artifacts/webshop-readiness.json
```

A failed preflight is an experiment-admission failure. Fix the environment or declare a new protocol; do not weaken the requirement after seeing measured outcomes.

## Phase 3: measured paired execution

Run baseline and treatment through the paired runner using the retained readiness evidence. The exact command should be archived with the experiment.

```bash
remem-paired-benchmark \
  --benchmark webshop \
  --episodes 100 \
  --max-steps 50 \
  --seeds 11,17,29,43,71 \
  --environment-factory your_package.environments:build_webshop \
  --success-evaluator your_package.metrics:is_success \
  --baseline-policy-factory your_package.policies:build_baseline \
  --treatment-policy-factory your_package.policies:build_remem \
  --strict-reproducibility \
  --require-code-revision <REMEM_COMMIT> \
  --require-clean-working-tree \
  --require-dependency-version torch==<PINNED_VERSION> \
  --source-checkout webshop=/path/to/webshop \
  --require-source-revision webshop=<WEBSHOP_COMMIT> \
  --require-preflight-evidence artifacts/webshop-readiness.json \
  --output artifacts/webshop-paired.json \
  --manifest artifacts/webshop-paired.json.manifest.json
```

The primary scientific unit is the independent seed, not an individual episode pooled across seeds. Preserve per-seed reports so paired deltas remain auditable.

## Phase 4: artifact verification

For single/repeated benchmark artifacts that include observability evidence, retain the report, manifest, aggregate observability snapshot, duration distribution sidecar, and verification attestation together.

Example verification shape:

```bash
remem-verify-benchmark \
  artifacts/webshop.json \
  --manifest artifacts/webshop.json.manifest.json \
  --preflight-evidence artifacts/webshop-readiness.json \
  --observability-sidecar artifacts/webshop.observability.json \
  --distribution-sidecar artifacts/webshop.duration-distribution.json \
  --attestation-output artifacts/webshop.verification.json \
  --json
```

Verification checks retained schemas and artifact identities and, when both telemetry sidecars are present, cross-checks completed-episode counts and measured-duration totals before producing the combined evidence digest. The digest is an integrity/association checksum, not a digital signature and not proof of model quality.

Never edit a measured JSON artifact by hand and then continue using its original manifest or verification attestation.

## Phase 5: paired analysis

For matched seed-level baseline-versus-treatment results:

1. report the per-seed values and paired deltas;
2. report descriptive summaries without hiding unfavorable seeds;
3. use the implemented exact paired sign-flip test where its assumptions and sample size are appropriate;
4. apply Holm-Bonferroni correction across the predeclared primary metric family;
5. report paired Cohen's *d_z* as an effect-size description rather than as a substitute for raw results;
6. distinguish exploratory secondary analyses from predeclared primary outcomes.

A positive mean delta alone is not sufficient evidence of a robust improvement.

## Phase 6: freeze the retained evidence set

After the experiment-specific validators have succeeded, create one machine-checkable index over the exact files that support the claim. The evidence record stores a versioned schema, the declared evidence level, the exact ReMemAgent commit SHA, safe relative artifact paths, exact byte counts, and SHA-256 digests. Publication is atomic and refuses to overwrite an existing record.

Keep the record in the same experiment directory as the files it indexes. Every `--artifact` path must resolve beneath the record's output directory so that the record is portable without embedding machine-specific absolute paths.

Example shape:

```bash
remem-research-evidence freeze \
  --experiment webshop-memory-study \
  --level E3 \
  --revision <REMEM_COMMIT> \
  --artifact readiness=artifacts/webshop-readiness.json \
  --artifact paired_report=artifacts/webshop-paired.json \
  --artifact manifest=artifacts/webshop-paired.json.manifest.json \
  --artifact verification=artifacts/webshop.verification.json \
  --note "predeclared seed set: 11,17,29,43,71" \
  --output artifacts/research-evidence.json

remem-research-evidence verify artifacts/research-evidence.json
```

The evidence record is an **index and exact-byte integrity contract**, not a semantic validator and not a digital signature. It does not decide whether `E3` has actually been earned. Run the specific preflight, benchmark, manifest, observability, and statistical checks first, then declare only the highest evidence level those retained artifacts genuinely support. If any indexed file is edited, replaced, deleted, or moved outside the retained experiment directory, record verification fails closed.

## Negative-transfer reporting

Because ReMemAgent explicitly studies whether memory can hurt decisions, external experiments should report negative transfer rather than only aggregate success. At minimum, preserve enough condition-level evidence to identify seeds or task subsets where memory-guided behavior underperforms the matched baseline.

Synthetic negative-transfer fixtures remain regression/evaluation tools. They must not be presented as official ALFWorld or WebShop benchmark results.

## Ablations

When resources permit, evaluate the full treatment against predeclared ablations that isolate the research mechanisms, such as reconstruction, trust/transferability, counterfactual routing, failure memory, and lifecycle/consolidation behavior. Keep the external policy/model runtime fixed across conditions unless the changed component is itself the subject of the ablation.

Ablations should use the same independent seed set and benchmark protocol as the full treatment when paired comparison is intended.

## Required retained evidence

For a result intended to be cited later, retain at least:

- exact command/configuration used for readiness and measurement;
- ReMemAgent revision and Quality workflow reference;
- upstream benchmark revision and clean-tree state;
- dependency/runtime provenance;
- policy/model/checkpoint identity and caller-owned inference configuration;
- readiness evidence;
- measured report and integrity manifest;
- per-seed paired results and statistical analysis inputs;
- observability/distribution sidecars when collected;
- persisted verification attestation when produced;
- experiment notes describing any interruption, retry, exclusion, or protocol deviation;
- a verified `remem-research-evidence` record indexing the exact retained files when the result is intended for later citation.

If any required evidence is missing, describe that limitation instead of reconstructing provenance from memory.

## Claim language

Acceptable claims should stay inside the measured evidence boundary. Examples:

- **Engineering only:** "The framework passed its repository Quality workflow at commit X."
- **Measured benchmark:** "Under the pinned protocol recorded in artifact X, treatment success was Y and baseline success was Z."
- **Repeated paired:** "Across the predeclared independent seeds, the paired delta was ...; the exact paired analysis produced ..."

Do not convert those statements into "production ready," "universally better," or "state of the art" without separate evidence supporting those claims.

## Current scientific boundary

At the time this protocol was added, ReMemAgent had engineering tests for its memory engine, synthetic evaluation, ALFWorld/WebShop adapters, paired execution, GRPO/verl boundaries, reproducibility, artifact verification, observability, and machine-checkable retained-evidence indexing. It did **not** yet contain real pinned multi-seed ALFWorld/WebShop effectiveness results or a completed real GRPO/verl optimization run. Those remain experimental work, not documentation placeholders to be filled with invented numbers.
