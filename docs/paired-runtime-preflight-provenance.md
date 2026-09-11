# Paired runtime preflight provenance

Paired benchmark artifacts produced by `remem-paired-benchmark` record whether the CLI's runtime preflight completed before measured execution and whether that preflight was reset-only or included a concrete probe action.

The launcher writes two string-valued runtime-provenance fields only after `run_paired_external_benchmarks_with_preflight(...)` returns successfully:

- `paired_runtime_preflight`: `completed`
- `paired_runtime_preflight_probe_action`: the exact `--probe-action` value, or `reset-only` when no action probe was requested

Because runtime provenance participates in paired experiment identity construction, this distinguishes otherwise identical paired artifacts that used different preflight probes.

## Controlled paired runtime gate

The paired CLI can additionally declare the runtime that is allowed to produce measured evidence:

- `--require-code-revision REVISION` requires an exact repository revision;
- `--require-clean-working-tree` rejects dirty or unknown checkout state;
- `--require-dependency-version PACKAGE==VERSION` requires an exact installed dependency version and may be supplied more than once.

When any of these requirements are declared, ReMemAgent collects runtime provenance once and validates the complete requirement set before paired callable resolution, environment construction, or condition-specific preflight begins. A revision, working-tree, missing-package, or version mismatch therefore prevents both baseline and treatment side effects rather than allowing a partially preflighted or partially measured paired experiment.

For example, a controlled experiment can be launched with values chosen and pinned by the experiment owner:

```text
remem-paired-benchmark \
  ... \
  --require-code-revision <exact-commit-sha> \
  --require-clean-working-tree \
  --require-dependency-version alfworld==<pinned-version> \
  --require-dependency-version <model-runtime-package>==<pinned-version>
```

The repository deliberately does not invent benchmark or model-runtime versions. The exact versions must come from the controlled environment used for the real experiment.

Runtime requirements and runtime provenance have different roles. Requirements are a fail-closed pre-measurement contract; provenance records what actually executed and is persisted with the paired artifact. Passing the requirement gate does not replace artifact provenance, integrity manifests, or repeated-seed experimental design.

This metadata and runtime gate are evidence about the ReMemAgent execution path, not an independent attestation that an upstream benchmark installation is scientifically correct. A successful reset-only preflight validates construction and reset. A successful action probe additionally validates one concrete step through the normalized adapter. Neither constitutes benchmark evidence, establishes model effectiveness, or demonstrates production readiness without the corresponding measured experiment.
