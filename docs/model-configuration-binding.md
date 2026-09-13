# Frozen model configuration binding

Real external benchmark measurements can be invalidated by changing a model checkpoint or inference setting while leaving the benchmark, seeds, and policy factories unchanged. ReMemAgent therefore treats caller-owned model configuration as part of the frozen research protocol when `--require-experiment-plan` is used.

## Measurement contract

A `ResearchExperimentPlan` may declare:

- `model_identity`: an exact caller-defined model/provider/checkpoint identity;
- `parameters`: named JSON scalar values that affect the caller-owned policy, such as temperature, maximum generated tokens, sampling enablement, or a checkpoint-specific switch.

During plan-bound measured execution, supply the same values to `remem-paired-benchmark`:

```bash
remem-paired-benchmark \
  ... \
  --require-preflight-evidence artifacts/webshop-readiness.json \
  --require-experiment-plan artifacts/webshop-experiment-plan.json \
  --model-identity 'Qwen/Qwen2.5-7B-Instruct@<PINNED_REVISION>' \
  --experiment-parameter temperature=0.0 \
  --experiment-parameter max_tokens=128 \
  --experiment-parameter do_sample=false
```

`--experiment-parameter` is repeatable and accepts `NAME=JSON_SCALAR`. Numbers, strings, booleans, and `null` are accepted. Arrays and objects are rejected so that the command-line contract stays explicit and deterministic. Duplicate parameter names are rejected.

Plan-bound admission requires exact equality between the supplied model identity/parameter mapping and the frozen plan. A missing value, extra value, checkpoint change, or parameter change blocks execution before measured episodes begin. The bridge-owned options are also rejected when no frozen plan is supplied, preventing apparently recorded model metadata from being silently ignored.

After successful admission, the measured report runtime provenance includes:

- `research_experiment_plan_sha256`;
- `research_model_identity`;
- `research_experiment_parameters`.

These fields make the actual admitted model declaration auditable next to the measured artifact rather than leaving it only in researcher notes.

## Retained artifact verification

After measurement, verify that the retained report still carries the exact model declaration from the frozen plan:

```bash
remem-verify-model-binding \
  artifacts/webshop-paired.json \
  artifacts/webshop-experiment-plan.json \
  --json
```

The verifier fails closed when the report plan digest, code revision, model identity, or parameter mapping differs from the retained plan. Parameter comparison uses canonical JSON rather than Python's loose numeric equality, so `128` and `128.0` are treated as different declarations. It also rejects missing provenance fields instead of interpreting absence as a match.

Successful JSON output includes the exact report byte count and SHA-256 together with the verified plan SHA-256, ReMemAgent revision, model identity, parameter mapping, and `model_configuration_binding_verified: true`.

The same semantic check is available directly through the retained research-evidence verifier when the evidence record contains the canonical `experiment_plan` and `paired_report` roles:

```bash
remem-research-evidence verify \
  artifacts/webshop-research-evidence.json \
  --require-model-binding \
  --json
```

`--require-complete-binding` now includes this model-configuration check together with the frozen-plan, exact-report, report-manifest, observability/distribution-sidecar, and verification-attestation bindings. A package cannot therefore claim complete semantic binding while omitting or drifting the measured model declaration.

The research-evidence JSON summary exposes `model_identity`, `experiment_parameters`, and `model_configuration_binding_verified: true` when this check succeeds. Retain that verified evidence record with the measured artifacts when publishing controlled external benchmark evidence.

## Boundary of the guarantee

This contract verifies **declaration consistency**. ReMemAgent cannot introspect an arbitrary caller-owned policy factory and prove that it actually loaded the declared checkpoint or applied every declared decoding parameter. The concrete model/policy integration must use those values faithfully, and external experiments should pin the model artifact or provider revision independently where possible.

A matching declaration therefore strengthens reproducibility evidence but is not proof of model behavior, benchmark effectiveness, statistical significance, or production readiness.
