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

## Boundary of the guarantee

This contract verifies **declaration consistency**. ReMemAgent cannot introspect an arbitrary caller-owned policy factory and prove that it actually loaded the declared checkpoint or applied every declared decoding parameter. The concrete model/policy integration must use those values faithfully, and external experiments should pin the model artifact or provider revision independently where possible.

A matching declaration therefore strengthens reproducibility evidence but is not proof of model behavior, benchmark effectiveness, statistical significance, or production readiness.
