# Research execution checklist

This checklist converts ReMemAgent's reporting contract into an executable research workflow. It does not imply that any benchmark or training result has already been obtained.

## 1. Freeze the run identity

Before executing an experiment, record:

- exact Git commit SHA;
- experiment configuration and policy variant;
- random seed(s);
- Python and dependency versions;
- environment/task version and split;
- model/provider identifiers where applicable;
- hardware/runtime metadata;
- output directory and artifact-manifest location.

Do not compare runs whose uncontrolled configuration differs.

## 2. Validate the deterministic baseline first

Run the repository quality suite before collecting research evidence. A benchmark run from a revision that fails tests, lint, typing, dependency validation, compilation, or packaging must not be promoted as evidence.

For synthetic negative-transfer experiments, establish the no-memory baseline before enabling memory components.

## 3. Execute controlled ablations

At minimum, preserve comparable runs for:

1. no memory;
2. retrieval/replay baseline;
3. reconstruction without trust/transferability gating;
4. reconstruction with trust/transferability;
5. counterfactual routing enabled;
6. full configured ReMemAgent policy.

When studying a component, change that component only. If a configuration change is unavoidable, report it explicitly rather than treating the runs as a controlled ablation.

## 4. Preserve per-seed evidence

Do not retain only aggregate summaries. Preserve per-seed raw outputs and the provenance needed to regenerate them. Report the aggregate required by `experiment-reporting.md` only after all planned seeds finish.

A failed or interrupted seed remains part of the experiment record. Do not silently replace it; record the failure reason and any rerun separately.

## 5. Attribute negative transfer

A memory-assisted decision is negative transfer only when the controlled memory condition performs worse than the matched no-memory decision under the repository's declared metric. Record enough routing and memory evidence to distinguish:

- harmful retrieval;
- incorrect transferability/trust assessment;
- harmful reconstruction;
- routing failure;
- downstream action/environment failure unrelated to memory.

Do not label every failed memory-assisted episode as negative transfer.

## 6. External environments

ALFWorld/WebShop adapter tests validate software boundaries, not benchmark performance. Before reporting external-environment results, preserve:

- the real environment version and setup;
- task/split identifiers;
- episode-level outcomes;
- model and decoding configuration;
- memory-policy configuration;
- seed/run identity;
- raw logs and artifact manifest.

Mock, fake, or adapter-only executions must be described as integration tests.

## 7. GRPO / verl-agent training

Integration code is not evidence of learned improvement. A GRPO claim requires a completed training run with:

- training configuration and exact revision;
- initialization/checkpoint identity;
- reward definition;
- training curves or raw trainer metrics;
- final checkpoint identity;
- held-out evaluation against the matched pre-training policy;
- repeated evaluation or uncertainty where required by the reporting contract.

Training failures and unstable runs must remain visible in the experiment record.

## 8. Verify artifacts before analysis

Use the repository's provenance and artifact-manifest facilities to verify that expected outputs exist and have not changed. Analysis should consume preserved artifacts rather than manually copied summary values whenever possible.

## 9. Report conservatively

A result can move into README/research documentation only when its evidence satisfies `docs/experiment-reporting.md`. Keep these categories distinct:

- software contract verified by tests;
- synthetic experimental evidence;
- external-environment evidence;
- learned-policy/training evidence.

Never convert a passing test into a benchmark claim or a single successful run into a general performance conclusion.

## 10. Completion criteria

The empirical milestone is complete only when the planned baseline matrix has finished, raw artifacts and provenance are preserved, failed runs are accounted for, aggregate metrics and uncertainty are computed from the preserved data, and the conclusions are scoped to what those experiments actually establish.
