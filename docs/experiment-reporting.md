# Experiment reporting contract

This document defines the minimum evidence required before an experimental result is reported for ReMemAgent. It is intentionally stricter than a run log: its purpose is to make a result traceable, comparable, and falsifiable.

## Required run identity

Every reported run must preserve:

- the exact Git commit SHA;
- experiment/configuration identity and random seed;
- model and environment identifiers, including versions when available;
- retrieval and memory budgets;
- the policy/ablation variant being evaluated;
- runtime provenance and the artifact manifest produced by the repository's reproducibility utilities;
- raw per-episode or per-task outputs used to compute aggregate metrics.

A missing or incomplete run is not a zero-valued result. Keep failures distinguishable from completed evaluations.

For automated reporting pipelines, use `EvidenceContract` to persist the exact run-relative files required for a result class and call `EvidenceContract.verify(...)` before accepting the result. Contracts are canonical, deterministic JSON and fail closed when required evidence is absent or its captured bytes have changed. `ArtifactManifest.require_paths(...)` and `ArtifactManifest.verify_required(...)` remain available for lower-level checks. Evidence paths reject absolute, parent-traversing, backslash-based, or otherwise non-normalized requirements so contracts remain portable and unambiguous.

When a persisted `ExperimentSummary` is used as reportable aggregate evidence, name the summary file in the same `EvidenceContract` as its required raw inputs and consume it through `verify_reportable_summary(...)`. That boundary verifies the complete evidence bundle before parsing the aggregate summary, so a valid-looking summary cannot bypass missing or mutated source evidence. Do not load a standalone summary and treat successful JSON/schema validation as proof that its underlying experiment evidence is intact.

## Baseline matrix

When the corresponding policies are supported, evaluate the same task set under:

| Variant | Purpose |
|---|---|
| Self-reasoning / no memory | Establish performance without retrieved experience. |
| Replay memory | Measure the conventional retrieve-and-inject baseline. |
| Reconstructive memory without routing | Isolate reconstruction from counterfactual routing. |
| Full ReMemAgent routing | Evaluate the complete deterministic memory policy. |

Do not compare variants that use different task sets, seeds, model settings, retrieval budgets, or memory budgets without explicitly labeling the comparison as uncontrolled.

## Metrics

Report task outcomes separately from memory behavior. At minimum, preserve task success, memory acceptance/rejection, negative-transfer events, routing decisions, and lifecycle changes when those signals exist for the experiment. Record latency, token usage, and memory growth when they are measurable; do not infer them from unrelated proxies.

Aggregate results must retain per-seed values. For repeated experiments, report the number of completed runs and an uncertainty measure appropriate to the sample rather than only a mean. Small samples should be presented as such, not as evidence of statistical significance.

## Negative transfer

A negative-transfer event requires a paired or otherwise controlled comparison showing that using memory produced a worse decision or outcome than the defined non-memory comparator. Retrieval of an irrelevant memory alone is not sufficient. The experiment must state the comparator and attribution rule used to count the event.

## External environments

ALFWorld and WebShop adapter/unit tests verify integration contracts only. An environment result requires an actual environment execution with task identifiers, environment version/configuration, model configuration, seed, raw trajectories or equivalent outputs, and the generating revision preserved.

## Training claims

GRPO/verl-agent integration code is not evidence of policy improvement. A training claim requires a completed training run, training configuration, seed, checkpoint or sufficient model provenance, training diagnostics, and evaluation against an appropriate pre-training or deterministic baseline under the same evaluation contract.

## Result table checklist

Before adding a quantitative table to research documentation, verify that every row can be traced to preserved raw artifacts and that the table states the generating commit, configuration, seeds, completed-run count, baseline definition, and metric definition. Do not manually copy numbers from console output when a machine-readable artifact is available.

## Claim language

Use narrow language that matches the evidence level. Synthetic experiments support conclusions about the recorded synthetic benchmark. External-environment runs support conclusions about the executed task/configuration set. General claims require repeated evidence across suitable settings. Unit tests and CI quality gates establish software behavior, not research effectiveness.

See `docs/research-status.md` for the project-wide claims policy and `docs/reproducibility.md` plus `docs/runtime-provenance.md` for run identity and provenance requirements.
