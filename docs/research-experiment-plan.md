# Frozen research experiment plans

`experiments.research_experiment_plan` provides a versioned, deterministic contract for declaring a paired external-benchmark protocol **before measured execution**. It closes the gap between a prose experiment checklist and a machine-checkable configuration identity.

## What the plan records

`ResearchExperimentPlan` records the behavior-affecting experiment inputs that must not silently drift between preflight and measurement:

- stable experiment name and exact 40-character ReMemAgent commit SHA;
- benchmark name, episode count, maximum steps, and ordered independent seed set;
- environment and success-evaluator callable specifications;
- exactly one baseline policy mode and exactly one treatment policy mode, using either a full policy factory or an action-policy factory for each condition;
- optional transfer-success evaluator, minimum-trust threshold, and explicit condition labels;
- exact upstream source revisions and exact dependency versions;
- caller-owned scalar inference/research parameters;
- optional model/checkpoint identity and immutable notes.

The constructor fails closed for missing source/dependency pins, duplicate or non-integer seeds, fewer than two paired seeds, ambiguous policy modes, malformed callable specifications, invalid Git revisions, non-finite parameters, and trust values outside `[0, 1]`.

## Deterministic identity

`canonical_research_experiment_plan_json(plan)` produces sorted, compact, newline-terminated JSON. `plan.sha256` hashes those exact canonical bytes. Mapping insertion order therefore does not change the experiment identity, while any recorded protocol change does.

`write_research_experiment_plan(...)` uses a temporary file, flush + `fsync`, and atomic no-overwrite publication. A pre-existing declaration is never silently replaced. `load_research_experiment_plan(...)` strictly validates the complete schema, and `verify_research_experiment_plan(...)` can additionally require the predeclared ReMemAgent revision and plan SHA-256.

Example Python usage:

```python
from pathlib import Path

from experiments.research_experiment_plan import (
    build_research_experiment_plan,
    verify_research_experiment_plan,
    write_research_experiment_plan,
)

plan = build_research_experiment_plan(
    experiment_name="webshop-memory-study",
    remem_revision="<40-character-remem-commit>",
    benchmark_name="webshop",
    episode_count=100,
    max_steps=50,
    seeds=(11, 17, 29, 43, 71),
    environment_factory="study.environments:build_webshop",
    success_evaluator="study.metrics:is_success",
    baseline_policy_factory="study.policies:build_baseline",
    treatment_action_policy_factory="study.policies:build_remem_action",
    source_revisions={"webshop": "<40-character-webshop-commit>"},
    dependency_versions={"torch": "<exact-version>"},
    minimum_trust=0.6,
    parameters={"temperature": 0.0, "do_sample": False},
    model_identity="provider/model@checkpoint",
)

path = Path("artifacts/webshop-experiment-plan.json")
write_research_experiment_plan(path, plan)
verify_research_experiment_plan(
    path,
    expected_revision=plan.remem_revision,
    expected_sha256=plan.sha256,
)
```

Replace the angle-bracket values with identities observed from the actual environment. They are illustrative placeholders, not benchmark provenance.

## Measurement admission

`remem-paired-benchmark` can bind measured controlled execution to a frozen plan with `--require-experiment-plan PATH`. Plan-bound measurement also requires `--require-preflight-evidence` and `--strict-reproducibility`; it is not accepted as a weaker alternative to readiness evidence.

Before measured execution begins, the bridge fails closed if the paired CLI differs from the frozen declaration in the benchmark name, episode count, maximum steps, ordered seeds, environment/evaluator callables, baseline or treatment policy mode, transfer evaluator, minimum-trust threshold, or condition labels. It also requires the exact ReMemAgent revision, a clean ReMemAgent working tree, exact dependency pins, exact source-checkout revisions, and clean source checkouts to match the plan.

When admission succeeds, the canonical plan SHA-256 is persisted under `research_experiment_plan_sha256` in the paired report's runtime provenance. The preflight-evidence SHA remains separately persisted, so the measured artifact retains both the readiness identity and the frozen protocol identity.

The generic paired CLI cannot inspect arbitrary model or decoding behavior hidden inside user policy factories. `parameters` and `model_identity` are therefore retained as reviewable protocol declarations, but their semantic enforcement remains the responsibility of the concrete policy/model integration. Do not interpret the plan digest as proof that opaque factory internals used those values.

## Evidence boundary

The frozen plan is a **pre-measurement configuration identity**, not scientific benchmark evidence by itself. Its digest does not prove that an upstream checkout contained scientifically correct code, that a benchmark implementation is valid, or that an effectiveness claim is supported. Controlled preflight, exact source-checkout admission, measured artifact verification, paired analysis, and retained research-evidence indexing remain separate contracts.

For controlled measurement, retain the frozen plan, readiness evidence, paired report, integrity manifest, verification outputs, and final research-evidence record together. A matching plan and green engineering gates establish reproducible execution constraints; they do not promote an experiment to E3 without genuine benchmark measurements and the required statistical analysis.
