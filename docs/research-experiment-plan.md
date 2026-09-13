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

## Evidence boundary

The frozen plan is a **pre-measurement configuration identity**, not benchmark evidence. Its digest does not prove that a runner actually used the declaration, that an upstream checkout contained the expected code, or that a scientific claim is valid. Controlled preflight, exact source-checkout admission, measured artifact verification, paired analysis, and retained research-evidence indexing remain separate contracts.

The next integration step is to bind `ResearchExperimentPlan` directly into `remem-paired-benchmark` admission so measured execution fails closed when CLI/runtime configuration differs from the frozen plan. Until that binding exists, retain the plan beside the exact command and readiness evidence and treat it as a reviewable declaration rather than execution attestation.
