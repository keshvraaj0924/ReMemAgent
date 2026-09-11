# Runtime requirements for controlled measurement

Runtime provenance records the environment that executed a benchmark, but recording drift after measurement is weaker than preventing the wrong runtime from measuring in the first place.

`experiments.runtime_requirements` adds a fail-closed validation boundary for controlled external experiments. `RuntimeRequirements` can require:

- an exact Git/code revision;
- a clean working tree;
- exact installed versions for selected benchmark, model-runtime, tokenizer, or inference packages.

`RuntimeRequirements` detaches, normalizes, and freezes its dependency-version mapping at construction. A caller cannot change a validated runtime contract later by mutating either the original input dictionary or the mapping exposed by the requirements object.

`validate_runtime_requirements(provenance, requirements)` compares those requirements against a collected `RuntimeProvenance` instance. Dependency names are matched case-insensitively after trimming surrounding whitespace, while versions use exact string equality.

Exact version equality is intentional. This layer is for reproducible measurement rather than dependency resolution, so a broad compatible-version range would still permit an experiment to run under a different runtime than the declared protocol.

A clean-worktree requirement rejects both `dirty` and `unknown` states. Likewise, an expected code revision rejects `unknown` or any other revision. Missing required dependencies and dependency version drift fail closed.

## Benchmark CLI

`remem-benchmark` exposes the same fail-closed contract directly. Controlled runs can declare:

- `--require-code-revision <full-commit-sha>`;
- `--require-clean-working-tree`;
- repeated `--require-dependency-version PACKAGE==VERSION` pins.

Declaring any runtime requirement on a measured run automatically routes execution through controlled runtime preflight before measurement starts; `--preflight-before-run` is not additionally required. Runtime-only probes (`--runtime-preflight` and `--repeated-runtime-preflight`) enforce the same contract. Callable-only `--preflight` rejects runtime requirements because it intentionally does not construct or inspect the benchmark runtime.

Example:

```bash
remem-benchmark \
  --benchmark alfworld \
  --episodes 20 \
  --max-steps 50 \
  --seed 100 \
  --environment-factory my_runtime:make_alfworld_environment \
  --policy-factory my_runtime:make_policy \
  --success-evaluator my_runtime:evaluate_success \
  --require-code-revision <full-commit-sha> \
  --require-clean-working-tree \
  --require-dependency-version alfworld==<pinned-version> \
  --require-dependency-version transformers==<pinned-version> \
  --output artifacts/alfworld-seed-100.json
```

The version placeholders are intentional. ReMemAgent does not fabricate or prescribe benchmark/model package versions; experiment owners must pin the versions actually selected for their controlled runtime.

## Single external benchmark preflight

`validate_controlled_external_benchmark_runtime(...)` and `run_external_benchmark_with_preflight(...)` apply the runtime requirement gate to one external benchmark run. Runtime provenance is collected and validated before the configured ALFWorld/WebShop environment is resolved or constructed. A failed revision, working-tree, or dependency check therefore cannot create benchmark probe side effects or start measured execution.

Example:

```python
from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.external_preflight import run_external_benchmark_with_preflight
from experiments.runtime_requirements import RuntimeRequirements

spec = ExternalBenchmarkSpec(
    benchmark_name="alfworld",
    episode_count=20,
    max_steps=50,
    environment_factory="my_runtime:make_alfworld_environment",
    policy_factory="my_runtime:make_policy",
    success_evaluator="my_runtime:evaluate_success",
    seed=100,
)
requirements = RuntimeRequirements(
    expected_code_revision="<full-commit-sha>",
    require_clean_working_tree=True,
    dependency_versions={
        "alfworld": "<pinned-version>",
        "transformers": "<pinned-version>",
    },
)

report = run_external_benchmark_with_preflight(
    spec,
    runtime_requirements=requirements,
)
```

## Repeated external benchmark preflight

`validate_repeated_external_benchmark_runtime(...)` and `run_repeated_external_benchmarks_with_preflight(...)` accept the same optional `runtime_requirements` argument. When supplied, ReMemAgent collects runtime provenance and validates the declared requirements **before constructing the first ALFWorld/WebShop environment**.

This ordering matters for controlled measurement: a wrong commit, dirty checkout, missing benchmark package, or dependency-version drift cannot create probe environments or start measured runs. Runtime validation is performed once for the repeated preflight, then each independent seed is probed through the existing environment and policy contract boundary.

Example:

```python
from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.external_preflight import run_repeated_external_benchmarks_with_preflight
from experiments.runtime_requirements import RuntimeRequirements

spec = ExternalBenchmarkSpec(
    benchmark_name="alfworld",
    episode_count=20,
    max_steps=50,
    environment_factory="my_runtime:make_alfworld_environment",
    policy_factory="my_runtime:make_policy",
    success_evaluator="my_runtime:evaluate_success",
)
requirements = RuntimeRequirements(
    expected_code_revision="<full-commit-sha>",
    require_clean_working_tree=True,
    dependency_versions={
        "alfworld": "<pinned-version>",
        "transformers": "<pinned-version>",
    },
)

reports = run_repeated_external_benchmarks_with_preflight(
    spec,
    seeds=(100, 200, 300),
    runtime_requirements=requirements,
)
```

The runtime requirement gate protects the benchmark CLI plus single-run and repeated external preflight orchestration through `experiments.external_preflight`. Direct calls to the lower-level execution functions in `experiments.external_benchmark` remain caller-owned and intentionally do not collect runtime provenance implicitly.

This validation is an engineering reproducibility gate, not evidence of benchmark effectiveness. Real ALFWorld/WebShop results still require measured multi-seed execution with the exact declared environment and model-policy runtime.
