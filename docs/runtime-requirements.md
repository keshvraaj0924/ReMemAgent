# Runtime requirements for controlled measurement

Runtime provenance records the environment that executed a benchmark, but recording drift after measurement is weaker than preventing the wrong runtime from measuring in the first place.

`experiments.runtime_requirements` adds a fail-closed validation boundary for controlled external experiments. `RuntimeRequirements` can require:

- an exact Git/code revision;
- a clean working tree;
- exact installed versions for selected benchmark, model-runtime, tokenizer, or inference packages.

`validate_runtime_requirements(provenance, requirements)` compares those requirements against a collected `RuntimeProvenance` instance. Dependency names are matched case-insensitively after trimming surrounding whitespace, while versions use exact string equality.

Exact version equality is intentional. This layer is for reproducible measurement rather than dependency resolution, so a broad compatible-version range would still permit an experiment to run under a different runtime than the declared protocol.

A clean-worktree requirement rejects both `dirty` and `unknown` states. Likewise, an expected code revision rejects `unknown` or any other revision. Missing required dependencies and dependency version drift fail closed.

## Repeated external benchmark preflight

`validate_repeated_external_benchmark_runtime(...)` and `run_repeated_external_benchmarks_with_preflight(...)` now accept an optional `runtime_requirements` argument. When supplied, ReMemAgent collects runtime provenance and validates the declared requirements **before constructing the first ALFWorld/WebShop environment**.

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

The version strings above are deliberately placeholders in documentation. ReMemAgent does not invent or prescribe benchmark/model package versions; the experiment owner must pin the versions actually selected for the controlled run.

The runtime requirement gate currently protects the repeated external preflight execution path. Direct low-level calls that bypass `experiments.external_preflight` remain caller-owned and must validate runtime requirements explicitly before constructing benchmark environments.

This validation is an engineering reproducibility gate, not evidence of benchmark effectiveness. Real ALFWorld/WebShop results still require measured multi-seed execution with the exact declared environment and model-policy runtime.