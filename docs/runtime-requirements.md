# Runtime requirements for controlled measurement

Runtime provenance records the environment that executed a benchmark, but recording drift after measurement is weaker than preventing the wrong runtime from measuring in the first place.

`experiments.runtime_requirements` adds a fail-closed validation boundary for controlled external experiments. `RuntimeRequirements` can require:

- an exact Git/code revision;
- a clean working tree;
- exact installed versions for selected benchmark, model-runtime, tokenizer, or inference packages.

`validate_runtime_requirements(provenance, requirements)` compares those requirements against a collected `RuntimeProvenance` instance before benchmark construction. Dependency names are matched case-insensitively after trimming surrounding whitespace, while versions use exact string equality.

Exact version equality is intentional. This layer is for reproducible measurement rather than dependency resolution, so a broad compatible-version range would still permit an experiment to run under a different runtime than the declared protocol.

A clean-worktree requirement rejects both `dirty` and `unknown` states. Likewise, an expected code revision rejects `unknown` or any other revision. Missing required dependencies and dependency version drift fail before measured execution when callers place this validation ahead of environment/model construction.

Example:

```python
from experiments.runtime_provenance import collect_runtime_provenance
from experiments.runtime_requirements import RuntimeRequirements, validate_runtime_requirements

provenance = collect_runtime_provenance()
requirements = RuntimeRequirements(
    expected_code_revision="<full-commit-sha>",
    require_clean_working_tree=True,
    dependency_versions={
        "alfworld": "<pinned-version>",
        "transformers": "<pinned-version>",
    },
)
validate_runtime_requirements(provenance, requirements)
```

The version strings above are deliberately placeholders in documentation. ReMemAgent does not invent or prescribe benchmark/model package versions; the experiment owner must pin the versions actually selected for the controlled run.

This validation is an engineering reproducibility gate, not evidence of benchmark effectiveness. Real ALFWorld/WebShop results still require measured multi-seed execution with the exact declared environment and model-policy runtime.
