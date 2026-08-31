# Reproducible experiments

ReMemAgent separates experiment configuration, seeded case generation, execution, and result persistence so a research run can be reconstructed without relying on process-global random state.

## Single run

`experiments.runner.run_reproducible_ablation()` creates a dedicated `random.Random` instance from `ExperimentConfig.seed`, generates the matched cases, records case and experiment fingerprints, and returns an immutable `ExperimentReport`.

```python
from experiments.runner import ExperimentConfig, run_reproducible_ablation

report = run_reproducible_ablation(make_cases, ExperimentConfig(seed=17))
```

Use `save_report()` to persist the report as deterministic JSON.

## Repeated seeds

`run_repeated_ablations()` executes the same protocol independently for an explicit sequence of unique seeds. Each report retains its own case fingerprint and experiment fingerprint, which prevents an aggregate result from hiding the provenance of individual runs.

```python
from experiments.runner import run_repeated_ablations, save_repeated_reports

reports = run_repeated_ablations(make_cases, [17, 23, 41, 59, 73])
save_repeated_reports(reports, "results/multi_seed.json")
```

The helper rejects an empty or duplicated seed list. `save_repeated_reports()` also rejects duplicate report seeds.

## Evidence boundary

The multi-seed runner establishes deterministic execution and traceable serialization; it does **not** establish statistical significance or benchmark improvement. Research conclusions still require real benchmark environments, declared model/configuration versions, repeated executions, and appropriate statistical analysis.

## Required provenance

A publishable result should retain at least:

- experiment configuration and seed set;
- code revision;
- benchmark/environment version;
- model/checkpoint identifier;
- serialized per-run metrics;
- case/input fingerprint;
- execution environment and dependency versions.

ReMemAgent's serializers deliberately do not infer missing provenance or manufacture benchmark results.
