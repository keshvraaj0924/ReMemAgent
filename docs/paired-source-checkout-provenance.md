# Paired source-checkout provenance

Controlled paired ALFWorld/WebShop experiments can depend on benchmark code that is executed directly from a Git checkout. Python package metadata alone cannot prove which external source revision was measured.

`experiments.paired_source_preflight.run_controlled_paired_external_benchmarks` provides a fail-closed admission boundary for that case. Before either paired condition resolves benchmark callables, constructs an environment, or starts measured execution, it:

1. collects the ReMemAgent runtime snapshot and validates the declared `RuntimeRequirements`;
2. collects each declared external source checkout's Git revision and working-tree state;
3. validates those observations against `SourceCheckoutRequirement` objects;
4. runs the normal paired environment/policy preflight without recollecting runtime state;
5. executes the paired measurement only after every admission check succeeds.

The returned `ControlledPairedBenchmarkResult` carries the exact `RuntimeProvenance` and immutable source-checkout provenance mapping that passed admission. This avoids recollecting mutable Git state after a benchmark has already finished.

## Python API example

Use the actual commit chosen for the controlled experiment. The placeholders below are intentionally not recommended or fabricated revisions.

```python
from pathlib import Path

from experiments.paired_artifacts import save_paired_execution_result
from experiments.paired_source_preflight import run_controlled_paired_external_benchmarks
from experiments.runtime_requirements import RuntimeRequirements
from experiments.source_checkouts import SourceCheckoutRequirement

runtime_requirements = RuntimeRequirements(
    expected_code_revision="<REMEMAGENT_COMMIT>",
    require_clean_working_tree=True,
)
source_requirements = {
    "webshop": SourceCheckoutRequirement(
        expected_revision="<WEBSHOP_COMMIT>",
        require_clean_working_tree=True,
    )
}

controlled = run_controlled_paired_external_benchmarks(
    baseline_spec,
    treatment_spec,
    seeds=(11, 17, 23),
    runtime_requirements=runtime_requirements,
    source_checkout_paths={
        "webshop": Path("/path/to/webshop"),
    },
    source_checkout_requirements=source_requirements,
)

save_paired_execution_result(
    controlled.paired_result,
    output_path,
    runtime_provenance=controlled.runtime_provenance.to_dict(),
    runtime_requirements=runtime_requirements,
    source_checkout_provenance=controlled.source_checkout_provenance,
    source_checkout_requirements=source_requirements,
)
```

A revision mismatch, dirty checkout when cleanliness is required, unknown Git state, missing source checkout, or runtime-contract mismatch prevents paired preflight and measurement from starting.

## Paired CLI source contracts

`remem-paired-benchmark` exposes the same controlled source admission boundary. Each external checkout must be declared with both a path and an exact Git revision:

```bash
remem-paired-benchmark \
  --benchmark webshop \
  --episodes 10 \
  --max-steps 30 \
  --seeds 11,17,23 \
  --environment-factory your_package.factories:make_webshop_environment \
  --success-evaluator your_package.evaluation:evaluate_webshop_success \
  --baseline-policy-factory your_package.policies:make_baseline_policy \
  --treatment-policy-factory your_package.policies:make_memory_policy \
  --source-checkout webshop=/path/to/webshop \
  --require-source-revision webshop=<WEBSHOP_COMMIT> \
  --require-code-revision <REMEMAGENT_COMMIT> \
  --require-clean-working-tree
```

`--source-checkout NAME=PATH` and `--require-source-revision NAME=REVISION` are repeatable and must declare the same case-insensitive name set. Source checkouts are required to be clean by default. `--allow-dirty-source-checkout NAME` is an explicit opt-out for a named source and should not be used for publication-grade measurements unless the local modification is independently captured and justified.

When any source checkout contract is declared, the CLI automatically uses `run_controlled_paired_external_benchmarks`. The exact source snapshot admitted before preflight is passed directly to `save_paired_execution_result`; Git state is not recollected after measurement. The legacy paired runner remains unchanged when no source checkout contract is supplied.

## Identity binding and verification

`experiments.source_checkouts` has canonical, schema-versioned serializers and deterministic SHA-256 fingerprints for both the declared source admission contract and the observed Git snapshot. Repository names are normalized case-insensitively for canonical identity while the validation boundary rejects ambiguous duplicates.

`save_paired_execution_result()` persists the full source requirement contract and observed checkout snapshot. Before the lower-level paired serializer constructs experiment identity, it injects both fingerprints into runtime provenance. Because paired identity already covers runtime provenance, changing either the required external revision/cleanliness policy or the observed source state changes the experiment identity.

Artifact verification reconstructs both persisted source mappings, re-validates the observed snapshot against the declared requirements, recomputes both SHA-256 fingerprints, and rejects missing, malformed, stale, or tampered evidence. Legacy artifacts without source evidence remain readable but do not gain this guarantee retroactively.

## Evidence boundary

The controlled Python API and `remem-paired-benchmark` can now produce paired artifacts cryptographically bound to both the admitted ReMemAgent runtime contract and named external source checkout contracts/snapshots.

This is a reproducibility guarantee, not a benchmark-effectiveness claim. Real ALFWorld/WebShop evidence still requires explicitly selected upstream revisions, real environment dependencies, a concrete model/policy runtime, and successful multi-seed execution. No upstream revision or benchmark result is inferred or fabricated by this framework.
