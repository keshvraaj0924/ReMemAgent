# Paired source-checkout provenance

Controlled paired ALFWorld/WebShop experiments can depend on benchmark code that is executed directly from a Git checkout. Python package metadata alone cannot prove which external source revision was measured.

`experiments.paired_source_preflight.run_controlled_paired_external_benchmarks` provides a fail-closed admission boundary for that case. Before either paired condition resolves benchmark callables, constructs an environment, or starts measured execution, it:

1. collects the ReMemAgent runtime snapshot and validates the declared `RuntimeRequirements`;
2. collects each declared external source checkout's Git revision and working-tree state;
3. validates those observations against `SourceCheckoutRequirement` objects;
4. runs the normal paired environment/policy preflight without recollecting runtime state;
5. executes the paired measurement only after every admission check succeeds.

The returned `ControlledPairedBenchmarkResult` carries the exact `RuntimeProvenance` and immutable source-checkout provenance mapping that passed admission. This avoids recollecting mutable Git state after a benchmark has already finished.

## Example

Use the actual commit chosen for the controlled experiment. The placeholder below is intentionally not a recommended or fabricated WebShop revision.

```python
from pathlib import Path

from experiments.paired_source_preflight import run_controlled_paired_external_benchmarks
from experiments.runtime_requirements import RuntimeRequirements
from experiments.source_checkouts import SourceCheckoutRequirement

result = run_controlled_paired_external_benchmarks(
    baseline_spec,
    treatment_spec,
    seeds=(11, 17, 23),
    runtime_requirements=RuntimeRequirements(
        expected_code_revision="<REMEMAGENT_COMMIT>",
        require_clean_working_tree=True,
    ),
    source_checkout_paths={
        "webshop": Path("/path/to/webshop"),
    },
    source_checkout_requirements={
        "webshop": SourceCheckoutRequirement(
            expected_revision="<WEBSHOP_COMMIT>",
            require_clean_working_tree=True,
        ),
    },
)
```

A revision mismatch, dirty checkout when cleanliness is required, unknown Git state, missing source checkout, or runtime-contract mismatch prevents paired preflight and measurement from starting.

## Evidence boundary

This increment establishes validated pre-measurement source provenance and exact snapshot handoff. It does **not** yet make source-checkout state part of the persisted paired artifact identity, and `remem-paired-benchmark` does not yet expose source-checkout contract flags. Until those two pieces are implemented, callers using the controlled Python API must persist the returned source-checkout snapshot separately and must not describe the paired artifact as cryptographically bound to the external benchmark revision.

The next reproducibility milestone is to add a canonical source-checkout contract/snapshot fingerprint, bind it into paired artifact identity, validate it on artifact load, and then expose the same controlled contract through the paired CLI before running real multi-seed ALFWorld/WebShop measurements.
