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

## Identity binding and verification

`experiments.source_checkouts` now has canonical, schema-versioned serializers and deterministic SHA-256 fingerprints for both the declared source admission contract and the observed Git snapshot. Repository names are normalized case-insensitively for canonical identity while the validation boundary still rejects ambiguous duplicates.

`save_paired_execution_result()` persists the full source requirement contract and observed checkout snapshot. Before the lower-level paired serializer constructs experiment identity, it injects both fingerprints into runtime provenance. Because paired identity already covers runtime provenance, changing either the required external revision/cleanliness policy or the observed source state changes the experiment identity.

Artifact verification reconstructs both persisted source mappings, re-validates the observed snapshot against the declared requirements, recomputes both SHA-256 fingerprints, and rejects missing, malformed, stale, or tampered evidence. Legacy artifacts without source evidence remain readable but do not gain this guarantee retroactively.

## Remaining boundary

The controlled Python API can now produce paired artifacts cryptographically bound to the admitted external source contract and snapshot. `remem-paired-benchmark` does **not yet** expose source-checkout path/revision flags or route them through this controlled source handoff automatically.

The next reproducibility milestone is therefore CLI integration: parse explicit source checkout paths and revisions, execute through the controlled paired admission path, and pass the exact admitted source snapshot and requirement contract into artifact persistence. Only after that CLI path is tested should real multi-seed ALFWorld/WebShop measurements be treated as fully reproducible command-line evidence.
