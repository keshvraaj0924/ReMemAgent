# Source checkout provenance

Some research dependencies are installed from source or executed directly from a Git checkout. WebShop is the clearest example in the current external benchmark path. A Python package version alone cannot prove which source revision was measured.

`experiments.source_checkouts` adds a small fail-closed provenance boundary for these dependencies.

## What it records

`collect_source_checkout_provenance()` accepts a mapping from a stable dependency name to a `pathlib.Path` and records, for each checkout:

- the exact `git rev-parse HEAD` revision;
- whether `git status --porcelain` reports the working tree as `clean` or `dirty`;
- `unknown` explicitly when Git state cannot be established.

The returned mapping is detached, deterministically ordered by dependency name, and immutable.

## Requirement validation

`SourceCheckoutRequirement` declares an exact expected revision and whether the working tree must be clean. `validate_source_checkout_requirements()` compares collected state against those requirements case-insensitively by dependency name and fails when:

- a required checkout was not collected;
- the observed Git revision differs;
- cleanliness is required but the checkout is dirty or unknown;
- duplicate dependency names become ambiguous after whitespace/case normalization.

Example:

```python
from pathlib import Path

from experiments.source_checkouts import (
    SourceCheckoutRequirement,
    collect_source_checkout_provenance,
    validate_source_checkout_requirements,
)

observed = collect_source_checkout_provenance({"WebShop": Path("/opt/benchmarks/webshop")})
validate_source_checkout_requirements(
    observed,
    {
        "WebShop": SourceCheckoutRequirement(
            expected_revision="<exact-webshop-git-sha>",
            require_clean_working_tree=True,
        )
    },
)
```

The placeholder revision above is intentional; ReMemAgent does not invent or bless a WebShop revision on behalf of an experiment.

## Canonical persistence and fingerprints

Source requirement contracts and observed checkout snapshots now have schema-versioned canonical JSON representations. Repository names are stripped, normalized case-insensitively, and sorted before serialization. Deterministic SHA-256 helpers are available for both the declared requirement contract and the observed source snapshot.

Persisted forms can be reconstructed into immutable mappings. Unknown schemas, malformed entries, empty repository sets, duplicate normalized names, and invalid checkout states fail closed rather than being silently accepted.

## Controlled external preflight

Single-run and repeated external preflight accept source-checkout paths and requirements as one admission contract. Both mappings must be supplied together. ReMemAgent collects and validates the declared checkout state before constructing the first third-party benchmark environment, so revision or cleanliness drift cannot create probe or measured-execution side effects.

```python
from pathlib import Path

from experiments.external_preflight import run_repeated_external_benchmarks_with_preflight
from experiments.source_checkouts import SourceCheckoutRequirement

reports = run_repeated_external_benchmarks_with_preflight(
    spec,
    seeds=(11, 17, 23),
    source_checkout_paths={"WebShop": Path("/opt/benchmarks/webshop")},
    source_checkout_requirements={
        "WebShop": SourceCheckoutRequirement(
            expected_revision="<exact-webshop-git-sha>",
            require_clean_working_tree=True,
        )
    },
)
```

Source-checkout collection occurs once before repeated environment probes; it is not repeated independently for every seed.

## Paired artifact evidence

Controlled paired execution can carry the exact admitted source snapshot through measurement. `save_paired_execution_result()` accepts that snapshot together with the source requirement contract, validates the pair again, persists both canonical forms, and injects their SHA-256 fingerprints into runtime provenance before paired experiment identity is constructed.

Artifact verification reconstructs the contract and snapshot, verifies that the observed Git state still satisfies the persisted requirements, and checks both fingerprints. Tampering with the required revision, cleanliness policy, observed revision, or working-tree state is therefore detectable and also changes experiment identity when artifacts are created correctly.

## Remaining boundary

Source state is now enforced during controlled Python preflight and can be cryptographically bound into paired artifacts. The remaining reproducibility gap is command-line orchestration: `remem-paired-benchmark` does not yet expose source-checkout path/revision options or automatically pass the admitted snapshot to persistence.

The next integration step is to wire those source controls through the paired CLI and cover the end-to-end handoff before running real multi-seed ALFWorld/WebShop measurements as reproducible evidence.
