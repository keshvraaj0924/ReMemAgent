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

## Evidence boundary

This layer provides real source-state collection and validation, but it is not yet automatically threaded through the paired benchmark CLI or persisted into the identity-bound runtime contract. Until that wiring is added, callers must invoke it explicitly before measurement and must not treat source-checkout validation as part of the persisted experiment identity.

The next integration step is to bind validated source-checkout requirements and observations into controlled preflight and artifact identity without breaking verification of existing runtime-requirement artifacts.
