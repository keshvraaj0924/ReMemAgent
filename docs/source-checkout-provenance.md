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

`experiments.controlled_external_benchmark` is the evidence-preserving execution boundary. It always captures the runtime snapshot before benchmark side effects, validates runtime requirements when supplied, optionally validates a complete source-checkout contract, and returns the exact admitted snapshots together with the measured report or reports. Runtime-only and source-only controlled executions therefore do not need to recollect mutable state after measurement.

## Command-line admission

Both external benchmark entry points expose the source checkout contract directly. `remem-benchmark` and `remem-paired-benchmark` accept:

- `--source-checkout NAME=PATH` to identify a source-installed benchmark repository;
- `--require-source-revision NAME=REVISION` to require the exact Git revision;
- `--allow-dirty-source-checkout NAME` to opt a declared checkout out of the default clean-tree requirement.

The source path and revision sets must declare the same names, compared case-insensitively. A source contract automatically selects controlled runtime preflight for single-run, repeated, and paired measured execution, so a caller cannot accidentally request pinned source evidence while bypassing admission. Callable-only `--preflight` remains intentionally insufficient because it does not inspect Git state.

Example:

```bash
remem-benchmark \
  --benchmark webshop \
  --episodes 20 \
  --max-steps 40 \
  --seeds 11,17,23 \
  --environment-factory package.module:make_webshop_environment \
  --policy-factory package.module:make_policy \
  --success-evaluator package.module:is_success \
  --source-checkout WebShop=/opt/benchmarks/webshop \
  --require-source-revision WebShop=<exact-webshop-git-sha> \
  --output artifacts/webshop.json
```

The revision remains an explicit experiment input. ReMemAgent does not substitute a guessed upstream commit.

## Artifact evidence

Controlled paired execution carries the exact admitted source snapshot through measurement. `save_paired_execution_result()` accepts that snapshot together with the source requirement contract, validates the pair again, persists both canonical forms, and injects their SHA-256 fingerprints into runtime provenance before paired experiment identity is constructed.

Standalone controlled execution has the equivalent persistence boundary in `experiments.controlled_benchmark_artifacts`. `save_controlled_benchmark_result()` and `save_controlled_repeated_benchmark_result()` accept the exact snapshots returned by controlled execution, validate source state against the declared contract again, inject runtime/source evidence fingerprints before ordinary experiment identity is constructed, and persist the canonical contracts and snapshots alongside the report.

`remem-benchmark` now routes any measured run that declares runtime requirements or a source-checkout contract through `experiments.controlled_external_benchmark` and then through the corresponding controlled artifact writer. The CLI therefore persists the exact pre-measurement runtime/source evidence admitted for that run instead of collecting mutable Git or package state again after measurement. Repeated controlled runs use the same admission snapshot for all requested seeds while statistics are computed from the measured reports returned by that controlled execution.

`validate_persisted_controlled_benchmark_artifact()` checks the same runtime/source evidence during artifact verification. Tampering with a required revision, cleanliness policy, observed revision, working-tree state, or runtime requirement digest is therefore detectable for artifacts created through the controlled persistence path.

## Evidence boundary

The standalone and paired controlled CLI paths now implement the complete admission → measurement → identity-bound persistence handoff for declared runtime/source contracts. This is a reproducibility and integrity guarantee only. It does not establish that ReMemAgent improves ALFWorld or WebShop performance, and it does not establish production readiness.

Real effectiveness evidence still requires explicitly pinned upstream benchmark revisions, a fixed model/policy runtime, matched multi-seed baseline-versus-memory runs, and the corresponding statistical analysis. ReMemAgent intentionally does not invent those revisions or benchmark results.