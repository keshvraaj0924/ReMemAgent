# Strict paired reproducibility

`remem-paired-benchmark` supports an opt-in `--strict-reproducibility` gate for baseline-versus-treatment experiments that are intended to become scientific evidence.

The normal CLI remains available for development runs, smoke checks, adapter validation, and exploratory experiments. Strict mode exists so a research run cannot accidentally start with an under-specified reproducibility contract.

Before any external environment is measured, strict mode requires all of the following:

- at least two independent seeds;
- an exact ReMemAgent code revision through `--require-code-revision`;
- `--require-clean-working-tree`;
- at least one exact installed dependency pin through `--require-dependency-version PACKAGE==VERSION`;
- at least one named external source checkout with an exact revision;
- no dirty-source exception for any declared source checkout;
- an explicit `--manifest` destination for exact-byte artifact verification.

The gate validates the declaration only. It does not claim that those requirements are satisfied. The existing controlled preflight still collects runtime provenance and external Git state immediately before measurement and fails closed when the observed machine state differs from the declaration.

A strict run therefore follows this order:

```text
parse experiment contract
        |
        v
strict declaration gate
        |
        v
collect + validate runtime/source provenance
        |
        v
runtime environment preflight
        |
        v
paired measured execution
        |
        v
identity-bound artifact persistence
        |
        v
exact-byte integrity manifest
```

Example shape:

```bash
remem-paired-benchmark \
  --benchmark webshop \
  --episodes 100 \
  --max-steps 50 \
  --seeds 11,17,29,43,71 \
  --environment-factory your_package.environments:build_webshop \
  --success-evaluator your_package.metrics:is_success \
  --baseline-policy-factory your_package.policies:build_baseline \
  --treatment-policy-factory your_package.policies:build_remem \
  --strict-reproducibility \
  --require-code-revision <REMEM_COMMIT> \
  --require-clean-working-tree \
  --require-dependency-version torch==<PINNED_VERSION> \
  --source-checkout webshop=/path/to/webshop \
  --require-source-revision webshop=<WEBSHOP_COMMIT> \
  --output artifacts/webshop-paired.json \
  --manifest artifacts/webshop-paired.json.manifest.json
```

The placeholders are deliberate. ReMemAgent does not invent upstream revisions, model versions, or measured results. Those values must come from the execution environment selected for the actual experiment.

Strict mode is a reproducibility safeguard, not a benchmark-quality claim. A successful strict run still needs the resulting artifact and manifest to be preserved and verified, and any scientific conclusion must be based on the measured outputs rather than on the existence of the gate itself.
