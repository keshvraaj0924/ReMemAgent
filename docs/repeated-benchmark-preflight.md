# Repeated external benchmark preflight

`experiments.external_preflight.validate_repeated_external_benchmark_runtime`
provides a fail-fast gate for multi-seed ALFWorld/WebShop launches.

## Contract

The helper accepts an `ExternalBenchmarkSpec` and an ordered sequence of
independent seeds. It rejects an empty seed sequence and duplicate seeds, then
runs the existing runtime preflight once per seed.

Each probe therefore uses the same environment adapter and policy boundary as a
measured run. The configured policy factory receives the seed used by that
probe, so seed-dependent checkpoint or model initialization failures can be
caught before the experiment begins.

Every probe environment is closed by the existing runtime-preflight lifecycle.
Probe results are returned as `EnvironmentContractReport` values; they are not
benchmark reports and must not be included as scientific measurements.

## CLI entry points

A preflight-only command is available without importing experiment modules:

```text
python -m experiments.benchmark_cli \
  --benchmark alfworld \
  --episodes 1 \
  --max-steps 1 \
  --seeds 11,17,23 \
  --environment-factory package.module:make_environment \
  --policy-factory package.module:make_policy \
  --success-evaluator package.module:is_success \
  --repeated-runtime-preflight
```

For a measured multi-seed launch, `--preflight-before-run` turns the same
runtime checks into a fail-fast launch gate:

```text
python -m experiments.benchmark_cli \
  --benchmark alfworld \
  --episodes 50 \
  --max-steps 20 \
  --seeds 11,17,23 \
  --environment-factory package.module:make_environment \
  --action-policy-factory package.module:make_policy \
  --success-evaluator package.module:is_success \
  --probe-action "look" \
  --preflight-before-run \
  --output artifacts/alfworld.json \
  --manifest artifacts/alfworld.json.manifest.json
```

The launch gate performs the probes first and only starts measured execution if
every seed passes. Probe environments and memory stores are isolated from the
measured runs, and probe results are never written into benchmark artifacts.

## Intended launch order

1. Resolve the configured callables with `validate_external_benchmark`.
2. Run `validate_repeated_external_benchmark_runtime` for every requested seed,
   either explicitly or through `--preflight-before-run`.
3. Launch `run_repeated_external_benchmarks` with the same seed sequence.
4. Persist and integrity-check the measured reports.
5. Perform statistical analysis only after the measured artifacts are available.

This separates integration readiness from benchmark evidence while avoiding a
single-seed preflight that can miss seed-specific integration failures.

`--preflight-before-run` is intentionally opt-in because runtime preflight may
load a model or checkpoint a second time. It should be enabled for expensive or
publication-bound runs where fail-fast integration validation is worth that
startup cost.
