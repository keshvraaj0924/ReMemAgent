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

## Intended launch order

1. Resolve the configured callables with `validate_external_benchmark`.
2. Run `validate_repeated_external_benchmark_runtime` for every requested seed.
3. Launch `run_repeated_external_benchmarks` with the same seed sequence.
4. Persist and integrity-check the measured reports.
5. Perform statistical analysis only after the measured artifacts are available.

This separates integration readiness from benchmark evidence while avoiding a
single-seed preflight that can miss seed-specific integration failures.
