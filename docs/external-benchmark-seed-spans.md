# External benchmark seed-span validation

ReMemAgent's supported ALFWorld and WebShop execution bridges rely on legacy NumPy-compatible episode seeds. The accepted seed domain is therefore `0` through `2**32 - 1` (`4294967295`), inclusive.

A benchmark run does not use only the configured run seed. `BenchmarkSuiteRunner` derives one episode seed per episode as:

```text
episode_seed = run_seed + episode_index
```

For that reason, validating only the initial run seed is insufficient. A configuration such as `run_seed=4294967294` with three episodes would begin with a valid seed and later derive `4294967296`, which is outside the supported RNG domain.

`ExternalBenchmarkSpec` now validates the complete derived episode seed span before a single-run experiment can execute. Repeated experiments perform the same span validation for every independent run seed before constructing any benchmark environment. This fail-closed behavior prevents partially measured runs caused solely by a predictable seed overflow.

Repeated-run overlap validation remains a separate invariant: independent run seed spans must both fit the supported domain and remain disjoint from one another.

This validation does not claim complete benchmark determinism. Dataset/index state, upstream package versions, model runtime behavior, hardware/native kernels, and other external inputs still need to be pinned and recorded for controlled ALFWorld/WebShop experiments.
